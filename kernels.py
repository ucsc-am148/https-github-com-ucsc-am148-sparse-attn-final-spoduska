"""STUDENT FILE: implement the three block-sparse rung functions.

Implement these three functions from the spec in ALGORITHMS.md -- no reference
code is shipped:

    dsd_matmul             (A1) block-sparse (BCSR) A @ dense B -> dense C
    sparse_flash_forward   (A2) block-sparse flash attention forward
    sparse_flash_backward  (A3) block-sparse flash attention backward

Your functions must match the signatures below: the SHAPES and DTYPES of the
inputs and outputs (each docstring states them; ALGORITHMS.md sec 0.1 collects
them). EVERYTHING ELSE IS YOURS -- how many @triton.jit kernels you write, the
grid, the (B, H) flatten, strides, output allocation, and the launch/tuning. The
grader asserts the returned shapes and dtypes, then checks correctness against an
fp64 reference.

ALGORITHMS.md is the complete spec: the BCSR layout and its two transpose views,
what each output equals, and the five backward equations.

When `python sanity_check.py` passes all three rungs, you're done.
"""
import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634

@triton.jit
def _dsd_kernel(
    values_ptr,         # (nnz, block, block) f32 -- A's packed live blocks
    row_offsets_ptr,    # (n+1,) i32
    col_indices_ptr,    # (nnz,) i32
    B_ptr,              # (K, N) f32
    C_ptr,              # (M, N) f32
    N,
    stride_vb, stride_vr, stride_vc,   # values: block, row, col
    stride_bk, stride_bn,             # B: row(K), col(N)
    stride_cm, stride_cn,             # C: row(M), col(N)
    block: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_KK: tl.constexpr,        # inner contraction tile (keeps shared mem bounded)
):
    pid_m = tl.program_id(0)    # which block-row i of A/C
    pid_n = tl.program_id(1)    # which N-tile of the output

    offs_m   = tl.arange(0, block)                      # output rows within this block-row
    offs_n   = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # this tile's output columns
    offs_kk  = tl.arange(0, BLOCK_KK)                   # within-chunk contraction index
    n_mask   = offs_n < N                               # guard ragged last N-tile

    acc = tl.zeros((block, BLOCK_N), dtype=tl.float32)  # fp32 accumulator

    # THE SPARSE K-LOOP: only this block-row's live blocks
    start = tl.load(row_offsets_ptr + pid_m)
    end   = tl.load(row_offsets_ptr + pid_m + 1)
    for idx in range(start, end):
        k = tl.load(col_indices_ptr + idx)  # logical K-block column

        # tile the block's inner dimension so only a slice is on-chip at a time
        for kk in range(0, block, BLOCK_KK):
            cur = kk + offs_kk

            # A slice = values[idx][:, cur] -> (block, BLOCK_KK)
            a_ptrs = (values_ptr + idx * stride_vb
                      + offs_m[:, None] * stride_vr
                      + cur[None, :] * stride_vc)
            a_sub = tl.load(a_ptrs)

            # B slice = rows [k*block + cur], columns offs_n -> (BLOCK_KK, BLOCK_N)
            rows_b = k * block + cur
            b_ptrs = (B_ptr + rows_b[:, None] * stride_bk
                      + offs_n[None, :] * stride_bn)
            b_sub = tl.load(b_ptrs, mask=n_mask[None, :], other=0.0)

            acc += tl.dot(a_sub, b_sub, allow_tf32=False)   # exact fp32, no TF32

    rows_c = pid_m * block + offs_m
    c_ptrs = (C_ptr + rows_c[:, None] * stride_cm
              + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=n_mask[None, :])

def dsd_matmul(values, row_offsets, column_indices, B, M, K, N, block):
    """A1 -- block-sparse C = A @ B. See ALGORITHMS.md sec 1-2.

    Inputs:
        values         (nnz, block, block)  fp32   A's live blocks, row-major
        row_offsets    (M//block + 1,)      int32  per block-row prefix sum of nnz
        column_indices (nnz,)               int32  K-block of each live block
        B              (K, N)               fp32   dense right operand
        M, K, N, block                      ints   dims and block size
    Returns:
        C              (M, N)               fp32

    fp32 throughout, allow_tf32=False.

    TODO: implement (done).
    """
    # raise NotImplementedError("TODO: implement dsd_matmul (A1)")

    C = torch.zeros((M, N), device=B.device, dtype=torch.float32)
    BLOCK_N = 64                       # tunable
    BLOCK_KK = min(block, 32)          # inner contraction tile -> bounded shared mem
    grid = (M // block, triton.cdiv(N, BLOCK_N))
    _dsd_kernel[grid](
        values, row_offsets, column_indices, B, C,
        N,
        values.stride(0), values.stride(1), values.stride(2),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        block=block,
        BLOCK_N=BLOCK_N,
        BLOCK_KK=BLOCK_KK,
        num_warps=8,
        num_stages=2,
    )
    return C

@triton.jit
def _fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    q_row_offsets_ptr, q_col_indices_ptr,
    qk_scale,                                  # = sm_scale * LOG2E  (base-2 fold)
    stride_qb, stride_qt, stride_qd,           # strides over (BH, T, d)
    stride_kb, stride_kt, stride_kd,
    stride_vb, stride_vt, stride_vd,
    stride_ob, stride_ot, stride_od,
    stride_lb, stride_lt,
    T,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, D: tl.constexpr,
):
    pid_q  = tl.program_id(0)    # which query block (like pid_m in A1)
    pid_bh = tl.program_id(1)    # which (b,h) head — the B·H flatten

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)    # this block's query rows
    offs_d = tl.arange(0, D)
    q_mask = offs_q < T

    # jump to this head's slice of each tensor
    Q_head = Q_ptr + pid_bh * stride_qb
    K_head = K_ptr + pid_bh * stride_kb
    V_head = V_ptr + pid_bh * stride_vb
    O_head = O_ptr + pid_bh * stride_ob

    # load the Q block ONCE; it stays on-chip for the whole loop
    q = tl.load(Q_head + offs_q[:, None] * stride_qt + offs_d[None, :] * stride_qd,
                mask=q_mask[:, None], other=0.0)            # (BLOCK_Q, D) f16

    # the three running totals (per query row)
    m   = tl.full((BLOCK_Q,), -float("inf"), tl.float32)    # running max
    l   = tl.zeros((BLOCK_Q,), tl.float32)                  # running denominator
    acc = tl.zeros((BLOCK_Q, D), tl.float32)                # running numerator (un-normalized O)

    # the BCSR sparse loop — identical shape to A1
    start = tl.load(q_row_offsets_ptr + pid_q)
    end   = tl.load(q_row_offsets_ptr + pid_q + 1)
    for idx in range(start, end):
        j = tl.load(q_col_indices_ptr + idx)                # which key block
        offs_k = j * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offs_k < T

        k = tl.load(K_head + offs_k[:, None] * stride_kt + offs_d[None, :] * stride_kd,
                    mask=k_mask[:, None], other=0.0)        # (BLOCK_K, D)
        v = tl.load(V_head + offs_k[:, None] * stride_vt + offs_d[None, :] * stride_vd,
                    mask=k_mask[:, None], other=0.0)        # (BLOCK_K, D)

        # scores in base-2 units:  σ·LOG2E·Q·Kᵀ
        s = tl.dot(q, tl.trans(k)) * qk_scale               # (BLOCK_Q, BLOCK_K) f32
        s = tl.where(k_mask[None, :], s, -float("inf"))     # dead keys -> -inf

        # --- the online-softmax update ---
        m_new = tl.maximum(m, tl.max(s, axis=1))
        alpha = tl.exp2(m - m_new)                          # correction for old totals
        p     = tl.exp2(s - m_new[:, None])                 # this block's weights
        l     = l * alpha + tl.sum(p, axis=1)
        acc   = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m     = m_new

    # finalize this query block
    acc   = acc / l[:, None]                                # numerator / denominator
    L_val = m + tl.log2(l)                                  # log of denominator (for A3)

    tl.store(O_head + offs_q[:, None] * stride_ot + offs_d[None, :] * stride_od,
             acc.to(O_ptr.dtype.element_ty), mask=q_mask[:, None])
    tl.store(L_ptr + pid_bh * stride_lb + offs_q * stride_lt,
             L_val, mask=q_mask)

def sparse_flash_forward(Q, K, V, q_row_offsets, q_col_indices,
                         sm_scale, BLOCK_Q, BLOCK_K):
    """A2 -- block-sparse flash attention forward. See ALGORITHMS.md sec 1, 3.

    Inputs:
        Q, K, V        (B, H, T, d)         fp16
        q_row_offsets  (T//block + 1,)      int32  query-block view: for query
        q_col_indices  (nnz,)               int32  block i, its live key blocks j
        sm_scale       float                       1/sqrt(d)
        BLOCK_Q, BLOCK_K  ints                     == block (the mask granularity)
    Returns:
        O              (B, H, T, d)         fp16
        L              (B, H, T)            fp32   log2 of the softmax denominator (sec 3)

    See ALGORITHMS.md sec 3 for O and L.

    TODO: implement.
    """
    # raise NotImplementedError("TODO: implement sparse_flash_forward (A2)")

    B, H, T, d = Q.shape
    Qf, Kf, Vf = (X.reshape(B * H, T, d) for X in (Q, K, V))   # flatten B,H
    O = torch.empty_like(Q); Of = O.reshape(B * H, T, d)
    L = torch.empty((B, H, T), device=Q.device, dtype=torch.float32)
    Lf = L.reshape(B * H, T)

    n_q  = (T + BLOCK_Q - 1) // BLOCK_Q
    grid = (n_q, B * H)                             # one program per (query block, head)
    _fwd_kernel[grid](
        Qf, Kf, Vf, Of, Lf,
        q_row_offsets, q_col_indices,
        sm_scale * LOG2E,
        Qf.stride(0), Qf.stride(1), Qf.stride(2),
        Kf.stride(0), Kf.stride(1), Kf.stride(2),
        Vf.stride(0), Vf.stride(1), Vf.stride(2),
        Of.stride(0), Of.stride(1), Of.stride(2),
        Lf.stride(0), Lf.stride(1),
        T,
        BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, D=d,
    )
    return O, L

@triton.jit
def _bwd_dkdv_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, Delta_ptr,
    dK_ptr, dV_ptr,
    k_row_offsets_ptr, k_col_indices_ptr,        # KEY-block view: for key j, its query blocks i
    qk_scale, sm_scale,
    stride_qb, stride_qt, stride_qd,
    stride_kb, stride_kt, stride_kd,
    stride_vb, stride_vt, stride_vd,
    stride_dob, stride_dot, stride_dod,
    stride_lb, stride_lt,                          # L and Delta share (BH, T) layout
    stride_dkb, stride_dkt, stride_dkd,
    stride_dvb, stride_dvt, stride_dvd,
    T,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, D: tl.constexpr,
):
    pid_j  = tl.program_id(0)    # which KEY block (the j we accumulate dK_j, dV_j for)
    pid_bh = tl.program_id(1)

    offs_k = pid_j * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, D)
    k_mask = offs_k < T

    K_head  = K_ptr  + pid_bh * stride_kb
    V_head  = V_ptr  + pid_bh * stride_vb
    Q_head  = Q_ptr  + pid_bh * stride_qb
    dO_head = dO_ptr + pid_bh * stride_dob
    L_head  = L_ptr     + pid_bh * stride_lb
    Del_head= Delta_ptr + pid_bh * stride_lb       # same layout as L

    # this key block stays on-chip for the whole loop
    k = tl.load(K_head + offs_k[:, None] * stride_kt + offs_d[None, :] * stride_kd,
                mask=k_mask[:, None], other=0.0)
    v = tl.load(V_head + offs_k[:, None] * stride_vt + offs_d[None, :] * stride_vd,
                mask=k_mask[:, None], other=0.0)

    dk = tl.zeros((BLOCK_K, D), tl.float32)
    dv = tl.zeros((BLOCK_K, D), tl.float32)

    start = tl.load(k_row_offsets_ptr + pid_j)
    end   = tl.load(k_row_offsets_ptr + pid_j + 1)
    for idx in range(start, end):
        i = tl.load(k_col_indices_ptr + idx)       # a query block that attends key j
        offs_q = i * BLOCK_Q + tl.arange(0, BLOCK_Q)
        q_mask = offs_q < T

        q  = tl.load(Q_head  + offs_q[:, None] * stride_qt  + offs_d[None, :] * stride_qd,
                     mask=q_mask[:, None], other=0.0)
        do = tl.load(dO_head + offs_q[:, None] * stride_dot + offs_d[None, :] * stride_dod,
                     mask=q_mask[:, None], other=0.0)
        li = tl.load(L_head   + offs_q * stride_lt, mask=q_mask, other=0.0)
        di = tl.load(Del_head + offs_q * stride_lt, mask=q_mask, other=0.0)

        # recover P_ij = exp2(σ·LOG2E·Q·Kᵀ − L_i)
        s = tl.dot(q, tl.trans(k)) * qk_scale          # (BLOCK_Q, BLOCK_K)
        p = tl.exp2(s - li[:, None])
        p = tl.where(q_mask[:, None] & k_mask[None, :], p, 0.0)   # kill padding

        # (2) dV_j += Σ_i P_ijᵀ · dO_i
        dv += tl.dot(tl.trans(p).to(do.dtype), do)

        # (3) dP = dO·Vᵀ   (4) dS = P·(dP − D_i)
        dp = tl.dot(do, tl.trans(v))                   # (BLOCK_Q, BLOCK_K)
        ds = p * (dp - di[:, None])

        # (5) dK_j += Σ_i dS_ijᵀ · Q_i
        dk += tl.dot(tl.trans(ds).to(q.dtype), q)

    dk *= sm_scale                                     # σ on dK only (dV has no σ)
    tl.store(dK_ptr + pid_bh * stride_dkb + offs_k[:, None] * stride_dkt + offs_d[None, :] * stride_dkd,
             dk.to(dK_ptr.dtype.element_ty), mask=k_mask[:, None])
    tl.store(dV_ptr + pid_bh * stride_dvb + offs_k[:, None] * stride_dvt + offs_d[None, :] * stride_dvd,
             dv.to(dV_ptr.dtype.element_ty), mask=k_mask[:, None])

@triton.jit
def _bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, Delta_ptr,
    dQ_ptr,
    q_row_offsets_ptr, q_col_indices_ptr,        # QUERY-block view: for query i, its key blocks j
    qk_scale, sm_scale,
    stride_qb, stride_qt, stride_qd,
    stride_kb, stride_kt, stride_kd,
    stride_vb, stride_vt, stride_vd,
    stride_dob, stride_dot, stride_dod,
    stride_lb, stride_lt,
    stride_dqb, stride_dqt, stride_dqd,
    T,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, D: tl.constexpr,
):
    pid_i  = tl.program_id(0)    # which QUERY block (the i we accumulate dQ_i for)
    pid_bh = tl.program_id(1)

    offs_q = pid_i * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, D)
    q_mask = offs_q < T

    Q_head  = Q_ptr  + pid_bh * stride_qb
    K_head  = K_ptr  + pid_bh * stride_kb
    V_head  = V_ptr  + pid_bh * stride_vb
    dO_head = dO_ptr + pid_bh * stride_dob
    L_head  = L_ptr     + pid_bh * stride_lb
    Del_head= Delta_ptr + pid_bh * stride_lb

    q  = tl.load(Q_head  + offs_q[:, None] * stride_qt  + offs_d[None, :] * stride_qd,
                 mask=q_mask[:, None], other=0.0)
    do = tl.load(dO_head + offs_q[:, None] * stride_dot + offs_d[None, :] * stride_dod,
                 mask=q_mask[:, None], other=0.0)
    li = tl.load(L_head   + offs_q * stride_lt, mask=q_mask, other=0.0)
    di = tl.load(Del_head + offs_q * stride_lt, mask=q_mask, other=0.0)

    dq = tl.zeros((BLOCK_Q, D), tl.float32)

    start = tl.load(q_row_offsets_ptr + pid_i)
    end   = tl.load(q_row_offsets_ptr + pid_i + 1)
    for idx in range(start, end):
        j = tl.load(q_col_indices_ptr + idx)       # a key block that query i attends
        offs_k = j * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offs_k < T

        k = tl.load(K_head + offs_k[:, None] * stride_kt + offs_d[None, :] * stride_kd,
                    mask=k_mask[:, None], other=0.0)
        v = tl.load(V_head + offs_k[:, None] * stride_vt + offs_d[None, :] * stride_vd,
                    mask=k_mask[:, None], other=0.0)

        s = tl.dot(q, tl.trans(k)) * qk_scale
        p = tl.exp2(s - li[:, None])
        p = tl.where(q_mask[:, None] & k_mask[None, :], p, 0.0)

        dp = tl.dot(do, tl.trans(v))               # (3)
        ds = p * (dp - di[:, None])                # (4)

        # (5) dQ_i += Σ_j dS_ij · K_j
        dq += tl.dot(ds.to(k.dtype), k)

    dq *= sm_scale
    tl.store(dQ_ptr + pid_bh * stride_dqb + offs_q[:, None] * stride_dqt + offs_d[None, :] * stride_dqd,
             dq.to(dQ_ptr.dtype.element_ty), mask=q_mask[:, None])

def sparse_flash_backward(Q, K, V, O, L, dO,
                          k_row_offsets, k_col_indices,   # key-block view (sec 1)
                          q_row_offsets, q_col_indices,   # query-block view (sec 1)
                          sm_scale, BLOCK_Q, BLOCK_K):
    """A3 -- block-sparse flash attention backward. See ALGORITHMS.md sec 1, 4.

    Inputs:
        Q, K, V, O, dO (B, H, T, d)         fp16   O, dO are the forward output and its grad
        L              (B, H, T)            fp32   the forward residual
        k_row_offsets  (T//block + 1,)      int32  key-block view: for key block j,
        k_col_indices  (nnz,)               int32  the query blocks i that attend it
        q_row_offsets  (T//block + 1,)      int32  query-block view: for query block i,
        q_col_indices  (nnz,)               int32  its key blocks j (same as forward)
        sm_scale       float
        BLOCK_Q, BLOCK_K  ints                     == block
    Returns:
        dQ, dK, dV     (B, H, T, d)         fp16

    See ALGORITHMS.md sec 4 for the five gradient equations.

    TODO: implement.
    """
    # raise NotImplementedError("TODO: implement sparse_flash_backward (A3)")

    B, H, T, d = Q.shape
    Qf, Kf, Vf, Of, dOf = (X.reshape(B * H, T, d) for X in (Q, K, V, O, dO))
    Lf = L.reshape(B * H, T)

    # (1) D_i = rowsum(dO_i ⊙ O_i) — O(T·d) memory, computed once
    Delta = (dOf.to(torch.float32) * Of.to(torch.float32)).sum(-1).contiguous()  # (BH, T) f32

    dQ = torch.empty_like(Q); dK = torch.empty_like(K); dV = torch.empty_like(V)
    dQf, dKf, dVf = (X.reshape(B * H, T, d) for X in (dQ, dK, dV))
    qk_scale = sm_scale * LOG2E

    grid_kv = (triton.cdiv(T, BLOCK_K), B * H)
    _bwd_dkdv_kernel[grid_kv](
        Qf, Kf, Vf, dOf, Lf, Delta, dKf, dVf,
        k_row_offsets, k_col_indices,
        qk_scale, sm_scale,
        Qf.stride(0), Qf.stride(1), Qf.stride(2),
        Kf.stride(0), Kf.stride(1), Kf.stride(2),
        Vf.stride(0), Vf.stride(1), Vf.stride(2),
        dOf.stride(0), dOf.stride(1), dOf.stride(2),
        Lf.stride(0), Lf.stride(1),
        dKf.stride(0), dKf.stride(1), dKf.stride(2),
        dVf.stride(0), dVf.stride(1), dVf.stride(2),
        T, BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, D=d,
    )

    grid_q = (triton.cdiv(T, BLOCK_Q), B * H)
    _bwd_dq_kernel[grid_q](
        Qf, Kf, Vf, dOf, Lf, Delta, dQf,
        q_row_offsets, q_col_indices,
        qk_scale, sm_scale,
        Qf.stride(0), Qf.stride(1), Qf.stride(2),
        Kf.stride(0), Kf.stride(1), Kf.stride(2),
        Vf.stride(0), Vf.stride(1), Vf.stride(2),
        dOf.stride(0), dOf.stride(1), dOf.stride(2),
        Lf.stride(0), Lf.stride(1),
        dQf.stride(0), dQf.stride(1), dQf.stride(2),
        T, BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, D=d,
    )
    return dQ, dK, dV