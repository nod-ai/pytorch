import torch
import triton
import triton.language as tl
from torch.library import register_kernel
torch.utils.rename_privateuse1_backend('zoom')

@triton.heuristics({
    'BLOCK_SIZE_M': lambda args: 128,
    'BLOCK_SIZE_N': lambda args: 64,
    'BLOCK_SIZE_K': lambda args: 32,
    'GROUP_SIZE_M': lambda args: 32,
    'EVEN_K': lambda args: args['K'] % args['BLOCK_SIZE_K'] == 0,
})
@triton.jit
def batched_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    B,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    a_scale_ptr,
    b_scale_ptr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    APPLY_SCALE: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """Kernel for computing the batched matmul C = A x B.
    A has shape (B, M, K), B has shape (B, K, N) and C has shape (B, M, N)
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_batch = num_pid_m * num_pid_n
    batch_id = pid // num_pid_in_batch
    pid_in_batch = pid % num_pid_in_batch

    if GROUP_SIZE_M == 1:
        pid_m = pid_in_batch // num_pid_n
        pid_n = pid_in_batch % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid_in_batch // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid_in_batch % group_size_m)
        pid_n = (pid_in_batch % num_pid_in_group) // group_size_m

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    a_ptrs = a_ptr + batch_id * stride_ab + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + batch_id * stride_bb + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    if APPLY_SCALE:
        a_scale = tl.load(a_scale_ptr)
        b_scale = tl.load(b_scale_ptr)

    acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if APPLY_SCALE:
        accumulator = accumulator * a_scale * b_scale

    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(c_ptr.type.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + batch_id * stride_cb + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

# Wrapper for batched gemm kernel
def batched_matmul(a, b, c, a_scale, b_scale, scale_a8_b8=False, activation=""):
    assert a.shape[2] == b.shape[1], "Incompatible matrix dimensions!!!"
    assert a.shape[0] == b.shape[0], "Incompatible batch dimensions!!!"
    assert a.dtype == b.dtype, "Mixed dtype GEMMs are not supported!!!"
    B, M, K = a.shape
    _, K, N = b.shape
    grid = lambda META: (B * triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    batched_matmul_kernel[grid](
        a,
        b,
        c,
        B,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        a_scale,
        b_scale,
        APPLY_SCALE=scale_a8_b8,
        ACTIVATION=activation,
    )

# Activation function.
@triton.jit
def leaky_relu(x):
    x = x + 1
    return tl.where(x >= 0, x, 0.01 * x)

name_to_torch_types = {
    'int8': torch.int8,
    'int32': torch.int32,
    'fp16': torch.float16,
    'fp32': torch.float32,
    'bf16': torch.bfloat16,
    'fp8e5': torch.float8_e5m2fnuz,
    'fp8e4': torch.float8_e4m3fnuz,
}

dtype_max = {
    dtype: (torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)).max
    for dtype in [
        torch.float8_e5m2fnuz,
        torch.float8_e4m3fnuz,
        torch.int8,
    ]
}

def mm_out_zoom(self, mat2, out):
    batched_matmul(self.unsqueeze(0), mat2.unsqueeze(0), out.unsqueeze(0), None, None, False)
    
def bmm_out_zoom(self, mat2, out):
    batched_matmul(self, mat2, out, None, None, False)

@register_kernel("aten::mm.out", "zoom")
def mm_out(self, mat2, out):
    mm_out_zoom(self, mat2, out)
    
@register_kernel("aten::mm", "zoom")
def mm(self, mat2):
    out = self.new_empty((self.size(0), mat2.size(1)))
    mm_out_zoom(self, mat2, out)
    return out
    
@register_kernel("aten::bmm.out", "zoom")
def bmm_out(self, mat2, out):
    bmm_out_zoom(self, mat2, out)
        
@register_kernel("aten::bmm", "zoom")
def bmm(self, mat2):
    out = self.new_empty((self.size(0), self.size(1), mat2.size(2)))
    bmm_out_zoom(self, mat2, out)
    return out
    