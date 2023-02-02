import torch
import math

import triton
import triton.language as tl

sqrt2pi = math.sqrt(2.0 / math.pi)

@triton.jit
def tanh(x):
    """Tanh activation function"""
    return tl.libdevice.tanh(x)

@triton.jit
def fast_gelu_kernel(x_ptr, output_ptr, n_elements,  BLOCK_SIZE: tl.constexpr,):
    """Fast approximation of the gelu function. May slightly decrease accuracy."""
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = 0.5 * x * (1 + tanh(sqrt2pi * (x + 0.044715 * x * x * x)))
    tl.store(output_ptr + offsets, output, mask=mask)

def fast_gelu(x: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda

    n_elements = output.numel()
    BLOCK_SIZE=1024
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fast_gelu_kernel[grid](x, output, n_elements, num_warps = num_warps, BLOCK_SIZE=BLOCK_SIZE)
    return output


torch.manual_seed(0)
size = 2 ** 24
x = torch.rand(size, device='cuda')
x1 = torch.rand(size+1, device='cuda')
output = fast_gelu(x)   # 1774 GBPS
output1 = fast_gelu(x1) # 959  GBPS
