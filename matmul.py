import os
import torch
from torch.utils.cpp_extension import load
from torch.utils.benchmark import Timer

torch.set_grad_enabled(False)
M, N, K = 4096, 4096, 4096  
os.environ["TORCH_CUDA_ARCH_LIST"] = '9.0+PTX'

cc_flag = []
cc_flag.append("-gencode")
cc_flag.append("arch=compute_90a,code=sm_90a")

lib = load(name='matmul', 
          sources=['csrc/matmul.cu'], 
          extra_cuda_cflags=[
                    "-O3",
                    "-std=c++17",
                    "-DNDEBUG",
                    "-D_USE_MATH_DEFINES",
                    "-Wno-deprecated-declarations",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--ptxas-options=-v,--register-usage-level=10"
                ] + cc_flag, 
          extra_cflags=["-O3", "-std=c++17", "-DNDEBUG", "-Wno-deprecated-declarations"])

def benchmark_matmul(func, name, A, B, dtype=torch.float32,):
    print("=" * 50)
    C = torch.zeros(M, N, device='cuda', dtype=dtype)
    
    # Warmup
    for _ in range(20):
        func(A, B, C)
    
    # Benchmark
    timer = Timer(
        stmt='func(A, B, C)',
        globals={'func': func, 'A': A, 'B': B, 'C': C},
        num_threads=1,
    )
    result = timer.blocked_autorange(min_run_time=5)
    
    # Calculate metrics
    time_in_s = result.mean
    time_in_ms = result.mean*1000

    tflops = (2 * M * N * K) / (time_in_s * 1e12)  # 2*M*N*K FLOPs
    print(f"{name}:")
    print(f"  Time: {time_in_ms:.3f} ms")
    print(f"  TFLOPS: {tflops:.2f}")
    print("=" * 50)

    return C

def torch_matmul(A, B, C):
    torch.mm(A, B, out=C)

print(f"Benchmarking {M}x{N}x{K} matrix multiplication for float32:")
print("=" * 50)

DTYPE = torch.float32
A = torch.randn(M, K, device='cuda', dtype=DTYPE)
B = torch.randn(K, N, device='cuda', dtype=DTYPE)

ref_result = benchmark_matmul(torch_matmul, "Torch GPU", A, B, DTYPE)

naive_result = benchmark_matmul(lib.mm_f32_naive, "CUDA Naive", A, B, DTYPE)

coalescent_result = benchmark_matmul(lib.mm_f32_coale, "CUDA Coalescent", A, B, DTYPE)

shared_result = benchmark_matmul(lib.mm_f32_share, "CUDA Shared Memory", A, B, DTYPE)

block_tile_1_result = benchmark_matmul(lib.mm_f32_blkt1, "CUDA Block Tile 1 Memory", A, B, DTYPE)


torch.testing.assert_close(naive_result, ref_result, rtol=1e-5, atol=1e-4)
torch.testing.assert_close(coalescent_result, ref_result, rtol=1e-5, atol=1e-4)
torch.testing.assert_close(shared_result, ref_result, rtol=1e-5, atol=1e-4)

print(f"Benchmarking {M}x{N}x{K} matrix multiplication for bfloat16:")
print("=" * 50)

DTYPE = torch.bfloat16
A = torch.randn(M, K, device='cuda', dtype=DTYPE)
B = torch.randn(K, N, device='cuda', dtype=DTYPE)

ref_result = benchmark_matmul(torch_matmul, "Torch GPU", A, B, DTYPE)

naive_result = benchmark_matmul(lib.mm_bf16_naive, "CUDA Naive", A, B, DTYPE)

coalescent_result = benchmark_matmul(lib.mm_bf16_naive, "CUDA Coalescent", A, B, DTYPE)

shared_result = benchmark_matmul(lib.mm_bf16_share, "CUDA Shared Memory", A, B, DTYPE)

block_tile_1_result = benchmark_matmul(lib.mm_bf16_bltk1, "CUDA Block Tile 1 Memory", A, B, DTYPE)

torch.testing.assert_close(naive_result, ref_result, rtol=1e-2, atol=1e-3)
torch.testing.assert_close(coalescent_result, ref_result, rtol=1e-2, atol=1e-3)
torch.testing.assert_close(shared_result, ref_result, rtol=1e-2, atol=1e-3)
torch.testing.assert_close(block_tile_1_result, ref_result, rtol=1e-2, atol=1e-3)