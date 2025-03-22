#include "c10/core/ScalarType.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>

// See https://siboehm.com/articles/22/CUDA-MMM

#define BLOCK 32
#define BLOCK_X BLOCK
#define BLOCK_Y BLOCK
#define BLOCK_Z 1
#define BM 64
#define BN 64
#define BK 8
#define TM 8
// Ceil N/B
#define CEIL(N, B) ((N - 1) / B + 1)
// See
// https://docs.nvidia.com/cuda/archive/12.4.1/cuda-math-api/group__CUDA__MATH__INTRINSIC__BFLOAT16.html#group__CUDA__MATH__INTRINSIC__BFLOAT16
#define bfloat16 __nv_bfloat16
#define b2f(x) (__bfloat162float(x))
#define f2b(x) (__float2bfloat16(x))

// A . B
// A: (M, K), B: (K, N)
__global__ void matmul_naive_kernel_float32(int M, int N, int K, float *A,
                                            float *B, float *C) {
  const uint col = blockIdx.x * BLOCK_X + threadIdx.x;
  const uint row = blockIdx.y * BLOCK_Y + threadIdx.y;

  if ((row < M) && (col < N)) {
    float tmp = 0.0f;
    for (int k = 0; k < K; k++) {
      // Row major
      // (row, k) -> row * K + k for A
      // (k, col) -> k * N + col for B
      // (row, col) -> row * N + col for C
      tmp += A[row * K + k] * B[k * N + col];
    }
    // Final result
    C[row * N + col] = tmp;
  }
}

// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy
// https://forums.developer.nvidia.com/t/any-elegant-way-to-calculate-lane-warp-id-and-warp-num-using-cooperative-groups/314784
__global__ void matmul_coalescent_kernel_float32(int M, int N, int K, float *A,
                                                 float *B, float *C) {
  const uint col = blockIdx.y * BLOCK + threadIdx.x % BLOCK;
  const uint row = blockIdx.x * BLOCK + threadIdx.x / BLOCK;

  if ((row < M) && (col < N)) {
    float tmp = 0.0f;
    for (int k = 0; k < K; k++) {
      // Row major
      // (row, k) -> row * K + k for A
      // (k, col) -> k * N + col for B
      // (row, col) -> row * N + col for C
      tmp += A[row * K + k] * B[k * N + col];
    }
    // Final result
    C[row * N + col] = tmp;
  }
}

// https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory
__global__ void matmul_shared_kernel_float32(int M, int N, int K, float *A,
                                             float *B, float *C) {
  __shared__ float sA[BLOCK * BLOCK];
  __shared__ float sB[BLOCK * BLOCK];
  const uint tR = threadIdx.x / BLOCK;
  const uint tC = threadIdx.x % BLOCK;
  const uint currentRow = blockIdx.x * BLOCK; // Start of row
  const uint currentCol = blockIdx.y * BLOCK; // Start of column
  // Shift pointer to current row
  A += currentRow * K;              // (currentRow, 0)
  B += currentCol;                  // (0, currentCol)
  C += currentRow * N + currentCol; // (currentRow, currentCol)
  // Loop over all blocks in the current Row/Column
  float tmp = 0.0;
  for (int b = 0; b < K; b += BLOCK) {
    // Threads collectively copy the current A block to shared memory
    sA[tR * BLOCK + tC] = A[tR * K + tC];
    // Threads collectively copy the current B block to shared memory
    sB[tR * BLOCK + tC] = B[tR * N + tC];
    // Wait for all threads in the block to arrive
    __syncthreads();
    // Step one block size to the right for A
    A += BLOCK;
    // Step one block size down for B
    B += BLOCK * N;
    // Accumulate product of current shared memory chunks.
    for (int i = 0; i < BLOCK; i++) {
      tmp += sA[tR * BLOCK + i] * sB[i * BLOCK + tC];
    }
    // Wait for all threads in the block to arrive
    __syncthreads();
  }
  // Populate C
  C[tR * N + tC] = tmp;
}

__global__ void matmul_block_tile_kernel_1_float32(int M, int N, int K,
                                                   float *A, float *B,
                                                   float *C) {
  __shared__ float sA[BM * BK];
  __shared__ float sB[BK * BN];
  const uint tR = threadIdx.x / BN;
  const uint tC = threadIdx.x % BN;
  const uint currentRow = blockIdx.y * BM; // Start of row
  const uint currentCol = blockIdx.x * BN; // Start of column
  // Shift pointer to current row
  A += currentRow * K;              // (currentRow, 0)
  B += currentCol;                  // (0, currentCol)
  C += currentRow * N + currentCol; // (currentRow, currentCol)
  // Access inner cols and rows on a warp level
  const uint innerColA = threadIdx.x % BK; // One lane for each column
  const uint innerRowA = threadIdx.x / BK; // One warp for each row
  const uint innerColB = threadIdx.x % BN;
  const uint innerRowB = threadIdx.x / BN;

  // Thread level cache
  float tmp[TM] = {0};
  for (int b = 0; b < K; b += BK) {
    // Threads collectively copy the current A block to shared memory
    sA[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    // Threads collectively copy the current B block to shared memory
    sB[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    // Wait for all threads in the block to arrive
    __syncthreads();
    // Step one block size to the right for A
    A += BK;
    // Step one block size down for B
    B += BK * N;
    // Accumulate product of current shared memory chunks.
    for (int i = 0; i < BK; i++) {
      float tmpB = sB[i * BN + tC]; // Cache entry, this has shape (BK, BN)
      for (int j = 0; j < TM; j++) {
        // Accumulate result
        // 3D indexing. (tR * TM + j) * BK + i =
        // tR * TM * BK + j * BK + i -> (tR, j, i)
        // i.e. we interpret (BM, BK) as (BM/TM, TM, BK)
        // That is because we know BK stays the column dimension
        // from the indexing and we further now the product
        // of each dimension must be constant.
        tmp[j] += sA[(tR * TM + j) * BK + i] * tmpB;
      }
    }
    // Wait for all threads in the block to arrive
    __syncthreads();
  }
  // Populate C
  for (int j = 0; j < TM; j++) {
    // Accumulate result
    // Interpret C as (M/TM, TM, N) -> (tR, j, i)
    C[(tR * TM + j) * N + tC] = tmp[j];
  }
}

__global__ void matmul_naive_kernel_bfloat16(int M, int N, int K, bfloat16 *A,
                                             bfloat16 *B, bfloat16 *C) {
  const uint col = blockIdx.x * BLOCK_X + threadIdx.x;
  const uint row = blockIdx.y * BLOCK_Y + threadIdx.y;

  if ((row < M) && (col < N)) {
    float tmp = 0.0f;
    for (int k = 0; k < K; k++) {
      // Multiply and accumulate in float precision
      tmp += b2f(A[row * K + k]) * b2f(B[k * N + col]);
    }
    // Final result
    C[row * N + col] = f2b(tmp);
  }
}

__global__ void matmul_coalescent_kernel_bfloat16(int M, int N, int K,
                                                  bfloat16 *A, bfloat16 *B,
                                                  bfloat16 *C) {
  const uint col = blockIdx.y * BLOCK + threadIdx.x % BLOCK;
  const uint row = blockIdx.x * BLOCK + threadIdx.x / BLOCK;

  if ((row < M) && (col < N)) {
    float tmp = 0.0f;
    for (int k = 0; k < K; k++) {
      // Multiply and accumulate in float precision
      tmp += b2f(A[row * K + k]) * b2f(B[k * N + col]);
    }
    // Final result
    C[row * N + col] = f2b(tmp);
  }
}

__global__ void matmul_shared_kernel_bfloat16(int M, int N, int K, bfloat16 *A,
                                              bfloat16 *B, bfloat16 *C) {
  __shared__ bfloat16 sA[BLOCK * BLOCK];
  __shared__ bfloat16 sB[BLOCK * BLOCK];
  const uint tR = threadIdx.x / BLOCK;
  const uint tC = threadIdx.x % BLOCK;
  const uint currentRow = blockIdx.x * BLOCK; // Start of row
  const uint currentCol = blockIdx.y * BLOCK; // Start of column
  A += currentRow * K;                        // (currentRow, 0)
  B += currentCol;                            // (0, currentCol)
  C += currentRow * N + currentCol;           // (currentRow, currentCol)
  float tmp = 0.0;
  for (int b = 0; b < K; b += BLOCK) {
    sA[tR * BLOCK + tC] = A[tR * K + tC];
    sB[tR * BLOCK + tC] = B[tR * N + tC];
    __syncthreads();
    A += BLOCK;
    B += BLOCK * N;
    for (int i = 0; i < BLOCK; i++) {
      tmp += b2f(sA[tR * BLOCK + i]) * b2f(sB[i * BLOCK + tC]);
    }
    __syncthreads();
  }
  C[tR * N + tC] = f2b(tmp);
}

__global__ void matmul_block_tile_kernel_1_bfloat16(int M, int N, int K,
                                                    bfloat16 *A, bfloat16 *B,
                                                    bfloat16 *C) {
  __shared__ bfloat16 sA[BM * BK];
  __shared__ bfloat16 sB[BK * BN];
  const uint tR = threadIdx.x / BN;
  const uint tC = threadIdx.x % BN;
  const uint currentRow = blockIdx.y * BM; // Start of row
  const uint currentCol = blockIdx.x * BN; // Start of column
  // Shift pointer to current row
  A += currentRow * K;              // (currentRow, 0)
  B += currentCol;                  // (0, currentCol)
  C += currentRow * N + currentCol; // (currentRow, currentCol)
  // Access inner cols and rows on a warp level
  const uint innerColA = threadIdx.x % BK; // One lane for each column
  const uint innerRowA = threadIdx.x / BK; // One warp for each row
  const uint innerColB = threadIdx.x % BN;
  const uint innerRowB = threadIdx.x / BN;

  // Thread level cache
  float tmp[TM] = {0};
  for (int b = 0; b < K; b += BK) {
    // Threads collectively copy the current A block to shared memory
    sA[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    // Threads collectively copy the current B block to shared memory
    sB[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    // Wait for all threads in the block to arrive
    __syncthreads();
    // Step one block size to the right for A
    A += BK;
    // Step one block size down for B
    B += BK * N;
    // Accumulate product of current shared memory chunks.
    for (int i = 0; i < BK; i++) {
      float tmpB = b2f(sB[i * BN + tC]); // Cache entry
      for (int j = 0; j < TM; j++) {
        // Accumulate result
        // Let one thread handle multiple values
        // tR * TM + j -> index into tiled matrix
        // (...) * BK + i -> index into block
        tmp[j] += b2f(sA[(tR * TM + j) * BK + i]) * tmpB;
      }
    }
    // Wait for all threads in the block to arrive
    __syncthreads();
  }
  // Populate C
  for (int j = 0; j < TM; j++) {
    // Accumulate result
    // Let one thread handle multiple values
    C[(tR * TM + j) * N + tC] = f2b(tmp[j]);
  }
}

// --------------------- PyTorch bindings for custom kernel
// -----------------------

// Adapted from
// https://github.com/deepseek-ai/FlashMLA/blob/main/csrc/flash_api.cpp
#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...)                                                    \
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}),                  \
              #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

void matmul_naive_float32(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
  TORCH_CHECK(A.dtype() == torch::kFloat32, "A must have dtype torch.float32");
  TORCH_CHECK(B.dtype() == torch::kFloat32, "B must have dtype torch.float32");
  TORCH_CHECK(C.dtype() == torch::kFloat32, "C must have dtype torch.float32");
  const int M = A.size(0);
  const int K = A.size(1);
  const int N = B.size(1);
  CHECK_SHAPE(A, M, K)
  CHECK_SHAPE(B, K, N)
  CHECK_SHAPE(C, M, N)

  dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
  dim3 grid(CEIL(N, BLOCK_X), CEIL(M, BLOCK_Y), 1);

  matmul_naive_kernel_float32<<<grid, block>>>(
      M, N, K, reinterpret_cast<float *>(A.data_ptr()),
      reinterpret_cast<float *>(B.data_ptr()),
      reinterpret_cast<float *>(C.data_ptr()));
}

void matmul_coalescent_float32(torch::Tensor A, torch::Tensor B,
                               torch::Tensor C) {
  TORCH_CHECK(A.dtype() == torch::kFloat32, "A must have dtype torch.float32");
  TORCH_CHECK(B.dtype() == torch::kFloat32, "B must have dtype torch.float32");
  TORCH_CHECK(C.dtype() == torch::kFloat32, "C must have dtype torch.float32");
  const int M = A.size(0);
  const int K = A.size(1);
  const int N = B.size(1);
  CHECK_SHAPE(A, M, K)
  CHECK_SHAPE(B, K, N)
  CHECK_SHAPE(C, M, N)

  dim3 block(BLOCK * BLOCK, 1, 1);
  dim3 grid(CEIL(M, BLOCK), CEIL(N, BLOCK), 1);

  matmul_coalescent_kernel_float32<<<grid, block>>>(
      M, N, K, reinterpret_cast<float *>(A.data_ptr()),
      reinterpret_cast<float *>(B.data_ptr()),
      reinterpret_cast<float *>(C.data_ptr()));
}

void matmul_shared_float32(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
  TORCH_CHECK(A.dtype() == torch::kFloat32, "A must have dtype torch.float32");
  TORCH_CHECK(B.dtype() == torch::kFloat32, "B must have dtype torch.float32");
  TORCH_CHECK(C.dtype() == torch::kFloat32, "C must have dtype torch.float32");
  const int M = A.size(0);
  const int K = A.size(1);
  const int N = B.size(1);
  CHECK_SHAPE(A, M, K)
  CHECK_SHAPE(B, K, N)
  CHECK_SHAPE(C, M, N)

  dim3 block(BLOCK * BLOCK, 1, 1);
  dim3 grid(CEIL(M, BLOCK), CEIL(N, BLOCK), 1);

  matmul_shared_kernel_float32<<<grid, block>>>(
      M, N, K, reinterpret_cast<float *>(A.data_ptr()),
      reinterpret_cast<float *>(B.data_ptr()),
      reinterpret_cast<float *>(C.data_ptr()));
}

void matmul_block_tile_1_float32(torch::Tensor A, torch::Tensor B,
                                 torch::Tensor C) {
  TORCH_CHECK(A.dtype() == torch::kFloat32, "A must have dtype torch.float32");
  TORCH_CHECK(B.dtype() == torch::kFloat32, "B must have dtype torch.float32");
  TORCH_CHECK(C.dtype() == torch::kFloat32, "C must have dtype torch.float32");
  const int M = A.size(0);
  const int K = A.size(1);
  const int N = B.size(1);
  CHECK_SHAPE(A, M, K)
  CHECK_SHAPE(B, K, N)
  CHECK_SHAPE(C, M, N)

  dim3 block((BM * BN) / TM, 1, 1);
  dim3 grid(CEIL(N, BN), CEIL(M, BM), 1);

  matmul_block_tile_kernel_1_float32<<<grid, block>>>(
      M, N, K, reinterpret_cast<float *>(A.data_ptr()),
      reinterpret_cast<float *>(B.data_ptr()),
      reinterpret_cast<float *>(C.data_ptr()));
}

void matmul_naive_bfloat16(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
  TORCH_CHECK(A.dtype() == torch::kBFloat16,
              "A must have dtype torch.bfloat16");
  TORCH_CHECK(B.dtype() == torch::kBFloat16,
              "B must have dtype torch.bfloat16");
  TORCH_CHECK(C.dtype() == torch::kBFloat16,
              "C must have dtype torch.bfloat16");
  const int M = A.size(0);
  const int K = A.size(1);
  const int N = B.size(1);
  CHECK_SHAPE(A, M, K)
  CHECK_SHAPE(B, K, N)
  CHECK_SHAPE(C, M, N)

  dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
  dim3 grid(CEIL(N, BLOCK_X), CEIL(M, BLOCK_Y), 1);

  matmul_naive_kernel_bfloat16<<<grid, block>>>(
      M, N, K, reinterpret_cast<bfloat16 *>(A.data_ptr()),
      reinterpret_cast<bfloat16 *>(B.data_ptr()),
      reinterpret_cast<bfloat16 *>(C.data_ptr()));
}

void matmul_coalescent_bfloat16(torch::Tensor A, torch::Tensor B,
                                torch::Tensor C) {
  TORCH_CHECK(A.dtype() == torch::kBFloat16,
              "A must have dtype torch.bfloat16");
  TORCH_CHECK(B.dtype() == torch::kBFloat16,
              "B must have dtype torch.bfloat16");
  TORCH_CHECK(C.dtype() == torch::kBFloat16,
              "C must have dtype torch.bfloat16");
  const int M = A.size(0);
  const int K = A.size(1);
  const int N = B.size(1);
  CHECK_SHAPE(A, M, K)
  CHECK_SHAPE(B, K, N)
  CHECK_SHAPE(C, M, N)

  dim3 block(BLOCK * BLOCK, 1, 1);
  dim3 grid(CEIL(M, BLOCK), CEIL(N, BLOCK), 1);

  matmul_coalescent_kernel_bfloat16<<<grid, block>>>(
      M, N, K, reinterpret_cast<bfloat16 *>(A.data_ptr()),
      reinterpret_cast<bfloat16 *>(B.data_ptr()),
      reinterpret_cast<bfloat16 *>(C.data_ptr()));
}

void matmul_shared_bfloat16(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
  TORCH_CHECK(A.dtype() == torch::kBFloat16,
              "A must have dtype torch.bfloat16");
  TORCH_CHECK(B.dtype() == torch::kBFloat16,
              "B must have dtype torch.bfloat16");
  TORCH_CHECK(C.dtype() == torch::kBFloat16,
              "C must have dtype torch.bfloat16");
  const int M = A.size(0);
  const int K = A.size(1);
  const int N = B.size(1);
  CHECK_SHAPE(A, M, K)
  CHECK_SHAPE(B, K, N)
  CHECK_SHAPE(C, M, N)

  dim3 block(BLOCK * BLOCK, 1, 1);
  dim3 grid(CEIL(M, BLOCK), CEIL(N, BLOCK), 1);

  matmul_shared_kernel_bfloat16<<<grid, block>>>(
      M, N, K, reinterpret_cast<bfloat16 *>(A.data_ptr()),
      reinterpret_cast<bfloat16 *>(B.data_ptr()),
      reinterpret_cast<bfloat16 *>(C.data_ptr()));
}

void matmul_block_tile_1_bfloat16(torch::Tensor A, torch::Tensor B,
                                  torch::Tensor C) {
  TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must have dtype torch.float32");
  TORCH_CHECK(B.dtype() == torch::kBFloat16, "B must have dtype torch.float32");
  TORCH_CHECK(C.dtype() == torch::kBFloat16, "C must have dtype torch.float32");
  const int M = A.size(0);
  const int K = A.size(1);
  const int N = B.size(1);
  CHECK_SHAPE(A, M, K)
  CHECK_SHAPE(B, K, N)
  CHECK_SHAPE(C, M, N)

  dim3 block((BM * BN) / TM, 1, 1);
  dim3 grid(CEIL(N, BN), CEIL(M, BM), 1);

  matmul_block_tile_kernel_1_bfloat16<<<grid, block>>>(
      M, N, K, reinterpret_cast<bfloat16 *>(A.data_ptr()),
      reinterpret_cast<bfloat16 *>(B.data_ptr()),
      reinterpret_cast<bfloat16 *>(C.data_ptr()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Matrix Multiplication";
  m.def("mm_f32_naive", &matmul_naive_float32);
  m.def("mm_f32_coale", &matmul_coalescent_float32);
  m.def("mm_f32_share", &matmul_shared_float32);
  m.def("mm_f32_blkt1", &matmul_block_tile_1_float32);
  m.def("mm_bf16_naive", &matmul_naive_bfloat16);
  m.def("mm_bf16_coale", &matmul_coalescent_bfloat16);
  m.def("mm_bf16_share", &matmul_shared_bfloat16);
  m.def("mm_bf16_bltk1", &matmul_block_tile_1_bfloat16);
}