/*
 * cuda_bf_matcher.cu — brute-force L2 kNN (k=2) using cuBLAS.
 *
 * Memory-efficient version: fuses distance computation and knn2 into one
 * kernel pass — avoids allocating the full n_q×n_t distance matrix.
 *
 *   Step 1  norms_q, norms_t  (small kernels)
 *   Step 2  P = Q * T^T       (cuBLAS sgemm — n_q×n_t dot products)
 *   Step 3  knn2_fused:       for each row i, compute dist on-the-fly from
 *                             P[i,*], norms_q[i], norms_t[*] and track top-2
 *
 * Peak CUDA memory: n_q×n_t×4 B (for P) + O(n_q+n_t) — no separate D matrix.
 */

#include "cuda_bf_matcher.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// ── error helpers ─────────────────────────────────────────────────────────────
#define CUDA_CHECK(x) do {                                                      \
    cudaError_t _e = (x);                                                       \
    if (_e != cudaSuccess)                                                      \
        throw std::runtime_error(std::string("CUDA ") + cudaGetErrorString(_e));\
} while(0)
#define CUBLAS_CHECK(x) do {                                                    \
    cublasStatus_t _s = (x);                                                    \
    if (_s != CUBLAS_STATUS_SUCCESS)                                            \
        throw std::runtime_error("cuBLAS error " + std::to_string(_s));         \
} while(0)

// ── kernels ───────────────────────────────────────────────────────────────────

__global__ void row_norms_sq(const float* __restrict__ M,
                              float* __restrict__ norms,
                              int n, int dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = 0.f;
    for (int k = 0; k < dim; ++k) { float v = M[i*dim+k]; s += v*v; }
    norms[i] = s;
}

// Fused: for each query row i, compute ||q_i-t_j||² from P[i,j] and norms,
// track 2 smallest without writing a full distance matrix.
__global__ void knn2_fused(const float* __restrict__ P,
                            const float* __restrict__ nq,
                            const float* __restrict__ nt,
                            int* __restrict__ idx0, float* __restrict__ d0,
                            int* __restrict__ idx1, float* __restrict__ d1,
                            int n_q, int n_t)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_q) return;
    float b0 = 1e30f, b1 = 1e30f;
    int   i0 = -1,    i1 = -1;
    float nqi = nq[i];
    const float* row = P + i * n_t;
    for (int j = 0; j < n_t; ++j) {
        float v = nqi + nt[j] - 2.f * row[j];
        if (v < 0.f) v = 0.f;
        if (v < b0)      { b1=b0; i1=i0; b0=v; i0=j; }
        else if (v < b1) { b1=v;  i1=j; }
    }
    idx0[i]=i0; d0[i]=b0;
    idx1[i]=i1; d1[i]=b1;
}

// ── public API ────────────────────────────────────────────────────────────────

void cuda_knn2_match(const cv::Mat& descs_q,
                     const cv::Mat& descs_t,
                     std::vector<std::vector<cv::DMatch>>& matches)
{
    const int n_q = descs_q.rows, n_t = descs_t.rows, dim = 128;

    float *d_Q=nullptr, *d_T=nullptr, *d_P=nullptr, *d_nq=nullptr, *d_nt=nullptr;
    int   *d_i0=nullptr, *d_i1=nullptr;
    float *d_d0=nullptr, *d_d1=nullptr;

    try {
        CUDA_CHECK(cudaMalloc(&d_Q,  (size_t)n_q*dim*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_T,  (size_t)n_t*dim*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_P,  (size_t)n_q*n_t*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_nq, (size_t)n_q*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_nt, (size_t)n_t*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_i0, (size_t)n_q*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_i1, (size_t)n_q*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_d0, (size_t)n_q*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_d1, (size_t)n_q*sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_Q, descs_q.ptr<float>(),
                              (size_t)n_q*dim*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_T, descs_t.ptr<float>(),
                              (size_t)n_t*dim*sizeof(float), cudaMemcpyHostToDevice));

        // norms
        int blk = 256;
        row_norms_sq<<<(n_q+blk-1)/blk, blk>>>(d_Q, d_nq, n_q, dim);
        row_norms_sq<<<(n_t+blk-1)/blk, blk>>>(d_T, d_nt, n_t, dim);
        CUDA_CHECK(cudaGetLastError());

        // P = Q * T^T  via cuBLAS (column-major: compute T * Q^T)
        cublasHandle_t h; CUBLAS_CHECK(cublasCreate(&h));
        const float alpha=1.f, beta=0.f;
        CUBLAS_CHECK(cublasSgemm(h,
            CUBLAS_OP_T, CUBLAS_OP_N,
            n_t, n_q, dim, &alpha,
            d_T, dim, d_Q, dim, &beta, d_P, n_t));
        cublasDestroy(h);

        // fused knn2
        knn2_fused<<<(n_q+blk-1)/blk, blk>>>(
            d_P, d_nq, d_nt, d_i0, d_d0, d_i1, d_d1, n_q, n_t);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // download
        std::vector<int>   h_i0(n_q), h_i1(n_q);
        std::vector<float> h_d0(n_q), h_d1(n_q);
        CUDA_CHECK(cudaMemcpy(h_i0.data(), d_i0, n_q*sizeof(int),   cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_i1.data(), d_i1, n_q*sizeof(int),   cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_d0.data(), d_d0, n_q*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_d1.data(), d_d1, n_q*sizeof(float), cudaMemcpyDeviceToHost));

        matches.resize(n_q);
        for (int i = 0; i < n_q; ++i) {
            matches[i].clear();
            if (h_i0[i] >= 0)
                matches[i].push_back(cv::DMatch(i, h_i0[i], std::sqrt(h_d0[i])));
            if (h_i1[i] >= 0)
                matches[i].push_back(cv::DMatch(i, h_i1[i], std::sqrt(h_d1[i])));
        }
    } catch (...) {
        // free whatever was allocated before re-throwing
        if (d_Q)  cudaFree(d_Q);  if (d_T)  cudaFree(d_T);
        if (d_P)  cudaFree(d_P);  if (d_nq) cudaFree(d_nq);
        if (d_nt) cudaFree(d_nt); if (d_i0) cudaFree(d_i0);
        if (d_i1) cudaFree(d_i1); if (d_d0) cudaFree(d_d0);
        if (d_d1) cudaFree(d_d1);
        throw;
    }

    cudaFree(d_Q);  cudaFree(d_T);  cudaFree(d_P);
    cudaFree(d_nq); cudaFree(d_nt);
    cudaFree(d_i0); cudaFree(d_i1);
    cudaFree(d_d0); cudaFree(d_d1);
}
