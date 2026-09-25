#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 4000
#define BLOCK_SIZE 16

// CUDA kernel for matrix multiplication: C = A * B
__global__ void matrixMulKernel(const float *A, const float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main() {
    size_t bytes = (size_t)N * N * sizeof(float);

    // Host memory allocation
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Host memory allocation failed!\n");
        return 1;
    }

    // Initialize matrices with 1.0
    for (size_t i = 0; i < (size_t)N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    // Device memory allocation
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, bytes);
    cudaMalloc((void **)&d_B, bytes);
    cudaMalloc((void **)&d_C, bytes);

    cudaEvent_t phase_start, phase_stop, kernel_start, kernel_stop;
    cudaEventCreate(&phase_start);
    cudaEventCreate(&phase_stop);
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);

    // Record total CUDA phase start (including memory transfers)
    cudaEventRecord(phase_start, 0);

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Define Grid and Block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch CUDA Kernel
    cudaEventRecord(kernel_start, 0);
    matrixMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaEventRecord(kernel_stop, 0);
    cudaEventSynchronize(kernel_stop);

    // Copy result matrix C from device to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    cudaEventRecord(phase_stop, 0);
    cudaEventSynchronize(phase_stop);

    float kernel_time_ms = 0.0f;
    float phase_time_ms = 0.0f;
    cudaEventElapsedTime(&kernel_time_ms, kernel_start, kernel_stop);
    cudaEventElapsedTime(&phase_time_ms, phase_start, phase_stop);

    printf("CUDA Matrix Multiplication Completed\n");
    printf("Matrix Size = %d x %d\n", N, N);
    printf("Grid Size = %d x %d blocks\n", gridDim.x, gridDim.y);
    printf("Block Size = %d x %d threads\n", blockDim.x, blockDim.y);
    printf("Kernel Execution Time = %f seconds\n", kernel_time_ms / 1000.0f);
    printf("Total CUDA Phase Time = %f seconds\n", phase_time_ms / 1000.0f);
    printf("Verification C[0][0] = %.2f\n", h_C[0]);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(phase_start);
    cudaEventDestroy(phase_stop);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
