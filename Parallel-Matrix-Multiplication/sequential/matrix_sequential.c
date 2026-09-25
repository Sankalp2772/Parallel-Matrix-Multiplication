#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 4000

int main() {
    printf("Starting Sequential Matrix Multiplication (%d x %d)...\n", N, N);

    // Allocate memory dynamically on the heap to avoid stack overflow
    double *A = (double *)malloc((size_t)N * N * sizeof(double));
    double *B = (double *)malloc((size_t)N * N * sizeof(double));
    double *C = (double *)calloc((size_t)N * N, sizeof(double));

    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "Error: Memory allocation failed!\n");
        return 1;
    }

    // Initialize matrices A and B with 1.0
    for (size_t i = 0; i < (size_t)N * N; i++) {
        A[i] = 1.0;
        B[i] = 1.0;
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Standard sequential matrix multiplication: C = A * B
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double execution_time = (end.tv_sec - start.tv_sec) + 
                            (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("Verification C[0][0] = %.2f\n", C[0]);
    printf("Measured Execution Time = %f seconds\n", execution_time);

    free(A);
    free(B);
    free(C);

    return 0;
}
