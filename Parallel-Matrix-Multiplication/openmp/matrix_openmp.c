#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 4000
#define NUM_THREADS 12

int main() {
    printf("Starting OpenMP Matrix Multiplication (%d x %d) with %d threads...\n", N, N, NUM_THREADS);

    double *A = (double *)malloc((size_t)N * N * sizeof(double));
    double *B = (double *)malloc((size_t)N * N * sizeof(double));
    double *C = (double *)calloc((size_t)N * N, sizeof(double));

    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "Error: Memory allocation failed!\n");
        return 1;
    }

    // Initialize matrices
    #pragma omp parallel for num_threads(NUM_THREADS)
    for (size_t i = 0; i < (size_t)N * N; i++) {
        A[i] = 1.0;
        B[i] = 1.0;
    }

    double start_time = omp_get_wtime();

    // Parallel matrix multiplication distributing rows across threads
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }

    double end_time = omp_get_wtime();
    double execution_time = end_time - start_time;

    printf("Number of Threads = %d\n", NUM_THREADS);
    printf("Verification C[0][0] = %.2f\n", C[0]);
    printf("Measured Execution Time = %f seconds\n", execution_time);

    free(A);
    free(B);
    free(C);

    return 0;
}
