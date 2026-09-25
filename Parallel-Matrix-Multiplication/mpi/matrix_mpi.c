#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 4000

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: Matrix dimension N (%d) must be divisible by number of processes (%d).\n", N, size);
        }
        MPI_Finalize();
        return 1;
    }

    int rows_per_proc = N / size;

    double *A = NULL;
    double *B = (double *)malloc((size_t)N * N * sizeof(double));
    double *C = NULL;

    double *local_A = (double *)malloc((size_t)rows_per_proc * N * sizeof(double));
    double *local_C = (double *)calloc((size_t)rows_per_proc * N, sizeof(double));

    if (local_A == NULL || local_C == NULL || B == NULL) {
        fprintf(stderr, "Process %d: Memory allocation failed!\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        A = (double *)malloc((size_t)N * N * sizeof(double));
        C = (double *)calloc((size_t)N * N, sizeof(double));

        if (A == NULL || C == NULL) {
            fprintf(stderr, "Rank 0: Matrix allocation failed!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Initialize A and B
        for (size_t i = 0; i < (size_t)N * N; i++) {
            A[i] = 1.0;
            B[i] = 1.0;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // 1. Scatter rows of matrix A to all processes
    MPI_Scatter(A, rows_per_proc * N, MPI_DOUBLE,
                local_A, rows_per_proc * N, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // 2. Broadcast matrix B to all processes
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 3. Local matrix computation: local_C = local_A * B
    for (int i = 0; i < rows_per_proc; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += local_A[i * N + k] * B[k * N + j];
            }
            local_C[i * N + j] = sum;
        }
    }

    // 4. Gather computed rows of local_C back into matrix C on rank 0
    MPI_Gather(local_C, rows_per_proc * N, MPI_DOUBLE,
               C, rows_per_proc * N, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    if (rank == 0) {
        double execution_time = end_time - start_time;
        printf("Number of MPI Processes = %d\n", size);
        printf("Verification C[0][0] = %.2f\n", C[0]);
        printf("Measured Execution Time = %f seconds\n", execution_time);

        free(A);
        free(C);
    }

    free(B);
    free(local_A);
    free(local_C);

    MPI_Finalize();
    return 0;
}
