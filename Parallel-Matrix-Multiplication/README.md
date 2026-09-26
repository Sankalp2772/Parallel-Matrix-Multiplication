# Parallel Matrix Multiplication using Sequential, OpenMP, MPI and CUDA

## Experiment 1 — Parallel Computing

This experiment implements the same matrix multiplication problem using four different computing models:

1. **Sequential CPU**
2. **OpenMP Shared-Memory Parallelism**
3. **MPI Distributed-Memory Parallelism**
4. **CUDA GPU Parallelism**

The objective is to understand how the same computational problem behaves under different parallel computing architectures and to compare their execution performance.

---

## Table of Contents

- [1. Objective](#1-objective)
- [2. Problem Definition](#2-problem-definition)
- [3. Experimental Environment](#3-experimental-environment)
- [4. Project Structure](#4-project-structure)
- [5. Sequential Implementation](#5-sequential-implementation)
- [6. OpenMP Implementation](#6-openmp-implementation)
- [7. MPI Implementation](#7-mpi-implementation)
- [8. CUDA Implementation](#8-cuda-implementation)
- [9. Results](#9-results)
- [10. Performance Comparison](#10-performance-comparison)
- [11. Speedup Analysis](#11-speedup-analysis)
- [12. Verification](#12-verification)
- [13. Observations](#13-observations)
- [14. Overall Comparison](#14-overall-comparison)
- [15. Conclusion](#15-conclusion)
- [Technologies Used](#technologies-used)
- [Experiment Summary](#experiment-summary)
- [Author](#author)

---

## 1. Objective

The objective of this experiment is to implement and compare matrix multiplication using different computing paradigms:

- Sequential execution on a CPU
- Shared-memory parallelism using OpenMP
- Distributed-memory parallelism using MPI
- GPU parallelism using CUDA

The experiment demonstrates how parallel computing techniques can reduce execution time for computationally intensive operations.

---

## 2. Problem Definition

Two square matrices are used:

```text
A = 4000 × 4000
B = 4000 × 4000
```

All elements of both matrices are initialized to:
```text
1.0
```

The result is calculated as:
```text
C = A × B
```

For every element:
```text
C[i][j] = Σ (A[i][k] × B[k][j])  for k = 0 to 3999
```

Since every element of $A$ and $B$ is `1.0`:
```text
C[i][j] = 1×1 + 1×1 + ... + 1×1  (4000 terms)
```

Therefore:
```text
C[i][j] = 4000.00
```

The program verifies the result using:
```text
C[0][0] = 4000.00
```

---

## 3. Experimental Environment

### Hardware
- **CPU**: Host system CPU
- **GPU**: NVIDIA GeForce RTX 2050
- **GPU Memory**: 4 GB
- **GPU Compute Capability**: 8.6
- **Cluster**: Four Ubuntu virtual machines for MPI

### Software
- **Operating Systems**: Windows, Ubuntu / WSL
- **Compilers & Toolchains**: GCC, Open MPI, NVIDIA CUDA Toolkit (`nvcc`), Visual Studio Build Tools
- **Libraries & Protocols**: OpenMP, OpenSSH
- **Virtualization**: VMware

---

## 4. Project Structure

The experiment is organized as follows:

```text
Parallel-Matrix-Multiplication/
│
├── README.md
│
├── sequential/
│   └── matrix_sequential.c
│
├── openmp/
│   └── matrix_openmp.c
│
├── mpi/
│   ├── matrix_mpi.c
│   └── hosts
│
├── cuda/
│   └── matrix_cuda.cu
│
└── results/
    └── comparison.md
```

---

## 5. Sequential Implementation

### 5.1 Description
The sequential implementation performs matrix multiplication using a single CPU execution flow.

The computation uses the standard three nested loops:
```c
for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++) {
            C[i][j] += A[i][k] * B[k][j];
        }
    }
}
```

This implementation serves as the baseline for comparing the parallel implementations.

### 5.2 Compilation
Inside the Ubuntu/WSL terminal:
```bash
gcc -O2 matrix_sequential.c -o matrix_sequential
```

### 5.3 Execution
```bash
./matrix_sequential
```

### 5.4 Result
- **Verification**: `C[0][0] = 4000.00`
- **Measured Execution Time**: `417.205920 seconds`

---

## 6. OpenMP Implementation

### 6.1 Description
OpenMP is used to implement shared-memory parallelism.

Multiple CPU threads work simultaneously on different portions of the matrix multiplication while sharing the matrices in the same memory space. The outer loop of the matrix multiplication is parallelized across available cores.

```text
CPU Shared Memory
       │
 ┌─────┼─────┐
 │     │     │
T1    T2    T3 ... Tn
 │     │     │
Rows  Rows  Rows
 │     │     │
 └─────┼─────┘
       │
       C
```

### 6.2 Compilation
```bash
gcc -O2 -fopenmp matrix_openmp.c -o matrix_openmp
```

### 6.3 Execution
```bash
./matrix_openmp
```

### 6.4 Configuration
- **Number of Threads**: `12`

### 6.5 Result
- **Verification**: `C[0][0] = 4000.00`
- **Measured Execution Time**: `230.801465 seconds`

---

## 7. MPI Implementation

### 7.1 Description
MPI (Message Passing Interface) is used to implement distributed-memory parallelism.

The experiment uses four Ubuntu virtual machines:
- **Master** (Rank 0)
- **Worker1** (Rank 1)
- **Worker2** (Rank 2)
- **Worker3** (Rank 3)

Each MPI process has its own memory space. The matrix rows are divided equally among the four MPI processes:

$$\frac{4000 \text{ rows}}{4} = 1000 \text{ rows per process}$$

- **Rank 0** → 1000 rows
- **Rank 1** → 1000 rows
- **Rank 2** → 1000 rows
- **Rank 3** → 1000 rows

### 7.2 MPI Cluster Setup

| MPI Rank | Hostname | IP Address | Rows Processed |
| :--- | :--- | :--- | :--- |
| **Rank 0** | master | 192.168.28.128 | 1000 |
| **Rank 1** | worker1 | 192.168.28.129 | 1000 |
| **Rank 2** | worker2 | 192.168.28.130 | 1000 |
| **Rank 3** | worker3 | 192.168.28.131 | 1000 |

### 7.3 MPI Data Flow

```text
                    Matrix A
                       │
                  MPI_Scatter
                       │
        ┌──────────────┼──────────────┐
        │              │              │
     Rank 0         Rank 1         Rank 2         Rank 3
   1000 rows      1000 rows      1000 rows      1000 rows
        │              │              │              │
        └──────────────┼──────────────┘
                       │
                  Local Compute (with Broadcasted Matrix B)
                       │
                  MPI_Gather
                       │
                       ▼
                Complete Matrix C
```

Matrix $B$ is distributed to all worker processes using `MPI_Bcast`.

### 7.4 Hostfile (`hosts`)
```text
master slots=1
worker1 slots=1
worker2 slots=1
worker3 slots=1
```

### 7.5 Compilation
```bash
mpicc -O2 matrix_mpi.c -o matrix_mpi
```
*(The executable is copied to all worker nodes)*

### 7.6 Execution
```bash
env -u DISPLAY mpirun -np 4 --hostfile hosts ./matrix_mpi
```

### 7.7 Result
- **Number of MPI Processes**: `4`
- **Verification**: `C[0][0] = 4000.00`
- **Measured Execution Time**: `138.772617 seconds`

---

## 8. CUDA Implementation

### 8.1 Description
CUDA is used to perform massively parallel matrix multiplication on an NVIDIA GPU.

The CUDA implementation assigns one logical GPU thread to calculate one output element of matrix $C$.

```text
             GPU
              │
       ┌──────┴──────┐
       │             │
     Block         Block
       │             │
   Threads       Threads
       │             │
       └──────┬──────┘
              │
              ▼
        Matrix C
```

### 8.2 GPU Configuration
- **GPU**: NVIDIA GeForce RTX 2050
- **Compute Capability**: 8.6
- **Target Architecture**: `sm_86`

### 8.3 Matrix Configuration
- **Matrix Size**: $4000 \times 4000$

### 8.4 CUDA Block Configuration
- **Block Dimensions**: $16 \times 16$ threads
- **Threads per Block**: $16 \times 16 = 256$ threads/block

### 8.5 CUDA Grid Configuration
$$\frac{4000}{16} = 250 \text{ blocks per dimension}$$

- **Grid Dimensions**: $250 \times 250$ blocks
- **Total Blocks**: $250 \times 250 = 62,500$ blocks
- **Logical Threads**: $62,500 \times 256 = 16,000,000$ logical threads

These 16,000,000 threads correspond directly to the $4000 \times 4000$ output elements.

### 8.6 Compilation
```bash
nvcc -O2 -arch=sm_86 matrix_cuda.cu -o matrix_cuda
```

### 8.7 Execution
```bash
matrix_cuda.exe
```

### 8.8 Result
```text
CUDA Matrix Multiplication Completed
Matrix Size = 4000 x 4000
Grid Size = 250 x 250 blocks
Block Size = 16 x 16 threads
Kernel Execution Time = 0.316872 seconds
Total CUDA Phase Time = 0.343028 seconds
Verification C[0][0] = 4000.00
```
- **Kernel Execution Time**: `0.316872 seconds`
- **Total CUDA Phase Time**: `0.343028 seconds` *(includes host-device memory transfers and computation)*

---

## 9. Results

All four implementations produced the exact verified result:
```text
C[0][0] = 4000.00
```

### Actual Experimental Results

| Implementation | Computing Model | Configuration | Execution Time |
| :--- | :--- | :--- | :--- |
| **Sequential** | Single CPU | 1 execution flow | **417.205920 s** |
| **OpenMP** | Shared Memory | 12 CPU threads | **230.801465 s** |
| **MPI** | Distributed Memory | 4 processes / 4 VMs | **138.772617 s** |
| **CUDA** | GPU Parallelism | NVIDIA RTX 2050 | **0.343028 s** |

---

## 10. Performance Comparison

The sequential execution time is used as the baseline:

$$\text{Speedup} = \frac{\text{Sequential Execution Time}}{\text{Parallel Execution Time}}$$

### 10.1 Speedup Results

| Implementation | Execution Time | Speedup |
| :--- | :--- | :--- |
| **Sequential** | 417.205920 s | **1.00×** (Baseline) |
| **OpenMP** | 230.801465 s | **1.81×** |
| **MPI** | 138.772617 s | **3.01×** |
| **CUDA** | 0.343028 s | **1216.24×** |

### 10.2 Performance Visualization

```text
Execution Time (seconds) - Lower is Better

Sequential   ██████████████████████████████████████████ 417.205920 s
OpenMP       ███████████████████████                    230.801465 s
MPI          ██████████████                             138.772617 s
CUDA         ▏                                            0.343028 s
```

---
![Performance Analysis](results/result%20analysis.png)

## 11. Speedup Analysis

### OpenMP
$$\text{Speedup} = \frac{417.205920}{230.801465} \approx 1.81\times$$
OpenMP reduced execution time by distributing row computation across 12 CPU threads on shared memory.

### MPI
$$\text{Speedup} = \frac{417.205920}{138.772617} \approx 3.01\times$$
MPI achieved a $3.01\times$ speedup across 4 virtual machines, scaling well across independent nodes despite network communication overhead (`MPI_Scatter` and `MPI_Gather`).

### CUDA
- **Total Phase Speedup**:
  $$\text{Speedup}_{\text{total}} = \frac{417.205920}{0.343028} \approx 1216.24\times$$
- **Kernel-Only Speedup**:
  $$\text{Speedup}_{\text{kernel}} = \frac{417.205920}{0.316872} \approx 1316.92\times$$

The massive parallelism of 16,000,000 GPU threads running on the RTX 2050 delivered over $1200\times$ speedup compared to single-core CPU execution.

---

## 12. Verification

The same mathematical operation was executed across all four implementations:

Since $A[i][k] = 1.0$ and $B[k][j] = 1.0$ for all elements:
$$C[i][j] = \sum_{k=0}^{3999} (1.0 \times 1.0) = 4000.00$$

### Verification Summary

| Implementation | Expected Value | Output `C[0][0]` | Status |
| :--- | :--- | :--- | :---: |
| **Sequential** | 4000.00 | 4000.00 | ✅ Passed |
| **OpenMP** | 4000.00 | 4000.00 | ✅ Passed |
| **MPI** | 4000.00 | 4000.00 | ✅ Passed |
| **CUDA** | 4000.00 | 4000.00 | ✅ Passed |

---

## 13. Observations

1. **Sequential**: Serves as the single-thread baseline ($417.21\text{ s}$). Without parallelism, CPU execution is bounded by serial loop throughput and cache hierarchy misses.
2. **OpenMP**: Achieved $1.81\times$ speedup with 12 threads ($230.80\text{ s}$). Memory bus contention and cache coherence traffic become primary bottlenecks when multiple threads access large memory structures concurrently.
3. **MPI**: Achieved $3.01\times$ speedup on 4 VMs ($138.77\text{ s}$). Distributed memory prevents cache-coherence bottlenecks, yielding near-linear scalability ($3.01/4 \approx 75\%$ parallel efficiency) despite communication overhead.
4. **CUDA**: Achieved dramatic acceleration ($0.343\text{ s}$, $1216\times$ speedup). The streaming multiprocessors (SMs) on the RTX 2050 hide memory latency through massive thread-level parallelism and high memory bandwidth.

---

## 14. Overall Comparison

```text
Sequential  (417.205920 s)  ──[ 1.00× ]── Baseline
     │
     ▼
  OpenMP    (230.801465 s)  ──[ 1.81× ]── Shared-Memory Parallelism
     │
     ▼
   MPI      (138.772617 s)  ──[ 3.01× ]── Distributed-Memory Parallelism
     │
     ▼
   CUDA       (0.343028 s)  ──[ 1216.24× ]── GPU Massively Parallel
```

---

## 15. Conclusion

This experiment evaluated a $4000 \times 4000$ matrix multiplication workload across four foundational parallel computing architectures:

- **Sequential CPU execution**: Baseline reference.
- **OpenMP shared-memory parallelism**: Multi-core threading with shared address space.
- **MPI distributed-memory parallelism**: Process-based message passing across distinct cluster nodes.
- **CUDA GPU parallelism**: Fine-grained massively parallel thread acceleration.

The results highlight that as the degree of parallelism increases and hardware is tailored for compute-dense linear algebra workloads, execution time drops from nearly 7 minutes to approximately a third of a second.

---

## Technologies Used

- **Languages**: C, C++
- **Parallel Frameworks**: OpenMP, Open MPI, NVIDIA CUDA
- **Compilers**: GCC, `nvcc`, Visual Studio Build Tools
- **Networking & Infra**: OpenSSH, VMware Workstation, Ubuntu, WSL

---

## Experiment Summary

| Parameter | Value |
| :--- | :--- |
| **Matrix Size** | $4000 \times 4000$ |
| **Input Values** | $1.0$ |
| **Expected Output** | $4000.00$ |
| **Sequential Time** | $417.205920\text{ s}$ |
| **OpenMP Time** | $230.801465\text{ s}$ |
| **MPI Time** | $138.772617\text{ s}$ |
| **CUDA Time** | $0.343028\text{ s}$ |
| **OpenMP Speedup** | $1.81\times$ |
| **MPI Speedup** | $3.01\times$ |
| **CUDA Speedup** | $1216.24\times$ |
| **CUDA GPU** | NVIDIA GeForce RTX 2050 (Compute 8.6) |
| **CUDA Block / Grid** | $16 \times 16$ threads / $250 \times 250$ blocks |
| **Verification** | `C[0][0] = 4000.00` |

---

## Author

**Sankalp Prakash Patil**  
*B.Tech — Computer Science & Artificial Intelligence*  
KLE Technological University
