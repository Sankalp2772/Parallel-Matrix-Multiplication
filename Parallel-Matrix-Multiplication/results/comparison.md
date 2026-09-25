# Experimental Performance Comparison

## Experiment Summary

This document summarizes the performance results obtained from running the $4000 \times 4000$ matrix multiplication problem across four distinct architectures.

### Experimental Hardware & Specs
- **CPU**: Host system CPU (12 logical threads tested in OpenMP)
- **GPU**: NVIDIA GeForce RTX 2050 (4 GB GDDR6, Compute Capability 8.6)
- **MPI Cluster**: 4 Virtual Machines (Ubuntu on VMware, 1 master + 3 workers)

---

## 1. Execution Times & Speedups

| Model | Implementation | Configuration | Execution Time (s) | Speedup vs Sequential |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | Sequential CPU | 1 thread / 1 process | `417.205920` | `1.00×` |
| **Shared Memory** | OpenMP | 12 threads | `230.801465` | `1.81×` |
| **Distributed Memory** | MPI | 4 nodes (1 master, 3 workers) | `138.772617` | `3.01×` |
| **Massively Parallel GPU** | CUDA | 250×250 blocks, 16×16 threads | `0.343028` (total) / `0.316872` (kernel) | `1216.24×` (total) / `1316.92×` (kernel) |

---

## 2. Verification

All models calculated:
$$C[i][j] = \sum_{k=0}^{3999} (1.0 \times 1.0) = 4000.00$$

Every implementation verified `C[0][0] = 4000.00`.

---

## 3. Key Findings

1. **Sequential CPU**: Bounded by single-core instruction throughput and memory latency.
2. **OpenMP (Multi-threading)**: Delivered $1.81\times$ speedup. Limited by memory bandwidth contention and cache snooping between cores.
3. **MPI (Distributed Memory)**: Achieved near-linear scaling ($3.01\times$ on 4 nodes, ~75% efficiency) because each VM operated on its own isolated memory space.
4. **CUDA (GPU)**: Showed tremendous acceleration ($>1200\times$), completing the computation in roughly 340 milliseconds due to high parallelism and wide memory buses.
