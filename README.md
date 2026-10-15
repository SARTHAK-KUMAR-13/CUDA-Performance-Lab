# CUDA Performance Lab

A small study comparing CPU and GPU execution using hand-written CUDA kernels.
The focus is on **correctness testing** and on **separating kernel time from
end-to-end time** (including host-device memory transfers) across input sizes.

> **Status: in progress.** See the checklist below for what is done.

## Goals
- Implement basic CUDA kernels and verify each against a CPU baseline.
- Measure how execution time scales with input size.
- Show how much of the end-to-end time is data movement vs. compute.

## Status
- [x] Vector addition: kernel, CPU baseline, correctness check, timing
- [ ] Parallel reduction
- [ ] Results table and analysis
- [ ] Extension (optional): tiled matrix multiplication

## Method
- **Correctness:** each GPU output is compared element by element with the CPU result.
- **Kernel time:** measured with CUDA events around the kernel launch only.
- **End-to-end time:** measured with CUDA events around host-to-device copies,
  the kernel, and the device-to-host copy.
- **CPU baseline time:** measured with `std::chrono`.
- **Warm-up:** one dummy launch before timing so first-call overhead is excluded.
- **Input sizes:** 2^20 to 2^26 elements.

## Files
| File | Description |
|---|---|
| `vec_add.cu` | Vector addition kernel, CPU baseline, correctness check, timing |

## Run (Google Colab, T4 GPU)
1. Runtime -> Change runtime type -> T4 GPU
2. Upload `vec_add.cu`, then run:

```bash
nvcc -O2 -arch=sm_75 -o vec_add vec_add.cu
./vec_add
```

## Results
Results will be added after measurement on a T4 GPU.

## Notes
- Written for learning purposes.
- Tested on Google Colab only.
