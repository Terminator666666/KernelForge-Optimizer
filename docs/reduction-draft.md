# Reduction Kernel Optimization - Implementation Draft

## Task Contract

- **Task name**: Reduction Kernel Optimization
- **Objective**: Optimize parallel reduction for 16M float elements
- **Correctness requirements**: Result must match expected sum within 0.01% error
- **Performance target**: Achieve >100 GB/s memory bandwidth
- **Validation command**: `./test_reduction`
- **Promotion criteria**: Correct result + fair baseline comparison

## Current Baseline Analysis

### Problem Identified
- **Current Naive Implementation**: Uses only 1 thread (serial)
- **Execution time**: 740.559 ms
- **Bandwidth**: 0.09 GB/s
- **Issue**: This is NOT a fair baseline - it's a CPU-like serial implementation

### Correct Baseline Should Be
- Use multiple threads in parallel
- Each thread processes a portion of data
- Use atomicAdd for final reduction
- Expected time: ~1-5 ms (not 740 ms)

## Risks and Unknowns

1. **Baseline Fairness**: Current baseline is unfair (serial vs parallel)
2. **Precision**: Float accumulation order affects precision
3. **Atomic Contention**: atomicAdd can be a bottleneck

## Candidate Implementation Directions

### Candidate 1: Fair Naive Baseline (Priority: HIGH)
- Use grid-stride loop with atomicAdd
- Each thread processes multiple elements
- Simple and correct
- Expected: 1-5 ms

### Candidate 2: Shared Memory Reduction (Current Optimized)
- Already implemented
- Uses shared memory + warp-level reduction
- Expected: 0.3-0.5 ms

### Candidate 3: Warp Shuffle Reduction
- Use __shfl_down_sync for warp reduction
- Avoid shared memory
- Expected: 0.2-0.4 ms

## Implementation Steps

1. ✅ Identify the problem (done)
2. ⏳ Implement fair naive baseline
3. ⏳ Re-run benchmarks
4. ⏳ Calculate realistic speedup
5. ⏳ Record results in candidates.jsonl
6. ⏳ Run NCU profiling
7. ⏳ Document findings

## Validation Commands

```bash
# Compile
nvcc -O3 -arch=sm_89 examples/reduction_optimized.cu -o test_reduction

# Run
./test_reduction

# Verify
# - Check result matches expected (16777216)
# - Check error < 0.01%
# - Check speedup is reasonable (5-10x, not 2000x)
```

## Evidence Required

- Execution time for both versions
- Memory bandwidth utilization
- Correctness verification
- NCU profiling data
- Speedup calculation

## Expected Realistic Speedup

- Fair Naive: ~2-5 ms
- Optimized: ~0.3 ms
- **Realistic Speedup: 7-15x** (not 2296x)
