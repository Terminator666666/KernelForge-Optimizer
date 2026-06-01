# 策略库概述

## 简介

Strategy Library 是 KernelForge-Optimizer 的核心组件，提供经过验证的 CUDA 优化策略模板库。每个策略包含：

- **适用场景**：算子类型、瓶颈类型、GPU 架构要求
- **参数规则**：如何选择最优参数（tile size、block size 等）
- **代码模板**：可直接使用的完整 CUDA 实现
- **性能预期**：预期加速比和优化效果

## 策略分类

### 1. 内存优化策略

针对内存带宽和延迟瓶颈的优化：

#### matmul_tiling（矩阵乘法分块）
- **原理**：使用共享内存缓存数据块，减少全局内存访问
- **适用**：矩阵乘法、GEMM
- **瓶颈**：内存带宽、内存延迟
- **加速**：2-5×
- **关键参数**：TILE_SIZE（16/32/64/128）

#### vectorized_memory（向量化访问）
- **原理**：使用 float4/int4 向量化指令，减少内存事务
- **适用**：Element-wise 操作、连续内存访问
- **瓶颈**：内存带宽
- **加速**：1.5-3×
- **关键参数**：VECTOR_SIZE（2/4）

#### bank_conflict_free（避免 Bank Conflict）
- **原理**：通过 padding 避免共享内存 bank conflict
- **适用**：矩阵转置、使用共享内存的算子
- **瓶颈**：内存延迟
- **加速**：1.5-3×
- **关键参数**：PADDING（0/1/2）

### 2. 计算优化策略

针对计算能力瓶颈的优化：

#### tensor_core（Tensor Core 优化）
- **原理**：使用 Tensor Core 专用硬件加速矩阵运算
- **适用**：矩阵乘法、卷积
- **瓶颈**：计算能力
- **加速**：2-10×（取决于精度）
- **关键参数**：PRECISION（FP16/BF16/INT8/FP8）、WMMA_M/N/K
- **要求**：compute capability ≥ 7.0（Volta+）

#### reduction_warp_primitives（Warp 级归约）
- **原理**：使用 warp shuffle 指令，消除共享内存原子操作
- **适用**：归约操作（sum、max、min）
- **瓶颈**：内存延迟、占用率
- **加速**：3-8×
- **关键参数**：BLOCK_SIZE（128/256/512）、ELEMENTS_PER_THREAD（1/2/4/8）

### 3. 算子融合策略

针对 kernel 启动开销和中间数据传输的优化：

#### kernel_fusion（算子融合）
- **原理**：合并多个 element-wise 操作为单个 kernel
- **适用**：串联的 element-wise 操作
- **瓶颈**：内存带宽、kernel 启动开销
- **加速**：2-4×
- **关键参数**：NUM_OPERATIONS（2-10）

### 4. 占用率优化策略

针对低占用率的优化：

#### occupancy_tuning（占用率优化）
- **原理**：调整 block size 和资源使用，提高 SM 占用率
- **适用**：任何算子
- **瓶颈**：低占用率
- **加速**：1.5-4×
- **关键参数**：BLOCK_SIZE（64/128/256/512/1024）、MAX_REGISTERS

#### cooperative_groups（协作组）
- **原理**：使用协作组提供更灵活的同步机制
- **适用**：复杂同步、多 block 协作
- **瓶颈**：占用率、同步开销
- **加速**：1.2-2×
- **要求**：compute capability ≥ 6.0

#### persistent_threads（持久线程）
- **原理**：线程持久化，减少 kernel 启动开销
- **适用**：大量小 kernel、流式处理
- **瓶颈**：kernel 启动开销
- **加速**：2-5×

## 策略选择流程

### 1. 识别瓶颈类型

根据 NCU 分析结果确定主要瓶颈：

```
memory_bandwidth → 内存带宽饱和
memory_latency   → 内存访问延迟高
compute_bound    → 计算能力限制
occupancy        → SM 占用率过低
```

### 2. 匹配适用策略

根据瓶颈类型和算子类型匹配策略：

**内存带宽瓶颈**：
1. vectorized_memory（优先）
2. kernel_fusion
3. matmul_tiling

**内存延迟瓶颈**：
1. matmul_tiling（优先）
2. reduction_warp_primitives
3. bank_conflict_free

**计算能力瓶颈**：
1. tensor_core（优先，如果 GPU 支持）
2. occupancy_tuning

**低占用率**：
1. occupancy_tuning（优先）
2. reduction_warp_primitives
3. cooperative_groups

### 3. 检查 GPU 架构兼容性

确保策略支持当前 GPU：

- **Kepler (3.0+)**：支持 warp shuffle
- **Volta (7.0+)**：支持 Tensor Cores
- **Ampere (8.0+)**：改进的 Tensor Cores、异步拷贝
- **Ada Lovelace (8.9+)**：第 4 代 Tensor Cores、巨大 L2 缓存
- **Hopper (9.0+)**：Transformer Engine、FP8 支持

### 4. 选择参数

根据硬件规格和数据大小选择最优参数：

**TILE_SIZE 选择**：
- 考虑共享内存大小：`TILE_SIZE² × sizeof(type) ≤ 48KB`
- 考虑寄存器压力：更大的 tile 需要更多寄存器
- 考虑矩阵大小：最好能整除矩阵维度
- 推荐值：16（小矩阵）、32（通用）、64（大矩阵）

**BLOCK_SIZE 选择**：
- 考虑占用率：更大的 block size 通常提高占用率
- 考虑资源限制：寄存器、共享内存
- 必须是 32 的倍数（warp size）
- 推荐值：128-512

**VECTOR_SIZE 选择**：
- 检查内存对齐：数据必须对齐到 `VECTOR_SIZE × sizeof(type)`
- 检查数组大小：必须是 VECTOR_SIZE 的倍数
- 推荐值：4（float4/int4）

## 策略组合

某些场景下可以组合多个策略：

### 组合 1：matmul_tiling + vectorized_memory
- **场景**：大矩阵乘法，内存带宽瓶颈
- **效果**：分块减少访问次数，向量化提高带宽利用率
- **预期加速**：5-8×

### 组合 2：matmul_tiling + tensor_core
- **场景**：大矩阵乘法，可以使用 FP16/BF16
- **效果**：分块 + Tensor Core 硬件加速
- **预期加速**：10-20×

### 组合 3：kernel_fusion + vectorized_memory
- **场景**：多个 element-wise 操作串联
- **效果**：减少 kernel 启动和中间数据传输，向量化提高带宽
- **预期加速**：4-6×

### 组合 4：reduction_warp_primitives + occupancy_tuning
- **场景**：大数组归约，占用率低
- **效果**：warp shuffle 减少同步，提高占用率隐藏延迟
- **预期加速**：8-12×

## GPU 架构特定建议

### Volta (7.0)
- **推荐策略**：tensor_core、matmul_tiling
- **特性**：首次支持 Tensor Cores、独立线程调度
- **注意**：Tensor Core 需要 FP16/BF16 精度

### Ampere (8.0)
- **推荐策略**：tensor_core、matmul_tiling、vectorized_memory
- **特性**：改进的 Tensor Cores、异步拷贝、大 L2 缓存
- **注意**：可以使用 async copy 进一步优化

### Ada Lovelace (8.9)
- **推荐策略**：tensor_core、matmul_tiling、kernel_fusion
- **特性**：第 4 代 Tensor Cores、巨大 L2 缓存、Shader Execution Reordering
- **注意**：L2 缓存很大（72MB），可以缓存更多数据

### Hopper (9.0)
- **推荐策略**：tensor_core、cooperative_groups、matmul_tiling
- **特性**：Transformer Engine、FP8 支持、Thread Block Clusters
- **注意**：FP8 可以获得最高加速（需要 Hopper）

## 性能验证

### 验证指标

优化后应该验证以下指标：

1. **加速比**：`优化后时间 / 优化前时间`
2. **带宽利用率**：`achieved_bandwidth / peak_bandwidth`
3. **计算利用率**：`achieved_flops / peak_flops`
4. **占用率**：`active_warps / max_warps`

### 验证方法

使用 NCU 进行性能分析：

```bash
# 分析优化前的 kernel
ncu --set full -o baseline ./program

# 分析优化后的 kernel
ncu --set full -o optimized ./program

# 对比结果
ncu --import baseline.ncu-rep optimized.ncu-rep
```

### 成功标准

- **加速比**：达到预期加速比的 70% 以上
- **带宽利用率**：
  - Memory-bound kernel：≥ 70%
  - Compute-bound kernel：≥ 50%
- **计算利用率**：
  - Compute-bound kernel：≥ 60%
  - Memory-bound kernel：≥ 30%
- **占用率**：≥ 50%（除非是 compute-bound）

## 常见问题

### Q1: 如何选择第一个优化策略？

**A**: 根据 NCU 分析的瓶颈类型：
- 内存带宽瓶颈 → vectorized_memory 或 matmul_tiling
- 内存延迟瓶颈 → matmul_tiling 或 reduction_warp_primitives
- 计算能力瓶颈 → tensor_core（如果支持）
- 低占用率 → occupancy_tuning

### Q2: 优化后性能反而下降了怎么办？

**A**: 可能的原因：
1. **参数选择不当**：重新调整 TILE_SIZE、BLOCK_SIZE
2. **引入了新的瓶颈**：使用 NCU 重新分析
3. **策略不适用**：尝试其他策略
4. **实现有误**：检查代码逻辑

### Q3: 可以同时应用多个策略吗？

**A**: 可以，但需要注意：
- 确保策略之间不冲突
- 逐个应用，验证每个策略的效果
- 某些策略天然组合（如 matmul_tiling + tensor_core）

### Q4: 如何知道优化已经到极限？

**A**: 当满足以下条件之一：
- 带宽利用率 ≥ 90%（memory-bound）
- 计算利用率 ≥ 80%（compute-bound）
- 性能接近理论峰值（Roofline 模型）
- 多次优化后性能提升 < 5%

### Q5: 不同 GPU 需要不同的参数吗？

**A**: 是的，主要考虑：
- **共享内存大小**：影响 TILE_SIZE
- **寄存器数量**：影响 BLOCK_SIZE 和展开因子
- **SM 数量**：影响 grid size
- **峰值带宽/算力**：影响性能预期

## 参考资料

- **CUDA C Best Practices Guide**: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- **CUDA C Programming Guide**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- **NVIDIA GPU Architecture Whitepapers**: https://www.nvidia.com/en-us/data-center/resources/gpu-architecture/
- **Nsight Compute Documentation**: https://docs.nvidia.com/nsight-compute/
- **Kernel Design Agents (KDA)**: https://github.com/mit-han-lab/kernel-design-agents

## 下一步

- 阅读具体策略的详细文档（01-05）
- 查看 CUDA 代码模板（templates/）
- 使用 NCU Interpreter 分析你的 kernel
- 根据诊断结果选择合适的策略
