# GPU Architecture Specifications

## Overview

GPU 架构规格是 NCU Interpreter 进行性能分析的基础数据。不同 GPU 架构的峰值性能、内存带宽、资源限制等参数差异很大，必须使用正确的规格才能得到准确的诊断结果。

## 关键规格参数

### 1. 峰值内存带宽 (Peak Memory Bandwidth)

**定义**：GPU 与全局内存之间的理论最大数据传输速度

**单位**：GB/s（Gigabytes per second）

**用途**：
- 计算内存带宽利用率
- Roofline 模型分析
- 判断 memory-bound 瓶颈

**测量方法**：
```
Peak Bandwidth = Memory Clock × Memory Bus Width × 2 (DDR) / 8
```

### 2. 峰值计算性能 (Peak Compute Throughput)

**定义**：GPU 的理论最大浮点运算速度

**单位**：TFLOPS（Tera Floating Point Operations Per Second）

**类型**：
- **FP32**：单精度浮点
- **FP16**：半精度浮点
- **INT8**：8位整数
- **Tensor Core**：专用矩阵运算单元

**用途**：
- 计算计算利用率
- Roofline 模型分析
- 判断 compute-bound 瓶颈

### 3. SM 数量 (Streaming Multiprocessor Count)

**定义**：GPU 上的流式多处理器数量

**用途**：
- 计算理论最大并行线程数
- 占用率分析
- 性能扩展性评估

### 4. 每 SM 最大线程数 (Max Threads per SM)

**定义**：单个 SM 可以同时调度的最大线程数

**典型值**：
- Ampere/Ada (RTX 30/40): 1536 threads
- Hopper (H100): 2048 threads
- Turing (RTX 20): 1024 threads

**用途**：
- 计算理论占用率
- 优化 block size

### 5. 寄存器文件大小 (Register File Size)

**定义**：每个 SM 的寄存器总数

**典型值**：65536 registers per SM

**用途**：
- 计算寄存器限制的占用率
- 识别寄存器瓶颈

### 6. 共享内存大小 (Shared Memory Size)

**定义**：每个 SM 的共享内存容量

**典型值**：
- Ampere/Ada: 100 KB (可配置)
- Hopper: 228 KB
- Turing: 64 KB

**用途**：
- 计算共享内存限制的占用率
- 优化 tile size

## 支持的 GPU 架构

### NVIDIA GeForce RTX 50 系列

#### RTX 5090 (Blackwell)
```yaml
name: RTX 5090
architecture: Blackwell
compute_capability: 10.0
peak_bandwidth_gbps: 1792
peak_tflops_fp32: 125.0
peak_tflops_fp16: 250.0
peak_tflops_tensor: 1000.0
sm_count: 170
max_threads_per_sm: 1536
max_registers_per_sm: 65536
max_shared_memory_per_sm: 102400
l2_cache_size_mb: 96
```

#### RTX 5080
```yaml
name: RTX 5080
architecture: Blackwell
compute_capability: 10.0
peak_bandwidth_gbps: 960
peak_tflops_fp32: 90.0
peak_tflops_fp16: 180.0
peak_tflops_tensor: 720.0
sm_count: 120
max_threads_per_sm: 1536
max_registers_per_sm: 65536
max_shared_memory_per_sm: 102400
l2_cache_size_mb: 64
```

### NVIDIA GeForce RTX 40 系列 (Ada Lovelace)

#### RTX 4090
```yaml
name: RTX 4090
architecture: Ada Lovelace
compute_capability: 8.9
peak_bandwidth_gbps: 1008
peak_tflops_fp32: 82.6
peak_tflops_fp16: 165.2
peak_tflops_tensor: 660.6
sm_count: 128
max_threads_per_sm: 1536
max_registers_per_sm: 65536
max_shared_memory_per_sm: 102400
l2_cache_size_mb: 72
```

#### RTX 4080
```yaml
name: RTX 4080
architecture: Ada Lovelace
compute_capability: 8.9
peak_bandwidth_gbps: 716
peak_tflops_fp32: 48.7
peak_tflops_fp16: 97.4
peak_tflops_tensor: 389.6
sm_count: 76
max_threads_per_sm: 1536
max_registers_per_sm: 65536
max_shared_memory_per_sm: 102400
l2_cache_size_mb: 64
```

#### RTX 4070 Ti
```yaml
name: RTX 4070 Ti
architecture: Ada Lovelace
compute_capability: 8.9
peak_bandwidth_gbps: 504
peak_tflops_fp32: 40.1
peak_tflops_fp16: 80.2
peak_tflops_tensor: 320.8
sm_count: 60
max_threads_per_sm: 1536
max_registers_per_sm: 65536
max_shared_memory_per_sm: 102400
l2_cache_size_mb: 48
```

### NVIDIA GeForce RTX 30 系列 (Ampere)

#### RTX 3090
```yaml
name: RTX 3090
architecture: Ampere
compute_capability: 8.6
peak_bandwidth_gbps: 936
peak_tflops_fp32: 35.6
peak_tflops_fp16: 71.0
peak_tflops_tensor: 284.0
sm_count: 82
max_threads_per_sm: 1536
max_registers_per_sm: 65536
max_shared_memory_per_sm: 102400
l2_cache_size_mb: 6
```

#### RTX 3080
```yaml
name: RTX 3080
architecture: Ampere
compute_capability: 8.6
peak_bandwidth_gbps: 760
peak_tflops_fp32: 29.8
peak_tflops_fp16: 59.5
peak_tflops_tensor: 238.0
sm_count: 68
max_threads_per_sm: 1536
max_registers_per_sm: 65536
max_shared_memory_per_sm: 102400
l2_cache_size_mb: 5
```

#### RTX 3070
```yaml
name: RTX 3070
architecture: Ampere
compute_capability: 8.6
peak_bandwidth_gbps: 448
peak_tflops_fp32: 20.3
peak_tflops_fp16: 40.6
peak_tflops_tensor: 162.6
sm_count: 46
max_threads_per_sm: 1536
max_registers_per_sm: 65536
max_shared_memory_per_sm: 102400
l2_cache_size_mb: 4
```

### NVIDIA Data Center GPUs

#### H100 (Hopper)
```yaml
name: H100
architecture: Hopper
compute_capability: 9.0
peak_bandwidth_gbps: 3350  # HBM3
peak_tflops_fp32: 67.0
peak_tflops_fp16: 1979.0
peak_tflops_tensor: 3958.0
sm_count: 132
max_threads_per_sm: 2048
max_registers_per_sm: 65536
max_shared_memory_per_sm: 233472
l2_cache_size_mb: 50
```

#### A100 (Ampere)
```yaml
name: A100
architecture: Ampere
compute_capability: 8.0
peak_bandwidth_gbps: 1555  # HBM2e
peak_tflops_fp32: 19.5
peak_tflops_fp16: 312.0
peak_tflops_tensor: 624.0
sm_count: 108
max_threads_per_sm: 2048
max_registers_per_sm: 65536
max_shared_memory_per_sm: 167936
l2_cache_size_mb: 40
```

#### V100 (Volta)
```yaml
name: V100
architecture: Volta
compute_capability: 7.0
peak_bandwidth_gbps: 900  # HBM2
peak_tflops_fp32: 15.7
peak_tflops_fp16: 125.0
peak_tflops_tensor: 125.0
sm_count: 80
max_threads_per_sm: 2048
max_registers_per_sm: 65536
max_shared_memory_per_sm: 98304
l2_cache_size_mb: 6
```

#### T4 (Turing)
```yaml
name: T4
architecture: Turing
compute_capability: 7.5
peak_bandwidth_gbps: 320
peak_tflops_fp32: 8.1
peak_tflops_fp16: 65.0
peak_tflops_tensor: 260.0
sm_count: 40
max_threads_per_sm: 1024
max_registers_per_sm: 65536
max_shared_memory_per_sm: 65536
l2_cache_size_mb: 4
```

## Ridge Point 计算

Ridge Point 是 Roofline 模型的关键参数，表示从 memory-bound 转换到 compute-bound 的算术强度临界值。

**计算公式**：
```
Ridge Point = Peak FP32 TFLOPS / Peak Bandwidth (GB/s)
```

**各 GPU 的 Ridge Point**：

| GPU | Peak FP32 (TFLOPS) | Peak BW (GB/s) | Ridge Point (FLOPs/Byte) |
|-----|-------------------|----------------|-------------------------|
| RTX 5090 | 125.0 | 1792 | 70 |
| RTX 4090 | 82.6 | 1008 | 82 |
| RTX 3090 | 35.6 | 936 | 38 |
| H100 | 67.0 | 3350 | 20 |
| A100 | 19.5 | 1555 | 13 |
| V100 | 15.7 | 900 | 17 |

**解释**：
- **低 Ridge Point (H100: 20)**：更容易达到 compute-bound，适合计算密集型任务
- **高 Ridge Point (RTX 4090: 82)**：更容易 memory-bound，需要高算术强度才能充分利用计算能力

## 架构差异

### Ampere vs Ada Lovelace

| 特性 | Ampere (RTX 30) | Ada Lovelace (RTX 40) |
|-----|----------------|---------------------|
| 制程 | Samsung 8nm | TSMC 4nm |
| SM 架构 | GA10x | AD10x |
| L2 Cache | 6 MB | 72 MB (12×) |
| FP32 性能 | 基准 | +2.3× |
| 功耗效率 | 基准 | +2× |
| Tensor Core | 3rd Gen | 4th Gen |

**关键改进**：
- **巨大的 L2 Cache**：72 MB vs 6 MB，显著提高缓存命中率
- **更高的时钟频率**：2.5+ GHz vs 1.9 GHz
- **改进的 Tensor Cores**：支持 FP8

### Consumer vs Data Center

| 特性 | Consumer (RTX) | Data Center (A/H100) |
|-----|---------------|---------------------|
| 内存类型 | GDDR6X | HBM2e/HBM3 |
| 内存带宽 | 1008 GB/s | 1555-3350 GB/s |
| ECC 内存 | 否 | 是 |
| 多 GPU 互连 | NVLink (部分) | NVLink/NVSwitch |
| 虚拟化支持 | 有限 | 完整 (MIG) |
| 价格 | $1000-2000 | $10000-30000 |

**使用场景**：
- **Consumer GPU**：个人开发、小规模训练、推理
- **Data Center GPU**：大规模训练、生产部署、多租户

## 使用 GPU 规格

### Python 实现

```python
def create_interpreter_for_gpu(gpu_name: str) -> NCUInterpreter:
    """根据 GPU 名称创建 NCU Interpreter"""
    
    gpu_specs_db = {
        'RTX 5090': {
            'peak_bandwidth_gbps': 1792,
            'peak_tflops_fp32': 125.0,
            'peak_tflops_fp16': 250.0,
            'sm_count': 170,
            'max_threads_per_sm': 1536
        },
        'RTX 4090': {
            'peak_bandwidth_gbps': 1008,
            'peak_tflops_fp32': 82.6,
            'peak_tflops_fp16': 165.2,
            'sm_count': 128,
            'max_threads_per_sm': 1536
        },
        'RTX 3090': {
            'peak_bandwidth_gbps': 936,
            'peak_tflops_fp32': 35.6,
            'peak_tflops_fp16': 71.0,
            'sm_count': 82,
            'max_threads_per_sm': 1536
        },
        'A100': {
            'peak_bandwidth_gbps': 1555,
            'peak_tflops_fp32': 19.5,
            'peak_tflops_fp16': 312,
            'sm_count': 108,
            'max_threads_per_sm': 2048
        },
        'H100': {
            'peak_bandwidth_gbps': 3350,
            'peak_tflops_fp32': 67,
            'peak_tflops_fp16': 1979,
            'sm_count': 132,
            'max_threads_per_sm': 2048
        },
        'V100': {
            'peak_bandwidth_gbps': 900,
            'peak_tflops_fp32': 15.7,
            'peak_tflops_fp16': 125,
            'sm_count': 80,
            'max_threads_per_sm': 2048
        }
    }
    
    if gpu_name not in gpu_specs_db:
        raise ValueError(f"Unknown GPU: {gpu_name}")
    
    return NCUInterpreter(gpu_specs_db[gpu_name])
```

### 自动检测 GPU

```python
import subprocess

def detect_current_gpu():
    """自动检测当前 GPU"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True,
            text=True
        )
        gpu_name = result.stdout.strip()
        
        # 简化 GPU 名称
        if 'RTX 4090' in gpu_name:
            return 'RTX 4090'
        elif 'RTX 3090' in gpu_name:
            return 'RTX 3090'
        elif 'A100' in gpu_name:
            return 'A100'
        # ... 其他 GPU
        
        return gpu_name
    except Exception as e:
        print(f"Failed to detect GPU: {e}")
        return None
```

## 规格数据来源

### 官方来源

1. **NVIDIA 官方规格表**
   - https://www.nvidia.com/en-us/geforce/graphics-cards/
   - https://www.nvidia.com/en-us/data-center/

2. **CUDA Toolkit 文档**
   - Compute Capability 表
   - GPU 架构白皮书

3. **nvidia-smi 工具**
   ```bash
   nvidia-smi --query-gpu=name,memory.total,clocks.max.sm --format=csv
   ```

### 第三方来源

1. **TechPowerUp GPU Database**
   - https://www.techpowerup.com/gpu-specs/

2. **WikiChip**
   - https://en.wikichip.org/wiki/nvidia

## 注意事项

### 1. 理论 vs 实际性能

**理论峰值**：在理想条件下的最大性能
**实际性能**：通常为理论峰值的 50-90%

**影响因素**：
- 内存访问模式
- 占用率
- 指令混合
- 温度和功耗限制

### 2. Boost Clock vs Base Clock

**Base Clock**：保证的最低时钟频率
**Boost Clock**：在温度和功耗允许时的最高频率

**建议**：使用 Boost Clock 计算峰值性能（更接近实际）

### 3. 不同精度的性能

| 精度 | RTX 4090 | A100 | 说明 |
|-----|---------|------|------|
| FP32 | 82.6 TFLOPS | 19.5 TFLOPS | 单精度 |
| FP16 | 165.2 TFLOPS | 312 TFLOPS | 半精度（2× FP32） |
| Tensor Core | 660.6 TFLOPS | 624 TFLOPS | 矩阵运算专用 |

**选择建议**：
- 科学计算：FP32
- 深度学习训练：FP16 + Tensor Cores
- 推理：FP16/INT8

## 参考资料

- NVIDIA GPU Architecture Whitepapers
- CUDA C Programming Guide - Compute Capabilities
- TechPowerUp GPU Database
- NVIDIA Data Center GPU Documentation
