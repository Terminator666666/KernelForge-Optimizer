#!/bin/bash
# 设置 CUDA 环境变量
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 运行优化引擎
cd /mnt/d/Agent/KernelForge-Optimizer
python automation/optimization_engine.py "$@"
