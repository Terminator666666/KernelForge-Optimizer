#!/bin/bash
# 编译和测试所有剩余算子的脚本

set -e  # 遇到错误立即退出

echo "================================================================================"
echo "KernelForge-Optimizer: 编译和测试剩余算子"
echo "================================================================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 创建输出目录
mkdir -p runs
mkdir -p profile
mkdir -p candidates

# 算子列表
OPERATORS=("softmax" "layernorm" "scan" "flash_attention" "fused_moe")

# 编译选项
NVCC_FLAGS="-O3 -arch=sm_89 -lcublas"

# 统计变量
TOTAL=0
COMPILED=0
PASSED=0
FAILED=0

echo "步骤 1: 编译所有算子"
echo "================================================================================"
echo ""

for op in "${OPERATORS[@]}"; do
    TOTAL=$((TOTAL + 1))

    echo "[$TOTAL/5] 编译 ${op}..."

    if [ -f "examples/${op}_baseline.cu" ]; then
        if nvcc examples/${op}_baseline.cu -o runs/test_${op} $NVCC_FLAGS 2>&1 | tee runs/${op}_compile.log; then
            echo -e "${GREEN}✅ ${op} 编译成功${NC}"
            COMPILED=$((COMPILED + 1))
        else
            echo -e "${RED}❌ ${op} 编译失败${NC}"
            echo "查看日志: runs/${op}_compile.log"
        fi
    else
        echo -e "${YELLOW}⚠️  ${op}_baseline.cu 不存在${NC}"
    fi
    echo ""
done

echo ""
echo "================================================================================"
echo "步骤 2: 运行性能测试"
echo "================================================================================"
echo ""

for op in "${OPERATORS[@]}"; do
    if [ -f "runs/test_${op}" ]; then
        echo "测试 ${op}..."
        echo "--------------------------------------------------------------------------------"

        if ./runs/test_${op} 2>&1 | tee runs/${op}_test.log; then
            echo -e "${GREEN}✅ ${op} 测试通过${NC}"
            PASSED=$((PASSED + 1))

            # 提取加速比（如果有）
            SPEEDUP=$(grep "加速比:" runs/${op}_test.log | grep -oP '\d+\.\d+' | head -1)
            if [ ! -z "$SPEEDUP" ]; then
                echo "   加速比: ${SPEEDUP}x"
            fi
        else
            echo -e "${RED}❌ ${op} 测试失败${NC}"
            FAILED=$((FAILED + 1))
        fi
        echo ""
    fi
done

echo ""
echo "================================================================================"
echo "步骤 3: 生成候选记录"
echo "================================================================================"
echo ""

for op in "${OPERATORS[@]}"; do
    if [ -f "runs/${op}_test.log" ]; then
        echo "记录 ${op} 候选方案..."

        # 提取性能数据
        BASELINE_TIME=$(grep "Baseline" runs/${op}_test.log | grep "执行时间" | grep -oP '\d+\.\d+' | head -1)
        OPT_TIME=$(grep "Optimized" runs/${op}_test.log | grep "执行时间" | grep -oP '\d+\.\d+' | head -1)
        SPEEDUP=$(grep "加速比:" runs/${op}_test.log | grep -oP '\d+\.\d+' | head -1)
        STATUS=$(grep "总体状态:" runs/${op}_test.log | grep -q "PASS" && echo "PASS" || echo "FAIL")

        # 生成 JSONL 记录
        cat >> candidates/${op}_candidates.jsonl << EOF
{"operator": "${op}", "timestamp": "$(date -Iseconds)", "baseline_time_ms": ${BASELINE_TIME:-0}, "optimized_time_ms": ${OPT_TIME:-0}, "speedup": ${SPEEDUP:-0}, "status": "${STATUS}", "file": "examples/${op}_baseline.cu"}
EOF

        echo -e "${GREEN}✅ ${op} 候选记录已保存${NC}"
    fi
done

echo ""
echo "================================================================================"
echo "步骤 4: 可选 - NCU Profiling"
echo "================================================================================"
echo ""
echo "NCU profiling 需要较长时间，是否运行？(y/N)"
read -t 10 -n 1 RUN_NCU || RUN_NCU="n"
echo ""

if [ "$RUN_NCU" = "y" ] || [ "$RUN_NCU" = "Y" ]; then
    for op in "${OPERATORS[@]}"; do
        if [ -f "runs/test_${op}" ]; then
            echo "NCU profiling: ${op}..."

            # Baseline profiling
            if ncu --set full -o profile/${op}_baseline runs/test_${op} 2>&1 | tee profile/${op}_ncu.log; then
                echo -e "${GREEN}✅ ${op} NCU profiling 完成${NC}"
            else
                echo -e "${YELLOW}⚠️  ${op} NCU profiling 失败（可选步骤）${NC}"
            fi
            echo ""
        fi
    done
else
    echo "跳过 NCU profiling"
    echo "提示：稍后可以手动运行："
    echo "  ncu --set full -o profile/<operator>_baseline runs/test_<operator>"
fi

echo ""
echo "================================================================================"
echo "📊 测试总结"
echo "================================================================================"
echo ""
echo "总计算子: ${TOTAL}"
echo -e "${GREEN}编译成功: ${COMPILED}${NC}"
echo -e "${GREEN}测试通过: ${PASSED}${NC}"
echo -e "${RED}测试失败: ${FAILED}${NC}"
echo ""

# 生成性能报告
echo "================================================================================"
echo "📈 性能报告"
echo "================================================================================"
echo ""
printf "%-20s %-15s %-15s %-10s %-10s\n" "算子" "Baseline (ms)" "Optimized (ms)" "加速比" "状态"
echo "--------------------------------------------------------------------------------"

for op in "${OPERATORS[@]}"; do
    if [ -f "runs/${op}_test.log" ]; then
        BASELINE_TIME=$(grep "Baseline" runs/${op}_test.log | grep "执行时间" | grep -oP '\d+\.\d+' | head -1)
        OPT_TIME=$(grep "Optimized" runs/${op}_test.log | grep "执行时间" | grep -oP '\d+\.\d+' | head -1)
        SPEEDUP=$(grep "加速比:" runs/${op}_test.log | grep -oP '\d+\.\d+' | head -1)
        STATUS=$(grep "总体状态:" runs/${op}_test.log | grep -q "PASS" && echo "✅ PASS" || echo "❌ FAIL")

        printf "%-20s %-15s %-15s %-10s %-10s\n" \
            "${op}" \
            "${BASELINE_TIME:-N/A}" \
            "${OPT_TIME:-N/A}" \
            "${SPEEDUP:-N/A}x" \
            "${STATUS}"
    fi
done

echo ""
echo "================================================================================"
echo "📁 输出文件"
echo "================================================================================"
echo ""
echo "编译产物: runs/test_*"
echo "测试日志: runs/*_test.log"
echo "候选记录: candidates/*_candidates.jsonl"
echo "NCU 报告: profile/*.ncu-rep (如果运行了 NCU)"
echo ""

echo "================================================================================"
echo "✅ 所有任务完成！"
echo "================================================================================"
echo ""

# 返回状态码
if [ $FAILED -eq 0 ]; then
    exit 0
else
    exit 1
fi
