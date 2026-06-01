#!/usr/bin/env python3
"""
NCU Report Analyzer - 解析 NCU 报告并提取关键指标

从 NCU CSV 或 JSON 格式的报告中提取性能指标，转换为标准化的字典格式。
"""

import json
import csv
from typing import Dict, Optional
from pathlib import Path


def parse_ncu_csv(csv_path: str) -> Dict[str, float]:
    """
    解析 NCU CSV 报告

    参数：
        csv_path: NCU CSV 报告文件路径

    返回：
        Dict[str, float]: 指标名称到数值的映射
    """
    metrics = {}

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # NCU CSV 格式：Metric Name, Metric Value, Metric Unit
            metric_name = row.get('Metric Name', '').strip()
            metric_value = row.get('Metric Value', '').strip()

            if metric_name and metric_value:
                try:
                    # 尝试转换为浮点数
                    value = float(metric_value.replace(',', ''))
                    metrics[metric_name] = value
                except ValueError:
                    # 如果无法转换，跳过
                    continue

    return metrics


def parse_ncu_json(json_path: str) -> Dict[str, float]:
    """
    解析 NCU JSON 报告

    参数：
        json_path: NCU JSON 报告文件路径

    返回：
        Dict[str, float]: 指标名称到数值的映射
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    metrics = {}

    # NCU JSON 格式可能有多种结构，这里处理常见格式
    if isinstance(data, list) and len(data) > 0:
        # 通常第一个元素包含 kernel 信息
        kernel_data = data[0]

        # 提取 metrics
        if 'metrics' in kernel_data:
            for metric in kernel_data['metrics']:
                name = metric.get('name', '')
                value = metric.get('value', 0)
                if name:
                    metrics[name] = float(value)

    return metrics


def parse_ncu_report(report_path: str) -> Dict[str, float]:
    """
    自动检测格式并解析 NCU 报告

    参数：
        report_path: NCU 报告文件路径（CSV 或 JSON）

    返回：
        Dict[str, float]: 指标名称到数值的映射
    """
    path = Path(report_path)

    if not path.exists():
        raise FileNotFoundError(f"NCU report not found: {report_path}")

    # 根据文件扩展名选择解析器
    if path.suffix.lower() == '.csv':
        return parse_ncu_csv(report_path)
    elif path.suffix.lower() == '.json':
        return parse_ncu_json(report_path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def extract_key_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    """
    从完整的 NCU 指标中提取关键指标

    参数：
        metrics: 完整的 NCU 指标字典

    返回：
        Dict[str, float]: 关键指标字典
    """
    key_metrics = {}

    # 关键指标映射（NCU 指标名 -> 简化名）
    metric_mapping = {
        # 时间和吞吐量
        'duration': 'duration',
        'gpu__time_duration.sum': 'duration',

        # 内存带宽
        'dram__bytes.sum': 'dram__bytes.sum',
        'dram__bytes_read.sum': 'dram__bytes_read.sum',
        'dram__bytes_write.sum': 'dram__bytes_write.sum',

        # L1 Cache
        'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum': 'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum',
        'l1tex__t_sectors_pipe_lsu_mem_global_op_ld_hit.sum': 'l1tex__t_sectors_pipe_lsu_mem_global_op_ld_hit.sum',

        # L2 Cache
        'lts__t_sectors_op_read.sum': 'lts__t_sectors_op_read.sum',
        'lts__t_sectors_op_read_hit.sum': 'lts__t_sectors_op_read_hit.sum',

        # 内存效率
        'smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct': 'smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct',
        'smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct': 'smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct',

        # 占用率
        'sm__warps_active.avg.pct_of_peak_sustained_active': 'sm__warps_active.avg.pct_of_peak_sustained_active',

        # SM 效率
        'smsp__cycles_active.avg.pct_of_peak_sustained_elapsed': 'smsp__cycles_active.avg.pct_of_peak_sustained_elapsed',

        # Warp 效率
        'smsp__thread_inst_executed_per_inst_executed.ratio': 'smsp__thread_inst_executed_per_inst_executed.ratio',

        # 浮点运算
        'smsp__sass_thread_inst_executed_op_fadd_pred_on.sum': 'smsp__sass_thread_inst_executed_op_fadd_pred_on.sum',
        'smsp__sass_thread_inst_executed_op_fmul_pred_on.sum': 'smsp__sass_thread_inst_executed_op_fmul_pred_on.sum',
        'smsp__sass_thread_inst_executed_op_ffma_pred_on.sum': 'smsp__sass_thread_inst_executed_op_ffma_pred_on.sum',
    }

    # 提取关键指标
    for ncu_name, simple_name in metric_mapping.items():
        if ncu_name in metrics:
            key_metrics[simple_name] = metrics[ncu_name]

    return key_metrics


def main():
    """命令行接口示例"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analyze_ncu_report.py <ncu_report.csv|json>")
        sys.exit(1)

    report_path = sys.argv[1]

    try:
        # 解析报告
        metrics = parse_ncu_report(report_path)
        print(f"Parsed {len(metrics)} metrics from {report_path}")

        # 提取关键指标
        key_metrics = extract_key_metrics(metrics)
        print(f"\nKey metrics ({len(key_metrics)}):")
        for name, value in key_metrics.items():
            print(f"  {name}: {value}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
