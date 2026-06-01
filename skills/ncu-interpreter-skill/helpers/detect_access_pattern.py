#!/usr/bin/env python3
"""
Access Pattern Detector - 检测内存访问模式

基于内存访问效率自动识别访问模式类型（coalesced/strided/mixed/random）。
"""

from typing import Dict


def detect_access_pattern(memory_analysis: Dict) -> str:
    """
    检测内存访问模式

    参数：
        memory_analysis: 内存子系统分析结果，包含 load_efficiency 和 store_efficiency

    返回：
        str: 访问模式类型
            - 'coalesced': 合并访问（效率 > 80%）
            - 'strided': 跨步访问（效率 50-80%）
            - 'mixed': 混合模式（效率 25-50%）
            - 'random': 随机访问（效率 < 25%）
    """
    load_eff = memory_analysis.get('load_efficiency', 100)
    store_eff = memory_analysis.get('store_efficiency', 100)
    avg_eff = (load_eff + store_eff) / 2

    if avg_eff > 80:
        return 'coalesced'
    elif avg_eff > 50:
        return 'strided'
    elif avg_eff > 25:
        return 'mixed'
    else:
        return 'random'


def get_pattern_description(pattern: str) -> str:
    """
    获取访问模式的描述信息

    参数：
        pattern: 访问模式类型

    返回：
        str: 模式描述
    """
    descriptions = {
        'coalesced': 'Coalesced Access - 连续线程访问连续内存，最优访问模式',
        'strided': 'Strided Access - 连续线程访问固定间隔内存，中等效率',
        'mixed': 'Mixed Access - 部分合并部分跨步，访问模式不规则',
        'random': 'Random Access - 线程访问不规则内存地址，效率很差'
    }

    return descriptions.get(pattern, 'Unknown access pattern')


def get_pattern_characteristics(pattern: str) -> Dict:
    """
    获取访问模式的特征信息

    参数：
        pattern: 访问模式类型

    返回：
        Dict: 特征信息
    """
    characteristics = {
        'coalesced': {
            'efficiency_range': '> 80%',
            'memory_transactions': '1-2× (最优)',
            'relative_performance': '100%',
            'optimization_priority': '低（已优化）'
        },
        'strided': {
            'efficiency_range': '50-80%',
            'memory_transactions': '2-4×',
            'relative_performance': '50-80%',
            'optimization_priority': '中'
        },
        'mixed': {
            'efficiency_range': '25-50%',
            'memory_transactions': '4-16×',
            'relative_performance': '25-50%',
            'optimization_priority': '高'
        },
        'random': {
            'efficiency_range': '< 25%',
            'memory_transactions': '16-32×',
            'relative_performance': '10-25%',
            'optimization_priority': '很高'
        }
    }

    return characteristics.get(pattern, {})


def get_optimization_suggestions(pattern: str) -> list:
    """
    根据访问模式获取优化建议

    参数：
        pattern: 访问模式类型

    返回：
        list: 优化建议列表
    """
    suggestions = {
        'coalesced': [
            '保持当前访问模式',
            '关注其他瓶颈（带宽、计算、占用率）'
        ],
        'strided': [
            '数据布局转换（AoS → SoA）',
            '使用共享内存进行数据重用',
            '转置数据改为行主序访问',
            '使用向量化访问（float4, int4）'
        ],
        'mixed': [
            '分析并分离规则和不规则访问',
            '使用共享内存缓存不规则访问',
            '重组计算以提高局部性'
        ],
        'random': [
            '排序或重组数据以提高局部性',
            '使用共享内存缓存频繁访问的数据',
            '考虑使用纹理内存（只读数据）',
            '批量处理相似的访问'
        ]
    }

    return suggestions.get(pattern, ['No specific suggestions available'])


def analyze_access_efficiency(load_eff: float, store_eff: float) -> Dict:
    """
    详细分析 load 和 store 效率

    参数：
        load_eff: Load 效率百分比
        store_eff: Store 效率百分比

    返回：
        Dict: 分析结果
    """
    analysis = {
        'load_efficiency': load_eff,
        'store_efficiency': store_eff,
        'average_efficiency': (load_eff + store_eff) / 2,
        'load_status': 'good' if load_eff > 80 else 'medium' if load_eff > 50 else 'poor',
        'store_status': 'good' if store_eff > 80 else 'medium' if store_eff > 50 else 'poor',
        'issues': []
    }

    # 识别问题
    if load_eff < 50:
        analysis['issues'].append('Poor load efficiency - memory reads are not coalesced')
    if store_eff < 50:
        analysis['issues'].append('Poor store efficiency - memory writes are not coalesced')
    if abs(load_eff - store_eff) > 30:
        analysis['issues'].append('Large difference between load and store efficiency')

    return analysis


def main():
    """命令行接口示例"""
    # 示例数据
    test_cases = [
        {'load_efficiency': 95, 'store_efficiency': 92, 'name': 'Coalesced'},
        {'load_efficiency': 65, 'store_efficiency': 58, 'name': 'Strided'},
        {'load_efficiency': 42, 'store_efficiency': 38, 'name': 'Mixed'},
        {'load_efficiency': 18, 'store_efficiency': 15, 'name': 'Random'}
    ]

    for case in test_cases:
        memory_analysis = {
            'load_efficiency': case['load_efficiency'],
            'store_efficiency': case['store_efficiency']
        }

        pattern = detect_access_pattern(memory_analysis)
        characteristics = get_pattern_characteristics(pattern)

        print(f"\n{'='*60}")
        print(f"Test Case: {case['name']}")
        print(f"{'='*60}")
        print(f"Detected Pattern: {pattern}")
        print(f"Description: {get_pattern_description(pattern)}")
        print(f"\nCharacteristics:")
        for key, value in characteristics.items():
            print(f"  {key}: {value}")
        print(f"\nOptimization Suggestions:")
        for i, suggestion in enumerate(get_optimization_suggestions(pattern), 1):
            print(f"  {i}. {suggestion}")


if __name__ == '__main__':
    main()
