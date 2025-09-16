#!/usr/bin/env python3
"""
深度分析原始Factory Detector的误报和漏报来源

This script provides detailed analysis of where false positives and false negatives
come from in the original factory detector.
"""

import json
import sys
import os
from collections import defaultdict
from typing import Dict, List

# Add project root to path to import factory_detector
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from factory_detector import ImprovedFactoryDetector


def analyze_false_positives_sources():
    """分析False Positives的具体来源"""
    
    print("=" * 80)
    print("原始DETECTOR FALSE POSITIVES 深度分析")
    print("=" * 80)
    
    # Load evaluation results
    with open('factory_detector_evaluation_results.json', 'r') as f:
        data = json.load(f)
    
    contracts = data['contracts']
    
    # Find false positives
    false_positives = [c for c in contracts if not c['is_factory_ground_truth'] and c['is_factory_detected']]
    
    print(f"总计False Positives: {len(false_positives)}个")
    
    # 按来源分类
    fp_by_source = defaultdict(list)
    for contract in false_positives:
        source = contract['source_type']
        fp_by_source[source].append(contract)
    
    print(f"\n按数据源分布:")
    for source, contracts_list in fp_by_source.items():
        print(f"  {source}: {len(contracts_list)}个 ({len(contracts_list)/len(false_positives)*100:.1f}%)")
    
    # 分析合约名称模式
    print(f"\n合约名称模式分析:")
    contract_names = []
    for contract in false_positives:
        if contract.get('verification_notes'):
            # Extract contract name from verification notes
            notes = contract['verification_notes']
            if 'Contract:' in notes:
                name = notes.split('Contract:')[-1].strip().split()[0]
                contract_names.append(name)
    
    # Count name patterns
    name_patterns = defaultdict(int)
    for name in contract_names:
        if 'Factory' in name:
            name_patterns['包含Factory关键字'] += 1
        elif 'Mint' in name:
            name_patterns['包含Mint关键字'] += 1
        elif 'Token' in name:
            name_patterns['包含Token关键字'] += 1
        elif 'Router' in name or 'Swap' in name:
            name_patterns['路由/交换合约'] += 1
        else:
            name_patterns['其他类型'] += 1
    
    print(f"  合约类型分布:")
    for pattern, count in name_patterns.items():
        print(f"    {pattern}: {count}个")
    
    # 详细分析前10个False Positives
    detector = ImprovedFactoryDetector()
    
    print(f"\n详细案例分析 (前10个):")
    for i, contract in enumerate(false_positives[:10]):
        print(f"\n{i+1}. 合约: {contract['address']}")
        
        # Extract contract name
        contract_name = "Unknown"
        if contract.get('verification_notes') and 'Contract:' in contract['verification_notes']:
            contract_name = contract['verification_notes'].split('Contract:')[-1].strip().split()[0]
        
        print(f"   合约名称: {contract_name}")
        print(f"   检测类型: {contract['factory_type']}")
        print(f"   CREATE操作: {contract.get('create_positions', 0)}个")
        print(f"   CREATE2操作: {contract.get('create2_positions', 0)}个")
        print(f"   执行时间: {contract.get('execution_time', 0)}ms")
        
        # 分析为什么被误判
        if 'Factory' in contract_name:
            print(f"   ❌ 误判原因: 合约名包含'Factory'但源码分析确认非工厂合约")
            print(f"   ❌ 问题: 字节码级检测与源码分析不一致")
        elif 'Mint' in contract_name:
            print(f"   ❌ 误判原因: Mint类合约可能包含CREATE用于铸造但非工厂模式")
        elif contract.get('create_positions', 0) > 0:
            print(f"   ❌ 误判原因: 检测到CREATE操作但实际用途非工厂创建")
        
        if contract.get('verification_notes'):
            print(f"   验证信息: {contract['verification_notes'][:100]}...")


def analyze_false_negatives_sources():
    """分析False Negatives的具体来源"""
    
    print("\n" + "=" * 80)
    print("原始DETECTOR FALSE NEGATIVES 深度分析")
    print("=" * 80)
    
    # Load evaluation results
    with open('factory_detector_evaluation_results.json', 'r') as f:
        data = json.load(f)
    
    contracts = data['contracts']
    
    # Find false negatives
    false_negatives = [c for c in contracts if c['is_factory_ground_truth'] and not c['is_factory_detected']]
    
    print(f"总计False Negatives: {len(false_negatives)}个")
    
    # 按来源分类
    fn_by_source = defaultdict(list)
    for contract in false_negatives:
        source = contract['source_type']
        fn_by_source[source].append(contract)
    
    print(f"\n按数据源分布:")
    for source, contracts_list in fn_by_source.items():
        print(f"  {source}: {len(contracts_list)}个 ({len(contracts_list)/len(false_negatives)*100:.1f}%)")
    
    # 分析CREATE操作执行次数
    print(f"\nCREATE操作执行统计:")
    create_counts = []
    for contract in false_negatives:
        if contract.get('verification_notes') and 'executed' in contract['verification_notes']:
            # Extract number of CREATE operations from notes
            notes = contract['verification_notes']
            if 'executed' in notes and 'CREATE' in notes:
                try:
                    # Look for pattern like "executed 4615 CREATE operations"
                    words = notes.split()
                    for i, word in enumerate(words):
                        if word == 'executed' and i+1 < len(words):
                            count = int(words[i+1])
                            create_counts.append(count)
                            break
                except:
                    pass
    
    if create_counts:
        create_counts.sort()
        print(f"  总计{len(create_counts)}个有执行统计的合约")
        print(f"  CREATE执行次数范围: {min(create_counts):,} - {max(create_counts):,}")
        print(f"  平均执行次数: {sum(create_counts)/len(create_counts):,.0f}")
        print(f"  中位数执行次数: {create_counts[len(create_counts)//2]:,}")
        
        # 按执行次数分组
        ranges = [
            (0, 100, "小规模 (≤100次)"),
            (101, 1000, "中规模 (101-1000次)"),
            (1001, 10000, "大规模 (1001-10000次)"),
            (10001, float('inf'), "超大规模 (>10000次)")
        ]
        
        for min_count, max_count, label in ranges:
            count = len([c for c in create_counts if min_count <= c <= max_count])
            if count > 0:
                print(f"  {label}: {count}个合约")
    
    # 分析合约地址模式
    print(f"\n合约地址模式分析:")
    address_patterns = defaultdict(int)
    
    for contract in false_negatives:
        addr = contract['address'].lower()
        if addr.startswith('0x000000000'):
            # Count leading zeros after 0x
            zero_count = 0
            for char in addr[2:]:
                if char == '0':
                    zero_count += 1
                else:
                    break
            if zero_count >= 8:
                address_patterns['CREATE2预计算地址 (≥8个前导0)'] += 1
            elif zero_count >= 4:
                address_patterns['可能的CREATE2地址 (4-7个前导0)'] += 1
            else:
                address_patterns['普通地址'] += 1
        else:
            address_patterns['普通地址'] += 1
    
    for pattern, count in address_patterns.items():
        print(f"  {pattern}: {count}个")
    
    # 详细分析字节码特征
    detector = ImprovedFactoryDetector()
    
    print(f"\n字节码分析 (前10个):")
    for i, contract in enumerate(false_negatives[:10]):
        print(f"\n{i+1}. 合约: {contract['address']}")
        
        # Extract CREATE count from notes
        create_executed = "Unknown"
        if contract.get('verification_notes'):
            notes = contract['verification_notes']
            if 'executed' in notes and 'CREATE' in notes:
                words = notes.split()
                for j, word in enumerate(words):
                    if word == 'executed' and j+1 < len(words):
                        try:
                            create_executed = f"{int(words[j+1]):,}次"
                            break
                        except:
                            pass
        
        print(f"   实际CREATE执行: {create_executed}")
        
        # Analyze bytecode
        bytecode = contract['bytecode']
        if bytecode.startswith('0x'):
            bytecode_clean = bytecode[2:]
        else:
            bytecode_clean = bytecode
            
        create_bytes = bytecode_clean.lower().count('f0')
        create2_bytes = bytecode_clean.lower().count('f5')
        
        print(f"   字节码中CREATE字节: {create_bytes}个")
        print(f"   字节码中CREATE2字节: {create2_bytes}个")
        
        # Check if it's a vanity address
        if contract['address'].lower().startswith('0x000000000'):
            print(f"   ⚠️  特征: CREATE2预计算vanity地址")
        
        # Get CFG analysis
        try:
            cfg_info = detector.get_basic_block_info(contract['bytecode'])
            factory_blocks = [b for b in cfg_info['blocks'] 
                            if b['contains_create'] or b['contains_create2']]
            reachable_factory_blocks = [b for b in factory_blocks if b['is_reachable']]
            
            print(f"   CFG分析: {cfg_info['total_blocks']}个基本块, {cfg_info['reachable_blocks']}个可达")
            print(f"   工厂块: {len(factory_blocks)}个总块, {len(reachable_factory_blocks)}个可达")
            
            if create_bytes > 0 or create2_bytes > 0:
                if len(factory_blocks) == 0:
                    print(f"   ❌ 问题: CREATE字节存在但未识别为工厂块 (可能在PUSH操作数中)")
                elif len(reachable_factory_blocks) == 0:
                    print(f"   ❌ 问题: 工厂块存在但标记为不可达")
                else:
                    print(f"   ❓ 疑问: 应该被检测为工厂但未被检测")
            else:
                print(f"   ❌ 问题: 无CREATE字节但实际执行了CREATE (可能通过代理或外部调用)")
                
        except Exception as e:
            print(f"   CFG分析错误: {e}")


def analyze_detection_patterns():
    """分析检测模式和边界情况"""
    
    print("\n" + "=" * 80)
    print("检测模式和边界情况分析")
    print("=" * 80)
    
    # Load evaluation results
    with open('factory_detector_evaluation_results.json', 'r') as f:
        data = json.load(f)
    
    contracts = data['contracts']
    
    # 统计各种检测结果
    true_positives = [c for c in contracts if c['is_factory_ground_truth'] and c['is_factory_detected']]
    false_positives = [c for c in contracts if not c['is_factory_ground_truth'] and c['is_factory_detected']]
    true_negatives = [c for c in contracts if not c['is_factory_ground_truth'] and not c['is_factory_detected']]
    false_negatives = [c for c in contracts if c['is_factory_ground_truth'] and not c['is_factory_detected']]
    
    print(f"检测结果分布:")
    print(f"  True Positives:  {len(true_positives):4d} (正确检测工厂)")
    print(f"  False Positives: {len(false_positives):4d} (误报非工厂为工厂)")
    print(f"  True Negatives:  {len(true_negatives):4d} (正确检测非工厂)")
    print(f"  False Negatives: {len(false_negatives):4d} (漏报工厂为非工厂)")
    
    # 分析检测类型分布
    print(f"\n检测类型分布:")
    factory_types = defaultdict(int)
    for contract in contracts:
        if contract['is_factory_detected']:
            factory_type = contract.get('factory_type', 'UNKNOWN')
            factory_types[factory_type] += 1
    
    for factory_type, count in factory_types.items():
        print(f"  {factory_type}: {count}个")
    
    # 分析执行时间模式
    print(f"\n执行时间分析:")
    fp_times = [c.get('execution_time', 0) for c in false_positives if c.get('execution_time')]
    fn_times = [c.get('execution_time', 0) for c in false_negatives if c.get('execution_time')]
    tp_times = [c.get('execution_time', 0) for c in true_positives if c.get('execution_time')]
    tn_times = [c.get('execution_time', 0) for c in true_negatives if c.get('execution_time')]
    
    time_categories = [
        ("False Positives", fp_times),
        ("False Negatives", fn_times),
        ("True Positives", tp_times),
        ("True Negatives", tn_times)
    ]
    
    for category, times in time_categories:
        if times:
            avg_time = sum(times) / len(times)
            print(f"  {category}: 平均 {avg_time:.1f}ms (范围: {min(times)}-{max(times)}ms)")
    
    # 关键发现总结
    print(f"\n关键发现:")
    print(f"1. False Positives主要来源:")
    print(f"   - {len([c for c in false_positives if c['source_type'] == 'etherscan'])}个来自Etherscan验证合约")
    print(f"   - 合约名称包含'Factory'但源码分析确认非工厂")
    print(f"   - 字节码级检测到CREATE但实际用途非工厂创建")
    
    print(f"\n2. False Negatives主要来源:")
    print(f"   - {len([c for c in false_negatives if c['source_type'] == 'traces'])}个来自交易追踪数据")
    print(f"   - 大量具有CREATE2预计算地址的合约")
    print(f"   - CREATE操作可能通过代理或动态生成")
    
    vanity_addresses = len([c for c in false_negatives 
                           if c['address'].lower().startswith('0x000000000')])
    print(f"   - {vanity_addresses}个使用vanity地址 (可能通过CREATE2)")


def main():
    """主分析函数"""
    print("开始深度分析原始Factory Detector的误报和漏报来源...")
    
    # 分析False Positives
    analyze_false_positives_sources()
    
    # 分析False Negatives
    analyze_false_negatives_sources()
    
    # 分析检测模式
    analyze_detection_patterns()
    
    print("\n" + "=" * 80)
    print("分析完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()