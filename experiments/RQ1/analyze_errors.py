#!/usr/bin/env python3
"""
详细分析Factory Detector的误报和漏报原因

This script analyzes the false positives and false negatives to understand
why the detector failed in these cases.
"""

import json
import sys
import os
from typing import Dict, List

# Add project root to path to import factory_detector
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from factory_detector import ImprovedFactoryDetector


def analyze_false_positives(contracts: List[Dict]) -> None:
    """分析False Positives (误报) - 非工厂合约被误检为工厂合约"""
    
    print("=" * 70)
    print("FALSE POSITIVES 分析 (误报 - 非工厂合约被误检为工厂合约)")
    print("=" * 70)
    
    false_positives = [c for c in contracts if not c['is_factory_ground_truth'] and c['is_factory_detected']]
    
    print(f"总计False Positives: {len(false_positives)}个")
    
    if not false_positives:
        print("没有False Positives!")
        return
    
    # 按factory_type分组分析
    fp_by_type = {}
    for contract in false_positives:
        factory_type = contract.get('factory_type', 'UNKNOWN')
        if factory_type not in fp_by_type:
            fp_by_type[factory_type] = []
        fp_by_type[factory_type].append(contract)
    
    print(f"\n按检测类型分组:")
    for factory_type, contracts_list in fp_by_type.items():
        print(f"  {factory_type}: {len(contracts_list)}个")
    
    # 详细分析前10个False Positives
    detector = ImprovedFactoryDetector()
    
    print(f"\n详细分析 (前10个):")
    for i, contract in enumerate(false_positives[:10]):
        print(f"\n{i+1}. 合约: {contract['address']}")
        print(f"   来源: {contract['source_type']}")
        print(f"   检测类型: {contract['factory_type']}")
        print(f"   CREATE数量: {contract.get('create_positions', 0)}")
        print(f"   CREATE2数量: {contract.get('create2_positions', 0)}")
        print(f"   执行时间: {contract.get('execution_time', 0)}ms")
        if contract.get('verification_notes'):
            print(f"   验证备注: {contract['verification_notes'][:100]}...")
        
        # 获取详细的CFG分析
        try:
            cfg_info = detector.get_basic_block_info(contract['bytecode'])
            print(f"   CFG分析: {cfg_info['total_instructions']}条指令, "
                  f"{cfg_info['total_blocks']}个基本块, "
                  f"{cfg_info['reachable_blocks']}个可达块")
            
            # 检查是否有可达的CREATE/CREATE2块
            factory_blocks = [b for b in cfg_info['blocks'] 
                            if b['is_reachable'] and (b['contains_create'] or b['contains_create2'])]
            if factory_blocks:
                print(f"   工厂块: {len(factory_blocks)}个可达块包含CREATE/CREATE2")
                for j, block in enumerate(factory_blocks[:3]):  # 显示前3个
                    print(f"     块{j+1}: PC {block['start_pc']}-{block['end_pc']}, "
                          f"CREATE={block['contains_create']}, CREATE2={block['contains_create2']}")
            
        except Exception as e:
            print(f"   CFG分析错误: {e}")


def analyze_false_negatives(contracts: List[Dict]) -> None:
    """分析False Negatives (漏报) - 工厂合约未被检测出"""
    
    print("\n" + "=" * 70)
    print("FALSE NEGATIVES 分析 (漏报 - 工厂合约未被检测出)")
    print("=" * 70)
    
    false_negatives = [c for c in contracts if c['is_factory_ground_truth'] and not c['is_factory_detected']]
    
    print(f"总计False Negatives: {len(false_negatives)}个")
    
    if not false_negatives:
        print("没有False Negatives!")
        return
    
    # 按source_type分组分析
    fn_by_source = {}
    for contract in false_negatives:
        source_type = contract.get('source_type', 'UNKNOWN')
        if source_type not in fn_by_source:
            fn_by_source[source_type] = []
        fn_by_source[source_type].append(contract)
    
    print(f"\n按数据源分组:")
    for source_type, contracts_list in fn_by_source.items():
        print(f"  {source_type}: {len(contracts_list)}个")
    
    # 详细分析前10个False Negatives
    detector = ImprovedFactoryDetector()
    
    print(f"\n详细分析 (前10个):")
    for i, contract in enumerate(false_negatives[:10]):
        print(f"\n{i+1}. 合约: {contract['address']}")
        print(f"   来源: {contract['source_type']}")
        print(f"   检测结果: {contract['factory_type']}")
        print(f"   执行时间: {contract.get('execution_time', 0)}ms")
        if contract.get('verification_notes'):
            print(f"   验证备注: {contract['verification_notes'][:100]}...")
        
        # 获取详细的CFG分析
        try:
            cfg_info = detector.get_basic_block_info(contract['bytecode'])
            print(f"   CFG分析: {cfg_info['total_instructions']}条指令, "
                  f"{cfg_info['total_blocks']}个基本块, "
                  f"{cfg_info['reachable_blocks']}个可达块")
            
            # 检查字节码中的CREATE/CREATE2
            bytecode = contract['bytecode']
            if bytecode.startswith('0x'):
                bytecode = bytecode[2:]
            
            create_count = bytecode.lower().count('f0')  # CREATE
            create2_count = bytecode.lower().count('f5')  # CREATE2
            print(f"   字节码中: {create_count}个CREATE字节, {create2_count}个CREATE2字节")
            
            # 检查是否有任何块包含CREATE/CREATE2
            all_factory_blocks = [b for b in cfg_info['blocks'] 
                                if b['contains_create'] or b['contains_create2']]
            reachable_factory_blocks = [b for b in cfg_info['blocks'] 
                                      if b['is_reachable'] and (b['contains_create'] or b['contains_create2'])]
            
            print(f"   工厂块: {len(all_factory_blocks)}个总块, {len(reachable_factory_blocks)}个可达块")
            
            if all_factory_blocks and not reachable_factory_blocks:
                print(f"   问题: 包含CREATE/CREATE2的块不可达!")
                for j, block in enumerate(all_factory_blocks[:3]):
                    print(f"     不可达块{j+1}: PC {block['start_pc']}-{block['end_pc']}, "
                          f"CREATE={block['contains_create']}, CREATE2={block['contains_create2']}")
            
        except Exception as e:
            print(f"   CFG分析错误: {e}")


def analyze_error_patterns(contracts: List[Dict]) -> None:
    """分析错误模式和潜在改进点"""
    
    print("\n" + "=" * 70)
    print("错误模式分析和改进建议")
    print("=" * 70)
    
    false_positives = [c for c in contracts if not c['is_factory_ground_truth'] and c['is_factory_detected']]
    false_negatives = [c for c in contracts if c['is_factory_ground_truth'] and not c['is_factory_detected']]
    
    print(f"False Positives分析:")
    print(f"  数量: {len(false_positives)}")
    
    if false_positives:
        # 分析误报的特征
        fp_etherscan = [c for c in false_positives if c['source_type'] == 'etherscan']
        print(f"  来自Etherscan验证合约: {len(fp_etherscan)}个")
        print(f"  这些合约通过源码分析确认不包含create/create2/new关键字")
        print(f"  但字节码级别检测到CREATE/CREATE2操作")
        print(f"  可能原因:")
        print(f"    1. 编译器优化产生了CREATE相关字节码但实际不执行")
        print(f"    2. 条件分支导致CREATE代码在某些路径下不可达")
        print(f"    3. CREATE用于非工厂用途(如自毁重建)")
    
    print(f"\nFalse Negatives分析:")
    print(f"  数量: {len(false_negatives)}")
    
    if false_negatives:
        # 分析漏报的特征
        fn_traces = [c for c in false_negatives if c['source_type'] == 'traces']
        print(f"  来自交易追踪: {len(fn_traces)}个")
        print(f"  这些合约在实际执行中创建了其他合约")
        print(f"  但字节码分析未检测到可达的CREATE/CREATE2")
        print(f"  可能原因:")
        print(f"    1. 动态跳转目标分析不完整")
        print(f"    2. 复杂的控制流导致可达性分析遗漏")
        print(f"    3. CREATE操作在深层嵌套的函数调用中")
        print(f"    4. 通过代理合约或委托调用执行CREATE")
    
    print(f"\n改进建议:")
    print(f"  1. 对于False Positives:")
    print(f"     - 增强上下文分析,区分CREATE的实际用途")
    print(f"     - 分析CREATE操作的执行条件")
    print(f"     - 检查CREATE是否在异常处理或清理代码中")
    
    print(f"  2. 对于False Negatives:")
    print(f"     - 改进动态跳转分析")
    print(f"     - 增强可达性分析的保守性")
    print(f"     - 考虑间接调用和代理模式")


def main():
    """主分析函数"""
    print("开始分析Factory Detector的误报和漏报...")
    
    # 加载评估结果
    with open('factory_detector_evaluation_results.json', 'r') as f:
        data = json.load(f)
    
    contracts = data['contracts']
    print(f"加载了{len(contracts):,}个合约的评估结果")
    
    # 分析False Positives
    analyze_false_positives(contracts)
    
    # 分析False Negatives  
    analyze_false_negatives(contracts)
    
    # 分析错误模式
    analyze_error_patterns(contracts)
    
    print("\n" + "=" * 70)
    print("分析完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()