#!/usr/bin/env python3
"""
This script extracts the complete list of false positive and false negative contracts
with their detailed information for further analysis.
"""

import json
import csv
from typing import List, Dict


def extract_false_positives() -> List[Dict]:
    
    # Load evaluation results
    with open('factory_detector_evaluation_results.json', 'r') as f:
        data = json.load(f)
    
    contracts = data['contracts']
    
    # Find false positives
    false_positives = [c for c in contracts if not c['is_factory_ground_truth'] and c['is_factory_detected']]
    
    return false_positives


def extract_false_negatives() -> List[Dict]:
    
    # Load evaluation results
    with open('factory_detector_evaluation_results.json', 'r') as f:
        data = json.load(f)
    
    contracts = data['contracts']
    
    # Find false negatives
    false_negatives = [c for c in contracts if c['is_factory_ground_truth'] and not c['is_factory_detected']]
    
    return false_negatives


def save_false_positives_list():
    
    false_positives = extract_false_positives()
    
    print("=" * 80)
    print(f"FALSE POSITIVES 完整列表 ({len(false_positives)}个合约)")
    print("=" * 80)
    
    csv_data = []
    
    for i, contract in enumerate(false_positives):
        contract_name = "Unknown"
        if contract.get('verification_notes') and 'Contract:' in contract['verification_notes']:
            try:
                contract_name = contract['verification_notes'].split('Contract:')[-1].strip().split()[0]
            except:
                pass
        
        create_count = contract.get('create_positions', 0)
        create2_count = contract.get('create2_positions', 0)
        
        print(f"{i+1:2d}. {contract['address']}")
        print(f"    合约名称: {contract_name}")
        print(f"    检测类型: {contract['factory_type']}")
        print(f"    CREATE操作: {create_count}个")
        print(f"    CREATE2操作: {create2_count}个")
        print(f"    执行时间: {contract.get('execution_time', 0)}ms")
        
        if contract.get('verification_notes'):
            notes = contract['verification_notes'][:150] + "..." if len(contract['verification_notes']) > 150 else contract['verification_notes']
            print(f"    验证信息: {notes}")
        print()
        
        csv_data.append({
            'address': contract['address'],
            'contract_name': contract_name,
            'factory_type': contract['factory_type'],
            'create_positions': create_count,
            'create2_positions': create2_count,
            'execution_time_ms': contract.get('execution_time', 0),
            'source_type': contract['source_type'],
            'verification_notes': contract.get('verification_notes', '')
        })
    
    with open('false_positives_list.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['address', 'contract_name', 'factory_type', 'create_positions', 'create2_positions', 
                     'execution_time_ms', 'source_type', 'verification_notes']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"False Positives列表已保存到: false_positives_list.csv")
    return false_positives


def save_false_negatives_list():
    
    false_negatives = extract_false_negatives()
    
    print("=" * 80)
    print(f"FALSE NEGATIVES 完整列表 ({len(false_negatives)}个合约)")
    print("=" * 80)
    
    csv_data = []
    
    for i, contract in enumerate(false_negatives):
        create_executed = "Unknown"
        if contract.get('verification_notes') and 'executed' in contract['verification_notes']:
            notes = contract['verification_notes']
            words = notes.split()
            for j, word in enumerate(words):
                if word == 'executed' and j+1 < len(words):
                    try:
                        create_executed = int(words[j+1])
                        break
                    except:
                        pass
        
        is_vanity = contract['address'].lower().startswith('0x000000000')
        vanity_zeros = 0
        if is_vanity:
            for char in contract['address'].lower()[2:]:
                if char == '0':
                    vanity_zeros += 1
                else:
                    break
        
        bytecode = contract['bytecode']
        if bytecode.startswith('0x'):
            bytecode_clean = bytecode[2:]
        else:
            bytecode_clean = bytecode
            
        create_bytes = bytecode_clean.lower().count('f0')
        create2_bytes = bytecode_clean.lower().count('f5')
        
        print(f"{i+1:2d}. {contract['address']}")
        print(f"    实际CREATE执行: {create_executed}次")
        print(f"    字节码CREATE字节: {create_bytes}个")
        print(f"    字节码CREATE2字节: {create2_bytes}个")
        print(f"    Vanity地址: {'是' if is_vanity else '否'} ({vanity_zeros}个前导0)" if is_vanity else "    Vanity地址: 否")
        print(f"    执行时间: {contract.get('execution_time', 0)}ms")
        
        if contract.get('verification_notes'):
            notes = contract['verification_notes'][:150] + "..." if len(contract['verification_notes']) > 150 else contract['verification_notes']
            print(f"    验证信息: {notes}")
        print()
        
        csv_data.append({
            'address': contract['address'],
            'actual_create_executions': create_executed,
            'bytecode_create_bytes': create_bytes,
            'bytecode_create2_bytes': create2_bytes,
            'is_vanity_address': is_vanity,
            'vanity_leading_zeros': vanity_zeros if is_vanity else 0,
            'execution_time_ms': contract.get('execution_time', 0),
            'source_type': contract['source_type'],
            'verification_notes': contract.get('verification_notes', '')
        })
    
    with open('false_negatives_list.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['address', 'actual_create_executions', 'bytecode_create_bytes', 'bytecode_create2_bytes',
                     'is_vanity_address', 'vanity_leading_zeros', 'execution_time_ms', 'source_type', 'verification_notes']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"False Negatives列表已保存到: false_negatives_list.csv")
    return false_negatives


def create_summary_analysis():
    
    false_positives = extract_false_positives()
    false_negatives = extract_false_negatives()
    
    print("=" * 80)
    print("误报和漏报总结分析")
    print("=" * 80)
    
    print(f"\n📊 FALSE POSITIVES 统计 ({len(false_positives)}个):")
    
    fp_with_factory = len([c for c in false_positives 
                          if c.get('verification_notes') and 'Factory' in c.get('verification_notes', '')])
    fp_with_mint = len([c for c in false_positives 
                       if c.get('verification_notes') and 'Mint' in c.get('verification_notes', '')])
    
    print(f"  - 合约名包含'Factory': {fp_with_factory}个")
    print(f"  - 合约名包含'Mint': {fp_with_mint}个")
    print(f"  - 其他类型: {len(false_positives) - fp_with_factory - fp_with_mint}个")
    
    fp_create_only = len([c for c in false_positives if c.get('factory_type') == 'CREATE_ONLY'])
    fp_create2_only = len([c for c in false_positives if c.get('factory_type') == 'CREATE2_ONLY'])
    fp_both = len([c for c in false_positives if c.get('factory_type') == 'BOTH_CREATE_CREATE2'])
    
    print(f"  - CREATE_ONLY: {fp_create_only}个")
    print(f"  - CREATE2_ONLY: {fp_create2_only}个") 
    print(f"  - BOTH_CREATE_CREATE2: {fp_both}个")
    
    print(f"\n📊 FALSE NEGATIVES 统计 ({len(false_negatives)}个):")
    
    fn_vanity = len([c for c in false_negatives if c['address'].lower().startswith('0x000000000')])
    fn_normal = len(false_negatives) - fn_vanity
    
    print(f"  - Vanity地址 (CREATE2预计算): {fn_vanity}个")
    print(f"  - 普通地址: {fn_normal}个")
    
    create_counts = []
    for contract in false_negatives:
        if contract.get('verification_notes') and 'executed' in contract['verification_notes']:
            notes = contract['verification_notes']
            words = notes.split()
            for j, word in enumerate(words):
                if word == 'executed' and j+1 < len(words):
                    try:
                        count = int(words[j+1])
                        create_counts.append(count)
                        break
                    except:
                        pass
    
    if create_counts:
        small_scale = len([c for c in create_counts if c <= 1000])
        medium_scale = len([c for c in create_counts if 1001 <= c <= 10000])
        large_scale = len([c for c in create_counts if c > 10000])
        
        print(f"  - 小规模 (≤1000次CREATE): {small_scale}个")
        print(f"  - 中规模 (1001-10000次): {medium_scale}个")
        print(f"  - 大规模 (>10000次): {large_scale}个")
        print(f"  - 平均CREATE执行: {sum(create_counts)/len(create_counts):,.0f}次")
    
    fn_with_create_bytes = len([c for c in false_negatives 
                               if c['bytecode'].lower().count('f0') > 0])
    fn_with_create2_bytes = len([c for c in false_negatives 
                                if c['bytecode'].lower().count('f5') > 0])
    fn_no_create_bytes = len([c for c in false_negatives 
                             if c['bytecode'].lower().count('f0') == 0 and c['bytecode'].lower().count('f5') == 0])
    
    print(f"  - 包含CREATE字节: {fn_with_create_bytes}个")
    print(f"  - 包含CREATE2字节: {fn_with_create2_bytes}个")
    print(f"  - 无CREATE字节: {fn_no_create_bytes}个")


def create_excel_compatible_files():
    
    false_positives = extract_false_positives()
    false_negatives = extract_false_negatives()
    
    fp_simple = []
    for i, contract in enumerate(false_positives):
        contract_name = "Unknown"
        if contract.get('verification_notes') and 'Contract:' in contract['verification_notes']:
            try:
                contract_name = contract['verification_notes'].split('Contract:')[-1].strip().split()[0]
            except:
                pass
        
        fp_simple.append([
            i+1,
            contract['address'],
            contract_name,
            contract['factory_type'],
            contract.get('create_positions', 0),
            contract.get('create2_positions', 0),
            contract.get('execution_time', 0)
        ])
    
    fn_simple = []
    for i, contract in enumerate(false_negatives):
        create_executed = "Unknown"
        if contract.get('verification_notes') and 'executed' in contract['verification_notes']:
            notes = contract['verification_notes']
            words = notes.split()
            for j, word in enumerate(words):
                if word == 'executed' and j+1 < len(words):
                    try:
                        create_executed = f"{int(words[j+1]):,}"
                        break
                    except:
                        pass
        
        is_vanity = contract['address'].lower().startswith('0x000000000')
        bytecode_clean = contract['bytecode'][2:] if contract['bytecode'].startswith('0x') else contract['bytecode']
        create_bytes = bytecode_clean.lower().count('f0')
        create2_bytes = bytecode_clean.lower().count('f5')
        
        fn_simple.append([
            i+1,
            contract['address'],
            create_executed,
            create_bytes,
            create2_bytes,
            "是" if is_vanity else "否",
            contract.get('execution_time', 0)
        ])
    
    with open('false_positives_table.txt', 'w', encoding='utf-8') as f:
        f.write("序号\t地址\t合约名称\t检测类型\tCREATE数\tCREATE2数\t执行时间(ms)\n")
        for row in fp_simple:
            f.write('\t'.join(map(str, row)) + '\n')
    
    with open('false_negatives_table.txt', 'w', encoding='utf-8') as f:
        f.write("序号\t地址\t实际CREATE执行\t字节码CREATE字节\t字节码CREATE2字节\tVanity地址\t执行时间(ms)\n")
        for row in fn_simple:
            f.write('\t'.join(map(str, row)) + '\n')
    
    print(f"\nExcel兼容文件已创建:")
    print(f"  - false_positives_table.txt (可直接复制到Excel)")
    print(f"  - false_negatives_table.txt (可直接复制到Excel)")


def main():
    print("提取原始Factory Detector的具体误报和漏报合约...")
    
    false_positives = save_false_positives_list()
    
    false_negatives = save_false_negatives_list()
    
    create_summary_analysis()
    
    create_excel_compatible_files()
    
    print("\n" + "=" * 80)
    print("所有误报和漏报合约信息已提取完成!")
    print("=" * 80)
    print("\n生成的文件:")
    print("  📄 false_positives_list.csv - False Positives详细信息")
    print("  📄 false_negatives_list.csv - False Negatives详细信息") 
    print("  📄 false_positives_table.txt - Excel兼容格式")
    print("  📄 false_negatives_table.txt - Excel兼容格式")


if __name__ == "__main__":
    main()
