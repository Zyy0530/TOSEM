#!/usr/bin/env python3
"""
Deep dive analysis of specific false negative contracts
"""

import json
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from enhanced_factory_detector import EnhancedFactoryDetector


def analyze_specific_contract(address: str, bytecode: str):
    """Analyze a specific contract in detail"""
    
    print(f"\nDeep Analysis of {address}")
    print("=" * 60)
    
    detector = EnhancedFactoryDetector()
    
    # Run detection
    result = detector.detect_factory_contract(bytecode)
    
    print(f"Detection Result: {result.is_factory_contract}")
    print(f"Factory Type: {result.factory_type}")
    print(f"Confidence: {result.validation_details.get('confidence_score', 0):.3f}")
    
    # Get detailed CFG info
    cfg_info = detector.get_basic_block_info(bytecode)
    
    print(f"\nCFG Information:")
    print(f"  Total Instructions: {cfg_info['total_instructions']}")
    print(f"  Total Blocks: {cfg_info['total_blocks']}")
    print(f"  Reachable Blocks: {cfg_info['reachable_blocks']}")
    
    # Check for CREATE/CREATE2 in bytecode
    if bytecode.startswith('0x'):
        bytecode_clean = bytecode[2:]
    else:
        bytecode_clean = bytecode
        
    create_count = bytecode_clean.lower().count('f0')
    create2_count = bytecode_clean.lower().count('f5')
    
    print(f"  CREATE bytes in bytecode: {create_count}")
    print(f"  CREATE2 bytes in bytecode: {create2_count}")
    
    # Find blocks with CREATE/CREATE2
    factory_blocks = [b for b in cfg_info['blocks'] if b['contains_create'] or b['contains_create2']]
    reachable_factory_blocks = [b for b in factory_blocks if b['is_reachable']]
    
    print(f"  Blocks with CREATE/CREATE2: {len(factory_blocks)}")
    print(f"  Reachable blocks with CREATE/CREATE2: {len(reachable_factory_blocks)}")
    
    if factory_blocks:
        print(f"\nFactory Blocks Details:")
        for i, block in enumerate(factory_blocks[:5]):  # Show first 5
            print(f"    Block {i+1}: PC {block['start_pc']}-{block['end_pc']}, "
                  f"Reachable: {block['is_reachable']}, "
                  f"CREATE: {block['contains_create']}, CREATE2: {block['contains_create2']}")
    
    # Check if there are unreachable CREATE blocks
    unreachable_factory_blocks = [b for b in factory_blocks if not b['is_reachable']]
    if unreachable_factory_blocks:
        print(f"\nUnreachable Factory Blocks: {len(unreachable_factory_blocks)}")
        print("This suggests the detector correctly identified CREATE/CREATE2 but marked them as unreachable")
    
    # Check if CREATE bytes are in data/unreachable sections
    if create_count > 0 or create2_count > 0:
        if not factory_blocks:
            print(f"\nPROBLEM: CREATE/CREATE2 bytes exist but no factory blocks detected!")
            print("This suggests the bytes are in PUSH operands or other data sections")


def main():
    """Analyze the most problematic false negative contracts"""
    
    # Load results
    with open('enhanced_factory_detector_evaluation_results.json', 'r') as f:
        data = json.load(f)
    
    contracts = data['contracts']
    
    # Find false negatives with 0 confidence
    zero_confidence_fns = [c for c in contracts 
                          if c['is_factory_ground_truth'] 
                          and not c['enhanced_is_factory_detected']
                          and c.get('enhanced_confidence', 0) == 0.0]
    
    print(f"Found {len(zero_confidence_fns)} false negatives with 0 confidence")
    
    # Analyze first few
    for i, contract in enumerate(zero_confidence_fns[:3]):
        analyze_specific_contract(contract['address'], contract['bytecode'])
        
        if i < 2:  # Add separator between contracts
            print("\n" + "="*80)


if __name__ == "__main__":
    main()