#!/usr/bin/env python3
"""
Test enhanced detector on a specific problematic contract
"""

import json
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from enhanced_factory_detector import EnhancedFactoryDetector


def test_specific_contract():
    """Test the problematic contract"""
    
    # Load the specific contract that should be detected as factory
    with open('enhanced_factory_detector_evaluation_results.json', 'r') as f:
        data = json.load(f)
    
    # Find the contract 0x000000000000006f6502b7f2bbac8c30a3f67e9a
    target_address = "0x000000000000006f6502b7f2bbac8c30a3f67e9a"
    target_contract = None
    
    for contract in data['contracts']:
        if contract['address'] == target_address:
            target_contract = contract
            break
    
    if not target_contract:
        print("Contract not found!")
        return
    
    print(f"Testing contract: {target_address}")
    print(f"Ground truth: Factory = {target_contract['is_factory_ground_truth']}")
    print(f"Original detection: {target_contract.get('is_factory_detected', 'N/A')}")
    print(f"Enhanced detection: {target_contract.get('enhanced_is_factory_detected', 'N/A')}")
    
    # Test with updated detector
    detector = EnhancedFactoryDetector()
    result = detector.detect_factory_contract(target_contract['bytecode'])
    
    print(f"\nUpdated detection result:")
    print(f"  Is Factory: {result.is_factory_contract}")
    print(f"  Factory Type: {result.factory_type}")
    print(f"  Confidence: {result.validation_details.get('confidence_score', 0):.3f}")
    print(f"  CREATE positions: {len(result.verified_create_positions)}")
    print(f"  CREATE2 positions: {len(result.verified_create2_positions)}")
    print(f"  Analysis time: {result.analysis_time_ms:.1f}ms")
    
    print(f"\nValidation details:")
    for key, value in result.validation_details.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    test_specific_contract()