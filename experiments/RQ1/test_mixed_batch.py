#!/usr/bin/env python3
"""
Test with a mixed batch including both factory and non-factory contracts
"""

import os
import sys
import time
from google.cloud import bigquery

# Add project root to path to import factory_detector
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from factory_detector import ImprovedFactoryDetector


def test_mixed_batch():
    """Test with a mixed batch of factory and non-factory contracts"""
    
    # Initialize BigQuery client and detector
    project_id = "ziyue-wang"
    dataset_id = "tosem_factory_analysis"
    table_id = "GroundTruthDataset"
    table_full_id = f"{project_id}.{dataset_id}.{table_id}"
    
    client = bigquery.Client(project=project_id)
    detector = ImprovedFactoryDetector()
    
    print("Testing Factory Detector with Mixed Batch")
    print("="*60)
    
    # Fetch a mixed batch - 5 factory + 5 non-factory contracts
    factory_query = f"""
    SELECT 
        address, bytecode, is_factory_ground_truth, source_type, verification_notes
    FROM `{table_full_id}`
    WHERE is_factory_ground_truth = true
    LIMIT 5
    """
    
    non_factory_query = f"""
    SELECT 
        address, bytecode, is_factory_ground_truth, source_type, verification_notes
    FROM `{table_full_id}`
    WHERE is_factory_ground_truth = false
    LIMIT 5
    """
    
    print("Fetching mixed batch (5 factory + 5 non-factory)...")
    
    contracts = []
    
    # Get factory contracts
    query_job = client.query(factory_query)
    for row in query_job.result():
        contracts.append({
            'address': row.address,
            'bytecode': row.bytecode,
            'is_factory_ground_truth': row.is_factory_ground_truth,
            'source_type': row.source_type,
            'verification_notes': row.verification_notes
        })
    
    # Get non-factory contracts
    query_job = client.query(non_factory_query)
    for row in query_job.result():
        contracts.append({
            'address': row.address,
            'bytecode': row.bytecode,
            'is_factory_ground_truth': row.is_factory_ground_truth,
            'source_type': row.source_type,
            'verification_notes': row.verification_notes
        })
    
    print(f"Fetched {len(contracts)} contracts")
    
    # Run factory detection
    print("\nRunning factory detection...")
    processed_contracts = []
    total_execution_time = 0
    
    for i, contract in enumerate(contracts):
        print(f"\nProcessing contract {i+1}/{len(contracts)}: {contract['address'][:12]}...")
        print(f"  Ground Truth: {contract['is_factory_ground_truth']} ({contract['source_type']})")
        if contract['verification_notes']:
            print(f"  Notes: {contract['verification_notes'][:80]}...")
        
        try:
            # Run factory detection
            start_time = time.perf_counter()
            detection_result = detector.detect_factory_contract(contract['bytecode'])
            execution_time_ms = int((time.perf_counter() - start_time) * 1000)
            
            # Update contract with detection results
            contract['is_factory_detected'] = detection_result.is_factory_contract
            contract['execution_time'] = execution_time_ms
            contract['factory_type'] = detection_result.factory_type
            contract['create_positions'] = len(detection_result.verified_create_positions)
            contract['create2_positions'] = len(detection_result.verified_create2_positions)
            
            total_execution_time += execution_time_ms
            processed_contracts.append(contract)
            
            result_icon = "✓" if (contract['is_factory_ground_truth'] == contract['is_factory_detected']) else "✗"
            print(f"  {result_icon} Detection Result: {contract['is_factory_detected']} ({detection_result.factory_type})")
            print(f"  Execution Time: {execution_time_ms}ms")
            if detection_result.is_factory_contract:
                print(f"  CREATE ops: {contract['create_positions']}, CREATE2 ops: {contract['create2_positions']}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            contract['is_factory_detected'] = False
            contract['execution_time'] = 0
            contract['factory_type'] = 'ERROR'
            processed_contracts.append(contract)
    
    # Calculate metrics
    print("\n" + "="*60)
    print("MIXED BATCH TEST RESULTS")
    print("="*60)
    
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    for contract in processed_contracts:
        ground_truth = contract['is_factory_ground_truth']
        detected = contract['is_factory_detected']
        
        if ground_truth and detected:
            true_positives += 1
        elif not ground_truth and detected:
            false_positives += 1
        elif not ground_truth and not detected:
            true_negatives += 1
        elif ground_truth and not detected:
            false_negatives += 1
    
    total_contracts = len(processed_contracts)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    avg_execution_time = total_execution_time / total_contracts if total_contracts > 0 else 0
    
    print(f"Total Contracts: {total_contracts}")
    print(f"Total Execution Time: {total_execution_time}ms")
    print(f"Average Execution Time: {avg_execution_time:.2f}ms")
    
    print("\nConfusion Matrix:")
    print(f"  True Positives (TP):  {true_positives}")
    print(f"  False Positives (FP): {false_positives}")
    print(f"  True Negatives (TN):  {true_negatives}")
    print(f"  False Negatives (FN): {false_negatives}")
    
    print("\nPerformance Metrics:")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1_score:.4f} ({f1_score*100:.2f}%)")
    
    print("\nDetailed Results:")
    for contract in processed_contracts:
        status = "✓" if (contract['is_factory_ground_truth'] == contract['is_factory_detected']) else "✗"
        print(f"  {status} {contract['address'][:12]}... | GT: {contract['is_factory_ground_truth']} | "
              f"Det: {contract['is_factory_detected']} | {contract['execution_time']}ms | {contract['source_type']}")
    
    print("="*60)
    
    return processed_contracts


if __name__ == "__main__":
    test_mixed_batch()