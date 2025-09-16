#!/usr/bin/env python3
"""
Factory Detector Evaluation Script - Small Batch Test

Test script for evaluating a small batch of contracts first.
"""

import os
import sys
import time
import json
from typing import Dict, List
from dataclasses import dataclass
from google.cloud import bigquery

# Add project root to path to import factory_detector
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from factory_detector import ImprovedFactoryDetector


@dataclass
class EvaluationResult:
    """Results of factory detector evaluation"""
    total_contracts: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    avg_execution_time: float
    total_execution_time: float


def test_small_batch():
    """Test with a small batch of contracts"""
    
    # Initialize BigQuery client and detector
    project_id = "ziyue-wang"
    dataset_id = "tosem_factory_analysis"
    table_id = "GroundTruthDataset"
    table_full_id = f"{project_id}.{dataset_id}.{table_id}"
    
    client = bigquery.Client(project=project_id)
    detector = ImprovedFactoryDetector()
    
    print("Testing Factory Detector Evaluation with Small Batch")
    print("="*60)
    
    # Fetch a small batch (first 10 contracts)
    query = f"""
    SELECT 
        address,
        bytecode,
        is_factory_ground_truth,
        is_factory_detected,
        execution_time,
        source_type
    FROM `{table_full_id}`
    LIMIT 10
    """
    
    print("Fetching small batch of contracts...")
    query_job = client.query(query)
    results = query_job.result()
    
    contracts = []
    for row in results:
        contracts.append({
            'address': row.address,
            'bytecode': row.bytecode,
            'is_factory_ground_truth': row.is_factory_ground_truth,
            'is_factory_detected': row.is_factory_detected,
            'execution_time': row.execution_time,
            'source_type': row.source_type
        })
    
    print(f"Fetched {len(contracts)} contracts")
    
    # Run factory detection on the small batch
    print("Running factory detection...")
    processed_contracts = []
    total_execution_time = 0
    
    for i, contract in enumerate(contracts):
        print(f"Processing contract {i+1}/{len(contracts)}: {contract['address'][:10]}...")
        
        try:
            # Run factory detection
            start_time = time.perf_counter()
            detection_result = detector.detect_factory_contract(contract['bytecode'])
            execution_time_ms = int((time.perf_counter() - start_time) * 1000)
            
            # Update contract with detection results
            contract['is_factory_detected'] = detection_result.is_factory_contract
            contract['execution_time'] = execution_time_ms
            
            total_execution_time += execution_time_ms
            processed_contracts.append(contract)
            
            print(f"  -> Ground Truth: {contract['is_factory_ground_truth']}, "
                  f"Detected: {contract['is_factory_detected']}, "
                  f"Time: {execution_time_ms}ms")
            
        except Exception as e:
            print(f"  -> Error: {e}")
            contract['is_factory_detected'] = False
            contract['execution_time'] = 0
            processed_contracts.append(contract)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    
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
    
    # Calculate metrics
    total_contracts = len(processed_contracts)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    avg_execution_time = total_execution_time / total_contracts if total_contracts > 0 else 0
    
    # Print results
    print("\n" + "="*60)
    print("SMALL BATCH TEST RESULTS")
    print("="*60)
    
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
              f"Det: {contract['is_factory_detected']} | {contract['execution_time']}ms")
    
    print("="*60)
    
    return processed_contracts


if __name__ == "__main__":
    test_small_batch()