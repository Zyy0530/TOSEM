#!/usr/bin/env python3
"""
Analyze Factory Detector Evaluation Results

This script loads the evaluation results and displays the final metrics.
"""

import json
from dataclasses import dataclass

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

def analyze_results(filename: str = "factory_detector_evaluation_results.json"):
    """Analyze the evaluation results"""
    
    print("Loading evaluation results...")
    with open(filename, 'r') as f:
        data = json.load(f)
    
    contracts = data['contracts']
    timestamp = data['timestamp']
    
    print(f"Results from: {timestamp}")
    print(f"Loaded data for {len(contracts):,} contracts")
    
    # Calculate metrics
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    execution_times = []
    
    for contract in contracts:
        ground_truth = contract['is_factory_ground_truth']
        detected = contract['is_factory_detected']
        exec_time = contract['execution_time'] or 0
        
        execution_times.append(exec_time)
        
        if ground_truth and detected:
            true_positives += 1
        elif not ground_truth and detected:
            false_positives += 1
        elif not ground_truth and not detected:
            true_negatives += 1
        elif ground_truth and not detected:
            false_negatives += 1
    
    # Calculate metrics
    total_contracts = len(contracts)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
    total_execution_time = sum(execution_times)
    accuracy = (true_positives + true_negatives) / total_contracts
    
    # Display results
    print("\n" + "="*70)
    print("FACTORY DETECTOR EVALUATION RESULTS")
    print("="*70)
    
    print(f"Total Contracts Evaluated: {total_contracts:,}")
    print(f"Total Execution Time: {total_execution_time:,.0f}ms ({total_execution_time/1000:.1f}s)")
    print(f"Average Execution Time: {avg_execution_time:.2f}ms")
    
    print("\nConfusion Matrix:")
    print(f"  True Positives (TP):  {true_positives:4,d} (Factory correctly detected)")
    print(f"  False Positives (FP): {false_positives:4,d} (Non-factory incorrectly detected as factory)")
    print(f"  True Negatives (TN):  {true_negatives:4,d} (Non-factory correctly detected)")
    print(f"  False Negatives (FN): {false_negatives:4,d} (Factory incorrectly detected as non-factory)")
    
    print("\nPerformance Metrics:")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1_score:.4f} ({f1_score*100:.2f}%)")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\nMetric Definitions:")
    print("  Precision = TP / (TP + FP) - How many detected factories are actually factories")
    print("  Recall = TP / (TP + FN) - How many actual factories are detected") 
    print("  F1-Score = 2 * (Precision * Recall) / (Precision + Recall)")
    print("  Accuracy = (TP + TN) / Total - Overall correctness")
    
    # Additional statistics
    print(f"\nDataset Statistics:")
    factory_contracts = true_positives + false_negatives
    non_factory_contracts = true_negatives + false_positives
    print(f"  Factory contracts in dataset: {factory_contracts:,} ({factory_contracts/total_contracts*100:.1f}%)")
    print(f"  Non-factory contracts in dataset: {non_factory_contracts:,} ({non_factory_contracts/total_contracts*100:.1f}%)")
    
    # Error analysis
    print(f"\nError Analysis:")
    false_positives_list = [c for c in contracts if not c['is_factory_ground_truth'] and c['is_factory_detected']]
    false_negatives_list = [c for c in contracts if c['is_factory_ground_truth'] and not c['is_factory_detected']]
    
    if false_positives_list:
        print(f"  False Positives ({len(false_positives_list)} contracts):")
        for i, contract in enumerate(false_positives_list[:3]):  # Show first 3
            print(f"    {i+1}. {contract['address']} ({contract['source_type']}) - {contract['factory_type']}")
            if len(false_positives_list) > 3 and i == 2:
                print(f"    ... and {len(false_positives_list)-3} more")
                break
    
    if false_negatives_list:
        print(f"  False Negatives ({len(false_negatives_list)} contracts):")
        for i, contract in enumerate(false_negatives_list[:3]):  # Show first 3
            print(f"    {i+1}. {contract['address']} ({contract['source_type']})")
            if len(false_negatives_list) > 3 and i == 2:
                print(f"    ... and {len(false_negatives_list)-3} more")
                break
    
    print("="*70)
    
    # Save summary
    summary_data = {
        'timestamp': timestamp,
        'total_contracts': total_contracts,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'avg_execution_time_ms': avg_execution_time,
        'total_execution_time_ms': total_execution_time
    }
    
    with open('factory_detector_evaluation_summary.json', 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\nSummary saved to: factory_detector_evaluation_summary.json")
    
    return summary_data

if __name__ == "__main__":
    analyze_results()