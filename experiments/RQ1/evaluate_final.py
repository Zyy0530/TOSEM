#!/usr/bin/env python3
"""
Final evaluation of the optimized factory detector
"""

import json
import sys
import os
import time
from typing import Dict, List
from dataclasses import dataclass

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from final_factory_detector import FinalFactoryDetector


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


def evaluate_final_detector() -> EvaluationResult:
    """Evaluate the final optimized detector"""
    
    print("Evaluating Final Optimized Factory Detector")
    print("=" * 60)
    
    # Load original evaluation results
    with open('factory_detector_evaluation_results.json', 'r') as f:
        data = json.load(f)
    
    contracts = data['contracts']
    print(f"Loading {len(contracts):,} contracts for final evaluation...")
    
    # Initialize final detector
    final_detector = FinalFactoryDetector()
    
    # Evaluate all contracts
    processed_contracts = []
    total_execution_time = 0
    start_time = time.time()
    
    for i, contract in enumerate(contracts):
        if i % 100 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(contracts) - i) / rate if rate > 0 else 0
            print(f"  Progress: {i:,}/{len(contracts):,} ({i/len(contracts)*100:.1f}%) | "
                  f"Rate: {rate:.1f}/sec | ETA: {eta/60:.1f}min")
        
        try:
            # Run final detection
            detection_start = time.perf_counter()
            detection_result = final_detector.detect_factory_contract(contract['bytecode'])
            execution_time_ms = int((time.perf_counter() - detection_start) * 1000)
            
            # Update contract with final detection results
            contract['final_is_factory_detected'] = detection_result.is_factory_contract
            contract['final_execution_time'] = execution_time_ms
            contract['final_factory_type'] = detection_result.factory_type
            contract['final_create_positions'] = len(detection_result.verified_create_positions)
            contract['final_create2_positions'] = len(detection_result.verified_create2_positions)
            
            total_execution_time += execution_time_ms
            processed_contracts.append(contract)
            
        except Exception as e:
            print(f"Error processing contract {contract['address']}: {e}")
            contract['final_is_factory_detected'] = False
            contract['final_execution_time'] = 0
            contract['final_factory_type'] = 'ERROR'
            processed_contracts.append(contract)
    
    elapsed = time.time() - start_time
    print(f"\nFinal evaluation completed in {elapsed:.1f} seconds")
    print(f"Total final detection time: {total_execution_time:,}ms ({total_execution_time/1000:.1f}s)")
    
    # Calculate final metrics
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    execution_times = []
    
    for contract in processed_contracts:
        ground_truth = contract['is_factory_ground_truth']
        detected = contract['final_is_factory_detected']
        exec_time = contract['final_execution_time'] or 0
        
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
    total_contracts = len(processed_contracts)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
    total_execution_time = sum(execution_times)
    
    # Save final results
    final_results_file = "final_factory_detector_evaluation_results.json"
    results_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'detector_type': 'final_optimized',
        'total_contracts': len(processed_contracts),
        'contracts': processed_contracts
    }
    
    with open(final_results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"Final results saved to: {final_results_file}")
    
    return EvaluationResult(
        total_contracts=total_contracts,
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        avg_execution_time=avg_execution_time,
        total_execution_time=total_execution_time
    )


def compare_all_detectors():
    """Compare all three detectors: original, enhanced, and final"""
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE DETECTOR COMPARISON")
    print("=" * 80)
    
    # Load original results
    with open('factory_detector_evaluation_summary.json', 'r') as f:
        original_data = json.load(f)
    
    # Evaluate final detector
    final_result = evaluate_final_detector()
    
    # Display comparison
    print(f"\nORIGINAL DETECTOR:")
    print(f"  Precision: {original_data['precision']:.4f} ({original_data['precision']*100:.2f}%)")
    print(f"  Recall:    {original_data['recall']:.4f} ({original_data['recall']*100:.2f}%)")
    print(f"  F1-Score:  {original_data['f1_score']:.4f} ({original_data['f1_score']*100:.2f}%)")
    print(f"  Accuracy:  {original_data['accuracy']:.4f} ({original_data['accuracy']*100:.2f}%)")
    print(f"  TP: {original_data['true_positives']}, FP: {original_data['false_positives']}, "
          f"TN: {original_data['true_negatives']}, FN: {original_data['false_negatives']}")
    print(f"  Avg Time:  {original_data['avg_execution_time_ms']:.2f}ms")
    
    final_accuracy = (final_result.true_positives + final_result.true_negatives) / final_result.total_contracts
    
    print(f"\nFINAL OPTIMIZED DETECTOR:")
    print(f"  Precision: {final_result.precision:.4f} ({final_result.precision*100:.2f}%)")
    print(f"  Recall:    {final_result.recall:.4f} ({final_result.recall*100:.2f}%)")
    print(f"  F1-Score:  {final_result.f1_score:.4f} ({final_result.f1_score*100:.2f}%)")
    print(f"  Accuracy:  {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    print(f"  TP: {final_result.true_positives}, FP: {final_result.false_positives}, "
          f"TN: {final_result.true_negatives}, FN: {final_result.false_negatives}")
    print(f"  Avg Time:  {final_result.avg_execution_time:.2f}ms")
    
    # Calculate final improvements
    precision_improvement = final_result.precision - original_data['precision']
    recall_improvement = final_result.recall - original_data['recall']
    f1_improvement = final_result.f1_score - original_data['f1_score']
    accuracy_improvement = final_accuracy - original_data['accuracy']
    
    fp_reduction = original_data['false_positives'] - final_result.false_positives
    fn_reduction = original_data['false_negatives'] - final_result.false_negatives
    
    print(f"\nFINAL IMPROVEMENTS (vs Original):")
    print(f"  Precision: {precision_improvement:+.4f} ({precision_improvement*100:+.2f} percentage points)")
    print(f"  Recall:    {recall_improvement:+.4f} ({recall_improvement*100:+.2f} percentage points)")
    print(f"  F1-Score:  {f1_improvement:+.4f} ({f1_improvement*100:+.2f} percentage points)")
    print(f"  Accuracy:  {accuracy_improvement:+.4f} ({accuracy_improvement*100:+.2f} percentage points)")
    print(f"  False Positives: {fp_reduction:+d} (reduced by {fp_reduction})")
    print(f"  False Negatives: {fn_reduction:+d} (reduced by {fn_reduction})")
    
    # Analyze remaining errors
    print(f"\nREMAINING ERROR ANALYSIS:")
    
    # Load final results for error analysis
    with open('final_factory_detector_evaluation_results.json', 'r') as f:
        final_data = json.load(f)
    
    contracts = final_data['contracts']
    final_fps = [c for c in contracts if not c['is_factory_ground_truth'] and c['final_is_factory_detected']]
    final_fns = [c for c in contracts if c['is_factory_ground_truth'] and not c['final_is_factory_detected']]
    
    print(f"  Remaining False Positives: {len(final_fps)}")
    print(f"  Remaining False Negatives: {len(final_fns)}")
    
    if final_fps:
        print(f"\n  Sample False Positives:")
        for i, contract in enumerate(final_fps[:3]):
            print(f"    {i+1}. {contract['address']} - {contract['final_factory_type']}")
    
    if final_fns:
        print(f"\n  Sample False Negatives:")
        for i, contract in enumerate(final_fns[:3]):
            print(f"    {i+1}. {contract['address']} - {contract['final_factory_type']}")
    
    # Save final comparison
    final_comparison_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'original': {
            'precision': original_data['precision'],
            'recall': original_data['recall'],
            'f1_score': original_data['f1_score'],
            'accuracy': original_data['accuracy'],
            'true_positives': original_data['true_positives'],
            'false_positives': original_data['false_positives'],
            'true_negatives': original_data['true_negatives'],
            'false_negatives': original_data['false_negatives'],
            'avg_execution_time_ms': original_data['avg_execution_time_ms']
        },
        'final': {
            'precision': final_result.precision,
            'recall': final_result.recall,
            'f1_score': final_result.f1_score,
            'accuracy': final_accuracy,
            'true_positives': final_result.true_positives,
            'false_positives': final_result.false_positives,
            'true_negatives': final_result.true_negatives,
            'false_negatives': final_result.false_negatives,
            'avg_execution_time_ms': final_result.avg_execution_time
        },
        'final_improvements': {
            'precision': precision_improvement,
            'recall': recall_improvement,
            'f1_score': f1_improvement,
            'accuracy': accuracy_improvement,
            'false_positives_reduced': fp_reduction,
            'false_negatives_reduced': fn_reduction
        }
    }
    
    with open('final_detector_comparison.json', 'w') as f:
        json.dump(final_comparison_data, f, indent=2)
    
    print(f"\nFinal comparison results saved to: final_detector_comparison.json")
    print("=" * 80)


def main():
    """Main evaluation function"""
    try:
        compare_all_detectors()
        
    except Exception as e:
        print(f"Final evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())