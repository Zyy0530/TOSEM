#!/usr/bin/env python3
"""
Evaluate Enhanced Factory Detector

This script evaluates the enhanced factory detector against the ground truth dataset
to measure improvements in precision and recall.
"""

import json
import sys
import os
import time
from typing import Dict, List
from dataclasses import dataclass

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from enhanced_factory_detector import EnhancedFactoryDetector


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


def evaluate_enhanced_detector() -> EvaluationResult:
    """Evaluate the enhanced factory detector"""
    
    print("Evaluating Enhanced Factory Detector")
    print("=" * 50)
    
    # Load original evaluation results
    with open('factory_detector_evaluation_results.json', 'r') as f:
        data = json.load(f)
    
    contracts = data['contracts']
    print(f"Loading {len(contracts):,} contracts for re-evaluation...")
    
    # Initialize enhanced detector
    enhanced_detector = EnhancedFactoryDetector()
    
    # Re-evaluate all contracts
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
            # Run enhanced factory detection
            detection_start = time.perf_counter()
            detection_result = enhanced_detector.detect_factory_contract(contract['bytecode'])
            execution_time_ms = int((time.perf_counter() - detection_start) * 1000)
            
            # Update contract with enhanced detection results
            contract['enhanced_is_factory_detected'] = detection_result.is_factory_contract
            contract['enhanced_execution_time'] = execution_time_ms
            contract['enhanced_factory_type'] = detection_result.factory_type
            contract['enhanced_confidence'] = detection_result.validation_details.get('confidence_score', 0.0)
            contract['enhanced_create_positions'] = len(detection_result.verified_create_positions)
            contract['enhanced_create2_positions'] = len(detection_result.verified_create2_positions)
            
            total_execution_time += execution_time_ms
            processed_contracts.append(contract)
            
        except Exception as e:
            print(f"Error processing contract {contract['address']}: {e}")
            # Mark as failed detection
            contract['enhanced_is_factory_detected'] = False
            contract['enhanced_execution_time'] = 0
            contract['enhanced_factory_type'] = 'ERROR'
            processed_contracts.append(contract)
    
    elapsed = time.time() - start_time
    print(f"\nEnhanced evaluation completed in {elapsed:.1f} seconds")
    print(f"Total enhanced detection time: {total_execution_time:,}ms ({total_execution_time/1000:.1f}s)")
    
    # Calculate enhanced metrics
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    execution_times = []
    
    for contract in processed_contracts:
        ground_truth = contract['is_factory_ground_truth']
        detected = contract['enhanced_is_factory_detected']
        exec_time = contract['enhanced_execution_time'] or 0
        
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
    
    # Save enhanced results
    enhanced_results_file = "enhanced_factory_detector_evaluation_results.json"
    results_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'detector_type': 'enhanced',
        'total_contracts': len(processed_contracts),
        'contracts': processed_contracts
    }
    
    with open(enhanced_results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"Enhanced results saved to: {enhanced_results_file}")
    
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


def compare_detectors():
    """Compare original vs enhanced detector performance"""
    
    print("\n" + "=" * 70)
    print("DETECTOR COMPARISON")
    print("=" * 70)
    
    # Load original results
    with open('factory_detector_evaluation_summary.json', 'r') as f:
        original_data = json.load(f)
    
    # Evaluate enhanced detector
    enhanced_result = evaluate_enhanced_detector()
    
    # Display comparison
    print(f"\nORIGINAL DETECTOR:")
    print(f"  Precision: {original_data['precision']:.4f} ({original_data['precision']*100:.2f}%)")
    print(f"  Recall:    {original_data['recall']:.4f} ({original_data['recall']*100:.2f}%)")
    print(f"  F1-Score:  {original_data['f1_score']:.4f} ({original_data['f1_score']*100:.2f}%)")
    print(f"  Accuracy:  {original_data['accuracy']:.4f} ({original_data['accuracy']*100:.2f}%)")
    print(f"  TP: {original_data['true_positives']}, FP: {original_data['false_positives']}, "
          f"TN: {original_data['true_negatives']}, FN: {original_data['false_negatives']}")
    print(f"  Avg Time:  {original_data['avg_execution_time_ms']:.2f}ms")
    
    enhanced_accuracy = (enhanced_result.true_positives + enhanced_result.true_negatives) / enhanced_result.total_contracts
    
    print(f"\nENHANCED DETECTOR:")
    print(f"  Precision: {enhanced_result.precision:.4f} ({enhanced_result.precision*100:.2f}%)")
    print(f"  Recall:    {enhanced_result.recall:.4f} ({enhanced_result.recall*100:.2f}%)")
    print(f"  F1-Score:  {enhanced_result.f1_score:.4f} ({enhanced_result.f1_score*100:.2f}%)")
    print(f"  Accuracy:  {enhanced_accuracy:.4f} ({enhanced_accuracy*100:.2f}%)")
    print(f"  TP: {enhanced_result.true_positives}, FP: {enhanced_result.false_positives}, "
          f"TN: {enhanced_result.true_negatives}, FN: {enhanced_result.false_negatives}")
    print(f"  Avg Time:  {enhanced_result.avg_execution_time:.2f}ms")
    
    # Calculate improvements
    precision_improvement = enhanced_result.precision - original_data['precision']
    recall_improvement = enhanced_result.recall - original_data['recall']
    f1_improvement = enhanced_result.f1_score - original_data['f1_score']
    accuracy_improvement = enhanced_accuracy - original_data['accuracy']
    
    fp_reduction = original_data['false_positives'] - enhanced_result.false_positives
    fn_reduction = original_data['false_negatives'] - enhanced_result.false_negatives
    
    print(f"\nIMPROVEMENTS:")
    print(f"  Precision: {precision_improvement:+.4f} ({precision_improvement*100:+.2f} percentage points)")
    print(f"  Recall:    {recall_improvement:+.4f} ({recall_improvement*100:+.2f} percentage points)")
    print(f"  F1-Score:  {f1_improvement:+.4f} ({f1_improvement*100:+.2f} percentage points)")
    print(f"  Accuracy:  {accuracy_improvement:+.4f} ({accuracy_improvement*100:+.2f} percentage points)")
    print(f"  False Positives: {fp_reduction:+d} (reduced by {fp_reduction})")
    print(f"  False Negatives: {fn_reduction:+d} (reduced by {fn_reduction})")
    
    # Save comparison summary
    comparison_data = {
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
        'enhanced': {
            'precision': enhanced_result.precision,
            'recall': enhanced_result.recall,
            'f1_score': enhanced_result.f1_score,
            'accuracy': enhanced_accuracy,
            'true_positives': enhanced_result.true_positives,
            'false_positives': enhanced_result.false_positives,
            'true_negatives': enhanced_result.true_negatives,
            'false_negatives': enhanced_result.false_negatives,
            'avg_execution_time_ms': enhanced_result.avg_execution_time
        },
        'improvements': {
            'precision': precision_improvement,
            'recall': recall_improvement,
            'f1_score': f1_improvement,
            'accuracy': accuracy_improvement,
            'false_positives_reduced': fp_reduction,
            'false_negatives_reduced': fn_reduction
        }
    }
    
    with open('detector_comparison_results.json', 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\nComparison results saved to: detector_comparison_results.json")
    print("=" * 70)


def analyze_enhanced_errors():
    """Analyze remaining errors in enhanced detector"""
    
    print("\n" + "=" * 70)
    print("ENHANCED DETECTOR ERROR ANALYSIS")
    print("=" * 70)
    
    # Load enhanced results
    with open('enhanced_factory_detector_evaluation_results.json', 'r') as f:
        data = json.load(f)
    
    contracts = data['contracts']
    
    # Find remaining false positives and false negatives
    enhanced_fps = [c for c in contracts if not c['is_factory_ground_truth'] and c['enhanced_is_factory_detected']]
    enhanced_fns = [c for c in contracts if c['is_factory_ground_truth'] and not c['enhanced_is_factory_detected']]
    
    print(f"Remaining False Positives: {len(enhanced_fps)}")
    print(f"Remaining False Negatives: {len(enhanced_fns)}")
    
    if enhanced_fps:
        print(f"\nRemaining False Positives (first 5):")
        for i, contract in enumerate(enhanced_fps[:5]):
            print(f"  {i+1}. {contract['address']} - {contract['enhanced_factory_type']} "
                  f"(confidence: {contract.get('enhanced_confidence', 0):.3f})")
    
    if enhanced_fns:
        print(f"\nRemaining False Negatives (first 5):")
        for i, contract in enumerate(enhanced_fns[:5]):
            print(f"  {i+1}. {contract['address']} - {contract['enhanced_factory_type']} "
                  f"(confidence: {contract.get('enhanced_confidence', 0):.3f})")
    
    print("=" * 70)


def main():
    """Main evaluation function"""
    try:
        compare_detectors()
        analyze_enhanced_errors()
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())