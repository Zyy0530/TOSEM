#!/usr/bin/env python3
"""
Local Factory Detector Evaluation - No BigQuery Updates

This approach:
1. Fetches all data from BigQuery once
2. Processes all contracts with factory_detector locally
3. Saves results to local JSON file
4. Calculates and displays metrics

This is the fastest approach - no BigQuery update bottleneck.
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


class LocalFactoryDetectorEvaluator:
    """Local factory detector evaluation - process locally, save locally"""
    
    def __init__(self, project_id: str = "ziyue-wang", dataset_id: str = "tosem_factory_analysis", 
                 table_id: str = "GroundTruthDataset"):
        """Initialize evaluator"""
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.table_full_id = f"{project_id}.{dataset_id}.{table_id}"
        
        # Initialize BigQuery client and detector
        self.client = bigquery.Client(project=project_id)
        self.detector = ImprovedFactoryDetector()
        
    def fetch_all_contracts(self) -> List[Dict]:
        """Fetch all contracts from BigQuery at once"""
        print("Fetching all contracts from BigQuery...")
        
        query = f"""
        SELECT 
            address,
            bytecode,
            is_factory_ground_truth,
            source_type,
            verification_notes
        FROM `{self.table_full_id}`
        ORDER BY address
        """
        
        query_job = self.client.query(query)
        results = query_job.result()
        
        contracts = []
        for row in results:
            contracts.append({
                'address': row.address,
                'bytecode': row.bytecode,
                'is_factory_ground_truth': row.is_factory_ground_truth,
                'source_type': row.source_type,
                'verification_notes': row.verification_notes,
                'is_factory_detected': None,  # Will be filled during processing
                'execution_time': None        # Will be filled during processing
            })
            
        print(f"Fetched {len(contracts):,} contracts")
        return contracts
        
    def process_all_contracts(self, contracts: List[Dict]) -> List[Dict]:
        """Process all contracts locally with progress tracking"""
        print(f"\nProcessing {len(contracts):,} contracts with factory detector...")
        
        total_execution_time = 0
        start_time = time.time()
        
        for i, contract in enumerate(contracts):
            # Progress reporting every 100 contracts
            if i % 100 == 0 and i > 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                eta = (len(contracts) - i) / rate if rate > 0 else 0
                print(f"  Progress: {i:,}/{len(contracts):,} ({i/len(contracts)*100:.1f}%) | "
                      f"Rate: {rate:.1f}/sec | ETA: {eta/60:.1f}min")
            
            try:
                # Run factory detection
                detection_start = time.perf_counter()
                detection_result = self.detector.detect_factory_contract(contract['bytecode'])
                execution_time_ms = int((time.perf_counter() - detection_start) * 1000)
                
                # Update contract with detection results
                contract['is_factory_detected'] = detection_result.is_factory_contract
                contract['execution_time'] = execution_time_ms
                contract['factory_type'] = detection_result.factory_type
                contract['create_positions'] = len(detection_result.verified_create_positions)
                contract['create2_positions'] = len(detection_result.verified_create2_positions)
                
                total_execution_time += execution_time_ms
                
            except Exception as e:
                print(f"Error processing contract {contract['address']}: {e}")
                # Mark as failed detection
                contract['is_factory_detected'] = False
                contract['execution_time'] = 0
                contract['factory_type'] = 'ERROR'
                contract['create_positions'] = 0
                contract['create2_positions'] = 0
        
        elapsed = time.time() - start_time
        print(f"\nProcessing completed in {elapsed:.1f} seconds")
        print(f"  Total factory detection time: {total_execution_time:,}ms ({total_execution_time/1000:.1f}s)")
        print(f"  Average per contract: {total_execution_time/len(contracts):.1f}ms")
        
        return contracts
        
    def save_results_to_file(self, contracts: List[Dict], filename: str = "factory_detector_evaluation_results.json") -> None:
        """Save all results to local JSON file"""
        print(f"\nSaving results to {filename}...")
        
        # Prepare data for saving
        results_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_contracts': len(contracts),
            'contracts': contracts
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
            
        print(f"Results saved to {filename} ({os.path.getsize(filename) / 1024 / 1024:.1f} MB)")
        
    def calculate_metrics(self, contracts: List[Dict]) -> EvaluationResult:
        """Calculate metrics from processed contracts"""
        print("\nCalculating evaluation metrics...")
        
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
        
    def print_evaluation_results(self, results: EvaluationResult) -> None:
        """Print detailed evaluation results"""
        print("\n" + "="*70)
        print("FACTORY DETECTOR EVALUATION RESULTS")
        print("="*70)
        
        print(f"Total Contracts Evaluated: {results.total_contracts:,}")
        print(f"Total Execution Time: {results.total_execution_time:,.0f}ms ({results.total_execution_time/1000:.1f}s)")
        print(f"Average Execution Time: {results.avg_execution_time:.2f}ms")
        
        print("\nConfusion Matrix:")
        print(f"  True Positives (TP):  {results.true_positives:4,d} (Factory correctly detected)")
        print(f"  False Positives (FP): {results.false_positives:4,d} (Non-factory incorrectly detected as factory)")
        print(f"  True Negatives (TN):  {results.true_negatives:4,d} (Non-factory correctly detected)")
        print(f"  False Negatives (FN): {results.false_negatives:4,d} (Factory incorrectly detected as non-factory)")
        
        print("\nPerformance Metrics:")
        print(f"  Precision: {results.precision:.4f} ({results.precision*100:.2f}%)")
        print(f"  Recall:    {results.recall:.4f} ({results.recall*100:.2f}%)")
        print(f"  F1-Score:  {results.f1_score:.4f} ({results.f1_score*100:.2f}%)")
        
        accuracy = (results.true_positives + results.true_negatives) / results.total_contracts
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print("\nMetric Definitions:")
        print("  Precision = TP / (TP + FP) - How many detected factories are actually factories")
        print("  Recall = TP / (TP + FN) - How many actual factories are detected") 
        print("  F1-Score = 2 * (Precision * Recall) / (Precision + Recall)")
        print("  Accuracy = (TP + TN) / Total - Overall correctness")
        
        # Additional statistics
        print(f"\nAdditional Statistics:")
        factory_contracts = results.true_positives + results.false_negatives
        non_factory_contracts = results.true_negatives + results.false_positives
        print(f"  Factory contracts in dataset: {factory_contracts:,} ({factory_contracts/results.total_contracts*100:.1f}%)")
        print(f"  Non-factory contracts in dataset: {non_factory_contracts:,} ({non_factory_contracts/results.total_contracts*100:.1f}%)")
        
        print("="*70)
        
    def analyze_errors(self, contracts: List[Dict]) -> None:
        """Analyze false positives and false negatives"""
        print("\nError Analysis:")
        print("="*50)
        
        false_positives = [c for c in contracts if not c['is_factory_ground_truth'] and c['is_factory_detected']]
        false_negatives = [c for c in contracts if c['is_factory_ground_truth'] and not c['is_factory_detected']]
        
        if false_positives:
            print(f"\nFalse Positives ({len(false_positives)} contracts):")
            for i, contract in enumerate(false_positives[:5]):  # Show first 5
                print(f"  {i+1}. {contract['address']} ({contract['source_type']}) - {contract['factory_type']}")
                if contract['verification_notes']:
                    print(f"     Notes: {contract['verification_notes'][:80]}...")
            if len(false_positives) > 5:
                print(f"     ... and {len(false_positives)-5} more")
        
        if false_negatives:
            print(f"\nFalse Negatives ({len(false_negatives)} contracts):")
            for i, contract in enumerate(false_negatives[:5]):  # Show first 5
                print(f"  {i+1}. {contract['address']} ({contract['source_type']})")
                if contract['verification_notes']:
                    print(f"     Notes: {contract['verification_notes'][:80]}...")
            if len(false_negatives) > 5:
                print(f"     ... and {len(false_negatives)-5} more")
        
        print("="*50)
        
    def run_local_evaluation(self) -> EvaluationResult:
        """Run local factory detector evaluation"""
        print("Starting Local Factory Detector Evaluation")
        print("="*70)
        
        # Step 1: Fetch all data
        contracts = self.fetch_all_contracts()
        
        # Step 2: Process all contracts locally
        processed_contracts = self.process_all_contracts(contracts)
        
        # Step 3: Save results to local file
        self.save_results_to_file(processed_contracts)
        
        # Step 4: Calculate metrics
        results = self.calculate_metrics(processed_contracts)
        
        # Step 5: Display results
        self.print_evaluation_results(results)
        
        # Step 6: Error analysis
        self.analyze_errors(processed_contracts)
        
        return results


def main():
    """Main evaluation function"""
    try:
        evaluator = LocalFactoryDetectorEvaluator()
        results = evaluator.run_local_evaluation()
        
        # Save summary metrics to separate file
        summary_file = "factory_detector_evaluation_summary.json"
        summary_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_contracts': results.total_contracts,
            'true_positives': results.true_positives,
            'false_positives': results.false_positives,
            'true_negatives': results.true_negatives,
            'false_negatives': results.false_negatives,
            'precision': results.precision,
            'recall': results.recall,
            'f1_score': results.f1_score,
            'accuracy': (results.true_positives + results.true_negatives) / results.total_contracts,
            'avg_execution_time_ms': results.avg_execution_time,
            'total_execution_time_ms': results.total_execution_time
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
            
        print(f"\nSummary metrics saved to: {summary_file}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())