#!/usr/bin/env python3
"""
Factory Detector Full Evaluation Script with Batch Processing

This script evaluates the factory_detector on the complete GroundTruth dataset
with optimized batch processing and progress tracking.
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


class BatchedFactoryDetectorEvaluator:
    """Factory detector evaluation with optimized batch processing"""
    
    def __init__(self, project_id: str = "ziyue-wang", dataset_id: str = "tosem_factory_analysis", 
                 table_id: str = "GroundTruthDataset", batch_size: int = 100):
        """Initialize evaluator with batch processing"""
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.table_full_id = f"{project_id}.{dataset_id}.{table_id}"
        self.batch_size = batch_size
        
        # Initialize BigQuery client and detector
        self.client = bigquery.Client(project=project_id)
        self.detector = ImprovedFactoryDetector()
        
    def get_total_count(self) -> int:
        """Get total number of contracts in the dataset"""
        query = f"SELECT COUNT(*) as total FROM `{self.table_full_id}`"
        query_job = self.client.query(query)
        result = list(query_job.result())[0]
        return result.total
        
    def get_contracts_batch(self, offset: int, limit: int) -> List[Dict]:
        """Fetch a batch of contracts"""
        query = f"""
        SELECT 
            address,
            bytecode,
            is_factory_ground_truth,
            is_factory_detected,
            execution_time,
            source_type,
            verification_notes
        FROM `{self.table_full_id}`
        ORDER BY address
        LIMIT {limit} OFFSET {offset}
        """
        
        query_job = self.client.query(query)
        results = query_job.result()
        
        contracts = []
        for row in results:
            contracts.append({
                'address': row.address,
                'bytecode': row.bytecode,
                'is_factory_ground_truth': row.is_factory_ground_truth,
                'is_factory_detected': row.is_factory_detected,
                'execution_time': row.execution_time,
                'source_type': row.source_type,
                'verification_notes': row.verification_notes
            })
            
        return contracts
        
    def process_contracts_batch(self, contracts: List[Dict]) -> List[Dict]:
        """Process a batch of contracts with factory detection"""
        processed_contracts = []
        
        for contract in contracts:
            try:
                # Skip if already processed
                if contract['is_factory_detected'] is not None:
                    processed_contracts.append(contract)
                    continue
                
                # Run factory detection
                start_time = time.perf_counter()
                detection_result = self.detector.detect_factory_contract(contract['bytecode'])
                execution_time_ms = int((time.perf_counter() - start_time) * 1000)
                
                # Update contract with detection results
                contract['is_factory_detected'] = detection_result.is_factory_contract
                contract['execution_time'] = execution_time_ms
                
                processed_contracts.append(contract)
                
            except Exception as e:
                print(f"Error processing contract {contract['address']}: {e}")
                # Mark as failed detection
                contract['is_factory_detected'] = False
                contract['execution_time'] = 0
                processed_contracts.append(contract)
                
        return processed_contracts
        
    def update_batch_to_bigquery(self, contracts: List[Dict]) -> None:
        """Update a batch of contracts in BigQuery using individual updates"""
        
        updates_to_process = []
        for contract in contracts:
            if contract['is_factory_detected'] is not None:
                updates_to_process.append(contract)
        
        if not updates_to_process:
            return
        
        # Process updates individually (more reliable than batch STRUCT operations)
        for contract in updates_to_process:
            update_query = f"""
            UPDATE `{self.table_full_id}`
            SET 
                is_factory_detected = @is_factory_detected,
                execution_time = @execution_time
            WHERE address = @address
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("address", "STRING", contract['address']),
                    bigquery.ScalarQueryParameter("is_factory_detected", "BOOL", contract['is_factory_detected']),
                    bigquery.ScalarQueryParameter("execution_time", "INT64", contract['execution_time'])
                ]
            )
            
            query_job = self.client.query(update_query, job_config=job_config)
            query_job.result()  # Wait for completion
        
    def calculate_final_metrics(self) -> EvaluationResult:
        """Calculate final metrics from all processed contracts"""
        query = f"""
        SELECT 
            is_factory_ground_truth,
            is_factory_detected,
            execution_time
        FROM `{self.table_full_id}`
        WHERE is_factory_detected IS NOT NULL
        """
        
        query_job = self.client.query(query)
        results = query_job.result()
        
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        execution_times = []
        
        for row in results:
            ground_truth = row.is_factory_ground_truth
            detected = row.is_factory_detected
            exec_time = row.execution_time or 0
            
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
        total_contracts = len(execution_times)
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
        print(f"  True Positives (TP):  {results.true_positives:,4d} (Factory correctly detected)")
        print(f"  False Positives (FP): {results.false_positives:,4d} (Non-factory incorrectly detected as factory)")
        print(f"  True Negatives (TN):  {results.true_negatives:,4d} (Non-factory correctly detected)")
        print(f"  False Negatives (FN): {results.false_negatives:,4d} (Factory incorrectly detected as non-factory)")
        
        print("\nPerformance Metrics:")
        print(f"  Precision: {results.precision:.4f} ({results.precision*100:.2f}%)")
        print(f"  Recall:    {results.recall:.4f} ({results.recall*100:.2f}%)")
        print(f"  F1-Score:  {results.f1_score:.4f} ({results.f1_score*100:.2f}%)")
        
        print(f"\nAccuracy: {(results.true_positives + results.true_negatives) / results.total_contracts:.4f} "
              f"({(results.true_positives + results.true_negatives) / results.total_contracts * 100:.2f}%)")
        
        print("\nMetric Definitions:")
        print("  Precision = TP / (TP + FP) - How many detected factories are actually factories")
        print("  Recall = TP / (TP + FN) - How many actual factories are detected") 
        print("  F1-Score = 2 * (Precision * Recall) / (Precision + Recall)")
        print("  Accuracy = (TP + TN) / Total - Overall correctness")
        
        print("="*70)
        
    def run_full_evaluation(self) -> EvaluationResult:
        """Run complete factory detector evaluation with batch processing"""
        print("Starting Full Factory Detector Evaluation")
        print("="*70)
        
        # Get total count
        total_contracts = self.get_total_count()
        print(f"Total contracts in dataset: {total_contracts:,}")
        print(f"Batch size: {self.batch_size}")
        print(f"Estimated batches: {(total_contracts + self.batch_size - 1) // self.batch_size}")
        
        # Process in batches
        processed_count = 0
        start_time = time.time()
        
        for batch_start in range(0, total_contracts, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_contracts)
            batch_num = (batch_start // self.batch_size) + 1
            total_batches = (total_contracts + self.batch_size - 1) // self.batch_size
            
            print(f"\nProcessing batch {batch_num}/{total_batches} "
                  f"(contracts {batch_start+1}-{batch_end})")
            
            # Fetch batch
            contracts = self.get_contracts_batch(batch_start, self.batch_size)
            
            # Process batch
            batch_start_time = time.time()
            processed_contracts = self.process_contracts_batch(contracts)
            batch_process_time = time.time() - batch_start_time
            
            # Update BigQuery
            self.update_batch_to_bigquery(processed_contracts)
            
            processed_count += len(contracts)
            elapsed_time = time.time() - start_time
            
            # Progress reporting
            contracts_per_second = processed_count / elapsed_time if elapsed_time > 0 else 0
            eta_seconds = (total_contracts - processed_count) / contracts_per_second if contracts_per_second > 0 else 0
            
            print(f"  Batch processed in {batch_process_time:.1f}s")
            print(f"  Progress: {processed_count:,}/{total_contracts:,} "
                  f"({processed_count/total_contracts*100:.1f}%)")
            print(f"  Speed: {contracts_per_second:.1f} contracts/second")
            print(f"  ETA: {eta_seconds/60:.1f} minutes")
            
        # Calculate final metrics
        print("\nCalculating final metrics...")
        results = self.calculate_final_metrics()
        
        # Display results
        self.print_evaluation_results(results)
        
        return results


def main():
    """Main evaluation function"""
    try:
        # Use smaller batch size for better progress tracking
        evaluator = BatchedFactoryDetectorEvaluator(batch_size=50)
        results = evaluator.run_full_evaluation()
        
        # Save results to file
        results_file = "factory_detector_full_evaluation_results.json"
        results_data = {
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
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
            
        print(f"\nResults saved to: {results_file}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())