

#!/usr/bin/env python3
"""
Factory Detector Evaluation Script

This script evaluates the factory_detector effectiveness using the GroundTruth dataset
from Google BigQuery. It measures:
1. Precision - How many detected factories are actually factories
2. Recall - How many actual factories are detected
3. Execution time - Performance metrics

The script:
1. Fetches GroundTruth dataset from BigQuery
2. Runs factory_detector on each contract
3. Updates BigQuery with detection results and execution times
4. Calculates and displays precision/recall metrics
"""

import os
import sys
import time
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from google.cloud import bigquery
from google.oauth2 import service_account

# Add project root to path to import factory_detector
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from factory_detector import ImprovedFactoryDetector, FactoryResult


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


class FactoryDetectorEvaluator:
    """Factory detector evaluation system"""
    
    def __init__(self, project_id: str = "ziyue-wang", dataset_id: str = "tosem_factory_analysis", 
                 table_id: str = "GroundTruthDataset"):
        """Initialize evaluator with BigQuery configuration"""
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.table_full_id = f"{project_id}.{dataset_id}.{table_id}"
        
        # Initialize BigQuery client
        self.client = self._init_bigquery_client()
        self.detector = ImprovedFactoryDetector()
        
    def _init_bigquery_client(self) -> bigquery.Client:
        """Initialize BigQuery client with proper authentication"""
        try:
            # Try to use default credentials first
            client = bigquery.Client(project=self.project_id)
            # Test the connection
            client.get_dataset(f"{self.project_id}.{self.dataset_id}")
            print("Successfully connected to BigQuery using default credentials")
            return client
        except Exception as e:
            print(f"Failed to connect with default credentials: {e}")
            print("Please ensure you are authenticated with Google Cloud:")
            print("Run: gcloud auth application-default login")
            raise
            
    def fetch_ground_truth_data(self) -> List[Dict]:
        """Fetch ground truth data from BigQuery"""
        query = f"""
        SELECT 
            address,
            bytecode,
            is_factory_ground_truth,
            is_factory_detected,
            execution_time,
            source_type,
            verification_notes,
            created_at
        FROM `{self.table_full_id}`
        ORDER BY created_at DESC
        """
        
        print(f"Fetching data from {self.table_full_id}...")
        query_job = self.client.query(query)
        results = query_job.result()
        
        data = []
        for row in results:
            data.append({
                'address': row.address,
                'bytecode': row.bytecode,
                'is_factory_ground_truth': row.is_factory_ground_truth,
                'is_factory_detected': row.is_factory_detected,
                'execution_time': row.execution_time,
                'source_type': row.source_type,
                'verification_notes': row.verification_notes,
                'created_at': row.created_at
            })
            
        print(f"Fetched {len(data)} contracts from ground truth dataset")
        return data
        
    def run_factory_detection(self, contracts: List[Dict]) -> List[Dict]:
        """Run factory detector on contracts and record results"""
        print("Running factory detection on contracts...")
        
        results = []
        total_execution_time = 0
        
        for i, contract in enumerate(contracts):
            if i % 100 == 0:
                print(f"Processing contract {i+1}/{len(contracts)}")
                
            try:
                # Run factory detection
                start_time = time.perf_counter()
                detection_result = self.detector.detect_factory_contract(contract['bytecode'])
                execution_time_ms = int((time.perf_counter() - start_time) * 1000)
                
                # Update contract with detection results
                contract['is_factory_detected'] = detection_result.is_factory_contract
                contract['execution_time'] = execution_time_ms
                
                total_execution_time += execution_time_ms
                results.append(contract)
                
            except Exception as e:
                print(f"Error processing contract {contract['address']}: {e}")
                # Mark as failed detection
                contract['is_factory_detected'] = False
                contract['execution_time'] = 0
                results.append(contract)
                
        print(f"Completed factory detection. Total execution time: {total_execution_time}ms")
        return results
        
    def update_bigquery_results(self, contracts: List[Dict]) -> None:
        """Update BigQuery table with detection results"""
        print("Updating BigQuery with detection results...")
        
        # Prepare update data
        update_data = []
        for contract in contracts:
            update_data.append({
                'address': contract['address'],
                'is_factory_detected': contract['is_factory_detected'],
                'execution_time': contract['execution_time']
            })
            
        # Use BigQuery DML to update records
        for i, data in enumerate(update_data):
            if i % 50 == 0:
                print(f"Updating record {i+1}/{len(update_data)}")
                
            update_query = f"""
            UPDATE `{self.table_full_id}`
            SET 
                is_factory_detected = @is_factory_detected,
                execution_time = @execution_time
            WHERE address = @address
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("address", "STRING", data['address']),
                    bigquery.ScalarQueryParameter("is_factory_detected", "BOOL", data['is_factory_detected']),
                    bigquery.ScalarQueryParameter("execution_time", "INT64", data['execution_time'])
                ]
            )
            
            query_job = self.client.query(update_query, job_config=job_config)
            query_job.result()  # Wait for completion
            
        print("Successfully updated BigQuery with all detection results")
        
    def calculate_metrics(self, contracts: List[Dict]) -> EvaluationResult:
        """Calculate precision, recall, and other metrics"""
        print("Calculating evaluation metrics...")
        
        true_positives = 0   # Factory contract detected as factory
        false_positives = 0  # Non-factory contract detected as factory  
        true_negatives = 0   # Non-factory contract detected as non-factory
        false_negatives = 0  # Factory contract detected as non-factory
        
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
        print("\n" + "="*60)
        print("FACTORY DETECTOR EVALUATION RESULTS")
        print("="*60)
        
        print(f"Total Contracts Evaluated: {results.total_contracts}")
        print(f"Total Execution Time: {results.total_execution_time:,.0f}ms ({results.total_execution_time/1000:.2f}s)")
        print(f"Average Execution Time: {results.avg_execution_time:.2f}ms")
        
        print("\nConfusion Matrix:")
        print(f"  True Positives (TP):  {results.true_positives:4d}")
        print(f"  False Positives (FP): {results.false_positives:4d}")
        print(f"  True Negatives (TN):  {results.true_negatives:4d}")
        print(f"  False Negatives (FN): {results.false_negatives:4d}")
        
        print("\nPerformance Metrics:")
        print(f"  Precision: {results.precision:.4f} ({results.precision*100:.2f}%)")
        print(f"  Recall:    {results.recall:.4f} ({results.recall*100:.2f}%)")
        print(f"  F1-Score:  {results.f1_score:.4f} ({results.f1_score*100:.2f}%)")
        
        print("\nMetric Definitions:")
        print("  Precision = TP / (TP + FP) - How many detected factories are actually factories")
        print("  Recall = TP / (TP + FN) - How many actual factories are detected")
        print("  F1-Score = 2 * (Precision * Recall) / (Precision + Recall)")
        
        print("="*60)
        
    def run_evaluation(self) -> EvaluationResult:
        """Run complete factory detector evaluation"""
        print("Starting Factory Detector Evaluation")
        print("="*60)
        
        # Step 1: Fetch ground truth data
        contracts = self.fetch_ground_truth_data()
        
        if not contracts:
            raise ValueError("No ground truth data found in BigQuery")
            
        # Check if detection has already been run
        needs_detection = any(contract['is_factory_detected'] is None for contract in contracts)
        
        if needs_detection:
            print("Detection results not found, running factory detection...")
            # Step 2: Run factory detection
            contracts = self.run_factory_detection(contracts)
            
            # Step 3: Update BigQuery with results
            self.update_bigquery_results(contracts)
        else:
            print("Detection results already present, skipping detection phase")
            
        # Step 4: Calculate metrics
        results = self.calculate_metrics(contracts)
        
        # Step 5: Display results
        self.print_evaluation_results(results)
        
        return results


def main():
    """Main evaluation function"""
    try:
        evaluator = FactoryDetectorEvaluator()
        results = evaluator.run_evaluation()
        
        # Save results to file
        results_file = "factory_detector_evaluation_results.json"
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
