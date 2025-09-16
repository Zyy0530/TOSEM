#!/usr/bin/env python3
"""
Optimized Factory Detector Evaluation - Process Locally, Update Once

This approach:
1. Fetches all data locally 
2. Processes all contracts with factory_detector locally
3. Updates BigQuery in one final operation
4. Calculates and displays metrics

This eliminates the BigQuery update bottleneck during processing.
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


class OptimizedFactoryDetectorEvaluator:
    """Optimized factory detector evaluation - process locally, update once"""
    
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
            is_factory_detected,
            execution_time,
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
                'is_factory_detected': row.is_factory_detected,
                'execution_time': row.execution_time,
                'source_type': row.source_type,
                'verification_notes': row.verification_notes
            })
            
        print(f"Fetched {len(contracts):,} contracts")
        return contracts
        
    def process_all_contracts(self, contracts: List[Dict]) -> List[Dict]:
        """Process all contracts locally with progress tracking"""
        print(f"\nProcessing {len(contracts):,} contracts with factory detector...")
        
        processed_contracts = []
        total_execution_time = 0
        processed_count = 0
        skipped_count = 0
        
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
                # Skip if already processed
                if contract['is_factory_detected'] is not None:
                    skipped_count += 1
                    processed_contracts.append(contract)
                    continue
                
                # Run factory detection
                detection_start = time.perf_counter()
                detection_result = self.detector.detect_factory_contract(contract['bytecode'])
                execution_time_ms = int((time.perf_counter() - detection_start) * 1000)
                
                # Update contract with detection results
                contract['is_factory_detected'] = detection_result.is_factory_contract
                contract['execution_time'] = execution_time_ms
                
                total_execution_time += execution_time_ms
                processed_count += 1
                processed_contracts.append(contract)
                
            except Exception as e:
                print(f"Error processing contract {contract['address']}: {e}")
                # Mark as failed detection
                contract['is_factory_detected'] = False
                contract['execution_time'] = 0
                processed_contracts.append(contract)
                processed_count += 1
        
        elapsed = time.time() - start_time
        print(f"\nProcessing completed in {elapsed:.1f} seconds")
        print(f"  Processed: {processed_count:,} contracts")
        print(f"  Skipped (already processed): {skipped_count:,} contracts")
        print(f"  Total factory detection time: {total_execution_time:,}ms ({total_execution_time/1000:.1f}s)")
        print(f"  Average per contract: {total_execution_time/processed_count:.1f}ms" if processed_count > 0 else "")
        
        return processed_contracts
        
    def update_all_to_bigquery(self, contracts: List[Dict]) -> None:
        """Update all contracts to BigQuery using CSV upload approach"""
        print("\nUpdating BigQuery with all results...")
        
        # Prepare updates
        updates_needed = [c for c in contracts if c['is_factory_detected'] is not None]
        
        if not updates_needed:
            print("No updates needed")
            return
            
        print(f"Updating {len(updates_needed):,} contracts...")
        
        # Create a temporary table with the updates and use MERGE
        temp_table_id = f"{self.table_id}_temp_updates"
        temp_table_full_id = f"{self.project_id}.{self.dataset_id}.{temp_table_id}"
        
        # Create temporary table schema
        temp_schema = [
            bigquery.SchemaField("address", "STRING"),
            bigquery.SchemaField("is_factory_detected", "BOOLEAN"),
            bigquery.SchemaField("execution_time", "INTEGER"),
        ]
        
        # Create temporary table
        temp_table = bigquery.Table(temp_table_full_id, schema=temp_schema)
        temp_table = self.client.create_table(temp_table, exists_ok=True)
        
        # Clear temporary table
        self.client.query(f"DELETE FROM `{temp_table_full_id}` WHERE TRUE").result()
        
        # Insert update data
        rows_to_insert = []
        for contract in updates_needed:
            rows_to_insert.append({
                'address': contract['address'],
                'is_factory_detected': contract['is_factory_detected'],
                'execution_time': contract['execution_time']
            })
        
        # Insert in chunks
        chunk_size = 1000
        for i in range(0, len(rows_to_insert), chunk_size):
            chunk = rows_to_insert[i:i + chunk_size]
            errors = self.client.insert_rows_json(temp_table, chunk)
            if errors:
                print(f"Error inserting chunk {i//chunk_size + 1}: {errors}")
            else:
                print(f"  Inserted chunk {i//chunk_size + 1}/{(len(rows_to_insert) + chunk_size - 1)//chunk_size}")
        
        # Use MERGE to update the main table
        merge_query = f"""
        MERGE `{self.table_full_id}` AS target
        USING `{temp_table_full_id}` AS source
        ON target.address = source.address
        WHEN MATCHED THEN
          UPDATE SET 
            is_factory_detected = source.is_factory_detected,
            execution_time = source.execution_time
        """
        
        print("Executing MERGE operation...")
        query_job = self.client.query(merge_query)
        query_job.result()
        
        # Clean up temporary table
        self.client.delete_table(temp_table)
        
        print("BigQuery update completed successfully")
        
    def calculate_metrics(self, contracts: List[Dict]) -> EvaluationResult:
        """Calculate metrics from processed contracts"""
        print("\nCalculating evaluation metrics...")
        
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        execution_times = []
        
        for contract in contracts:
            if contract['is_factory_detected'] is None:
                continue
                
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
        
        accuracy = (results.true_positives + results.true_negatives) / results.total_contracts
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print("\nMetric Definitions:")
        print("  Precision = TP / (TP + FP) - How many detected factories are actually factories")
        print("  Recall = TP / (TP + FN) - How many actual factories are detected") 
        print("  F1-Score = 2 * (Precision * Recall) / (Precision + Recall)")
        print("  Accuracy = (TP + TN) / Total - Overall correctness")
        
        print("="*70)
        
    def run_optimized_evaluation(self) -> EvaluationResult:
        """Run optimized factory detector evaluation"""
        print("Starting Optimized Factory Detector Evaluation")
        print("="*70)
        
        # Step 1: Fetch all data
        contracts = self.fetch_all_contracts()
        
        # Step 2: Process all contracts locally
        processed_contracts = self.process_all_contracts(contracts)
        
        # Step 3: Update BigQuery once
        self.update_all_to_bigquery(processed_contracts)
        
        # Step 4: Calculate metrics
        results = self.calculate_metrics(processed_contracts)
        
        # Step 5: Display results
        self.print_evaluation_results(results)
        
        return results


def main():
    """Main evaluation function"""
    try:
        evaluator = OptimizedFactoryDetectorEvaluator()
        results = evaluator.run_optimized_evaluation()
        
        # Save results to file
        results_file = "factory_detector_optimized_evaluation_results.json"
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