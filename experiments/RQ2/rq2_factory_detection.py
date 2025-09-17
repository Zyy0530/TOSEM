#!/usr/bin/env python3
"""
RQ2 Large-Scale Factory Contract Detection Experiment

This script performs comprehensive factory contract detection across Ethereum and Polygon blockchains:
- Fetches all contracts deployed before June 1, 2025 from BigQuery
- Runs factory_detector.py on contract bytecode  
- Stores results with progress tracking and resume capability
- Generates data for daily deployment analysis, bytecode deduplication, and transaction analysis

Author: Research Team
Date: 2025
"""

import os
import sys
import json
import logging
import hashlib
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add parent directory to path for factory_detector import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from google.cloud import bigquery
    from google.cloud.exceptions import NotFound
    import pandas as pd
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Please install: pip install google-cloud-bigquery pandas")
    sys.exit(1)

try:
    from factory_detector import ImprovedFactoryDetector, detect_factory_improved
except ImportError as e:
    print(f"Cannot import factory_detector: {e}")
    print("Please ensure factory_detector.py is in the parent directory")
    sys.exit(1)

@dataclass
class ChainConfig:
    """Configuration for blockchain datasets"""
    chain_name: str
    display_name: str
    dataset_name: str
    contracts_table: str
    genesis_date: str
    address_field: str = "address"
    bytecode_field: str = "bytecode"
    timestamp_field: str = "block_timestamp"
    block_number_field: str = "block_number"
    tx_hash_field: str = "transaction_hash"

@dataclass
class DetectionResult:
    """Factory detection result for a single contract"""
    chain: str
    address: str
    is_factory: bool
    is_create2_only: bool
    is_create_only: bool
    is_both: bool
    execution_time_ms: float
    bytecode_hash: str
    deployment_date: str
    block_number: int
    processed_at: str

class ProgressTracker:
    """Thread-safe progress tracking"""
    
    def __init__(self, total_contracts: int):
        self.total_contracts = total_contracts
        self.processed = 0
        self.factories_found = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
        
    def update(self, found_factory: bool = False):
        with self.lock:
            self.processed += 1
            if found_factory:
                self.factories_found += 1
                
    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            elapsed = time.time() - self.start_time
            progress_pct = (self.processed / self.total_contracts) * 100
            rate = self.processed / elapsed if elapsed > 0 else 0
            eta = (self.total_contracts - self.processed) / rate if rate > 0 else 0
            
            return {
                'processed': self.processed,
                'total': self.total_contracts,
                'progress_pct': progress_pct,
                'factories_found': self.factories_found,
                'elapsed_hours': elapsed / 3600,
                'rate_per_hour': rate * 3600,
                'eta_hours': eta / 3600
            }

class RQ2FactoryDetectionExperiment:
    """Main experiment class for large-scale factory detection"""
    
    def __init__(self, config_path: str = None):
        # Load configuration
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.project_id = config['project_id']
        self.result_dataset = config['result_dataset']
        self.result_table = "rq2_factory_detection_results"
        self.progress_table = "rq2_experiment_progress"
        self.cutoff_date = config['cutoff_date']
        self.batch_size = config['batch_size']
        self.max_workers = config['max_workers']
        
        # Initialize BigQuery client
        self.client = bigquery.Client(project=self.project_id)
        
        # Initialize factory detector
        self.detector = ImprovedFactoryDetector()
        
        # Chain configurations - load from config
        self.chains = {}
        for chain_name, chain_config in config['chains'].items():
            self.chains[chain_name] = ChainConfig(
                chain_name=chain_config['chain_name'],
                display_name=chain_config['display_name'],
                dataset_name=chain_config['dataset_name'],
                contracts_table=chain_config['contracts_table'],
                genesis_date=chain_config['genesis_date'],
                address_field=chain_config['query_fields']['address_field'],
                bytecode_field=chain_config['query_fields']['bytecode_field'],
                timestamp_field=chain_config['query_fields']['timestamp_field'],
                block_number_field=chain_config['query_fields']['block_number_field'],
                tx_hash_field=chain_config['query_fields']['tx_hash_field']
            )
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        os.makedirs('/Users/mac/ResearchSpace/TOSEM/experiments/RQ2/logs', exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger('RQ2Experiment')
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f'/Users/mac/ResearchSpace/TOSEM/experiments/RQ2/logs/rq2_experiment_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_file)
        console_handler = logging.StreamHandler()
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"RQ2 Factory Detection Experiment started - Log file: {log_file}")
    
    def setup_result_tables(self):
        """Create BigQuery tables for storing results and progress"""
        
        # Create dataset if it doesn't exist
        dataset_ref = self.client.dataset(self.result_dataset)
        try:
            self.client.get_dataset(dataset_ref)
            self.logger.info(f"Dataset {self.result_dataset} already exists")
        except NotFound:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = "US"
            dataset.description = "TOSEM Factory Analysis Results"
            self.client.create_dataset(dataset)
            self.logger.info(f"Created dataset {self.result_dataset}")
        
        # Create results table
        results_schema = [
            bigquery.SchemaField("chain", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("address", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("is_factory", "BOOLEAN", mode="REQUIRED"),
            bigquery.SchemaField("is_create2_only", "BOOLEAN", mode="REQUIRED"),
            bigquery.SchemaField("is_create_only", "BOOLEAN", mode="REQUIRED"),
            bigquery.SchemaField("is_both", "BOOLEAN", mode="REQUIRED"),
            bigquery.SchemaField("execution_time_ms", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("bytecode_hash", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("deployment_date", "DATE", mode="REQUIRED"),
            bigquery.SchemaField("block_number", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("processed_at", "TIMESTAMP", mode="REQUIRED"),
        ]
        
        self.create_table_if_not_exists(self.result_table, results_schema)
        
        # Create progress table
        progress_schema = [
            bigquery.SchemaField("chain", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("last_processed_block", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("total_contracts", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("processed_contracts", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("factories_found", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("last_updated", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("status", "STRING", mode="REQUIRED"),
        ]
        
        self.create_table_if_not_exists(self.progress_table, progress_schema)
        
    def create_table_if_not_exists(self, table_name: str, schema: List[bigquery.SchemaField]):
        """Create BigQuery table if it doesn't exist"""
        table_ref = self.client.dataset(self.result_dataset).table(table_name)
        
        try:
            self.client.get_table(table_ref)
            self.logger.info(f"Table {table_name} already exists")
        except NotFound:
            table = bigquery.Table(table_ref, schema=schema)
            table.clustering_fields = ["chain"] if table_name == self.result_table else None
            self.client.create_table(table)
            self.logger.info(f"Created table {table_name}")
    
    def get_chain_progress(self, chain_name: str) -> Dict[str, Any]:
        """Get current processing progress for a chain"""
        query = f"""
        SELECT 
            last_processed_block,
            total_contracts,
            processed_contracts,
            factories_found,
            status
        FROM `{self.project_id}.{self.result_dataset}.{self.progress_table}`
        WHERE chain = @chain_name
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("chain_name", "STRING", chain_name)
            ]
        )
        
        result = self.client.query(query, job_config=job_config).to_dataframe()
        
        if result.empty:
            return {
                'last_processed_block': 0,
                'total_contracts': 0,
                'processed_contracts': 0,
                'factories_found': 0,
                'status': 'not_started'
            }
        else:
            return result.iloc[0].to_dict()
    
    def get_total_contracts_count(self, chain_name: str) -> int:
        """Get total number of contracts to process for a chain"""
        chain_config = self.chains[chain_name]
        
        query = f"""
        SELECT COUNT(*) as total_contracts
        FROM `{chain_config.dataset_name}.{chain_config.contracts_table}`
        WHERE {chain_config.timestamp_field} < TIMESTAMP(@cutoff_date)
        AND {chain_config.bytecode_field} IS NOT NULL
        AND LENGTH({chain_config.bytecode_field}) > 2
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("cutoff_date", "DATE", self.cutoff_date)
            ]
        )
        
        result = self.client.query(query, job_config=job_config).to_dataframe()
        return int(result.iloc[0]['total_contracts'])
    
    def fetch_contract_batch(self, chain_name: str, offset: int, limit: int) -> pd.DataFrame:
        """Fetch a batch of contracts from BigQuery"""
        chain_config = self.chains[chain_name]
        
        query = f"""
        SELECT 
            {chain_config.address_field} as address,
            {chain_config.bytecode_field} as bytecode,
            {chain_config.timestamp_field} as deployment_timestamp,
            {chain_config.block_number_field} as block_number
        FROM `{chain_config.dataset_name}.{chain_config.contracts_table}`
        WHERE {chain_config.timestamp_field} < TIMESTAMP(@cutoff_date)
        AND {chain_config.bytecode_field} IS NOT NULL
        AND LENGTH({chain_config.bytecode_field}) > 2
        ORDER BY {chain_config.block_number_field}
        LIMIT @limit OFFSET @offset
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("cutoff_date", "DATE", self.cutoff_date),
                bigquery.ScalarQueryParameter("limit", "INTEGER", limit),
                bigquery.ScalarQueryParameter("offset", "INTEGER", offset)
            ]
        )
        
        return self.client.query(query, job_config=job_config).to_dataframe()
    
    def detect_factory_batch(self, contracts_df: pd.DataFrame, chain_name: str) -> List[DetectionResult]:
        """Process a batch of contracts through factory detector"""
        results = []
        
        for _, row in contracts_df.iterrows():
            try:
                start_time = time.time()
                
                # Run factory detection
                detection_result = self.detector.detect_factory_contract(row['bytecode'])
                
                execution_time = (time.time() - start_time) * 1000  # Convert to ms
                
                # Calculate bytecode hash for deduplication analysis
                bytecode_hash = hashlib.sha256(row['bytecode'].encode()).hexdigest()
                
                # Determine factory type from FactoryResult
                is_factory = detection_result.is_factory_contract
                
                # Parse factory type to determine CREATE/CREATE2 usage
                has_create = len(detection_result.verified_create_positions) > 0
                has_create2 = len(detection_result.verified_create2_positions) > 0
                
                is_create_only = is_factory and has_create and not has_create2
                is_create2_only = is_factory and has_create2 and not has_create
                is_both = is_factory and has_create and has_create2
                
                # Create result object
                result = DetectionResult(
                    chain=chain_name,
                    address=row['address'],
                    is_factory=is_factory,
                    is_create2_only=is_create2_only,
                    is_create_only=is_create_only,
                    is_both=is_both,
                    execution_time_ms=execution_time,
                    bytecode_hash=bytecode_hash,
                    deployment_date=str(row['deployment_timestamp'].date()),
                    block_number=int(row['block_number']),
                    processed_at=datetime.now(timezone.utc).isoformat()
                )
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error processing contract {row['address']}: {e}")
                continue
        
        return results
    
    def save_results_batch(self, results: List[DetectionResult]):
        """Save detection results to BigQuery"""
        if not results:
            return
        
        # Convert results to DataFrame
        data = [asdict(result) for result in results]
        df = pd.DataFrame(data)
        
        # Ensure proper data types
        df['deployment_date'] = pd.to_datetime(df['deployment_date']).dt.date
        df['processed_at'] = pd.to_datetime(df['processed_at'])
        df['block_number'] = df['block_number'].astype('int64')
        df['execution_time_ms'] = df['execution_time_ms'].astype('float64')
        
        # Configure job
        table_ref = self.client.dataset(self.result_dataset).table(self.result_table)
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION]
        )
        
        # Load data
        job = self.client.load_table_from_dataframe(df, table_ref, job_config=job_config)
        job.result()  # Wait for completion
        
        self.logger.info(f"Saved {len(results)} detection results to BigQuery")
    
    def update_progress(self, chain_name: str, processed_contracts: int, factories_found: int, 
                       last_block: int, total_contracts: int, status: str):
        """Update progress tracking in BigQuery"""
        
        # Delete existing progress for this chain
        delete_query = f"""
        DELETE FROM `{self.project_id}.{self.result_dataset}.{self.progress_table}`
        WHERE chain = @chain_name
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("chain_name", "STRING", chain_name)
            ]
        )
        
        self.client.query(delete_query, job_config=job_config).result()
        
        # Insert updated progress
        progress_data = [{
            'chain': chain_name,
            'last_processed_block': last_block,
            'total_contracts': total_contracts,
            'processed_contracts': processed_contracts,
            'factories_found': factories_found,
            'last_updated': datetime.now(timezone.utc),
            'status': status
        }]
        
        df = pd.DataFrame(progress_data)
        table_ref = self.client.dataset(self.result_dataset).table(self.progress_table)
        
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND
        )
        
        job = self.client.load_table_from_dataframe(df, table_ref, job_config=job_config)
        job.result()
    
    def process_chain(self, chain_name: str):
        """Process all contracts for a specific chain"""
        self.logger.info(f"Starting processing for {chain_name}")
        
        # Get current progress
        progress = self.get_chain_progress(chain_name)
        total_contracts = self.get_total_contracts_count(chain_name)
        
        self.logger.info(f"{chain_name}: Total contracts to process: {total_contracts:,}")
        self.logger.info(f"{chain_name}: Already processed: {progress['processed_contracts']:,}")
        
        # Initialize progress tracker
        remaining = total_contracts - progress['processed_contracts']
        tracker = ProgressTracker(remaining)
        
        # Start from where we left off
        offset = progress['processed_contracts']
        processed_count = progress['processed_contracts']
        factories_count = progress['factories_found']
        
        try:
            while offset < total_contracts:
                # Fetch batch
                self.logger.info(f"{chain_name}: Fetching batch at offset {offset:,}")
                batch_df = self.fetch_contract_batch(chain_name, offset, self.batch_size)
                
                if batch_df.empty:
                    self.logger.warning(f"{chain_name}: No more contracts to process")
                    break
                
                # Process batch
                self.logger.info(f"{chain_name}: Processing {len(batch_df)} contracts")
                results = self.detect_factory_batch(batch_df, chain_name)
                
                # Save results
                if results:
                    self.save_results_batch(results)
                    
                    # Update counters
                    batch_factories = sum(1 for r in results if r.is_factory)
                    factories_count += batch_factories
                    processed_count += len(results)
                    
                    # Update progress tracker
                    for _ in range(len(results)):
                        tracker.update(False)  # We'll count factories separately
                    for _ in range(batch_factories):
                        tracker.update(True)
                
                # Update progress in database
                last_block = int(batch_df['block_number'].max()) if not batch_df.empty else 0
                self.update_progress(
                    chain_name, processed_count, factories_count, 
                    last_block, total_contracts, "processing"
                )
                
                # Log progress
                stats = tracker.get_stats()
                self.logger.info(
                    f"{chain_name}: Progress {stats['progress_pct']:.2f}% "
                    f"({stats['processed']:,}/{stats['total']:,}) - "
                    f"Factories: {stats['factories_found']:,} - "
                    f"Rate: {stats['rate_per_hour']:.0f}/hr - "
                    f"ETA: {stats['eta_hours']:.1f}h"
                )
                
                offset += self.batch_size
                
                # Small delay to avoid overwhelming BigQuery
                time.sleep(1)
        
        except Exception as e:
            self.logger.error(f"Error processing {chain_name}: {e}")
            self.update_progress(
                chain_name, processed_count, factories_count,
                0, total_contracts, "error"
            )
            raise
        
        # Mark as completed
        self.update_progress(
            chain_name, processed_count, factories_count,
            0, total_contracts, "completed"
        )
        
        self.logger.info(f"{chain_name}: Processing completed - {factories_count:,} factories found")
    
    def run_experiment(self, chains: Optional[List[str]] = None):
        """Run the complete RQ2 experiment"""
        if chains is None:
            chains = list(self.chains.keys())
        
        self.logger.info("=== Starting RQ2 Factory Detection Experiment ===")
        self.logger.info(f"Processing chains: {chains}")
        self.logger.info(f"Cutoff date: {self.cutoff_date}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Max workers: {self.max_workers}")
        
        # Setup tables
        self.setup_result_tables()
        
        # Process each chain
        for chain_name in chains:
            try:
                self.process_chain(chain_name)
            except Exception as e:
                self.logger.error(f"Failed to process {chain_name}: {e}")
                continue
        
        self.logger.info("=== RQ2 Experiment Completed ===")
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of experiment results"""
        query = f"""
        SELECT 
            chain,
            COUNT(*) as total_contracts,
            SUM(CASE WHEN is_factory THEN 1 ELSE 0 END) as total_factories,
            SUM(CASE WHEN is_create_only THEN 1 ELSE 0 END) as create_only,
            SUM(CASE WHEN is_create2_only THEN 1 ELSE 0 END) as create2_only,
            SUM(CASE WHEN is_both THEN 1 ELSE 0 END) as both_types,
            AVG(execution_time_ms) as avg_execution_time
        FROM `{self.project_id}.{self.result_dataset}.{self.result_table}`
        GROUP BY chain
        ORDER BY chain
        """
        
        return self.client.query(query).to_dataframe()

def main():
    """Main execution function"""
    
    # Initialize experiment (config loaded automatically)
    experiment = RQ2FactoryDetectionExperiment()
    
    # Check command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="RQ2 Factory Detection Experiment")
    parser.add_argument("--chains", nargs="+", default=["ethereum", "polygon"],
                       help="Chains to process (default: ethereum polygon)")
    parser.add_argument("--summary", action="store_true",
                       help="Show experiment summary and exit")
    
    args = parser.parse_args()
    
    if args.summary:
        # Show summary
        try:
            summary = experiment.get_experiment_summary()
            print("\n=== RQ2 Experiment Summary ===")
            print(summary.to_string(index=False))
        except Exception as e:
            print(f"Error getting summary: {e}")
        return
    
    # Run experiment
    try:
        experiment.run_experiment(args.chains)
    except KeyboardInterrupt:
        experiment.logger.info("Experiment interrupted by user")
    except Exception as e:
        experiment.logger.error(f"Experiment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()