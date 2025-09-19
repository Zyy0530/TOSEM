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
import multiprocessing
from multiprocessing import Process, Queue
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

def _detect_worker(bytecode: str, result_queue: Queue) -> None:
    """Subprocess worker: run detection and send back flags + exec time.

    Using a subprocess allows enforcing a hard timeout by terminating the process
    if it exceeds the allowed time budget.
    """
    try:
        # Import inside the subprocess to avoid pickling detector state
        from factory_detector import ImprovedFactoryDetector
        detector = ImprovedFactoryDetector()
        res = detector.detect_factory_contract(bytecode)

        has_create = len(res.verified_create_positions) > 0
        has_create2 = len(res.verified_create2_positions) > 0
        flags = {
            'is_factory': res.is_factory_contract,
            'is_create_only': res.is_factory_contract and has_create and not has_create2,
            'is_create2_only': res.is_factory_contract and has_create2 and not has_create,
            'is_both': res.is_factory_contract and has_create and has_create2,
        }
        result_queue.put(("ok", flags, float(res.analysis_time_ms)))
    except Exception as e:  # Return error to parent
        try:
            result_queue.put(("error", str(e)))
        except Exception:
            pass

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
        self.dedup_table_prefix = "rq2_unique_bytecodes_"
        self.cutoff_date = config['cutoff_date']
        self.batch_size = config['batch_size']
        self.max_workers = config['max_workers']
        # Per-contract analysis timeout (seconds); default 10s if not in config
        self.per_contract_timeout_sec = int(config.get('per_contract_timeout_sec', 10))
        
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

    def get_dedup_table_name(self, chain_name: str) -> str:
        return f"{self.dedup_table_prefix}{chain_name}"

    def ensure_dedup_table(self, chain_name: str):
        """Ensure a per-chain deduplicated unique-bytecode table exists.

        Strategy: build by shards using FARM_FINGERPRINT(bytecode) % N to limit per-job memory.
        """
        table_name = self.get_dedup_table_name(chain_name)
        table_ref = self.client.dataset(self.result_dataset).table(table_name)
        chain_config = self.chains[chain_name]

        try:
            table = self.client.get_table(table_ref)
            self.logger.info(f"Dedup table {table_name} already exists; using it")
            return
        except NotFound:
            self.logger.info(f"Creating dedup table {table_name} (this may take a while)...")

        # Create empty table with schema
        schema = [
            bigquery.SchemaField("bytecode", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("address", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("deployment_timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("block_number", "INTEGER", mode="REQUIRED"),
        ]
        table = bigquery.Table(table_ref, schema=schema)
        self.client.create_table(table)

        # Build content by shards to reduce memory footprint
        shards = 64
        for shard in range(shards):
            self.logger.info(f"{chain_name}: Building dedup shard {shard+1}/{shards}")
            query = f"""
            INSERT INTO `{self.project_id}.{self.result_dataset}.{table_name}` (bytecode, address, deployment_timestamp, block_number)
            WITH filtered AS (
              SELECT 
                {chain_config.bytecode_field} AS bytecode,
                {chain_config.address_field} AS address,
                {chain_config.timestamp_field} AS deployment_timestamp,
                {chain_config.block_number_field} AS block_number
              FROM `{chain_config.dataset_name}.{chain_config.contracts_table}`
              WHERE {chain_config.timestamp_field} < TIMESTAMP(@cutoff_date)
                AND {chain_config.bytecode_field} IS NOT NULL
                AND LENGTH({chain_config.bytecode_field}) > 2
                AND MOD(ABS(FARM_FINGERPRINT({chain_config.bytecode_field})), {shards}) = @shard
            ), ranked AS (
              SELECT *,
                ROW_NUMBER() OVER (
                  PARTITION BY bytecode
                  ORDER BY block_number, deployment_timestamp, address
                ) AS rn
              FROM filtered
            )
            SELECT bytecode, address, deployment_timestamp, block_number
            FROM ranked
            WHERE rn = 1
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("cutoff_date", "DATE", self.cutoff_date),
                    bigquery.ScalarQueryParameter("shard", "INT64", shard),
                ]
            )

            job = self.client.query(query, job_config=job_config)
            job.result()
        
        self.logger.info(f"Dedup table {table_name} built successfully")
    
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
        """Get total number of UNIQUE bytecodes to process for a chain (from dedup table)"""
        table_name = self.get_dedup_table_name(chain_name)
        query = f"""
        SELECT COUNT(*) AS total_contracts
        FROM `{self.project_id}.{self.result_dataset}.{table_name}`
        """
        result = self.client.query(query).to_dataframe()
        return int(result.iloc[0]['total_contracts'])
    
    def fetch_contract_batch(self, chain_name: str, offset: int, limit: int) -> pd.DataFrame:
        """Fetch a batch of UNIQUE bytecodes (representative rows) from the per-chain dedup table"""
        table_name = self.get_dedup_table_name(chain_name)
        query = f"""
        SELECT address, bytecode, deployment_timestamp, block_number
        FROM `{self.project_id}.{self.result_dataset}.{table_name}`
        ORDER BY block_number
        LIMIT @limit OFFSET @offset
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("limit", "INTEGER", limit),
                bigquery.ScalarQueryParameter("offset", "INTEGER", offset)
            ]
        )
        return self.client.query(query, job_config=job_config).to_dataframe()

    def fetch_all_contracts_for_bytecodes(self, chain_name: str, bytecodes: List[str]) -> pd.DataFrame:
        """Fetch all contracts whose bytecode is in the provided list"""
        if not bytecodes:
            import pandas as pd  # local import to avoid top-level dependency issues
            return pd.DataFrame(columns=["address", "bytecode", "deployment_timestamp", "block_number"])  # type: ignore

        chain_config = self.chains[chain_name]
        query = f"""
        SELECT 
          {chain_config.address_field} AS address,
          {chain_config.bytecode_field} AS bytecode,
          {chain_config.timestamp_field} AS deployment_timestamp,
          {chain_config.block_number_field} AS block_number
        FROM `{chain_config.dataset_name}.{chain_config.contracts_table}`
        WHERE {chain_config.timestamp_field} < TIMESTAMP(@cutoff_date)
          AND {chain_config.bytecode_field} IN UNNEST(@bytecodes)
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("cutoff_date", "DATE", self.cutoff_date),
                bigquery.ArrayQueryParameter("bytecodes", "STRING", bytecodes)
            ]
        )

        return self.client.query(query, job_config=job_config).to_dataframe()

    def fetch_existing_results_for_hashes(self, chain_name: str, bytecode_hashes: List[str]) -> pd.DataFrame:
        """Fetch any existing results for the given bytecode_hash set (per chain)"""
        if not bytecode_hashes:
            import pandas as pd
            return pd.DataFrame(columns=[
                "bytecode_hash", "is_factory", "is_create_only", "is_create2_only", "is_both"
            ])

        query = f"""
        SELECT bytecode_hash, ANY_VALUE(is_factory) AS is_factory,
               ANY_VALUE(is_create_only) AS is_create_only,
               ANY_VALUE(is_create2_only) AS is_create2_only,
               ANY_VALUE(is_both) AS is_both
        FROM `{self.project_id}.{self.result_dataset}.{self.result_table}`
        WHERE chain = @chain
          AND bytecode_hash IN UNNEST(@hashes)
        GROUP BY bytecode_hash
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("chain", "STRING", chain_name),
                bigquery.ArrayQueryParameter("hashes", "STRING", bytecode_hashes)
            ]
        )
        return self.client.query(query, job_config=job_config).to_dataframe()

    def fetch_existing_addresses(self, chain_name: str, addresses: List[str]) -> pd.DataFrame:
        """Fetch addresses that already have results (per chain)"""
        if not addresses:
            import pandas as pd
            return pd.DataFrame(columns=["address"])

        query = f"""
        SELECT address
        FROM `{self.project_id}.{self.result_dataset}.{self.result_table}`
        WHERE chain = @chain
          AND address IN UNNEST(@addresses)
        GROUP BY address
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("chain", "STRING", chain_name),
                bigquery.ArrayQueryParameter("addresses", "STRING", addresses)
            ]
        )
        return self.client.query(query, job_config=job_config).to_dataframe()
    
    def detect_factory_batch(self, contracts_df: pd.DataFrame, chain_name: str) -> Tuple[List[DetectionResult], int]:
        """Process a batch of UNIQUE-bytecode contracts, avoid re-running detection for already-result bytecodes,
        and only append missing addresses to results.
        """
        results: List[DetectionResult] = []

        # Prepare batch maps
        batch_entries = []
        for _, row in contracts_df.iterrows():
            bytecode = row['bytecode']
            bch = hashlib.sha256(str(bytecode).encode()).hexdigest()
            batch_entries.append({
                'bytecode_norm': (bytecode or '').lower(),
                'bytecode_hash': bch,
                'rep_address': row['address'],
            })

        hashes = [e['bytecode_hash'] for e in batch_entries]
        # Query existing results by bytecode_hash
        existing_df = self.fetch_existing_results_for_hashes(chain_name, hashes)
        existing_map = {str(r['bytecode_hash']): {
            'is_factory': bool(r['is_factory']),
            'is_create_only': bool(r['is_create_only']),
            'is_create2_only': bool(r['is_create2_only']),
            'is_both': bool(r['is_both'])
        } for _, r in existing_df.iterrows()} if not existing_df.empty else {}

        # Map bytecode_norm -> (flags, exec_time_ms)
        det_map: Dict[str, Tuple[Dict[str, bool], float]] = {}

        for e in batch_entries:
            key = e['bytecode_norm']
            bch = e['bytecode_hash']
            if bch in existing_map:
                # Reuse existing flags; no detection run
                flags = existing_map[bch]
                det_map[key] = (flags, 0.0)
            else:
                # Run detector with hard timeout via subprocess
                q: Queue = multiprocessing.Queue()
                p: Process = multiprocessing.Process(target=_detect_worker, args=(key, q))
                try:
                    p.start()
                    p.join(self.per_contract_timeout_sec)
                    if p.is_alive():
                        # Timed out: terminate and record as non-factory with timeout exec time
                        p.terminate()
                        p.join()
                        self.logger.warning(
                            f"{chain_name}: Detection timeout after {self.per_contract_timeout_sec}s for bytecode (rep {e['rep_address']})"
                        )
                        det_map[key] = ({
                            'is_factory': False,
                            'is_create_only': False,
                            'is_create2_only': False,
                            'is_both': False,
                        }, float(self.per_contract_timeout_sec * 1000))
                    else:
                        try:
                            msg = q.get(timeout=1.0)
                        except Exception:
                            msg = ("error", "No result from worker")

                        if msg and isinstance(msg, tuple) and msg[0] == "ok":
                            _, flags, exec_ms = msg
                            det_map[key] = (flags, float(exec_ms))
                        else:
                            err_desc = msg[1] if isinstance(msg, tuple) and len(msg) > 1 else "unknown error"
                            self.logger.error(
                                f"{chain_name}: Detection error for bytecode (rep {e['rep_address']}): {err_desc}"
                            )
                            det_map[key] = ({
                                'is_factory': False,
                                'is_create_only': False,
                                'is_create2_only': False,
                                'is_both': False,
                            }, 0.0)
                except Exception as e2:
                    self.logger.error(f"{chain_name}: Exception starting detection (rep {e['rep_address']}): {e2}")
                finally:
                    try:
                        q.close()
                    except Exception:
                        pass

        # Unique factory count for this batch
        unique_factories = sum(1 for (_, (flags, _)) in det_map.items() if flags['is_factory'])

        # Fetch all addresses for these bytecodes
        bytecodes_list = list(det_map.keys())
        expanded_df = self.fetch_all_contracts_for_bytecodes(chain_name, bytecodes_list)

        # Filter out addresses that already have results
        all_addresses = expanded_df['address'].astype(str).tolist() if not expanded_df.empty else []
        existing_addr_df = self.fetch_existing_addresses(chain_name, all_addresses) if all_addresses else None
        existing_addr_set = set(existing_addr_df['address'].astype(str)) if existing_addr_df is not None and not existing_addr_df.empty else set()

        for _, row in expanded_df.iterrows():
            addr = str(row['address'])
            if addr in existing_addr_set:
                continue  # skip writing duplicates

            bc_norm = (row['bytecode'] or '').lower()
            if bc_norm not in det_map:
                continue
            flags, exec_ms = det_map[bc_norm]

            bytecode_hash = hashlib.sha256(str(row['bytecode']).encode()).hexdigest()

            try:
                result = DetectionResult(
                    chain=chain_name,
                    address=addr,
                    is_factory=flags['is_factory'],
                    is_create2_only=flags['is_create2_only'],
                    is_create_only=flags['is_create_only'],
                    is_both=flags['is_both'],
                    execution_time_ms=float(exec_ms),
                    bytecode_hash=bytecode_hash,
                    deployment_date=str(pd.to_datetime(row['deployment_timestamp']).date()),
                    block_number=int(row['block_number']),
                    processed_at=datetime.now(timezone.utc).isoformat()
                )
                results.append(result)
            except Exception as e3:
                self.logger.error(f"Error building result for address {addr}: {e3}")
                continue

        return results, unique_factories
    
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
        
        # Ensure we have a dedup table for this chain
        self.ensure_dedup_table(chain_name)

        # Get current progress
        progress = self.get_chain_progress(chain_name)
        total_contracts = self.get_total_contracts_count(chain_name)
        
        self.logger.info(f"{chain_name}: Total UNIQUE bytecodes to process: {total_contracts:,}")
        self.logger.info(f"{chain_name}: Already processed (unique): {progress['processed_contracts']:,}")
        
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
                self.logger.info(f"{chain_name}: Processing {len(batch_df)} UNIQUE bytecodes")
                results, unique_factories = self.detect_factory_batch(batch_df, chain_name)

                # Save results (all addresses for bytecodes in batch)
                if results:
                    self.save_results_batch(results)

                    # Update counters: track progress by UNIQUE bytecodes
                    factories_count += unique_factories
                    processed_count += len(batch_df)

                    # Update progress tracker by UNIQUE bytecodes
                    for _ in range(len(batch_df)):
                        tracker.update(False)
                    for _ in range(unique_factories):
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
                    f"({stats['processed']:,}/{stats['total']:,} unique) - "
                    f"Factory bytecodes: {stats['factories_found']:,} - "
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
        """Run the complete RQ2 experiment.

        Changes:
        - Process multiple chains concurrently (e.g., ethereum and polygon).
        - Preserve per-chain logic; BigQuery operations remain per-chain.
        """
        if chains is None:
            chains = list(self.chains.keys())

        self.logger.info("=== Starting RQ2 Factory Detection Experiment ===")
        self.logger.info(f"Processing chains: {chains}")
        self.logger.info(f"Cutoff date: {self.cutoff_date}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Max workers: {self.max_workers}")
        self.logger.info(f"Per-contract timeout: {self.per_contract_timeout_sec}s")

        # Setup tables (once)
        self.setup_result_tables()

        # Run chains concurrently, up to max_workers or number of chains
        max_chain_workers = max(1, min(self.max_workers, len(chains)))
        self.logger.info(f"Running up to {max_chain_workers} chains concurrently")

        with ThreadPoolExecutor(max_workers=max_chain_workers) as executor:
            future_map = {executor.submit(self.process_chain, chain): chain for chain in chains}
            for future in as_completed(future_map):
                chain_name = future_map[future]
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Failed to process {chain_name}: {e}")

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
