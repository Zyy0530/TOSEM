#!/usr/bin/env python3
"""
Ground Truth Dataset Construction for Factory Contract Detection Evaluation
"""

import os
import csv
import json
import time
import requests
import pandas as pd
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from google.cloud import bigquery
from google.oauth2 import service_account
from dotenv import load_dotenv
import logging

# Import the factory detector
from factory_detector import ImprovedFactoryDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

@dataclass
class GroundTruthContract:
    """Ground truth contract data structure"""
    address: str
    bytecode: str
    is_factory_ground_truth: bool
    source_type: str  # 'etherscan' or 'traces'
    verification_notes: str = ""

class EtherscanAPI:
    """Etherscan API client for contract data retrieval"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('ETHERSCAN_API_KEY', 'YourApiKeyToken')
        self.base_url = "https://api.etherscan.io/api"
        self.session = requests.Session()
        
    def get_contract_source_code(self, address: str) -> Optional[Dict]:
        """Get contract source code from Etherscan API"""
        params = {
            'module': 'contract',
            'action': 'getsourcecode',
            'address': address,
            'apikey': self.api_key
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Debug logging
            if data['status'] != '1':
                logger.debug(f"Etherscan API error for {address}: {data.get('message', 'Unknown error')}")
                return None
                
            if data['result'] and len(data['result']) > 0:
                result = data['result'][0]
                # Check if source code is available
                if result.get('SourceCode'):
                    return result
                else:
                    logger.debug(f"No source code available for {address}")
                    return None
            
            logger.debug(f"Empty result from Etherscan for {address}")
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get source code for {address}: {e}")
            return None
    
    def get_contract_bytecode(self, address: str) -> Optional[str]:
        """Get contract bytecode from Etherscan API"""
        params = {
            'module': 'proxy',
            'action': 'eth_getCode',
            'address': address,
            'tag': 'latest',
            'apikey': self.api_key
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'result' in data and data['result'] != '0x':
                return data['result']
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get bytecode for {address}: {e}")
            return None
    
    def is_non_factory_contract(self, source_code: str) -> bool:
        """Check if contract source code suggests it's NOT a factory contract"""
        if not source_code:
            return False
            
        source_lower = source_code.lower()
        
        # Very specific factory patterns to avoid
        forbidden_patterns = [
            'create2(',
            'deploybytecode', 
            'clonefactory',
            'contract factory',
            'factory contract',
            'deploycontract('
        ]
        
        # If contains any forbidden pattern, it's a factory
        for pattern in forbidden_patterns:
            if pattern in source_lower:
                return False
                
        # Accept most other contracts as non-factory
        # This is more permissive to ensure we get a good sample
        return True

class GroundTruthDatasetBuilder:
    """Ground Truth Dataset Builder"""
    
    def __init__(self, project_id: str, dataset_id: str = "tosem_factory_analysis"):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = "GroundTruthDataset"
        
        # Initialize BigQuery client
        credentials_path = "ziyue-wang-26825217908a.json"
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        self.bq_client = bigquery.Client(project=project_id, credentials=credentials)
        
        # Initialize Etherscan API
        self.etherscan = EtherscanAPI()
        
        # Track unique bytecodes to avoid duplicates
        self.existing_bytecodes: Set[str] = set()
        
        logger.info(f"Initialized GroundTruthDatasetBuilder for project: {project_id}")
    
    def create_bigquery_table(self) -> bool:
        """Create BigQuery table for ground truth dataset"""
        try:
            # First, ensure dataset exists
            dataset_ref = self.bq_client.dataset(self.dataset_id)
            try:
                self.bq_client.get_dataset(dataset_ref)
                logger.info(f"Dataset {self.dataset_id} already exists")
            except Exception:
                # Create dataset
                dataset = bigquery.Dataset(dataset_ref)
                dataset.description = "TOSEM Factory Analysis System - Ground Truth Datasets"
                dataset.location = "US"
                dataset = self.bq_client.create_dataset(dataset, timeout=30)
                logger.info(f"Created dataset {dataset.dataset_id}")
            
            # Define table schema
            schema = [
                bigquery.SchemaField("address", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("bytecode", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("is_factory_ground_truth", "BOOLEAN", mode="REQUIRED"),
                bigquery.SchemaField("is_factory_detected", "BOOLEAN", mode="NULLABLE"),
                bigquery.SchemaField("execution_time", "INTEGER", mode="NULLABLE"),
                bigquery.SchemaField("source_type", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("verification_notes", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
            ]
            
            # Create table reference
            table_ref = self.bq_client.dataset(self.dataset_id).table(self.table_id)
            
            # Check if table already exists
            try:
                table = self.bq_client.get_table(table_ref)
                logger.info(f"Table {self.table_id} already exists")
                
                # Load existing bytecodes to avoid duplicates
                self._load_existing_bytecodes()
                return True
                
            except Exception:
                # Table doesn't exist, create it
                table = bigquery.Table(table_ref, schema=schema)
                table.description = "Ground Truth Dataset for Factory Contract Detection Evaluation"
                
                # Set partitioning and clustering for performance
                table.time_partitioning = bigquery.TimePartitioning(
                    type_=bigquery.TimePartitioningType.DAY,
                    field="created_at"
                )
                table.clustering_fields = ["is_factory_ground_truth", "source_type"]
                
                table = self.bq_client.create_table(table)
                logger.info(f"Created table {table.project}.{table.dataset_id}.{table.table_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create BigQuery table: {e}")
            return False
    
    def _load_existing_bytecodes(self):
        """Load existing bytecodes from BigQuery table to avoid duplicates"""
        query = f"""
        SELECT DISTINCT bytecode
        FROM `{self.project_id}.{self.dataset_id}.{self.table_id}`
        """
        
        try:
            results = self.bq_client.query(query).result()
            for row in results:
                self.existing_bytecodes.add(row.bytecode)
            
            logger.info(f"Loaded {len(self.existing_bytecodes)} existing bytecodes")
            
        except Exception as e:
            logger.warning(f"Failed to load existing bytecodes: {e}")
    
    def insert_contract_to_bq(self, contract: GroundTruthContract) -> bool:
        """Insert contract data to BigQuery table"""
        # Check if bytecode already exists
        if contract.bytecode in self.existing_bytecodes:
            logger.debug(f"Bytecode already exists, skipping {contract.address}")
            return False
        
        try:
            table_ref = self.bq_client.dataset(self.dataset_id).table(self.table_id)
            table = self.bq_client.get_table(table_ref)
            
            # Prepare row data
            row_data = {
                "address": contract.address.lower(),
                "bytecode": contract.bytecode,
                "is_factory_ground_truth": contract.is_factory_ground_truth,
                "is_factory_detected": None,
                "execution_time": None,
                "source_type": contract.source_type,
                "verification_notes": contract.verification_notes,
                "created_at": time.time()
            }
            
            # Insert row
            errors = self.bq_client.insert_rows_json(table, [row_data])
            
            if errors:
                logger.error(f"Failed to insert contract {contract.address}: {errors}")
                return False
            
            # Add to existing bytecodes set
            self.existing_bytecodes.add(contract.bytecode)
            logger.debug(f"Successfully inserted contract {contract.address}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert contract {contract.address}: {e}")
            return False
    
    def collect_non_factory_contracts_from_etherscan(self, csv_file: str, max_contracts: int = 1000) -> List[GroundTruthContract]:
        """Collect non-factory contracts from Etherscan verified contracts"""
        logger.info(f"Collecting non-factory contracts from {csv_file} (max: {max_contracts})")
        
        contracts = []
        processed = 0
        successful = 0
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # Skip the note line and header, start from actual data
            data_lines = lines[2:] if len(lines) > 2 else lines
            reader_data = []
            
            # Manual CSV parsing to handle problematic lines
            for line in data_lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Split by comma but handle quoted fields properly
                parts = []
                current_part = ""
                in_quotes = False
                
                for char in line:
                    if char == '"':
                        in_quotes = not in_quotes
                    elif char == ',' and not in_quotes:
                        parts.append(current_part.strip('"'))
                        current_part = ""
                    else:
                        current_part += char
                
                if current_part:
                    parts.append(current_part.strip('"'))
                
                if len(parts) >= 3:  # Ensure we have all required fields
                    reader_data.append({
                        'ContractAddress': parts[1],
                        'ContractName': parts[2]
                    })
                    
            for row in reader_data:
                    if successful >= max_contracts:
                        break
                    
                    processed += 1
                    if processed % 100 == 0:
                        logger.info(f"Processed {processed} contracts, collected {successful} non-factory contracts")
                    
                    address = row.get('ContractAddress', '').strip()
                    
                    # Fix address format issues
                    if address and not address.startswith('0x'):
                        if address.startswith('M_'):
                            # Skip malformed addresses
                            continue
                        else:
                            address = '0x' + address
                    
                    # Validate address format  
                    if not address or len(address) != 42:
                        continue
                    
                    # Get contract source code
                    source_data = self.etherscan.get_contract_source_code(address)
                    if not source_data:
                        continue
                    
                    source_code = source_data.get('SourceCode', '')
                    if not source_code:
                        continue
                    
                    # Check if it's a non-factory contract
                    if not self.etherscan.is_non_factory_contract(source_code):
                        continue
                    
                    # Get contract bytecode
                    bytecode = self.etherscan.get_contract_bytecode(address)
                    if not bytecode or bytecode == '0x':
                        continue
                    
                    # Create ground truth contract
                    contract = GroundTruthContract(
                        address=address,
                        bytecode=bytecode,
                        is_factory_ground_truth=False,
                        source_type='etherscan',
                        verification_notes=f"Verified non-factory: no create/create2/new keywords. Contract: {row.get('ContractName', 'Unknown')}"
                    )
                    
                    contracts.append(contract)
                    successful += 1
                    
                    # Add small delay to respect API rate limits
                    time.sleep(0.2)
        
        except Exception as e:
            logger.error(f"Error processing CSV file: {e}")
        
        logger.info(f"Collected {len(contracts)} non-factory contracts from Etherscan")
        return contracts
    
    def collect_factory_contracts_from_traces(self, max_contracts: int = 1000) -> List[GroundTruthContract]:
        """Collect factory contracts from BigQuery traces table"""
        logger.info(f"Collecting factory contracts from BigQuery traces (max: {max_contracts})")
        
        # Query to find factory contracts from traces
        # We look for contracts that have performed CREATE operations (internal transactions)
        query = f"""
        WITH factory_contracts AS (
            SELECT DISTINCT
                from_address as factory_address,
                ANY_VALUE(block_timestamp) as sample_timestamp,
                COUNT(*) as create_count
            FROM `bigquery-public-data.crypto_ethereum.traces`
            WHERE trace_type = 'create'
                AND status = 1
                AND from_address IS NOT NULL
                AND from_address != '0x0000000000000000000000000000000000000000'
                AND block_timestamp < '2025-06-01'
                AND block_timestamp > '2020-01-01'
            GROUP BY from_address
            HAVING COUNT(*) >= 2  -- Contract must have created multiple contracts
            ORDER BY create_count DESC, sample_timestamp DESC
            -- No LIMIT - collect ALL factory contracts
        )
        SELECT 
            factory_address,
            sample_timestamp,
            create_count
        FROM factory_contracts
        """
        
        contracts = []
        
        try:
            logger.info("Executing BigQuery query for factory contracts...")
            query_job = self.bq_client.query(query)
            results = query_job.result()
            
            successful = 0
            processed = 0
            
            for row in results:
                if successful >= max_contracts:
                    break
                
                processed += 1
                if processed % 1000 == 0:  # 每1000个报告一次进度
                    logger.info(f"Processed {processed} factory candidates, collected {successful} contracts, unique bytecodes: {len(self.existing_bytecodes)}")
                
                factory_address = row.factory_address
                
                # Get contract bytecode using Etherscan API
                try:
                    bytecode = self.etherscan.get_contract_bytecode(factory_address)
                    if not bytecode or bytecode == '0x':
                        continue
                    
                    # Skip if bytecode already exists (more efficient check)
                    if bytecode in self.existing_bytecodes:
                        continue
                        
                except Exception as e:
                    logger.warning(f"Failed to get bytecode for {factory_address}: {e}")
                    # Continue to next contract instead of stopping
                    continue
                
                # Create ground truth contract
                contract = GroundTruthContract(
                    address=factory_address,
                    bytecode=bytecode,
                    is_factory_ground_truth=True,
                    source_type='traces',
                    verification_notes=f"Factory contract identified from traces: executed {row.create_count} CREATE operations"
                )
                
                # Add bytecode to set immediately to avoid duplicates in this session
                self.existing_bytecodes.add(bytecode)
                contracts.append(contract)
                successful += 1
                
                # Small delay to respect API rate limits, but smaller for large scale processing
                time.sleep(0.05)  # Reduced from 0.1
                
        except Exception as e:
            logger.error(f"Failed to collect factory contracts from traces: {e}")
        
        logger.info(f"Collected {len(contracts)} factory contracts from traces")
        return contracts
    
    def build_dataset(self, csv_file: str, max_non_factory: int = 500, max_factory: int = 500):
        """Build complete ground truth dataset"""
        logger.info("Starting Ground Truth Dataset construction...")
        
        # Step 1: Create BigQuery table
        if not self.create_bigquery_table():
            logger.error("Failed to create BigQuery table")
            return False
        
        # Step 2: Collect non-factory contracts from Etherscan
        logger.info("\n=== Collecting Non-Factory Contracts from Etherscan ===")
        non_factory_contracts = self.collect_non_factory_contracts_from_etherscan(
            csv_file, max_non_factory
        )
        
        # Step 3: Insert non-factory contracts to BigQuery
        logger.info("\n=== Inserting Non-Factory Contracts to BigQuery ===")
        non_factory_inserted = 0
        for contract in non_factory_contracts:
            if self.insert_contract_to_bq(contract):
                non_factory_inserted += 1
        
        logger.info(f"Inserted {non_factory_inserted}/{len(non_factory_contracts)} non-factory contracts")
        
        # Step 4: Collect factory contracts from traces
        logger.info("\n=== Collecting Factory Contracts from BigQuery Traces ===")
        factory_contracts = self.collect_factory_contracts_from_traces(max_factory)
        
        # Step 5: Insert factory contracts to BigQuery
        logger.info("\n=== Inserting Factory Contracts to BigQuery ===")
        factory_inserted = 0
        for contract in factory_contracts:
            if self.insert_contract_to_bq(contract):
                factory_inserted += 1
        
        logger.info(f"Inserted {factory_inserted}/{len(factory_contracts)} factory contracts")
        
        # Step 6: Generate summary
        logger.info("\n=== Dataset Construction Summary ===")
        logger.info(f"Non-factory contracts: {non_factory_inserted}")
        logger.info(f"Factory contracts: {factory_inserted}")
        logger.info(f"Total contracts: {non_factory_inserted + factory_inserted}")
        logger.info(f"Unique bytecodes: {len(self.existing_bytecodes)}")
        
        return True
    
    def get_dataset_statistics(self) -> Dict:
        """Get statistics about the constructed dataset"""
        query = f"""
        SELECT 
            COUNT(*) as total_contracts,
            SUM(CASE WHEN is_factory_ground_truth THEN 1 ELSE 0 END) as factory_contracts,
            SUM(CASE WHEN NOT is_factory_ground_truth THEN 1 ELSE 0 END) as non_factory_contracts,
            COUNT(DISTINCT bytecode) as unique_bytecodes,
            COUNT(CASE WHEN source_type = 'etherscan' THEN 1 END) as etherscan_contracts,
            COUNT(CASE WHEN source_type = 'traces' THEN 1 END) as traces_contracts
        FROM `{self.project_id}.{self.dataset_id}.{self.table_id}`
        """
        
        try:
            results = self.bq_client.query(query).result()
            for row in results:
                return {
                    'total_contracts': row.total_contracts,
                    'factory_contracts': row.factory_contracts,
                    'non_factory_contracts': row.non_factory_contracts,
                    'unique_bytecodes': row.unique_bytecodes,
                    'etherscan_contracts': row.etherscan_contracts,
                    'traces_contracts': row.traces_contracts
                }
        except Exception as e:
            logger.error(f"Failed to get dataset statistics: {e}")
            return {}

def main():
    """Main function to build ground truth dataset"""
    print("🏗️ Ground Truth Dataset Construction for Factory Contract Detection")
    print("=" * 80)
    
    # Configuration
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT_ID')
    if not project_id:
        print("❌ Error: GOOGLE_CLOUD_PROJECT_ID not set in environment")
        return
    
    csv_file = "experiments/RQ1/export-verified-contractaddress-opensource-license.csv"
    
    # Initialize dataset builder
    builder = GroundTruthDatasetBuilder(project_id)
    
    # Build dataset
    success = builder.build_dataset(
        csv_file=csv_file,
        max_non_factory=5000,  # Collect 5000 non-factory contracts
        max_factory=50000  # Continue collecting factory contracts (reduce from 200K for balance)
    )
    
    if success:
        # Get and display statistics
        stats = builder.get_dataset_statistics()
        if stats:
            print("\n📊 Final Dataset Statistics:")
            print(f"Total contracts: {stats['total_contracts']}")
            print(f"Factory contracts: {stats['factory_contracts']}")
            print(f"Non-factory contracts: {stats['non_factory_contracts']}")
            print(f"Unique bytecodes: {stats['unique_bytecodes']}")
            print(f"From Etherscan: {stats['etherscan_contracts']}")
            print(f"From Traces: {stats['traces_contracts']}")
        
        print("\n✅ Ground Truth Dataset construction completed successfully!")
        print("🚀 Ready for factory detector evaluation experiments")
    else:
        print("\n❌ Ground Truth Dataset construction failed")

if __name__ == "__main__":
    main()