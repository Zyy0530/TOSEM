#!/usr/bin/env python3
"""
RQ2 Experiment Test Script

This script validates the setup and connections before running the main RQ2 experiment.
It performs the following checks:
1. BigQuery connection and authentication
2. Access to public datasets (Ethereum and Polygon)
3. Factory detector functionality
4. Result dataset creation
5. Small batch processing test

Author: Research Team
Date: 2025
"""

import os
import sys
import json
import logging
from datetime import datetime

# Add parent directory to path for factory_detector import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from google.cloud import bigquery
    from google.cloud.exceptions import NotFound
    import pandas as pd
except ImportError as e:
    print(f"❌ Missing required dependencies: {e}")
    print("Please install: pip install google-cloud-bigquery pandas")
    sys.exit(1)

try:
    from factory_detector import ImprovedFactoryDetector
except ImportError as e:
    print(f"❌ Cannot import factory_detector: {e}")
    print("Please ensure factory_detector.py is in the parent directory")
    sys.exit(1)

class RQ2TestSuite:
    """Test suite for RQ2 experiment validation"""
    
    def __init__(self):
        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.project_id = self.config['project_id']
        self.result_dataset = self.config['result_dataset']
        
        # Initialize BigQuery client
        try:
            self.client = bigquery.Client(project=self.project_id)
            print(f"✅ BigQuery client initialized for project: {self.project_id}")
        except Exception as e:
            print(f"❌ Failed to initialize BigQuery client: {e}")
            sys.exit(1)
        
        # Initialize factory detector
        try:
            self.detector = ImprovedFactoryDetector()
            print("✅ Factory detector initialized")
        except Exception as e:
            print(f"❌ Failed to initialize factory detector: {e}")
            sys.exit(1)
    
    def test_bigquery_authentication(self):
        """Test BigQuery authentication and basic access"""
        print("\n🔐 Testing BigQuery authentication...")
        
        try:
            # Simple query to test authentication
            query = "SELECT 1 as test"
            result = self.client.query(query).to_dataframe()
            if not result.empty and result.iloc[0]['test'] == 1:
                print("✅ BigQuery authentication successful")
                return True
            else:
                print("❌ BigQuery authentication failed - unexpected result")
                return False
        except Exception as e:
            print(f"❌ BigQuery authentication failed: {e}")
            return False
    
    def test_public_dataset_access(self):
        """Test access to public Ethereum and Polygon datasets"""
        print("\n🌐 Testing public dataset access...")
        
        success = True
        
        for chain_name, chain_config in self.config['chains'].items():
            try:
                # Test dataset access
                dataset_name = chain_config['dataset_name']
                table_name = chain_config['contracts_table']
                
                query = f"""
                SELECT COUNT(*) as total_contracts
                FROM `{dataset_name}.{table_name}`
                LIMIT 1
                """
                
                result = self.client.query(query).to_dataframe()
                total_contracts = result.iloc[0]['total_contracts']
                
                print(f"✅ {chain_name}: {total_contracts:,} contracts accessible")
                
            except Exception as e:
                print(f"❌ {chain_name}: Failed to access dataset - {e}")
                success = False
        
        return success
    
    def test_factory_detector(self):
        """Test factory detector with known bytecode samples"""
        print("\n🏭 Testing factory detector...")
        
        # Test with known factory bytecode (simplified example)
        test_cases = [
            {
                'name': 'Simple CREATE factory',
                'bytecode': '0x608060405234801561001057600080fd5b50610150806100206000396000f3fe6080604052600436106100295760003560e01c80634e71e0c81461002e578063f8a8fd6d14610050575b600080fd5b34801561003a57600080fd5b5061004e610049366004610089565b610070565b005b34801561005c57600080fd5b5061004e61006b366004610089565b6100c1565b6000818051906020012090506000f080156100895761008681610114565b50505b50565b60006020828403121561009e57600080fd5b813567ffffffffffffffff8111156100b557600080fd5b82018360208201111561012157600080fd5b8035906020019184600183028401116401000000008311171561014257600080fd5b509092915050565b6000819050919050565b61014f81610114565b82525050565b600060208201905061016a600083018461014a565b9291505056fea26469706673582212208b',
                'expected_factory': True
            },
            {
                'name': 'Non-factory contract',
                'bytecode': '0x608060405234801561001057600080fd5b50610150806100206000396000f3fe608060405260043610610029576000803560e01c806381623e8a1461002e578063a87d942c14610050575b600080fd5b610036610070565b604051610047919061010a565b60405180910390f35b610058610076565b60405161006591906100ef565b60405180910390f35b60005481565b60606040518060200160405280600081525090500',
                'expected_factory': False
            }
        ]
        
        success = True
        
        for test_case in test_cases:
            try:
                result = self.detector.detect_factory_contract(test_case['bytecode'])
                is_factory = result.is_factory_contract
                
                if is_factory == test_case['expected_factory']:
                    print(f"✅ {test_case['name']}: Correctly detected as {'factory' if is_factory else 'non-factory'}")
                else:
                    print(f"❌ {test_case['name']}: Expected {test_case['expected_factory']}, got {is_factory}")
                    success = False
                    
            except Exception as e:
                print(f"❌ {test_case['name']}: Detection failed - {e}")
                success = False
        
        return success
    
    def test_result_dataset_creation(self):
        """Test creation of result dataset and tables"""
        print("\n📊 Testing result dataset creation...")
        
        try:
            # Check if dataset exists
            dataset_ref = self.client.dataset(self.result_dataset)
            try:
                self.client.get_dataset(dataset_ref)
                print(f"✅ Dataset {self.result_dataset} exists")
            except NotFound:
                # Create dataset
                dataset = bigquery.Dataset(dataset_ref)
                dataset.location = "US"
                dataset.description = "TOSEM Factory Analysis Results (Test)"
                self.client.create_dataset(dataset)
                print(f"✅ Created dataset {self.result_dataset}")
            
            # Test table creation (create a test table)
            test_table_name = f"rq2_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            schema = [
                bigquery.SchemaField("test_field", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("test_value", "INTEGER", mode="REQUIRED"),
            ]
            
            table_ref = self.client.dataset(self.result_dataset).table(test_table_name)
            table = bigquery.Table(table_ref, schema=schema)
            self.client.create_table(table)
            
            # Test data insertion
            test_data = [{'test_field': 'test', 'test_value': 123}]
            df = pd.DataFrame(test_data)
            
            job_config = bigquery.LoadJobConfig(
                write_disposition=bigquery.WriteDisposition.WRITE_APPEND
            )
            
            job = self.client.load_table_from_dataframe(df, table_ref, job_config=job_config)
            job.result()
            
            # Verify data
            query = f"SELECT COUNT(*) as count FROM `{self.project_id}.{self.result_dataset}.{test_table_name}`"
            result = self.client.query(query).to_dataframe()
            
            if result.iloc[0]['count'] == 1:
                print("✅ Table creation and data insertion successful")
                
                # Clean up test table
                self.client.delete_table(table_ref)
                print("✅ Test table cleaned up")
                
                return True
            else:
                print("❌ Data insertion failed")
                return False
                
        except Exception as e:
            print(f"❌ Dataset/table creation failed: {e}")
            return False
    
    def test_small_batch_processing(self):
        """Test processing a small batch of real contracts"""
        print("\n🔄 Testing small batch processing...")
        
        try:
            # Get a small batch of contracts from Ethereum
            query = """
            SELECT 
                address,
                bytecode,
                block_timestamp,
                block_number
            FROM `bigquery-public-data.crypto_ethereum.contracts`
            WHERE block_timestamp < TIMESTAMP('2025-06-01')
            AND bytecode IS NOT NULL
            AND LENGTH(bytecode) > 2
            ORDER BY block_number DESC
            LIMIT 5
            """
            
            contracts_df = self.client.query(query).to_dataframe()
            
            if contracts_df.empty:
                print("❌ No contracts retrieved for testing")
                return False
            
            print(f"✅ Retrieved {len(contracts_df)} test contracts")
            
            # Process each contract
            factories_found = 0
            for idx, row in contracts_df.iterrows():
                try:
                    result = self.detector.detect_factory_contract(row['bytecode'])
                    is_factory = result.is_factory_contract
                    
                    if is_factory:
                        factories_found += 1
                        print(f"  📍 Factory found: {row['address']}")
                    else:
                        print(f"  📄 Non-factory: {row['address']}")
                        
                except Exception as e:
                    print(f"  ❌ Error processing {row['address']}: {e}")
            
            print(f"✅ Batch processing completed: {factories_found}/{len(contracts_df)} factories found")
            return True
            
        except Exception as e:
            print(f"❌ Batch processing test failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all test cases"""
        print("🧪 RQ2 Experiment Test Suite")
        print("=" * 50)
        
        tests = [
            ("BigQuery Authentication", self.test_bigquery_authentication),
            ("Public Dataset Access", self.test_public_dataset_access),
            ("Factory Detector", self.test_factory_detector),
            ("Result Dataset Creation", self.test_result_dataset_creation),
            ("Small Batch Processing", self.test_small_batch_processing),
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                success = test_func()
                results.append((test_name, success))
            except Exception as e:
                print(f"❌ {test_name}: Unexpected error - {e}")
                results.append((test_name, False))
        
        # Summary
        print("\n📋 Test Results Summary:")
        print("=" * 50)
        
        all_passed = True
        for test_name, success in results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} {test_name}")
            if not success:
                all_passed = False
        
        print("\n" + "=" * 50)
        if all_passed:
            print("🎉 All tests passed! Ready to run RQ2 experiment.")
            print("\nTo start the experiment, run:")
            print("python rq2_factory_detection.py --chains ethereum polygon")
        else:
            print("⚠️  Some tests failed. Please fix issues before running the experiment.")
        
        return all_passed

def main():
    """Main test execution"""
    try:
        test_suite = RQ2TestSuite()
        test_suite.run_all_tests()
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
    except Exception as e:
        print(f"💥 Test suite failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()