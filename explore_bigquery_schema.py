#!/usr/bin/env python3
"""
BigQuery表结构探索脚本
检查Arbitrum、Optimism、Avalanche的表结构和字段名
"""

import json
import os
from google.cloud import bigquery
from google.oauth2 import service_account
from dotenv import load_dotenv

load_dotenv()

def explore_table_schema(project_id: str, credentials_path: str):
    """探索BigQuery表结构"""
    
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    client = bigquery.Client(project=project_id, credentials=credentials)
    
    datasets_to_check = [
        "bigquery-public-data.goog_blockchain_arbitrum_one_us",
        "bigquery-public-data.goog_blockchain_optimism_mainnet_us", 
        "bigquery-public-data.goog_blockchain_avalanche_contract_chain_us"
    ]
    
    for dataset_name in datasets_to_check:
        print(f"\n{'='*60}")
        print(f"检查数据集: {dataset_name}")
        print("="*60)
        
        try:
            # 获取数据集中的表列表
            dataset_ref = client.dataset(dataset_name.split('.')[1], project=dataset_name.split('.')[0])
            tables = list(client.list_tables(dataset_ref))
            
            print(f"数据集中的表:")
            for table in tables:
                print(f"  - {table.table_id}")
            
            # 检查transactions和receipts表的结构
            for table_name in ['transactions', 'receipts']:
                if any(t.table_id == table_name for t in tables):
                    print(f"\n{table_name} 表结构:")
                    table_ref = dataset_ref.table(table_name)
                    table = client.get_table(table_ref)
                    
                    for field in table.schema:
                        print(f"  {field.name}: {field.field_type}")
                        
                    # 获取一条样本数据
                    query = f"SELECT * FROM `{dataset_name}.{table_name}` LIMIT 1"
                    try:
                        results = client.query(query).result()
                        print(f"\n{table_name} 样本数据:")
                        for row in results:
                            for key, value in row.items():
                                if isinstance(value, str) and len(str(value)) > 50:
                                    print(f"  {key}: {str(value)[:50]}...")
                                else:
                                    print(f"  {key}: {value}")
                            break
                    except Exception as e:
                        print(f"  无法获取样本数据: {e}")
                else:
                    print(f"\n❌ {table_name} 表不存在")
                    
        except Exception as e:
            print(f"❌ 无法访问数据集 {dataset_name}: {e}")

if __name__ == "__main__":
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT_ID')
    credentials_path = "ziyue-wang-26825217908a.json"
    
    explore_table_schema(project_id, credentials_path)