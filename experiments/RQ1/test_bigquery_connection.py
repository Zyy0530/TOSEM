#!/usr/bin/env python3
"""
Simple test to verify BigQuery connection and dataset access
"""

from google.cloud import bigquery

def test_bigquery_connection():
    """Test BigQuery connection and dataset access"""
    try:
        # Initialize client
        project_id = "ziyue-wang"
        dataset_id = "tosem_factory_analysis"
        table_id = "GroundTruthDataset"
        
        client = bigquery.Client(project=project_id)
        print(f"Connected to BigQuery project: {project_id}")
        
        # Test dataset access
        dataset = client.get_dataset(f"{project_id}.{dataset_id}")
        print(f"Successfully accessed dataset: {dataset.dataset_id}")
        
        # Test table access and get basic info
        table = client.get_table(f"{project_id}.{dataset_id}.{table_id}")
        print(f"Successfully accessed table: {table.table_id}")
        print(f"Table has {table.num_rows} rows")
        
        # Get first few rows to test query
        query = f"""
        SELECT address, is_factory_ground_truth, is_factory_detected, source_type
        FROM `{project_id}.{dataset_id}.{table_id}`
        LIMIT 5
        """
        
        query_job = client.query(query)
        results = query_job.result()
        
        print("\nSample data:")
        for i, row in enumerate(results):
            print(f"  Row {i+1}: {row.address[:10]}..., Ground Truth: {row.is_factory_ground_truth}, Detected: {row.is_factory_detected}, Source: {row.source_type}")
            
        print("\nBigQuery connection test successful!")
        return True
        
    except Exception as e:
        print(f"BigQuery connection test failed: {e}")
        return False

if __name__ == "__main__":
    test_bigquery_connection()