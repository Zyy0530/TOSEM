#!/usr/bin/env python3
"""
Check ground truth distribution in the dataset
"""

from google.cloud import bigquery

def check_ground_truth_distribution():
    """Check the distribution of factory vs non-factory contracts"""
    
    project_id = "ziyue-wang"
    dataset_id = "tosem_factory_analysis"
    table_id = "GroundTruthDataset"
    table_full_id = f"{project_id}.{dataset_id}.{table_id}"
    
    client = bigquery.Client(project=project_id)
    
    # Get overall statistics
    query = f"""
    SELECT 
        is_factory_ground_truth,
        source_type,
        COUNT(*) as count
    FROM `{table_full_id}`
    GROUP BY is_factory_ground_truth, source_type
    ORDER BY is_factory_ground_truth, source_type
    """
    
    print("Ground Truth Dataset Distribution")
    print("="*50)
    
    query_job = client.query(query)
    results = query_job.result()
    
    total_contracts = 0
    factory_contracts = 0
    non_factory_contracts = 0
    
    for row in results:
        total_contracts += row.count
        print(f"  {row.source_type}: is_factory={row.is_factory_ground_truth} -> {row.count} contracts")
        
        if row.is_factory_ground_truth:
            factory_contracts += row.count
        else:
            non_factory_contracts += row.count
    
    print(f"\nSummary:")
    print(f"  Total contracts: {total_contracts}")
    print(f"  Factory contracts: {factory_contracts} ({factory_contracts/total_contracts*100:.1f}%)")
    print(f"  Non-factory contracts: {non_factory_contracts} ({non_factory_contracts/total_contracts*100:.1f}%)")
    
    # Get some examples of factory contracts
    if factory_contracts > 0:
        factory_query = f"""
        SELECT address, source_type, verification_notes
        FROM `{table_full_id}`
        WHERE is_factory_ground_truth = true
        LIMIT 5
        """
        
        print(f"\nSample Factory Contracts:")
        query_job = client.query(factory_query)
        results = query_job.result()
        
        for row in results:
            notes = row.verification_notes or "No notes"
            print(f"  {row.address} ({row.source_type}) - {notes}")
    
    # Get some examples of non-factory contracts  
    non_factory_query = f"""
    SELECT address, source_type, verification_notes
    FROM `{table_full_id}`
    WHERE is_factory_ground_truth = false
    LIMIT 5
    """
    
    print(f"\nSample Non-Factory Contracts:")
    query_job = client.query(non_factory_query)
    results = query_job.result()
    
    for row in results:
        notes = row.verification_notes or "No notes"
        print(f"  {row.address} ({row.source_type}) - {notes}")
    
    print("="*50)

if __name__ == "__main__":
    check_ground_truth_distribution()