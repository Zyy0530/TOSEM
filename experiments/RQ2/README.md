# RQ2 Factory Contract Detection Experiment

This directory contains the comprehensive implementation for RQ2 - large-scale factory contract detection across Ethereum and Polygon blockchains.

## Overview

The RQ2 experiment performs factory contract detection on all contracts deployed before June 1, 2025 across Ethereum and Polygon networks. It generates data for:

1. **Daily deployment analysis** - CREATE/CREATE2/both factory types with stacked bar charts
2. **Bytecode deduplication analysis** - Distribution and CDF of bytecode repetition 
3. **Transaction volume analysis** - Factory contract transaction count distributions
4. **Temporal trend analysis** - Evolution of factory contract deployment patterns

## Files Structure

```
RQ2/
├── rq2_factory_detection.py   # Main experiment script
├── test_setup.py              # Validation and testing script  
├── config.json                # Configuration settings
├── requirements.md            # Experiment requirements
├── logs/                      # Execution logs (auto-created)
└── README.md                  # This file
```

## Prerequisites

### 1. Dependencies
```bash
pip install google-cloud-bigquery pandas
```

### 2. Google Cloud Setup
- Set up Google Cloud authentication
- Ensure access to BigQuery public datasets
- Configure project ID in `config.json`

### 3. Factory Detector
- Ensure `factory_detector.py` is available in the parent directory
- The script automatically imports from `../../factory_detector.py`

## Usage

### Step 1: Validate Setup
Before running the main experiment, validate your setup:

```bash
cd /Users/mac/ResearchSpace/TOSEM/experiments/RQ2
python test_setup.py
```

This runs comprehensive tests:
- ✅ BigQuery authentication
- ✅ Public dataset access (Ethereum/Polygon)
- ✅ Factory detector functionality  
- ✅ Result dataset creation
- ✅ Small batch processing test

### Step 2: Run Main Experiment

**Full experiment (both chains):**
```bash
python rq2_factory_detection.py
```

**Single chain:**
```bash
python rq2_factory_detection.py --chains ethereum
python rq2_factory_detection.py --chains polygon
```

**Resume capability:**
The script automatically resumes from where it left off. If interrupted, simply restart with the same command.

**Check progress:**
```bash
python rq2_factory_detection.py --summary
```

## Configuration

Edit `config.json` to customize:

```json
{
  "project_id": "your-project-id",
  "result_dataset": "tosem_factory_analysis", 
  "cutoff_date": "2025-06-01",
  "batch_size": 1000,
  "max_workers": 4
}
```

## Output

### BigQuery Tables Created

**Results Table: `rq2_factory_detection_results`**
- `chain` - Blockchain name (ethereum/polygon)
- `address` - Contract address
- `is_factory` - Boolean factory detection result
- `is_create2_only` - Uses only CREATE2 operations
- `is_create_only` - Uses only CREATE operations  
- `is_both` - Uses both CREATE and CREATE2
- `execution_time_ms` - Detection execution time
- `bytecode_hash` - SHA256 hash for deduplication analysis
- `deployment_date` - Contract deployment date
- `block_number` - Deployment block number
- `processed_at` - Processing timestamp

**Progress Table: `rq2_experiment_progress`**
- Tracks processing progress per chain
- Enables resume capability
- Monitors execution statistics

### Log Files
Detailed execution logs are saved to:
```
logs/rq2_experiment_YYYYMMDD_HHMMSS.log
```

## Expected Scale

### Ethereum
- **Estimated contracts**: ~50M contracts (as of June 2025)
- **Expected factories**: ~100K-500K factories
- **Processing time**: 20-40 hours (depending on BigQuery quotas)

### Polygon  
- **Estimated contracts**: ~20M contracts (as of June 2025)
- **Expected factories**: ~50K-200K factories
- **Processing time**: 10-20 hours

## Performance Features

### 🚀 **Optimizations**
- **Batch processing**: 1000 contracts per batch
- **Resume capability**: Automatic progress tracking
- **Parallel processing**: Multi-threaded detection
- **BigQuery optimizations**: Efficient querying and loading
- **Memory management**: Streaming batch processing

### 📊 **Progress Tracking**
- Real-time progress percentage
- Factories found counter
- Processing rate (contracts/hour)
- ETA estimation
- Comprehensive logging

### 🔄 **Error Handling**
- Automatic retry on transient failures
- Individual contract error isolation
- Progress preservation on interruption
- Comprehensive error logging

## Analysis Queries

After the experiment completes, use these queries for analysis:

### Daily Deployment Counts
```sql
SELECT 
  deployment_date,
  chain,
  SUM(CASE WHEN is_create_only THEN 1 ELSE 0 END) as create_only,
  SUM(CASE WHEN is_create2_only THEN 1 ELSE 0 END) as create2_only, 
  SUM(CASE WHEN is_both THEN 1 ELSE 0 END) as both_types
FROM `project.tosem_factory_analysis.rq2_factory_detection_results`
WHERE is_factory = true
GROUP BY deployment_date, chain
ORDER BY deployment_date, chain
```

### Bytecode Deduplication Analysis
```sql
SELECT 
  bytecode_hash,
  COUNT(*) as duplicate_count,
  chain
FROM `project.tosem_factory_analysis.rq2_factory_detection_results`
WHERE is_factory = true
GROUP BY bytecode_hash, chain
HAVING COUNT(*) > 1
ORDER BY duplicate_count DESC
```

## Troubleshooting

### Common Issues

**Authentication Errors:**
```bash
gcloud auth application-default login
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
```

**BigQuery Quota Limits:**
- Reduce `batch_size` in config.json
- Add delays between batches
- Use multiple processing sessions

**Memory Issues:**
- Reduce `batch_size` 
- Ensure sufficient disk space for logs
- Monitor system resources

**Resume Not Working:**
- Check progress table exists
- Verify dataset permissions
- Check for schema mismatches

### Support

For issues:
1. Check log files in `logs/` directory
2. Run `test_setup.py` to validate configuration
3. Review BigQuery quotas and permissions
4. Ensure factory_detector.py is accessible

## Expected Results

Upon completion, you'll have comprehensive data for:
- 📈 **Daily factory deployment trends** across both chains
- 🔢 **Factory type distribution** (CREATE/CREATE2/both)
- 📊 **Bytecode deduplication patterns** 
- ⏱️ **Processing performance metrics**
- 🌐 **Cross-chain comparison analysis**

This dataset enables the four analysis components specified in the RQ2 requirements:
1. Daily deployment stacked bar charts
2. Bytecode repetition CDF analysis  
3. Transaction volume distributions
4. Temporal trend analysis