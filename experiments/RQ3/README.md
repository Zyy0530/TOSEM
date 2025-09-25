# RQ3 Factory Contract Clustering (Offline-First)

This experiment clusters factory contracts (Ethereum + Polygon) using multi-view features from verified source code (Sourcify), with optional product/runtime signals. It produces per-chain 2D semantic maps and JSON summaries.

- Primary script: `experiments/RQ3/factory_clustering_analysis.py`
- Config: `experiments/RQ3/config.json`
- Outputs: `experiments/RQ3/outputs/`

## Data Inputs

1) Factory addresses (local CSV by default):
- Path: `experiments/RQ3/inputs/factory_addresses.csv`
- Columns (header required):
  - `chain` (ethereum|polygon)
  - `factory_address` (hex, 0x-lowercase preferred)
  - `total_creations` (optional)
  - `unique_creations` (optional)
  - `first_seen_date` (optional)
  - `last_seen_date` (optional)

2) Verified sources (local Sourcify cache):
- Directory: `experiments/RQ3/data/sources/{chain}/{address}.json`
- JSON schema:
```
{
  "chain": "ethereum",
  "address": "0x...",
  "source_status": "full_match",  // or partial_match
  "compiler": "v0.8.20+commit.1",
  "evm_version": "london",
  "verified_at": "2025-01-01T00:00:00Z",
  "origin": "sourcify",
  "files": [
    {"path": "contracts/Factory.sol", "content": "pragma solidity ^0.8.20; ..."},
    {"path": "lib/Clones.sol", "content": "..."}
  ]
}
```

Network access is disabled by default; the script does not fetch from Sourcify. Pre-populate the `data/sources` directory with verified packages.

## Running

- Default (offline, CSV + local sources):
```
python experiments/RQ3/factory_clustering_analysis.py --log INFO
```
- Override chains:
```
python experiments/RQ3/factory_clustering_analysis.py --chains ethereum polygon
```

Outputs per chain:
- Figure: `experiments/RQ3/outputs/clustered_semantic_space_{chain}.pdf`
- Summary: `experiments/RQ3/outputs/clusters_{chain}.json`

## BigQuery (optional)
Set `bigquery.use=true` and fill `project_id`/`result_dataset` in `experiments/RQ3/config.json`,
then set `read_mode` to `"bigquery"`. Requires network and GCP credentials.

## Features & Clustering

- Features include: counts (functions/events/mappings/arrays), access control (Ownable/AccessControl/onlyOwner), mechanism flags (create/create2/clone/predict/salt/delegatecall), OpenZeppelin usage, public/external ratios, and a business bag-of-words (dex/token/nft/wallet/dao/beacon/proxy/uups/etc.).
- Orthogonalization: PCA + Varimax rotation.
- Clustering: HDBSCAN (if available); KMeans fallback with model selection (k in config).
- Embedding: UMAP preferred; fallback to t-SNE or 2D PCA.

## Notes

- If you want to weight points by activity, include `unique_creations` (or `total_creations`) in the CSV; it will be used to scale marker sizes.
- Product/runtime-side features can be integrated by extending the extractor to join on `rq2_factory_creations.runtime_code` exports if locally available.

## Orchestrator (phased)

Use `experiments/RQ3/run_rq3.py` to run the pipeline in separate steps to ensure each key phase finishes before the next starts:

- Prepare inputs CSV (materialise from BigQuery if enabled, else validate local CSV):
```
python experiments/RQ3/run_rq3.py --log INFO prepare
```

- Fetch verified sources from Sourcify static repo to local cache:
```
python experiments/RQ3/run_rq3.py --log INFO fetch --chains ethereum polygon --max-workers 8
```
  - Add `--no-skip` to re-fetch even if cache files exist.
  - Only factories (unique_creations > 0):
```
python experiments/RQ3/run_rq3.py --log INFO fetch --chains ethereum polygon --factory-only --max-workers 64
```
  - Top-5000 globally by total_creations across ETH+Polygon:
```
python experiments/RQ3/run_rq3.py --log INFO fetch --chains ethereum polygon \
  --factory-only --sort-by total_creations --top-n 5000 --global-top --max-workers 64
```

- Run clustering using the local cache:
```
python experiments/RQ3/run_rq3.py --log INFO cluster --chains ethereum polygon --size-weight none
```
  - Only factories: add `--factory-only`.
  - Combined ETH+Polygon clustering into one figure/summary:
```
python experiments/RQ3/run_rq3.py --log INFO cluster --chains ethereum polygon --factory-only --combined --size-weight none
```
  - Force KMeans k（例如 3/4）并且不按活跃度加权点大小：
```
python experiments/RQ3/run_rq3.py --log INFO cluster --chains ethereum polygon --combined --k 3 --size-weight none
```

This split avoids running everything at once and lets you validate the completion of each critical step (especially the fetch) before clustering.
