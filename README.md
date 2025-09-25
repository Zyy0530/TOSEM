This is Repo for Our Paper: “Empirical Analysis of Smart Contract Factories on EVM-Compatible Chains”

Overview
- Lightweight, CFG-based detector for EVM factory contracts, plus scripts and results to reproduce paper RQs (RQ1–RQ3).
- Includes detector implementations, experiment pipelines, and produced artifacts (figures/tables/JSON summaries).

Project Structure
- `factory_detector.py` — Baseline CFG-based factory detector (CREATE/CREATE2 reachability aware).
- `enhanced_factory_detector.py` — Improved variant with refined jump/context analysis.
- `analysis_all_chain_factory.py` — BigQuery-based large-scale scanning across five EVM chains.
- `blockchain_statistics.py` — Contract/bytecode statistics collection via BigQuery; uses `blockchain_config.json`.
- `bytecode` — Example bytecode for quick local tests.
- `experiments/`
  - `RQ1/` — Detector evaluation on ground-truth, error analysis, runtime plots, and result JSONs.
    - Scripts: `evaluation.py`, `evaluate_final.py`, `local_evaluation.py`, `analyze_errors.py`, `analyze_original_errors.py`, `extract_error_contracts.py`, `ground_truth_builder.py`
    - Results: `*evaluation_results.json`, `execution_time_cdf*.{pdf,png}`, tables under this folder
  - `RQ2/` — Trace-based factory detection and metric export.
    - Scripts: `rq2_factory_detection.py`, `build_factory_table_from_traces.py`, `export_factory_metrics.py`, `plot_factory_metrics.py`
    - Config/Outputs: `config.json`, `logs/`, `outputs/`
  - `RQ3/` — Clustering/analysis of factory families.
    - Scripts: `run_rq3.py`, `factory_clustering_analysis.py`, helpers under `scripts/`
    - Config/IO: `config.json`, `inputs/`, `outputs/`
- `requirements.txt` — Minimal Python dependencies.

Quick Start
- Python: 3.9+
- Install: `pip install -r requirements.txt`
- BigQuery credentials (for RQ1/RQ2/RQ3 pipelines):
  - `GOOGLE_CLOUD_PROJECT_ID=<your_gcp_project>`
  - `GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json` (or use ADC via `gcloud auth application-default login`).

Run the Detector
- Programmatic example
  - `python -c "from factory_detector import ImprovedFactoryDetector; d=ImprovedFactoryDetector(); print(d.detect_factory_contract(open('bytecode').read().strip()))"`
- CLI demos (use `bytecode` file)
  - Baseline: `python factory_detector.py`
  - Enhanced: `python enhanced_factory_detector.py`

Large-Scale Scanning (BigQuery)
- Edit `analysis_all_chain_factory.py` → set `BIGQUERY_CONFIG['project_id']` and auth method.
- Run: `python analysis_all_chain_factory.py`
- Output: writes to your BigQuery dataset/tables; logs in `bigquery_factory_analysis.log`.

Experiments (RQ Scripts)
- RQ1: detector effectiveness and runtime
  - Build/demo ground-truth: `experiments/RQ1/ground_truth_builder.py`
  - Evaluate on ground-truth table: `experiments/RQ1/evaluation.py`
  - Error analysis: `experiments/RQ1/analyze_errors.py`, `experiments/RQ1/analyze_original_errors.py`
  - Export error lists/tables: `experiments/RQ1/extract_error_contracts.py`
  - Results and figures live under `experiments/RQ1/`
- RQ2: trace-based detection
  - Run: `python experiments/RQ2/rq2_factory_detection.py --config experiments/RQ2/config.json`
  - Metrics/plots written to `experiments/RQ2/outputs/` (logs in `experiments/RQ2/logs/`)
- RQ3: factory family clustering/analysis
  - Run: `python experiments/RQ3/run_rq3.py --config experiments/RQ3/config.json`
  - Outputs under `experiments/RQ3/outputs/`

Notes
- Local `.env` and private credentials are intentionally excluded; configure cloud access via env vars or ADC.
- BigQuery public datasets are in `US` location; adjust configs if needed.
- Results referenced in the paper are under `experiments/` (JSON summaries, plots, and tables).
