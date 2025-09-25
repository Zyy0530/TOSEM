#!/usr/bin/env python3
"""
RQ3 Orchestrator (phased): prepare inputs, fetch sources, then cluster.

Subcommands (run in order):
- prepare: Materialise/verify the factory address CSV (BigQuery optional).
- fetch:   Fetch verified sources from Sourcify static repo into local cache.
- cluster: Run per-chain clustering using local cache and produce outputs.

Notes:
- Steps are intentionally separated so you can ensure each phase completes
  before starting the next one (e.g., finish fetch, then run cluster).
- The clustering implementation and plotting live in
  `experiments/RQ3/factory_clustering_analysis.py`.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import argparse
from types import SimpleNamespace as _NS

# Defer heavy imports (e.g., pandas, sklearn) to specific steps

# Make project root importable when running as a script
import sys as _sys
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from experiments.RQ3.sourcify_fetcher import fetch_for_addresses


def ensure_inputs_csv(cfg) -> Path:
    """Ensure we have a local CSV of factory addresses. If BigQuery is enabled,
    try to query and persist it to the configured CSV path. Otherwise, verify the CSV exists."""
    out_path = Path(cfg.factory_addresses_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if getattr(cfg, "read_mode", "local_csv") == "bigquery" and getattr(getattr(cfg, "bigquery", _NS(use=False)), "use", False):
        try:
            # Lazy import to avoid heavy deps unless required
            from experiments.RQ3.factory_clustering_analysis import load_factory_addresses
            df = load_factory_addresses(cfg)  # type: ignore[arg-type]
            df.to_csv(out_path, index=False)
            logging.info("Saved factory address list to %s", out_path)
            # Switch to local_csv for subsequent steps
            setattr(cfg, "read_mode", "local_csv")
            return out_path
        except Exception as e:
            logging.warning("BigQuery read failed (%s). Falling back to local CSV: %s", e, out_path)

    # Otherwise, just check existence; if missing, try reading RQ2 BigQuery config as fallback
    if not out_path.exists():
        rq2_cfg_path = Path(os.path.dirname(__file__)).parent / "RQ2" / "config.json"
        if rq2_cfg_path.exists():
            try:
                import json as _json
                from google.cloud import bigquery as _bq  # type: ignore
                _cfg = _json.loads(rq2_cfg_path.read_text(encoding="utf-8"))
                project_id = _cfg["project_id"]
                dataset = _cfg["result_dataset"]
                table = f"`{project_id}.{dataset}.rq2_factory_creations`"
                client = _bq.Client(project=project_id)
                sql = f"""
                SELECT chain, LOWER(factory_address) AS factory_address,
                       COUNT(*) AS total_creations,
                       COUNT(DISTINCT created_contract_address) AS unique_creations
                FROM {table}
                WHERE chain IN ('ethereum','polygon')
                GROUP BY chain, factory_address
                """
                df = client.query(sql).result().to_dataframe(create_bqstorage_client=False)
                df.to_csv(out_path, index=False)
                logging.info("Saved factory address list via RQ2 config to %s", out_path)
                return out_path
            except Exception as e:
                logging.warning("Fallback BigQuery via RQ2 config failed: %s", e)
        raise FileNotFoundError(f"Factory addresses CSV missing at {out_path}. Provide it or enable BigQuery.")
    return out_path


def step_prepare(cfg) -> Path:
    """Prepare and validate the factory addresses CSV.

    - If BigQuery is enabled, query and persist to CSV, then switch cfg.read_mode to local_csv.
    - Otherwise, verify the CSV exists.
    Returns the path to the CSV.
    """
    csv_path = ensure_inputs_csv(cfg)
    logging.info("Prepared inputs CSV at %s", csv_path)
    return csv_path


def step_fetch(
    cfg,
    chains: List[str],
    max_workers: int = 8,
    skip_existing: bool = True,
    min_unique: float = None,
    offset: int = 0,
    limit: int = None,
    factory_only: bool = False,
    sort_by: str = None,
    top_n: int = None,
    global_top: bool = False,
) -> None:
    """Fetch verified sources from Sourcify static repo into local cache, per chain."""
    csv_path = ensure_inputs_csv(cfg)
    import pandas as pd  # local import to avoid heavy import during prepare/cluster init
    df = pd.read_csv(csv_path)
    if "chain" not in df.columns or "factory_address" not in df.columns:
        raise ValueError("Input CSV must contain 'chain' and 'factory_address' columns")

    sources_dir = Path(cfg.local_sources_dir)
    total = {"fetched": 0, "skipped": 0, "missing": 0}
    # Determine ranking column
    numeric_sort = None
    if sort_by is not None:
        if sort_by not in df.columns:
            logging.warning("sort_by column '%s' not found; will try fallbacks", sort_by)
            sort_by = None
    if sort_by is None:
        if "total_creations" in df.columns:
            sort_by = "total_creations"
        elif "unique_creations" in df.columns:
            sort_by = "unique_creations"
        else:
            sort_by = None

    # Prepare per-chain address selection (possibly from global ranking)
    per_chain_addrs = {}
    if global_top and top_n is not None and sort_by is not None:
        # Apply factory filter first if requested
        df_sel = df[df["chain"].str.lower().isin([c.lower() for c in chains])].copy()
        if min_unique is not None and "unique_creations" in df_sel.columns:
            df_sel = df_sel[pd.to_numeric(df_sel["unique_creations"], errors="coerce") >= float(min_unique)]
        elif factory_only:
            if "unique_creations" in df_sel.columns:
                df_sel = df_sel[pd.to_numeric(df_sel["unique_creations"], errors="coerce") >= 1]
            elif "total_creations" in df_sel.columns:
                df_sel = df_sel[pd.to_numeric(df_sel["total_creations"], errors="coerce") >= 1]
        # Rank globally
        df_sel[sort_by] = pd.to_numeric(df_sel[sort_by], errors="coerce")
        df_sel = df_sel.dropna(subset=[sort_by])
        df_sel = df_sel.sort_values(by=sort_by, ascending=False).head(int(top_n))
        for ch in chains:
            df_ch = df_sel[df_sel["chain"].str.lower() == ch.lower()]
            per_chain_addrs[ch.lower()] = df_ch["factory_address"].dropna().astype(str).str.lower().unique().tolist()

    for chain in chains:
        df_chain = df[df["chain"].str.lower() == chain.lower()]
        # Optional filter by unique_creations / factory-only
        if min_unique is not None and "unique_creations" in df_chain.columns:
            try:
                df_chain = df_chain[pd.to_numeric(df_chain["unique_creations"], errors="coerce") >= float(min_unique)]
            except Exception:
                pass
        elif factory_only:
            if "unique_creations" in df_chain.columns:
                df_chain = df_chain[pd.to_numeric(df_chain["unique_creations"], errors="coerce") >= 1]
            elif "total_creations" in df_chain.columns:
                df_chain = df_chain[pd.to_numeric(df_chain["total_creations"], errors="coerce") >= 1]
        # Apply per-chain ranking if not using global_top
        if not global_top and top_n is not None and sort_by is not None and sort_by in df_chain.columns:
            df_chain[sort_by] = pd.to_numeric(df_chain[sort_by], errors="coerce")
            df_chain = df_chain.dropna(subset=[sort_by]).sort_values(by=sort_by, ascending=False).head(int(top_n))
        # Apply offset/limit for batching
        if offset:
            df_chain = df_chain.iloc[offset:]
        if limit is not None:
            df_chain = df_chain.iloc[:limit]
        if per_chain_addrs:
            addrs = per_chain_addrs.get(chain.lower(), [])
        else:
            addrs = df_chain["factory_address"].dropna().astype(str).str.lower().unique().tolist()
        if not addrs:
            logging.info("No addresses for chain=%s; skipping fetch.", chain)
            continue
        logging.info(
            "Fetching verified sources for chain=%s (addresses=%d, skip_existing=%s, max_workers=%d, min_unique=%s, offset=%d, limit=%s, factory_only=%s, sort_by=%s, top_n=%s, global_top=%s)",
            chain,
            len(addrs),
            skip_existing,
            max_workers,
            str(min_unique) if min_unique is not None else "-",
            offset or 0,
            str(limit) if limit is not None else "-",
            str(factory_only),
            sort_by or "-",
            str(top_n) if top_n is not None else "-",
            str(global_top),
        )
        fetched, skipped, missing = fetch_for_addresses(
            chain, addrs, sources_dir, skip_existing=skip_existing, max_workers=max_workers
        )
        logging.info("Fetch summary for %s: fetched=%d skipped=%d missing=%d", chain, fetched, skipped, missing)
        total["fetched"] += fetched
        total["skipped"] += skipped
        total["missing"] += missing
    logging.info("Overall fetch summary: %s", total)


def step_cluster(cfg, chains: List[str], factory_only: bool = False, combined: bool = False, size_weight: str = "none", k: int = None, palette: List[str] = None) -> None:
    """Run clustering for the specified chains using local CSV + local sources cache.

    Forces cfg.read_mode to 'local_csv' so we do not accidentally hit BigQuery
    when running clustering in isolation.
    """
    # Heavy imports only when clustering
    from experiments.RQ3.factory_clustering_analysis import run_for_chain as _run_for_chain
    from experiments.RQ3.factory_clustering_analysis import run_for_chains_combined as _run_combined
    cfg.read_mode = "local_csv"
    if combined:
        try:
            _run_combined(chains, cfg, factory_only=factory_only, k_override=k, size_weight=size_weight, palette_codes=palette)  # type: ignore[arg-type]
        except Exception as e:
            logging.error("Combined clustering failed for chains=%s: %s", ",".join(chains), e)
    else:
        for chain in chains:
            try:
                _run_for_chain(chain, cfg, factory_only=factory_only, k_override=k, size_weight=size_weight, palette_codes=palette)  # type: ignore[arg-type]
            except Exception as e:
                logging.error("Clustering failed for chain=%s: %s", chain, e)


def run() -> None:
    parser = argparse.ArgumentParser(description="RQ3 Orchestrator (phased)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "config.json"),
        help="Path to RQ3 config.json",
    )
    parser.add_argument("--log", default="INFO", help="Logging level (DEBUG, INFO, WARNING)")

    # prepare
    sub.add_parser("prepare", help="Materialise/verify factory addresses CSV")

    # fetch
    p_fetch = sub.add_parser("fetch", help="Fetch verified sources into local cache")
    p_fetch.add_argument("--chains", nargs="*", help="Override chains (e.g., ethereum polygon)")
    p_fetch.add_argument("--max-workers", type=int, default=8)
    p_fetch.add_argument("--no-skip", action="store_true", help="Do not skip existing cache files")
    p_fetch.add_argument("--min-unique", type=float, help="Fetch only factories with unique_creations >= this value (if column exists)")
    p_fetch.add_argument("--offset", type=int, default=0, help="Row offset per chain before selecting addresses")
    p_fetch.add_argument("--limit", type=int, help="Limit number of addresses per chain")
    p_fetch.add_argument("--factory-only", action="store_true", help="Restrict to addresses with creations (unique/total > 0)")
    p_fetch.add_argument("--sort-by", choices=["total_creations","unique_creations"], help="Sort column for top selection")
    p_fetch.add_argument("--top-n", type=int, help="Only fetch top-N addresses by sort-by")
    p_fetch.add_argument("--global-top", action="store_true", help="Select top-N across all chosen chains globally")

    # cluster
    p_cluster = sub.add_parser("cluster", help="Run clustering using local cache")
    p_cluster.add_argument("--chains", nargs="*", help="Override chains (e.g., ethereum polygon)")
    p_cluster.add_argument("--factory-only", action="store_true", help="Restrict to addresses with creations (unique/total > 0)")
    p_cluster.add_argument("--combined", action="store_true", help="Cluster across chains as a single dataset")
    p_cluster.add_argument("--size-weight", choices=["none","unique","total"], default="none", help="Marker size weighting (default: none)")
    p_cluster.add_argument("--k", type=int, help="Force KMeans with k clusters")
    p_cluster.add_argument("--palette", type=str, help="Comma-separated HEX colors for clusters (e.g., #F8D675,#3D7FBE,#EA3323,#56BCF9)")

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO), format="[%(levelname)s] %(message)s")

    # Lightweight config loading to avoid importing heavy modules unless needed
    with open(args.config, "r", encoding="utf-8") as _fh:
        _raw = json.load(_fh)
    _bq = _raw.get("bigquery", {}) or {}
    _cl = _raw.get("clustering", {}) or {}
    _emb = _raw.get("embedding", {}) or {}
    cfg = _NS(
        read_mode=_raw.get("read_mode", "local_csv"),
        factory_addresses_csv=_raw.get("factory_addresses_csv", "experiments/RQ3/inputs/factory_addresses.csv"),
        local_sources_dir=_raw.get("local_sources_dir", "experiments/RQ3/data/sources"),
        output_dir=_raw.get("output_dir", "experiments/RQ3/outputs"),
        chains=list(_raw.get("chains", ["ethereum", "polygon"])),
        bigquery=_NS(
            use=_bq.get("use", False),
            project_id=_bq.get("project_id", ""),
            result_dataset=_bq.get("result_dataset", ""),
            table_name=_bq.get("table_name", "rq2_factory_creations"),
        ),
        clustering=_NS(
            use_hdbscan=_cl.get("use_hdbscan", True),
            min_cluster_size=int(_cl.get("min_cluster_size", 8) or 8),
            min_samples=int(_cl.get("min_samples", 5) or 5),
            use_kmeans_fallback=_cl.get("use_kmeans_fallback", True),
            k_range=tuple(_cl.get("k_range", [2, 12]) or [2, 12]),
        ),
        embedding=_NS(
            prefer_umap=_emb.get("prefer_umap", True),
            prefer_tsne=_emb.get("prefer_tsne", True),
            random_state=int(_emb.get("random_state", 42) or 42),
        ),
    )

    if args.cmd == "prepare":
        step_prepare(cfg)
        return

    if args.cmd == "fetch":
        chains = [c.lower() for c in (args.chains if hasattr(args, "chains") and args.chains else cfg.chains)]
        step_fetch(
            cfg,
            chains,
            max_workers=getattr(args, "max_workers", 8),
            skip_existing=not getattr(args, "no_skip", False),
            min_unique=getattr(args, "min_unique", None),
            offset=getattr(args, "offset", 0),
            limit=getattr(args, "limit", None),
            factory_only=getattr(args, "factory_only", False),
            sort_by=getattr(args, "sort_by", None),
            top_n=getattr(args, "top_n", None),
            global_top=getattr(args, "global_top", False),
        )
        return

    if args.cmd == "cluster":
        chains = [c.lower() for c in (args.chains if hasattr(args, "chains") and args.chains else cfg.chains)]
        palette = None
        pal_arg = getattr(args, "palette", None)
        if pal_arg:
            palette = [c.strip() for c in str(pal_arg).split(',') if c.strip()]

        step_cluster(
            cfg,
            chains,
            factory_only=getattr(args, "factory_only", False),
            combined=getattr(args, "combined", False),
            size_weight=getattr(args, "size_weight", "none"),
            k=getattr(args, "k", None),
            palette=palette,
        )
        return


if __name__ == "__main__":
    run()
