#!/usr/bin/env python3
"""
Sourcify verified source fetcher (offline cache builder).

Fetches verified contract sources from Sourcify's static repo:
  https://repo.sourcify.dev/contracts/{status}/{chainId}/{address}/

We prefer the static repo over the server API to reduce coupling and leverage
CDN caching. For each address we try `full_match` then `partial_match`.

Outputs per contract a JSON file at:
  experiments/RQ3/data/sources/{chain}/{address}.json
with the schema expected by RQ3 clustering.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests


CHAIN_NAME_TO_ID: Dict[str, int] = {
    "ethereum": 1,
    "polygon": 137,
}

REPO_BASE = "https://repo.sourcify.dev/contracts"


@dataclass
class FetchResult:
    chain: str
    address: str
    status: str  # full_match | partial_match | missing
    files: List[Tuple[str, str]]  # (path, content)
    compiler: Optional[str]
    evm_version: Optional[str]
    verified_at: Optional[str]


def _http_get(url: str, timeout: float = 15.0) -> Optional[requests.Response]:
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            return resp
        return None
    except Exception:
        return None


def try_fetch_metadata(chain_id: int, address: str, status: str) -> Optional[dict]:
    url = f"{REPO_BASE}/{status}/{chain_id}/{address}/metadata.json"
    resp = _http_get(url)
    if resp is None:
        return None
    try:
        return resp.json()
    except Exception:
        return None


def fetch_single(chain: str, address: str, max_retries: int = 2, backoff: float = 0.5) -> FetchResult:
    chain_l = chain.lower()
    chain_id = CHAIN_NAME_TO_ID.get(chain_l)
    address_l = address.lower()
    if chain_id is None:
        return FetchResult(chain_l, address_l, "missing", [], None, None, None)

    meta = None
    status = "missing"
    for st in ("full_match", "partial_match"):
        for attempt in range(max_retries + 1):
            meta = try_fetch_metadata(chain_id, address_l, st)
            if meta is not None:
                status = st
                break
            time.sleep(backoff * (attempt + 1))
        if meta is not None:
            break

    if meta is None:
        return FetchResult(chain_l, address_l, status, [], None, None, None)

    sources = meta.get("sources", {})
    files: List[Tuple[str, str]] = []
    for src_path in sources.keys():
        # Prefer fetching content from repo to avoid depending on metadata inline content
        # URL-encode src_path by replacing spaces and ensuring forward slashes preserved
        # requests will handle simple quoted paths if we pass in quotes manually
        url = f"{REPO_BASE}/{status}/{chain_id}/{address_l}/sources/{src_path}"
        resp = _http_get(url)
        if resp is not None:
            files.append((src_path, resp.text))
        else:
            # Fallback to metadata embedded content if present
            content = sources.get(src_path, {}).get("content")
            if isinstance(content, str) and content:
                files.append((src_path, content))

    compiler = None
    evm_version = None
    verified_at = None
    try:
        compiler = meta.get("compiler", {}).get("version")
        evm_version = meta.get("settings", {}).get("evmVersion")
        verified_at = meta.get("metadata", {}).get("createdAt") or meta.get("sourcesTimestamp")
    except Exception:
        pass

    return FetchResult(chain_l, address_l, status, files, compiler, evm_version, verified_at)


def save_result(res: FetchResult, out_dir: Path) -> Optional[Path]:
    if res.status == "missing" or not res.files:
        return None
    chain_dir = out_dir / res.chain
    chain_dir.mkdir(parents=True, exist_ok=True)
    path = chain_dir / f"{res.address}.json"
    obj = {
        "chain": res.chain,
        "address": res.address,
        "source_status": res.status,
        "compiler": res.compiler,
        "evm_version": res.evm_version,
        "verified_at": res.verified_at,
        "origin": "sourcify",
        "files": [{"path": p, "content": c} for p, c in res.files],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False)
    return path


def fetch_for_addresses(chain: str, addresses: Iterable[str], out_dir: Path, skip_existing: bool = True, max_workers: int = 8) -> Tuple[int, int, int]:
    """Return (fetched, skipped, missing)."""
    chain_l = chain.lower()
    chain_dir = out_dir / chain_l
    chain_dir.mkdir(parents=True, exist_ok=True)

    addrs = [a.lower() for a in addresses]
    tasks = []
    fetched = 0
    skipped = 0
    missing = 0

    if skip_existing:
        remaining = []
        for a in addrs:
            if (chain_dir / f"{a}.json").exists():
                skipped += 1
            else:
                remaining.append(a)
        addrs = remaining

    if not addrs:
        return 0, skipped, 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for addr in addrs:
            tasks.append(ex.submit(fetch_single, chain_l, addr))
        for fut in concurrent.futures.as_completed(tasks):
            res = fut.result()
            if res.status == "missing" or not res.files:
                missing += 1
                continue
            path = save_result(res, out_dir)
            if path is not None:
                fetched += 1

    return fetched, skipped, missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch verified sources from Sourcify static repo")
    parser.add_argument("chain", choices=list(CHAIN_NAME_TO_ID.keys()), help="Chain name (ethereum|polygon)")
    parser.add_argument("addresses_file", help="Path to CSV with column 'factory_address' or a text file with one address per line")
    parser.add_argument("--out", default="experiments/RQ3/data/sources", help="Output directory")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--no-skip", action="store_true", help="Do not skip existing cache files")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    path = Path(args.addresses_file)
    addrs: List[str] = []
    if path.suffix.lower() == ".csv":
        import pandas as pd
        df = pd.read_csv(path)
        col = "factory_address" if "factory_address" in df.columns else df.columns[0]
        addrs = [str(x) for x in df[col].dropna().astype(str).tolist()]
    else:
        addrs = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    out_dir = Path(args.out)
    fetched, skipped, missing = fetch_for_addresses(args.chain, addrs, out_dir, skip_existing=not args.no_skip, max_workers=args.max_workers)
    logging.info("Fetched=%d, Skipped=%d, Missing(or not verified)=%d", fetched, skipped, missing)


if __name__ == "__main__":
    main()

