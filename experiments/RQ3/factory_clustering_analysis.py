#!/usr/bin/env python3
"""
RQ3 Factory Contract Clustering (offline-first)

Pipeline outline:
1) Load factory addresses by chain (local CSV by default; BigQuery optional).
2) Load verified source code packages from local Sourcify cache.
3) Extract multi-view features (mechanism, semantics, product signals [optional]).
4) Orthogonalise/standardise features and cluster (HDBSCAN preferred; KMeans fallback).
5) Embed to 2D (UMAP/TSNE/PCA) and save per-chain PDFs and JSON summaries.

Notes:
- Network access is restricted in this environment; this script does not fetch
  from Sourcify by default. Place pre-fetched Sourcify results under
  experiments/RQ3/data/sources/{chain}/{address}.json using the schema defined in
  `SourceRecord` below.
- If BigQuery access is needed, set bigquery.use=true in config and ensure
  Google Cloud credentials and network access are available.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional imports with graceful fallbacks
try:
    from google.cloud import bigquery  # type: ignore
except Exception:  # pragma: no cover - optional in offline mode
    bigquery = None  # type: ignore

try:
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover - optional
    hdbscan = None  # type: ignore

try:
    import umap  # type: ignore
except Exception:  # pragma: no cover - optional
    umap = None  # type: ignore

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE


# ------------------------------ Config types ------------------------------ #


@dataclass
class BQConfig:
    use: bool
    project_id: str
    result_dataset: str
    table_name: str


@dataclass
class ClusterConfig:
    use_hdbscan: bool
    min_cluster_size: int
    min_samples: int
    use_kmeans_fallback: bool
    k_range: Tuple[int, int]


@dataclass
class EmbeddingConfig:
    prefer_umap: bool
    prefer_tsne: bool
    random_state: int


@dataclass
class AppConfig:
    read_mode: str  # "local_csv" | "bigquery"
    factory_addresses_csv: str
    local_sources_dir: str
    output_dir: str
    chains: List[str]
    bigquery: BQConfig
    clustering: ClusterConfig
    embedding: EmbeddingConfig

    @staticmethod
    def load(path: str) -> "AppConfig":
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        bq = BQConfig(**raw.get("bigquery", {}))
        cl = raw.get("clustering", {})
        cl_cfg = ClusterConfig(
            use_hdbscan=cl.get("use_hdbscan", True),
            min_cluster_size=int(cl.get("min_cluster_size", 8)),
            min_samples=int(cl.get("min_samples", 5)),
            use_kmeans_fallback=cl.get("use_kmeans_fallback", True),
            k_range=tuple(cl.get("k_range", [2, 12])),
        )
        emb = raw.get("embedding", {})
        emb_cfg = EmbeddingConfig(
            prefer_umap=emb.get("prefer_umap", True),
            prefer_tsne=emb.get("prefer_tsne", True),
            random_state=int(emb.get("random_state", 42)),
        )
        return AppConfig(
            read_mode=raw.get("read_mode", "local_csv"),
            factory_addresses_csv=raw.get("factory_addresses_csv", "experiments/RQ3/inputs/factory_addresses.csv"),
            local_sources_dir=raw.get("local_sources_dir", "experiments/RQ3/data/sources"),
            output_dir=raw.get("output_dir", "experiments/RQ3/outputs"),
            chains=list(raw.get("chains", ["ethereum", "polygon"])),
            bigquery=bq,
            clustering=cl_cfg,
            embedding=emb_cfg,
        )


# --------------------------- Data model helpers --------------------------- #


@dataclass
class SourceFile:
    path: str
    content: str


@dataclass
class SourceRecord:
    """Local cache schema for a single verified factory contract.

    Expected JSON structure stored at
    experiments/RQ3/data/sources/{chain}/{address}.json

    {
      "chain": "ethereum",
      "address": "0x...",
      "source_status": "full_match",  // or partial_match
      "compiler": "v0.8.20+commit.1",
      "evm_version": "london",
      "verified_at": "2025-01-01T00:00:00Z",
      "origin": "sourcify",
      "files": [ {"path": "contracts/Factory.sol", "content": "pragma ..."}, ... ]
    }
    """

    chain: str
    address: str
    source_status: str
    files: List[SourceFile]
    compiler: Optional[str] = None
    evm_version: Optional[str] = None
    verified_at: Optional[str] = None
    origin: str = "sourcify"

    @staticmethod
    def from_json(obj: dict) -> "SourceRecord":
        files = [SourceFile(**f) for f in obj.get("files", [])]
        return SourceRecord(
            chain=obj.get("chain", ""),
            address=obj.get("address", ""),
            source_status=obj.get("source_status", "unknown"),
            files=files,
            compiler=obj.get("compiler"),
            evm_version=obj.get("evm_version"),
            verified_at=obj.get("verified_at"),
            origin=obj.get("origin", "sourcify"),
        )


# ------------------------------ IO functions ------------------------------ #


def load_factory_addresses(cfg: AppConfig) -> pd.DataFrame:
    """Load factory address summary by chain.

    Returns DataFrame with at least columns: chain, factory_address, total_creations, unique_creations.
    Accepts CSV with headers: chain,factory_address,total_creations,unique_creations[,first_seen_date,last_seen_date]
    """
    if cfg.read_mode == "local_csv":
        path = Path(cfg.factory_addresses_csv)
        if not path.exists():
            raise FileNotFoundError(
                f"Factory addresses CSV not found: {path}. Create a CSV with columns: "
                "chain,factory_address,total_creations,unique_creations[,first_seen_date,last_seen_date]"
            )
        df = pd.read_csv(path, dtype={"chain": str, "factory_address": str})
        df["factory_address"] = df["factory_address"].str.lower()
        if "total_creations" not in df.columns:
            df["total_creations"] = np.nan
        if "unique_creations" not in df.columns:
            df["unique_creations"] = np.nan
        return df[df["chain"].isin(cfg.chains)].reset_index(drop=True)

    if cfg.read_mode == "bigquery":
        if not cfg.bigquery.use:
            raise RuntimeError("BigQuery read_mode set but bigquery.use=false in config.")
        if bigquery is None:
            raise RuntimeError("google-cloud-bigquery not available in this environment.")
        table = f"{cfg.bigquery.project_id}.{cfg.bigquery.result_dataset}.{cfg.bigquery.table_name}"
        client = bigquery.Client(project=cfg.bigquery.project_id)  # type: ignore
        sql = f"""
        SELECT chain, LOWER(factory_address) AS factory_address,
               COUNT(*) AS total_creations,
               COUNT(DISTINCT created_contract_address) AS unique_creations
        FROM `{table}`
        WHERE chain IN UNNEST(@chains)
        GROUP BY chain, factory_address
        """
        job = client.query(sql, job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("chains", "STRING", cfg.chains)]  # type: ignore
        ))
        return job.result().to_dataframe(create_bqstorage_client=False)

    raise ValueError(f"Unsupported read_mode: {cfg.read_mode}")


def iter_local_sources(local_dir: Path, chain: str) -> Iterable[SourceRecord]:
    """Yield SourceRecord objects from local cache directory for a chain."""
    chain_dir = local_dir / chain
    if not chain_dir.exists():
        return []
    for p in chain_dir.glob("*.json"):
        try:
            with open(p, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
            rec = SourceRecord.from_json(obj)
            if rec.chain.lower() != chain.lower():
                # tolerate missing chain in file; enforce chain param
                rec.chain = chain
            if not rec.address:
                rec.address = p.stem.lower()
            rec.address = rec.address.lower()
            yield rec
        except Exception as e:  # pragma: no cover - tolerant reader
            logging.warning("Failed to read %s: %s", p, e)


def map_sources_by_address(local_dir: Path, chain: str) -> Dict[str, SourceRecord]:
    return {rec.address: rec for rec in iter_local_sources(local_dir, chain)}


# ----------------------------- Feature extraction ----------------------------- #


class FeatureExtractor:
    """Extracts mechanism/semantic features from source files and optional runtime signals."""

    def __init__(self) -> None:
        self._re_function = re.compile(r"\bfunction\s+[a-zA-Z0-9_]+\s*\(")
        self._re_event = re.compile(r"\bevent\s+[a-zA-Z0-9_]+\s*\(")
        self._re_mapping = re.compile(r"\bmapping\s*\(")
        self._re_array = re.compile(r"\[[^\]]*\]")
        self._re_contract = re.compile(r"\bcontract\s+([A-Za-z_][A-Za-z0-9_]*)\b")
        self._re_interface = re.compile(r"\binterface\s+([A-Za-z_][A-Za-z0-9_]*)\b")
        self._re_abstract = re.compile(r"\babstract\s+contract\b")
        self._re_constructor = re.compile(r"\bconstructor\s*\(")
        self._re_modifier_decl = re.compile(r"\bmodifier\s+[A-Za-z_][A-Za-z0-9_]*\s*\(")
        self._re_only_owner = re.compile(r"\bonlyOwner\b")
        self._re_access_control = re.compile(r"\bAccessControl\b")
        self._re_ownable = re.compile(r"\bOwnable\b")
        self._re_create = re.compile(r"\bcreate\s*\(", re.IGNORECASE)
        self._re_create2 = re.compile(r"\bcreate2\s*\(", re.IGNORECASE)
        self._re_clone = re.compile(r"\bclone\s*\(", re.IGNORECASE)
        self._re_predict = re.compile(r"\bpredict\w*\s*\(", re.IGNORECASE)
        self._re_salt = re.compile(r"\bbytes32\s+salt\b|CREATE2", re.IGNORECASE)
        self._re_delegatecall = re.compile(r"\bdelegatecall\b", re.IGNORECASE)
        self._re_public = re.compile(r"\bpublic\b")
        self._re_external = re.compile(r"\bexternal\b")
        self._re_openzeppelin = re.compile(r"openzeppelin", re.IGNORECASE)
        self._re_using = re.compile(r"\busing\s+([A-Za-z_][A-Za-z0-9_]*)\s+for\b")
        self._re_library_decl = re.compile(r"\blibrary\s+([A-Za-z_][A-Za-z0-9_]*)\b")
        self._re_selector_hex = re.compile(r"0x[a-fA-F0-9]{8}")

        # Business keywords for bag-of-words (extendable)
        self.business_terms = [
            "dex", "amm", "pool", "router", "factory", "pair",
            "token", "erc20", "erc721", "erc1155", "nft", "mint", "burn",
            "wallet", "account", "safe", "signature", "multisig", "aa",
            "dao", "govern", "vote", "vault", "stake", "yield",
            "beacon", "proxy", "uups", "upgradeable", "upgrade",
        ]
        self._bow_vectorizer = CountVectorizer(vocabulary=self.business_terms, lowercase=True)

        # Contract name tokens to capture semantics from names
        self.name_tokens = [
            "factory", "router", "manager", "token", "vault", "proxy", "beacon",
            "registry", "pool", "pair", "sale", "market", "bridge", "oracle",
            "staking", "govern", "wallet", "swap", "dex", "nft", "airdrop",
            "launch", "lottery", "crowd", "locker", "escrow"
        ]

    @staticmethod
    def _concat_sources(files: List[SourceFile]) -> str:
        return "\n".join(f.content for f in files)

    def extract(self, rec: SourceRecord) -> Dict[str, float]:
        text = self._concat_sources(rec.files)
        if not text:
            return {}

        # Basic counts
        n_funcs = len(self._re_function.findall(text))
        n_events = len(self._re_event.findall(text))
        n_mappings = len(self._re_mapping.findall(text))
        n_arrays = len(self._re_array.findall(text))
        n_public = len(self._re_public.findall(text))
        n_external = len(self._re_external.findall(text))
        n_lines = float(len(text.splitlines()))
        n_chars = float(len(text))
        n_imports = len(re.findall(r"\bimport\b", text))
        n_modifiers = len(self._re_modifier_decl.findall(text))
        n_constructors = len(self._re_constructor.findall(text))

        # Contract declarations and name-derived semantics
        contracts = self._re_contract.findall(text)
        n_contracts = len(contracts)
        n_interfaces = len(self._re_interface.findall(text))
        n_abstract = len(self._re_abstract.findall(text))
        # name token hits across declared contract names
        name_token_counts: Dict[str, float] = {}
        for tok in self.name_tokens:
            cnt = 0
            for nm in contracts:
                if tok in nm.lower():
                    cnt += 1
            name_token_counts[f"name_{tok}"] = float(cnt)

        # Mechanism flags
        has_create = 1.0 if self._re_create.search(text) else 0.0
        has_create2 = 1.0 if self._re_create2.search(text) else 0.0
        has_clone = 1.0 if self._re_clone.search(text) else 0.0
        has_predict = 1.0 if self._re_predict.search(text) else 0.0
        has_salt = 1.0 if self._re_salt.search(text) else 0.0
        has_delegatecall = 1.0 if self._re_delegatecall.search(text) else 0.0
        has_oz = 1.0 if self._re_openzeppelin.search(text) else 0.0

        # Library usage
        using_libs = [m for m in self._re_using.findall(text)]
        library_decls = self._re_library_decl.findall(text)
        n_using = float(len(using_libs))
        n_library_decl = float(len(library_decls))
        n_distinct_using = float(len(set(using_libs)))
        # Presence flags for common libraries
        lib_names = [
            "SafeMath", "Address", "ECDSA", "Strings", "Clones", "Create2",
            "Math", "EnumerableSet", "EnumerableMap"
        ]
        lib_flags = {f"lib_{ln}": (1.0 if re.search(rf"\b{ln}\b", text) else 0.0) for ln in lib_names}

        has_ownable = 1.0 if self._re_ownable.search(text) else 0.0
        has_access_control = 1.0 if self._re_access_control.search(text) else 0.0
        has_only_owner = 1.0 if self._re_only_owner.search(text) else 0.0

        # Ratios
        total_vis = n_public + n_external if (n_public + n_external) > 0 else 1.0
        public_ratio = n_public / total_vis
        external_ratio = n_external / total_vis

        # Business BoW
        bow_counts = self._bow_vectorizer.transform([text]).toarray()[0]
        bow_features = {f"bow_{t}": float(c) for t, c in zip(self.business_terms, bow_counts)}

        feats: Dict[str, float] = {
            "n_functions": float(n_funcs),
            "n_events": float(n_events),
            "n_mappings": float(n_mappings),
            "n_arrays": float(n_arrays),
            "n_lines": n_lines,
            "n_chars": n_chars,
            "n_imports": float(n_imports),
            "n_modifiers": float(n_modifiers),
            "n_constructors": float(n_constructors),
            "n_contracts": float(n_contracts),
            "n_interfaces": float(n_interfaces),
            "n_abstract": float(n_abstract),
            "has_create": has_create,
            "has_create2": has_create2,
            "has_clone": has_clone,
            "has_predict": has_predict,
            "has_salt": has_salt,
            "has_delegatecall": has_delegatecall,
            "has_openzeppelin": has_oz,
            "has_ownable": has_ownable,
            "has_access_control": has_access_control,
            "has_only_owner": has_only_owner,
            "public_ratio": float(public_ratio),
            "external_ratio": float(external_ratio),
            "n_using": n_using,
            "n_library_decl": n_library_decl,
            "n_distinct_using": n_distinct_using,
        }
        feats.update(bow_features)
        feats.update(name_token_counts)
        feats.update(lib_flags)
        return feats


# ----------------------------- Dimensionality utils ----------------------------- #


def varimax(Phi: np.ndarray, gamma: float = 1.0, q: int = 20, tol: float = 1e-6) -> np.ndarray:
    """Varimax rotation for PCA loadings (orthogonal rotation).

    Uses a numerically stable formulation: diag(diag(L.T @ L)) == diag(sum(L**2, axis=0)).
    """
    p, k = Phi.shape
    R = np.eye(k)
    d = 0.0
    for _ in range(q):
        d_old = d
        Lambda = Phi @ R
        # Compute rotation via SVD of the gradient
        col_norm_diag = np.diag(np.sum(Lambda ** 2, axis=0))
        M = Phi.T @ (Lambda ** 3 - (gamma / p) * (Lambda @ col_norm_diag))
        u, s, vh = np.linalg.svd(M, full_matrices=False)
        R = u @ vh
        d = float(np.sum(s))
        if d_old != 0.0 and (d - d_old) < tol:
            break
    return Phi @ R


def orthogonalise_features(X: np.ndarray, n_components: Optional[int] = None, apply_varimax: bool = True) -> Tuple[np.ndarray, PCA]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    if n_components is None:
        n_components = min(12, Xs.shape[1])

    pca = PCA(n_components=n_components, random_state=42)
    Z = pca.fit_transform(Xs)

    if apply_varimax:
        loadings = pca.components_.T  # shape (features, components)
        rotated = varimax(loadings)
        # Project Xs onto rotated axes
        Z = np.dot(Xs, rotated)

    return Z, pca


# -------------------------------- Clustering -------------------------------- #


def run_clustering(Z: np.ndarray, cfg: ClusterConfig, random_state: int = 42, k_override: int = None) -> Tuple[np.ndarray, Dict[str, float], str]:
    """Return labels array (shape [n_samples]), metrics dict, and method name.

    If k_override is provided, force KMeans with that k and skip HDBSCAN/model selection.
    """
    # If user forces K, run fixed-K KMeans regardless of HDBSCAN availability
    if k_override is not None:
        k = max(1, int(k_override))
        km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        labels = km.fit_predict(Z)
        try:
            sil = silhouette_score(Z, labels)
        except Exception:
            sil = -1.0
        try:
            db = davies_bouldin_score(Z, labels)
        except Exception:
            db = float("inf")
        metrics = {"n_clusters": float(k), "silhouette_db_score": float(sil - 0.1 * db), "davies_bouldin": float(db)}
        return labels, metrics, "kmeans"

    if cfg.use_hdbscan and hdbscan is not None:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=cfg.min_cluster_size, min_samples=cfg.min_samples)
        labels = clusterer.fit_predict(Z)
        # HDBSCAN labels -1 for noise
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        metrics = {"n_clusters": float(n_clusters), "noise_fraction": float(np.mean(labels == -1))}
        return labels, metrics, "hdbscan"

    # Fallback to KMeans with model selection over k
    best_score = -math.inf
    best_labels = None
    best_k = None
    best_db = math.inf
    k_min, k_max = cfg.k_range
    for k in range(max(2, k_min), max(3, k_max) + 1):
        km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        labels_k = km.fit_predict(Z)
        # Skip degenerate results
        if len(set(labels_k)) < 2:
            continue
        try:
            sil = silhouette_score(Z, labels_k)
        except Exception:
            sil = -1.0
        try:
            db = davies_bouldin_score(Z, labels_k)
        except Exception:
            db = math.inf
        score = sil - 0.1 * db
        if score > best_score:
            best_score = score
            best_labels = labels_k
            best_k = k
            best_db = db
    if best_labels is None:
        # Final fallback: single cluster
        best_labels = np.zeros(Z.shape[0], dtype=int)
        best_k = 1
        best_db = float("nan")
    metrics = {"n_clusters": float(best_k), "silhouette_db_score": float(best_score), "davies_bouldin": float(best_db)}
    return best_labels, metrics, "kmeans"


# -------------------------------- Embeddings -------------------------------- #


def embed_2d(X: np.ndarray, emb_cfg: EmbeddingConfig) -> Tuple[np.ndarray, str]:
    if umap is not None and emb_cfg.prefer_umap:
        reducer = umap.UMAP(n_components=2, random_state=emb_cfg.random_state)
        return reducer.fit_transform(X), "umap"
    if emb_cfg.prefer_tsne:
        tsne = TSNE(n_components=2, random_state=emb_cfg.random_state, init="pca", learning_rate="auto")
        return tsne.fit_transform(X), "tsne"
    # fallback to PCA 2D
    pca2 = PCA(n_components=2, random_state=emb_cfg.random_state)
    return pca2.fit_transform(X), "pca"


def _rotate_points(X2: np.ndarray, degrees: float, center: str = "mean") -> np.ndarray:
    """Rotate 2D points by `degrees` around a chosen center.

    center:
      - 'mean' (default): rotate around centroid (mean of points)
      - 'bbox': rotate around bounding-box center
    """
    if degrees % 360 == 0:
        return X2
    theta = np.deg2rad(degrees)
    c, s = float(np.cos(theta)), float(np.sin(theta))
    R = np.array([[c, -s], [s, c]], dtype=float)
    if center == "bbox":
        cx = (float(np.min(X2[:, 0])) + float(np.max(X2[:, 0]))) / 2.0
        cy = (float(np.min(X2[:, 1])) + float(np.max(X2[:, 1]))) / 2.0
    else:
        cx = float(np.mean(X2[:, 0]))
        cy = float(np.mean(X2[:, 1]))
    ctr = np.array([cx, cy], dtype=float)
    return (X2 - ctr) @ R.T + ctr


# --------------------------------- Plotting --------------------------------- #


def save_scatter_pdf(
    X2: np.ndarray,
    labels: np.ndarray,
    weights: Optional[np.ndarray],
    title: str,
    out_path: Path,
    palette: Optional[List[str]] = None,
    draw_hulls: bool = False,
    show_centroids: bool = False,
    max_points_per_cluster: Optional[int] = None,
    outlier_fraction: Optional[float] = None,
    figsize: Tuple[float, float] = (5, 5),
    show_legend: bool = True,
    random_state: int = 42,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    uniq = sorted(set(labels))
    cmap = plt.get_cmap("tab20")

    # Default canvas
    plt.figure(figsize=figsize, dpi=150)
    rng = None
    removed_total = 0.0
    if max_points_per_cluster is not None:
        import numpy as _np
        rng = _np.random.RandomState(random_state)

    for i, lab in enumerate(uniq):
        lab = int(lab)
        mask = labels == lab
        idx = np.where(mask)[0]
        # Remove far outliers per cluster by distance to centroid
        if outlier_fraction is not None and outlier_fraction > 0 and idx.size >= 10:
            idx_all = idx
            pts_all = X2[idx_all]
            cx, cy = float(np.mean(pts_all[:, 0])), float(np.mean(pts_all[:, 1]))
            d = np.sqrt((pts_all[:, 0] - cx) ** 2 + (pts_all[:, 1] - cy) ** 2)
            q = 1.0 - float(outlier_fraction)
            q = min(max(q, 0.0), 1.0)
            thr = np.quantile(d, q)
            keep = d <= thr
            if weights is not None:
                removed_total += float(np.sum(weights[idx_all[~keep]]))
            else:
                removed_total += float(np.sum(~keep))
            idx = idx_all[keep]
        if max_points_per_cluster is not None and idx.size > max_points_per_cluster:
            idx = rng.choice(idx, size=max_points_per_cluster, replace=False)  # type: ignore
        pts = X2[idx]
        if palette is not None and len(palette) > 0:
            if lab == -1:
                color = "#BDBDBD"  # grey for noise if present
            else:
                color = palette[i % len(palette)]
        else:
            color = cmap(i % 20)
        # Use a uniform marker size for all points (ignore weights for size)
        size = 12
        alpha = 0.8
        lbl = f"cluster {lab}" if lab != -1 else "noise"
        plt.scatter(pts[:, 0], pts[:, 1], s=size, alpha=alpha, c=[color], label=lbl, edgecolors="none")

        # Optional convex hull to delineate cluster area
        if draw_hulls and lab != -1 and pts.shape[0] >= 3:
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(pts)
                verts = pts[hull.vertices]
                import matplotlib.pyplot as _plt
                _plt.fill(verts[:, 0], verts[:, 1], facecolor=color, alpha=0.08, edgecolor=color, linewidth=1.0)
            except Exception:
                pass

        # Optional centroid marker
        if show_centroids and lab != -1 and pts.shape[0] >= 1:
            cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
            plt.scatter([cx], [cy], s=120, marker='X', c=[color], edgecolors='black', linewidths=0.5, alpha=0.9)

    plt.title(title)
    if show_legend:
        plt.legend(markerscale=1.2, fontsize=8, loc="best", framealpha=0.6)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    if removed_total:
        print(f"Removed outliers (not plotted, weighted): {removed_total}")


# ------------------------------ Cluster summary ------------------------------ #


def summarize_clusters(labels: np.ndarray, feature_names: List[str], X: np.ndarray, top_k: int = 5) -> Dict[str, dict]:
    df = pd.DataFrame(X, columns=feature_names)
    df["cluster"] = labels
    clusters: Dict[str, dict] = {}
    for cl in sorted(set(labels)):
        subset = df[df["cluster"] == cl].drop(columns=["cluster"]) if cl != -1 else df[df["cluster"] == cl].drop(columns=["cluster"]) if (-1 in set(labels)) else None
        if subset is None or subset.empty:
            continue
        means = subset.mean()
        top = means.abs().sort_values(ascending=False).head(top_k)
        clusters[str(cl)] = {
            "size": int(subset.shape[0]),
            "top_features": [{"name": k, "value": float(means[k])} for k in top.index],
        }
    return clusters


# --------------------------------- Pipeline --------------------------------- #


def run_for_chain(
    chain: str,
    cfg: AppConfig,
    factory_only: bool = False,
    k_override: int = None,
    size_weight: str = "none",
    palette_codes: Optional[List[str]] = None,
    rotate_deg: float = 90.0,
    outlier_fraction: Optional[float] = 0.10,
    dedup: bool = True,
) -> None:
    logger = logging.getLogger("RQ3")
    sources_dir = Path(cfg.local_sources_dir)
    out_dir = Path(cfg.output_dir)

    logger.info("Loading factory addresses from %s", cfg.read_mode)
    df_all = load_factory_addresses(cfg)
    df_all = df_all[df_all["chain"].str.lower() == chain.lower()].reset_index(drop=True)
    if factory_only:
        # Keep addresses that have evidence of creations
        if "unique_creations" in df_all.columns and "total_creations" in df_all.columns:
            df_all = df_all[(pd.to_numeric(df_all["unique_creations"], errors="coerce") >= 1) |
                            (pd.to_numeric(df_all["total_creations"], errors="coerce") >= 1)]
        elif "unique_creations" in df_all.columns:
            df_all = df_all[pd.to_numeric(df_all["unique_creations"], errors="coerce") >= 1]
        elif "total_creations" in df_all.columns:
            df_all = df_all[pd.to_numeric(df_all["total_creations"], errors="coerce") >= 1]
    if df_all.empty:
        logger.warning("No factory addresses found for chain=%s. Skipping.", chain)
        return

    # Map local sources by address
    logger.info("Indexing local source cache for chain=%s", chain)
    addr_to_source = map_sources_by_address(sources_dir, chain)
    df_all["address"] = df_all["factory_address"].str.lower()
    df = df_all[df_all["address"].isin(addr_to_source.keys())].reset_index(drop=True)
    logger.info("Found %d verified sources (Sourcify local cache)", len(df))

    if df.empty:
        logger.warning("No verified sources available locally for chain=%s. Nothing to cluster.", chain)
        return

    # Extract features
    extractor = FeatureExtractor()
    rows = []
    feature_names: Optional[List[str]] = None
    for _, row in df.iterrows():
        addr = row["address"]
        rec = addr_to_source.get(addr)
        if rec is None:
            continue
        feats = extractor.extract(rec)
        if not feats:
            continue
        if feature_names is None:
            feature_names = sorted(feats.keys())
        # Ensure consistent ordering / fill missing keys with 0.0
        vec = [float(feats.get(k, 0.0)) for k in feature_names]
        rows.append({"address": addr, **{k: v for k, v in zip(feature_names, vec)}})

    if not rows or feature_names is None:
        logger.warning("No features extracted for chain=%s.", chain)
        return

    feat_df = pd.DataFrame(rows)
    X = feat_df[feature_names].to_numpy(dtype=float)

    # Orthogonalise / standardise via PCA (+Varimax)
    Z, _pca = orthogonalise_features(X, n_components=min(12, X.shape[1]), apply_varimax=True)

    # Cluster
    labels, metrics, method = run_clustering(Z, cfg.clustering, k_override=k_override)
    logger.info("Clustering method=%s, metrics=%s", method, metrics)

    # 2D embedding
    X2, emb_method = embed_2d(Z, cfg.embedding)
    if rotate_deg:
        X2 = _rotate_points(X2, rotate_deg, center="mean")
    logger.info("Embedding method=%s", emb_method)

    # Figure and JSON outputs
    # Marker size weighting (default: none) — can be set to 'unique' or 'total'
    weights = None
    if size_weight == "unique" and "unique_creations" in df.columns:
        w_map = dict(zip(df["address"], df["unique_creations"].fillna(1.0)))
        weights = np.array([w_map.get(addr, 1.0) for addr in feat_df["address"]], dtype=float)
    elif size_weight == "total" and "total_creations" in df.columns:
        w_map = dict(zip(df["address"], df["total_creations"].fillna(1.0)))
        weights = np.array([w_map.get(addr, 1.0) for addr in feat_df["address"]], dtype=float)

    # Optional deduplication for plotting only
    labels_orig = labels.copy()
    if dedup:
        from collections import defaultdict, Counter
        X_round = np.round(X.astype(float), 6)
        groups = defaultdict(list)
        for i, row in enumerate(X_round):
            groups[tuple(row.tolist())].append(i)
        if len(groups) < X.shape[0]:
            nX2, nlbl, nw = [], [], []
            for idxs in groups.values():
                arr = np.array(idxs, dtype=int)
                nX2.append(np.mean(X2[arr], axis=0))
                cnt = Counter(labels[arr].tolist())
                nlbl.append(int(max(cnt.items(), key=lambda kv: kv[1])[0]))
                if weights is not None:
                    nw.append(float(np.sum(weights[arr])))
                else:
                    nw.append(float(arr.size))
            X2_plot = np.vstack(nX2)
            labels_plot = np.array(nlbl, dtype=int)
            weights_plot = np.array(nw, dtype=float)
        else:
            X2_plot, labels_plot, weights_plot = X2, labels, (weights if weights is not None else None)
    else:
        X2_plot, labels_plot, weights_plot = X2, labels, (weights if weights is not None else None)

    # Default palette (red, yellow, bright blue, green)
    if not palette_codes:
        palette_codes = ["#EA3323", "#F8D675", "#1E90FF", "#2CA02C"]

    title = f"Factory Cluster Map - {chain}"
    fig_path = out_dir / f"clustered_semantic_space_{chain}.pdf"
    save_scatter_pdf(
        X2_plot,
        labels_plot,
        weights_plot,
        title,
        fig_path,
        palette=palette_codes,
        outlier_fraction=outlier_fraction,
        figsize=(4, 4),
        show_legend=False,
    )
    logger.info("Saved figure: %s", fig_path)

    # Summaries (use original labels)
    clusters = summarize_clusters(labels_orig, feature_names, X)
    per_addr = {addr: int(lab) for addr, lab in zip(feat_df["address"], labels_orig.tolist())}
    summary = {
        "chain": chain,
        "method": method,
        "embedding": emb_method,
        "metrics": metrics,
        "clusters": clusters,
        "address_labels": per_addr,
        "feature_names": feature_names,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"clusters_{chain}.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    logger.info("Saved summary: %s", out_dir / f"clusters_{chain}.json")


def run_for_chains_combined(chains: List[str], cfg: AppConfig, factory_only: bool = False, k_override: int = None, size_weight: str = "none", palette_codes: Optional[List[str]] = None, rotate_deg: float = 0.0, outlier_fraction: Optional[float] = None, dedup: bool = True) -> None:
    logger = logging.getLogger("RQ3")
    sources_dir = Path(cfg.local_sources_dir)
    out_dir = Path(cfg.output_dir)

    logger.info("Loading factory addresses from %s", cfg.read_mode)
    df_all = load_factory_addresses(cfg)
    df_all["chain"] = df_all["chain"].str.lower()
    chains_l = [c.lower() for c in chains]
    df_all = df_all[df_all["chain"].isin(chains_l)].reset_index(drop=True)
    if factory_only:
        if "unique_creations" in df_all.columns and "total_creations" in df_all.columns:
            df_all = df_all[(pd.to_numeric(df_all["unique_creations"], errors="coerce") >= 1) |
                            (pd.to_numeric(df_all["total_creations"], errors="coerce") >= 1)]
        elif "unique_creations" in df_all.columns:
            df_all = df_all[pd.to_numeric(df_all["unique_creations"], errors="coerce") >= 1]
        elif "total_creations" in df_all.columns:
            df_all = df_all[pd.to_numeric(df_all["total_creations"], errors="coerce") >= 1]
    if df_all.empty:
        logger.warning("No factory addresses found for chains=%s. Skipping.", ",".join(chains_l))
        return

    # Build source maps for each chain and aggregate
    addr_to_source_by_chain: Dict[str, Dict[str, SourceRecord]] = {}
    for ch in chains_l:
        logger.info("Indexing local source cache for chain=%s", ch)
        addr_to_source_by_chain[ch] = map_sources_by_address(sources_dir, ch)

    df_all["address"] = df_all["factory_address"].str.lower()

    # Extract features for addresses with available sources
    extractor = FeatureExtractor()
    rows = []
    feature_names: Optional[List[str]] = None
    for _, row in df_all.iterrows():
        ch = row["chain"]
        addr = row["address"]
        rec = addr_to_source_by_chain.get(ch, {}).get(addr)
        if rec is None:
            continue
        feats = extractor.extract(rec)
        if not feats:
            continue
        if feature_names is None:
            feature_names = sorted(feats.keys())
        vec = [float(feats.get(k, 0.0)) for k in feature_names]
        rows.append({"key": f"{ch}:{addr}", "chain": ch, "address": addr, **{k: v for k, v in zip(feature_names, vec)}})

    if not rows or feature_names is None:
        logger.warning("No features extracted for combined chains=%s.", ",".join(chains_l))
        return

    feat_df = pd.DataFrame(rows)
    X = feat_df[feature_names].to_numpy(dtype=float)

    # Orthogonalise / standardise via PCA (+Varimax)
    Z, _pca = orthogonalise_features(X, n_components=min(12, X.shape[1]), apply_varimax=True)

    # Cluster
    labels, metrics, method = run_clustering(Z, cfg.clustering, k_override=k_override)
    logger.info("Clustering method=%s, metrics=%s", method, metrics)

    # 2D embedding
    X2, emb_method = embed_2d(Z, cfg.embedding)
    # Optional rotation around centroid to align the global structure (e.g., 90 degrees)
    if rotate_deg:
        X2 = _rotate_points(X2, rotate_deg, center="mean")
    logger.info("Embedding method=%s", emb_method)

    # Marker size weighting (default: none)
    weights = None
    if size_weight in ("unique", "total"):
        col = "unique_creations" if size_weight == "unique" else "total_creations"
        if col in df_all.columns:
            w_map = {(str(r["chain"]).lower(), str(r["address"]).lower()): r[col] for _, r in df_all.fillna(1.0).iterrows()}
            weights = np.array([w_map.get((c, a), 1.0) for c, a in zip(feat_df["chain"], feat_df["address"])], dtype=float)

    # Keep original labels for summaries/JSON
    labels_orig = labels.copy()

    # Deduplicate identical feature vectors (rounded) and aggregate weights for plotting only
    if dedup:
        from collections import defaultdict, Counter
        X_round = np.round(X.astype(float), 6)
        groups = defaultdict(list)
        for i, row in enumerate(X_round):
            groups[tuple(row.tolist())].append(i)
        if len(groups) < X.shape[0]:
            new_X2, new_labels, new_weights = [], [], []
            for idxs in groups.values():
                arr = np.array(idxs, dtype=int)
                # average embedding for visual placement
                new_X2.append(np.mean(X2[arr], axis=0))
                # majority vote for label
                cnt = Counter(labels[arr].tolist())
                new_labels.append(int(max(cnt.items(), key=lambda kv: kv[1])[0]))
                if weights is not None:
                    new_weights.append(float(np.sum(weights[arr])))
                else:
                    new_weights.append(float(arr.size))
            X2_plot = np.vstack(new_X2)
            labels_plot = np.array(new_labels, dtype=int)
            weights_plot = np.array(new_weights, dtype=float)
        else:
            X2_plot, labels_plot, weights_plot = X2, labels, (weights if weights is not None else None)
    else:
        X2_plot, labels_plot, weights_plot = X2, labels, (weights if weights is not None else None)

    # Default palette for combined mode if none provided (Red, Yellow, Blue, Green)
    if not palette_codes:
        palette_codes = [
            "#EA3323",  # red
            "#F8D675",  # yellow
            "#1E90FF",  # bright blue (DodgerBlue)
            "#2CA02C",  # green
        ]

    name = "_".join(chains_l)
    # Use a short, simple title as requested (3 words)
    title = "Factory Cluster Map"
    fig_path = out_dir / f"clustered_semantic_space_combined_{name}.pdf"
    save_scatter_pdf(
        X2_plot,
        labels_plot,
        weights_plot,
        title,
        fig_path,
        palette=palette_codes,
        outlier_fraction=outlier_fraction,
        figsize=(4, 4),
        show_legend=False,
    )
    logger.info("Saved figure: %s", fig_path)

    clusters = summarize_clusters(labels_orig, feature_names, X)
    per_key = {key: int(lab) for key, lab in zip(feat_df["key"], labels_orig.tolist())}
    summary = {
        "chains": chains_l,
        "method": method,
        "embedding": emb_method,
        "metrics": metrics,
        "clusters": clusters,
        "key_labels": per_key,
        "feature_names": feature_names,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"clusters_combined_{name}.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    logger.info("Saved summary: %s", out_dir / f"clusters_combined_{name}.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ3 Factory Clustering Analysis (offline-first)")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.json"), help="Path to RQ3 config.json")
    parser.add_argument("--chains", nargs="*", help="Override chains to process (e.g., ethereum polygon)")
    parser.add_argument("--log", default="INFO", help="Logging level (DEBUG, INFO, WARNING)")
    parser.add_argument("--factory-only", action="store_true", help="Filter to addresses with creations (unique/total > 0)")
    parser.add_argument("--combined", action="store_true", help="Cluster across chains as a single dataset")
    parser.add_argument("--k", type=int, help="Force KMeans with k clusters (overrides HDBSCAN/model selection)")
    parser.add_argument("--palette", type=str, help="Comma-separated HEX colors for clusters (e.g., #F8D675,#3D7FBE,#EA3323,#56BCF9)")
    parser.add_argument("--size-weight", choices=["none","unique","total"], default="none", help="Marker size weighting (default: none)")
    parser.add_argument("--outlier-frac", type=float, help="Hide far outliers per cluster by dropping top fraction of distances (e.g., 0.02)")
    parser.add_argument("--rotate", type=float, help="Rotate embedding by degrees around centroid (e.g., 90)")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO), format="[%(levelname)s] %(message)s")
    cfg = AppConfig.load(args.config)
    if args.chains:
        cfg.chains = [c.lower() for c in args.chains]

    palette_codes = None
    if getattr(args, "palette", None):
        palette_codes = [c.strip() for c in str(args.palette).split(",") if c.strip()]

    if args.combined:
        # Default to 90-degree rotation for combined to improve visual alignment unless explicitly set
        rotate_deg = args.rotate if args.rotate is not None else 90.0
        outlier_frac = args.outlier_frac if args.outlier_frac is not None else 0.10
        run_for_chains_combined(cfg.chains, cfg, factory_only=args.factory_only, k_override=args.k, size_weight=args.size_weight, palette_codes=palette_codes, rotate_deg=rotate_deg, outlier_fraction=outlier_frac, dedup=True)
    else:
        for chain in cfg.chains:
            run_for_chain(chain, cfg, factory_only=args.factory_only, k_override=args.k, size_weight=args.size_weight, palette_codes=palette_codes)


if __name__ == "__main__":
    main()
