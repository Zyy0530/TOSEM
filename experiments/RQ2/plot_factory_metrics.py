#!/usr/bin/env python3
"""Generate publication-ready figures for factory contract metrics."""

import os
from pathlib import Path
from typing import Dict, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "outputs"
PLOT_DIR = DATA_DIR / "plots"

FILES = {
    "daily_active": DATA_DIR / "daily_active_factories.csv",
    "daily_traces": DATA_DIR / "daily_factory_create_traces.csv",
    "bytecode": DATA_DIR / "bytecode_reuse_distribution.csv",
    "factory_activity": DATA_DIR / "factory_activity_distribution.csv",
}

COLORS = {
    "daily_active": "#1b9e77",
    "daily_traces": "#d95f02",
    "bytecode": "#7570b3",
    "factory_activity": "#e7298a",
}

DISTRIBUTION_COLORS = {
    # Slightly darkened blue for bars; keep CDF color
    "bytecode": {"bar": "#9BB6D1", "cdf": "#D59792"},
    "factory_activity": {"bar": "#9BB6D1", "cdf": "#D59792"},
}

CHAINS = ["ethereum", "polygon"]

# Global figure size (width, height in inches) to unify aspect ratio across all plots
# 8x4 gives a consistent 2:1 ratio; adjust here if needed.
FIG_SIZE = (8, 4)

# Global font sizes for improved readability across figures
FONT_SIZES = {
    "font.size": 14,            # Base font size
    "axes.titlesize": 16,      # Axes title size
    "axes.labelsize": 14,      # X/Y label size
    "xtick.labelsize": 12,     # X tick labels
    "ytick.labelsize": 12,     # Y tick labels
    "legend.fontsize": 12,     # Legend text size
    "figure.titlesize": 16,    # Figure-level title
}

BIN_DEFS = [
    (1, 1, "1"),
    (2, 2, "2"),
    (3, 3, "3"),
    (4, 4, "4"),
    (5, 5, "5"),
    (6, 6, "6"),
    (7, 7, "7"),
    (8, 8, "8"),
    (9, 9, "9"),
    (10, 10, "10"),
    (11, 11, "11"),
    (12, 12, "12"),
    (13, 13, "13"),
    (14, 14, "14"),
    (15, 15, "15"),
    (16, 16, "16"),
    (17, 17, "17"),
    (18, 18, "18"),
    (19, 19, "19"),
    (20, 20, "20"),
    (21, 50, "21-50"),
    (51, 100, "51-100"),
    (101, 200, "101-200"),
    (201, 500, "201-500"),
    (501, 1000, "501-1k"),
    (1001, 2000, "1k-2k"),
    (2001, 5000, "2k-5k"),
    (5001, 10000, "5k-10k"),
    (10001, 50000, "10k-50k"),
    (50001, 100000, "50k-100k"),
    (100001, 500000, "100k-500k"),
    (500001, 1000000, "500k-1M"),
    (1000001, 2000000, "1M-2M"),
    (2000001, 5000000, "2M-5M"),
    (5000001, float("inf"), ">5M"),
]

BIN_ORDER = {label: idx for idx, (_, _, label) in enumerate(BIN_DEFS)}


def _ensure_plot_dir() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected CSV file: {path}")
    return pd.read_csv(path)


def _format_date_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.tick_params(axis="x", rotation=0)


def plot_daily_metric(df: pd.DataFrame, value_column: str, title_template: str,
                      filename_template: str, color_key: str) -> None:
    color = COLORS[color_key]
    df = df.copy()
    df["block_date"] = pd.to_datetime(df["block_date"])
    df["chain"] = df["chain"].str.lower()

    chain_frames: Dict[str, pd.DataFrame] = {}
    for chain in CHAINS:
        chain_df = df[df["chain"] == chain].sort_values("block_date")
        if not chain_df.empty:
            chain_frames[chain] = chain_df

    if not chain_frames:
        return

    x_min = min(frame["block_date"].min() for frame in chain_frames.values())
    x_max = max(frame["block_date"].max() for frame in chain_frames.values())
    y_max = max(frame[value_column].max() for frame in chain_frames.values())
    positive_values = [
        v for frame in chain_frames.values() for v in frame[value_column] if v > 0
    ]
    use_log_scale = bool(positive_values)
    y_limit = y_max * 1.1 if y_max > 0 else 1
    if use_log_scale:
        min_positive = min(positive_values)
        lower_bound = 1 if min_positive >= 1 else min_positive
        upper_bound = max(y_limit, lower_bound * 1.1)
    else:
        lower_bound = 0
        upper_bound = y_limit

    for chain, chain_df in chain_frames.items():
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.bar(chain_df["block_date"], chain_df[value_column], width=1.0,
               color=color, edgecolor="none", align="center")

        ax.set_title(title_template.format(chain=chain.title()))
        ax.set_xlabel("Date")
        ax.set_ylabel("Count")
        if not use_log_scale:
            ax.ticklabel_format(style="plain", axis="y")
            formatter = ScalarFormatter(useOffset=False, useMathText=False)
            formatter.set_scientific(False)
            ax.yaxis.set_major_formatter(formatter)
        _format_date_axis(ax)
        # For Polygon, crop the x-axis to the interval where counts > 0
        if chain == "polygon":
            nz = chain_df[chain_df[value_column] > 0]
            if not nz.empty:
                chain_x_min = nz["block_date"].min()
                chain_x_max = nz["block_date"].max()
                ax.set_xlim(chain_x_min, chain_x_max)
            else:
                ax.set_xlim(x_min, x_max)
        else:
            ax.set_xlim(x_min, x_max)
        if use_log_scale:
            ax.set_yscale("log")
            ax.set_ylim(lower_bound, upper_bound)
        else:
            ax.set_ylim(lower_bound, upper_bound)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

        out_path = PLOT_DIR / filename_template.format(chain=chain)
        fig.tight_layout()
        fig.savefig(out_path, format="pdf")
        plt.close(fig)


def _assign_bin(value: float) -> str:
    for low, high, label in BIN_DEFS:
        if low <= value <= high:
            return label
    return BIN_DEFS[-1][2]


def _compute_log_widths(values: np.ndarray) -> np.ndarray:
    log_values = np.log10(values)
    if len(values) > 1:
        edges = np.empty(len(values) + 1)
        edges[1:-1] = (log_values[:-1] + log_values[1:]) / 2
        edges[0] = log_values[0] - (log_values[1] - log_values[0]) / 2
        edges[-1] = log_values[-1] + (log_values[-1] - log_values[-2]) / 2
    else:
        edges = np.array([log_values[0] - 0.5, log_values[0] + 0.5])
    linear_edges = np.power(10, edges)
    return linear_edges[1:] - linear_edges[:-1]


def _nice_log_upper_bound(value: float) -> float:
    if value <= 0:
        return 1.0
    exponent = np.floor(np.log10(value))
    base = 10 ** exponent
    scaled = value / base
    for candidate in (1, 2, 5, 10):
        if scaled <= candidate:
            return candidate * base
    return 10 * base


def plot_distribution(df: pd.DataFrame, x_column: str, count_column: str,
                      title_template: str,
                      x_label: str, y_label: str, cdf_label: str,
                      filename_template: str, color_key: str,
                      use_log_x: bool = False,
                      use_raw_counts: bool = False) -> None:
    df = df.copy()
    df["chain"] = df["chain"].str.lower()

    palettes = DISTRIBUTION_COLORS.get(
        color_key, {"bar": COLORS[color_key], "cdf": COLORS[color_key]}
    )

    chain_aggs: Dict[str, pd.DataFrame] = {}
    global_y_max = 0
    global_y_min_positive: Optional[float] = None

    if use_raw_counts:
        x_caps = {
            "bytecode": 10 ** 2,
            "factory_activity": 500,
        }
        x_cap = x_caps.get(color_key)

        all_x_values = set()
        for chain in CHAINS:
            chain_df = df[df["chain"] == chain]
            if chain_df.empty:
                continue

            agg = (
                chain_df.groupby(x_column, as_index=False)[count_column]
                .sum()
                .sort_values(x_column)
                .reset_index(drop=True)
            )
            agg = agg[agg[x_column] > 0]
            if x_cap is not None:
                agg = agg[agg[x_column] <= x_cap]
            if agg.empty:
                continue

            total = agg[count_column].sum()
            agg["cdf"] = agg[count_column].cumsum() / total if total > 0 else 0.0

            chain_aggs[chain] = agg
            current_max = agg[count_column].max()
            if current_max > global_y_max:
                global_y_max = current_max
            pos = agg[count_column][agg[count_column] > 0]
            if not pos.empty:
                cand = pos.min()
                if global_y_min_positive is None or cand < global_y_min_positive:
                    global_y_min_positive = cand
            all_x_values.update(agg[x_column].tolist())

        if not chain_aggs:
            return

        y_limit = global_y_max * 1.05 if global_y_max > 0 else 1
        x_min = min(all_x_values)
        x_max = max(all_x_values)
        if x_cap is not None:
            x_upper_bound = max(x_cap, x_min)
        else:
            x_upper_bound = _nice_log_upper_bound(x_max * 1.05) if use_log_x else x_max

        for chain, agg in chain_aggs.items():
            fig, ax = plt.subplots(figsize=FIG_SIZE)
            formatter_left = ScalarFormatter(useOffset=False, useMathText=False)
            formatter_left.set_scientific(False)
            formatter_right = ScalarFormatter(useOffset=False, useMathText=False)
            formatter_right.set_scientific(False)

            x_values = agg[x_column].to_numpy(dtype=float)
            widths = _compute_log_widths(x_values) if use_log_x else 0.8

            ax.bar(x_values, agg[count_column], color=palettes["bar"],
                   edgecolor="none", alpha=0.9, width=widths, align="center")
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title_template.format(chain=chain.title()))
            if use_log_x:
                ax.set_xscale("log")
            ax.set_xlim(x_min, x_upper_bound)
            # Use log-scale y-axis for counts if we have positive values
            if global_y_min_positive is not None and global_y_min_positive > 0:
                ax.set_yscale("log")
                # Avoid y-limits inversion: ensure upper > lower
                y_lower = global_y_min_positive
                if y_limit <= y_lower:
                    y_limit = y_lower * 1.1
                ax.set_ylim(y_lower, y_limit)
            else:
                ax.set_ylim(0, y_limit)
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
            # Only apply ScalarFormatter on linear scale
            if ax.get_yscale() == "linear":
                ax.yaxis.set_major_formatter(formatter_left)

            ax_cdf = ax.twinx()
            ax_cdf.plot(x_values, agg["cdf"], color=palettes["cdf"],
                        linewidth=1.2, linestyle="-")
            ax_cdf.set_ylabel(cdf_label)
            ax_cdf.set_ylim(0, 1.05)
            if use_log_x:
                ax_cdf.set_xscale("log")
            ax_cdf.set_xlim(ax.get_xlim())
            ax_cdf.yaxis.set_major_formatter(formatter_right)
            ax_cdf.grid(False)

            out_path = PLOT_DIR / filename_template.format(chain=chain)
            fig.tight_layout()
            fig.savefig(out_path, format="pdf")
            plt.close(fig)
        return

    df["bin_label"] = df[x_column].apply(_assign_bin)

    ordered_bins = sorted(df["bin_label"].unique(), key=BIN_ORDER.get)
    if not ordered_bins:
        return

    for chain in CHAINS:
        chain_df = df[df["chain"] == chain]
        if chain_df.empty:
            continue

        agg = (
            chain_df.groupby("bin_label", as_index=False)[count_column]
            .sum()
            .set_index("bin_label")
            .reindex(ordered_bins, fill_value=0)
            .reset_index()
        )
        agg["order"] = agg["bin_label"].map(BIN_ORDER)
        agg = agg.sort_values("order").reset_index(drop=True)
        total = agg[count_column].sum()
        if total > 0:
            agg["cdf"] = agg[count_column].cumsum() / total
        else:
            agg["cdf"] = 0.0

        chain_aggs[chain] = agg
        current_max = agg[count_column].max()
        if current_max > global_y_max:
            global_y_max = current_max

    if not chain_aggs:
        return

    positions = list(range(len(ordered_bins)))
    y_limit = global_y_max * 1.05 if global_y_max > 0 else 1

    for chain, agg in chain_aggs.items():
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        formatter_left = ScalarFormatter(useOffset=False, useMathText=False)
        formatter_left.set_scientific(False)
        formatter_right = ScalarFormatter(useOffset=False, useMathText=False)
        formatter_right.set_scientific(False)

        ax.bar(positions, agg[count_column], color=palettes["bar"],
               edgecolor="none", alpha=0.9, width=0.8)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title_template.format(chain=chain.title()))
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.yaxis.set_major_formatter(formatter_left)
        ax.set_xticks(positions)
        ax.set_xticklabels(agg["bin_label"], rotation=45, ha="right")
        ax.set_xlim(-0.5, len(positions) - 0.5)
        ax.set_ylim(0, y_limit)

        ax_cdf = ax.twinx()
        ax_cdf.plot(positions, agg["cdf"], color=palettes["cdf"],
                    linewidth=1.2, linestyle="-")
        ax_cdf.set_ylabel(cdf_label)
        ax_cdf.set_ylim(0, 1.05)
        ax_cdf.yaxis.set_major_formatter(formatter_right)
        ax_cdf.grid(False)

        out_path = PLOT_DIR / filename_template.format(chain=chain)
        fig.tight_layout()
        fig.savefig(out_path, format="pdf")
        plt.close(fig)


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    # Apply global font sizes after style to ensure our sizes take precedence
    plt.rcParams.update(FONT_SIZES)
    _ensure_plot_dir()

    daily_active_df = _load_csv(FILES["daily_active"])
    plot_daily_metric(
        daily_active_df,
        value_column="active_factory_count",
        title_template="Daily Factory Deployments ({chain})",
        filename_template="daily_factory_deployments_{chain}.pdf",
        color_key="daily_active",
    )

    daily_traces_df = _load_csv(FILES["daily_traces"])
    plot_daily_metric(
        daily_traces_df,
        value_column="create_trace_count",
        title_template="Daily Factory CREATE/CREATE2 Transactions ({chain})",
        filename_template="daily_factory_create_traces_{chain}.pdf",
        color_key="daily_traces",
    )

    bytecode_df = _load_csv(FILES["bytecode"])
    plot_distribution(
        bytecode_df,
        x_column="contract_reuse_count",
        count_column="runtime_group_count",
        title_template="Runtime Bytecode Reuse ({chain})",
        x_label="Contracts per Runtime Bytecode",
        y_label="Runtime Bytecodes",
        cdf_label="CDF (Runtime Bytecodes)",
        filename_template="bytecode_reuse_distribution_{chain}.pdf",
        color_key="bytecode",
        use_log_x=False,
        use_raw_counts=True,
    )

    activity_df = _load_csv(FILES["factory_activity"])
    plot_distribution(
        activity_df,
        x_column="create_count",
        count_column="factory_group_count",
        title_template="Factory CREATE Activity Distribution ({chain})",
        x_label="CREATE/CREATE2 Transactions per Factory",
        y_label="Factory Contracts",
        cdf_label="CDF (Factories)",
        filename_template="factory_activity_distribution_{chain}.pdf",
        color_key="factory_activity",
        use_log_x=False,
        use_raw_counts=True,
    )


if __name__ == "__main__":
    main()
