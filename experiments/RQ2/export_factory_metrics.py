#!/usr/bin/env python3
"""Export daily and distribution metrics for factory contract activity.

This script reads from the `rq2_factory_creations` BigQuery table (populated by
`build_factory_table_from_traces.py`) and materialises four CSV files required
for the RQ2 analysis:

1. Daily count of active factory contracts per chain (unique factory addresses
   executing CREATE on a given day).
2. Daily count of CREATE traces per chain.
3. Distribution of runtime bytecode reuse among factory-created contracts, with
   cumulative distribution data to support CDF plots.
4. Distribution of CREATE counts per factory contract, with cumulative
   distribution data for CDF plots.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
from google.cloud import bigquery


@dataclass(frozen=True)
class ExportTarget:
    filename: str
    query: str


class FactoryMetricsExporter:
    """Run parameterised BigQuery queries and persist CSV outputs."""

    def __init__(self, config_path: str, table_name: str = "rq2_factory_creations") -> None:
        with open(config_path, "r", encoding="utf-8") as fh:
            config = json.load(fh)

        self.project_id: str = config["project_id"]
        self.dataset_id: str = config["result_dataset"]
        self.cutoff_date: str = config["cutoff_date"]
        self.table_fqn: str = f"`{self.project_id}.{self.dataset_id}.{table_name}`"

        self.client = bigquery.Client(project=self.project_id)

    def _run_query(self, query: str) -> pd.DataFrame:
        job = self.client.query(query)
        return job.result().to_dataframe(create_bqstorage_client=False)

    def export(self, targets: List[ExportTarget], output_dir: Path) -> Dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        exported: Dict[str, Path] = {}

        for target in targets:
            df = self._run_query(target.query)
            output_path = output_dir / target.filename
            df.to_csv(output_path, index=False)
            exported[target.filename] = output_path

        return exported


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
DEST_TABLE = "rq2_factory_creations"
OUTPUT_DIR = Path(os.path.dirname(__file__)) / "outputs"

QUERY_TARGETS = [
    ExportTarget(
        filename="daily_active_factories.csv",
        query="""
            SELECT
              chain,
              block_date,
              COUNT(DISTINCT factory_address) AS active_factory_count
            FROM {table}
            WHERE block_date < DATE '{cutoff}'
            GROUP BY chain, block_date
            ORDER BY chain, block_date
        """,
    ),
    ExportTarget(
        filename="daily_factory_create_traces.csv",
        query="""
            SELECT
              chain,
              block_date,
              COUNT(*) AS create_trace_count
            FROM {table}
            WHERE block_date < DATE '{cutoff}'
            GROUP BY chain, block_date
            ORDER BY chain, block_date
        """,
    ),
    ExportTarget(
        filename="bytecode_reuse_distribution.csv",
        query="""
            WITH runtime_counts AS (
              SELECT
                chain,
                runtime_code,
                COUNT(*) AS occurrences
              FROM {table}
              WHERE runtime_code IS NOT NULL AND runtime_code != ''
              GROUP BY chain, runtime_code
            ),
            group_summary AS (
              SELECT
                chain,
                occurrences,
                COUNT(*) AS runtime_group_count,
                SUM(occurrences) AS contracts_in_group
              FROM runtime_counts
              GROUP BY chain, occurrences
            ),
            ordered AS (
              SELECT
                chain,
                occurrences,
                runtime_group_count,
                contracts_in_group,
                SUM(contracts_in_group) OVER (PARTITION BY chain ORDER BY occurrences) AS cumulative_contracts,
                SUM(contracts_in_group) OVER (PARTITION BY chain) AS total_contracts,
                SUM(runtime_group_count) OVER (PARTITION BY chain ORDER BY occurrences) AS cumulative_runtime_groups,
                SUM(runtime_group_count) OVER (PARTITION BY chain) AS total_runtime_groups
              FROM group_summary
            )
            SELECT
              chain,
              occurrences AS contract_reuse_count,
              runtime_group_count,
              contracts_in_group,
              cumulative_contracts,
              total_contracts,
              SAFE_DIVIDE(cumulative_contracts, total_contracts) AS cumulative_contract_fraction,
              cumulative_runtime_groups,
              total_runtime_groups,
              SAFE_DIVIDE(cumulative_runtime_groups, total_runtime_groups) AS cumulative_runtime_fraction
            FROM ordered
            ORDER BY chain, contract_reuse_count
        """,
    ),
    ExportTarget(
        filename="factory_activity_distribution.csv",
        query="""
            WITH factory_activity AS (
              SELECT
                chain,
                factory_address,
                COUNT(*) AS create_count
              FROM {table}
              GROUP BY chain, factory_address
            ),
            grouped AS (
              SELECT
                chain,
                create_count,
                COUNT(*) AS factory_group_count
              FROM factory_activity
              GROUP BY chain, create_count
            ),
            ordered AS (
              SELECT
                chain,
                create_count,
                factory_group_count,
                SUM(factory_group_count) OVER (PARTITION BY chain ORDER BY create_count) AS cumulative_factories,
                SUM(factory_group_count) OVER (PARTITION BY chain) AS total_factories
              FROM grouped
            )
            SELECT
              chain,
              create_count,
              factory_group_count,
              cumulative_factories,
              total_factories,
              SAFE_DIVIDE(cumulative_factories, total_factories) AS cumulative_fraction
            FROM ordered
            ORDER BY chain, create_count
        """,
    ),
]


if __name__ == "__main__":
    exporter = FactoryMetricsExporter(CONFIG_PATH, table_name=DEST_TABLE)
    parametrised_targets = [
        ExportTarget(filename=target.filename, query=target.query.format(cutoff=exporter.cutoff_date, table=exporter.table_fqn))
        for target in QUERY_TARGETS
    ]
    exporter.export(parametrised_targets, OUTPUT_DIR)
