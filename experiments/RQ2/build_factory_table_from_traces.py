#!/usr/bin/env python3
"""Build a unified BigQuery table of factory contract creations for RQ2.

This script scans the Google BigQuery public datasets for Ethereum and
Polygon traces, extracts successful CREATE operations initiated by smart
contracts (trace depth > 0), and persists the results into a dedicated
factory table inside the research project.

The resulting table is the data foundation for daily deployment counts,
transaction activity analysis, and bytecode deduplication studies that
follow in the experiment pipeline.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List

from google.cloud import bigquery
from google.cloud.exceptions import NotFound

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


@dataclass
class ChainDatasetConfig:
    """Minimal configuration required to query a chain's traces table."""

    chain_name: str
    dataset_name: str
    cutoff_date: str

    @property
    def traces_table(self) -> str:
        return f"{self.dataset_name}.traces"


class FactoryTableBuilder:
    """Orchestrates dataset/table creation and data refresh operations."""

    def __init__(self, config_path: str, table_name: str = "rq2_factory_creations") -> None:
        self.config_path = config_path
        self.table_name = table_name

        with open(config_path, "r", encoding="utf-8") as fp:
            config = json.load(fp)

        self.project_id: str = config["project_id"]
        self.dataset_id: str = config["result_dataset"]
        cutoff_date: str = config.get("cutoff_date", "2025-06-01")

        chains_cfg: Dict[str, Dict] = config.get("chains", {})
        if not chains_cfg:
            raise ValueError("No chain configuration found in config JSON")

        self.chains: Dict[str, ChainDatasetConfig] = {
            name: ChainDatasetConfig(
                chain_name=details["chain_name"],
                dataset_name=details["dataset_name"],
                cutoff_date=cutoff_date,
            )
            for name, details in chains_cfg.items()
        }

        self.client = bigquery.Client(project=self.project_id)
        self.logger = logging.getLogger("FactoryTableBuilder")

    # ------------------------------------------------------------------
    # BigQuery infrastructure helpers
    # ------------------------------------------------------------------

    def ensure_dataset(self) -> None:
        dataset_ref = bigquery.DatasetReference(self.project_id, self.dataset_id)
        try:
            self.client.get_dataset(dataset_ref)
            self.logger.info("Dataset %s already exists", dataset_ref.path)
        except NotFound:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = "US"
            dataset.description = "Factory contract analysis artifacts"
            self.client.create_dataset(dataset)
            self.logger.info("Created dataset %s", dataset_ref.path)

    def _schema_signature(self, schema: List[bigquery.SchemaField]) -> List[tuple]:
        return [(field.name, field.field_type, field.mode) for field in schema]

    def ensure_table(self) -> None:
        table_ref = bigquery.TableReference(
            bigquery.DatasetReference(self.project_id, self.dataset_id),
            self.table_name,
        )

        schema = [
            bigquery.SchemaField("chain", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("factory_address", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("created_contract_address", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("transaction_hash", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("block_number", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("block_timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("block_date", "DATE", mode="REQUIRED"),
            bigquery.SchemaField("trace_address", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("trace_depth", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("value", "NUMERIC", mode="NULLABLE"),
            bigquery.SchemaField("init_code", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("runtime_code", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("gas", "INT64", mode="NULLABLE"),
            bigquery.SchemaField("gas_used", "INT64", mode="NULLABLE"),
            bigquery.SchemaField("status", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("ingested_at", "TIMESTAMP", mode="REQUIRED"),
        ]

        table = bigquery.Table(table_ref, schema=schema)
        table.description = (
            "Successful CREATE traces (depth > 0) for Ethereum/Polygon factory activity"
        )
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="block_date",
        )
        table.clustering_fields = ["chain", "factory_address"]

        try:
            existing = self.client.get_table(table_ref)
            if self._schema_signature(existing.schema) != self._schema_signature(schema):
                self.logger.info(
                    "Schema mismatch detected for %s.%s; recreating table",
                    self.dataset_id,
                    self.table_name,
                )
                self.client.delete_table(table_ref)
                self.client.create_table(table)
                self.logger.info(
                    "Recreated table %s.%s with updated schema",
                    self.dataset_id,
                    self.table_name,
                )
            else:
                self.logger.info(
                    "Table %s.%s already exists", self.dataset_id, self.table_name
                )
        except NotFound:
            self.client.create_table(table)
            self.logger.info(
                "Created table %s.%s", self.dataset_id, self.table_name
            )

    # ------------------------------------------------------------------
    # Data refresh
    # ------------------------------------------------------------------

    def refresh_chain(self, chain_key: str) -> None:
        if chain_key not in self.chains:
            raise ValueError(f"Unknown chain key: {chain_key}")

        cfg = self.chains[chain_key]
        table_fqn = (
            f"`{self.project_id}.{self.dataset_id}.{self.table_name}`"
        )

        query = f"""
        DELETE FROM {table_fqn}
        WHERE chain = @chain_name;

        INSERT INTO {table_fqn} (
            chain,
            factory_address,
            created_contract_address,
            transaction_hash,
            block_number,
            block_timestamp,
            block_date,
            trace_address,
            trace_depth,
            value,
            init_code,
            runtime_code,
            gas,
            gas_used,
            status,
            ingested_at
        )
        SELECT
            @chain_name AS chain,
            LOWER(from_address) AS factory_address,
            LOWER(to_address) AS created_contract_address,
            transaction_hash,
            block_number,
            block_timestamp,
            DATE(block_timestamp) AS block_date,
            trace_address,
            IFNULL(ARRAY_LENGTH(SPLIT(NULLIF(trace_address, ''), ',')), 0) AS trace_depth,
            SAFE_CAST(value AS NUMERIC) AS value,
            input AS init_code,
            output AS runtime_code,
            gas,
            gas_used,
            status,
            CURRENT_TIMESTAMP() AS ingested_at
        FROM `{cfg.traces_table}`
        WHERE trace_type = 'create'
          AND status = 1
          AND to_address IS NOT NULL
          AND from_address IS NOT NULL
          AND from_address != '0x0000000000000000000000000000000000000000'
          AND block_timestamp < TIMESTAMP(@cutoff_date)
          AND IFNULL(ARRAY_LENGTH(SPLIT(NULLIF(trace_address, ''), ',')), 0) >= 1
        """

        job_config = bigquery.QueryJobConfig(
            use_legacy_sql=False,
            query_parameters=[
                bigquery.ScalarQueryParameter("chain_name", "STRING", cfg.chain_name),
                bigquery.ScalarQueryParameter("cutoff_date", "STRING", cfg.cutoff_date),
            ],
        )

        self.logger.info(
            "Refreshing factory table for %s using %s", cfg.chain_name, cfg.traces_table
        )
        job = self.client.query(query, job_config=job_config)
        job.result()  # Wait for completion
        self.logger.info(
            "Completed refresh for %s (job ID: %s)", cfg.chain_name, job.job_id
        )

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self, chains: Iterable[str]) -> None:
        selected = list(chains)
        if not selected:
            selected = list(self.chains.keys())

        self.ensure_dataset()
        self.ensure_table()

        for chain in selected:
            self.refresh_chain(chain)


# ---------------------------------------------------------------------------
# Script entry point with explicit configuration variables
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
    DESTINATION_TABLE = "rq2_factory_creations"
    TARGET_CHAINS: List[str] = ["ethereum", "polygon"]  # empty list => all chains in config
    LOG_LEVEL = "INFO"

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    builder = FactoryTableBuilder(CONFIG_PATH, table_name=DESTINATION_TABLE)
    builder.run(TARGET_CHAINS)
