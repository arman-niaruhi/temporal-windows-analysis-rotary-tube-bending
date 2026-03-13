from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.logging.log_utils import log_function


DEFAULT_INDEX_TABLES = [
    "machine_and_movement",
    "movement",
    "sensor",
    "machine",
]


class DataLoader:
    def __init__(self, storage_path: str) -> None:
        """
        Initialize the loader with a directory used to store ETL CSV tables.

        If an old SQLite-like path such as ``tube_geometry.db`` is passed, the
        suffix is stripped and the CSV tables are stored in ``tube_geometry/``.
        """
        self.storage_path = storage_path
        self.storage_dir = self._resolve_storage_dir(storage_path)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resolve_storage_dir(storage_path: str) -> Path:
        path = Path(storage_path)
        if path.suffix.lower() == ".db":
            return path.with_suffix("")
        return path

    @staticmethod
    def _prepare_frame_for_write(df: pd.DataFrame, include_index: bool) -> pd.DataFrame:
        if include_index:
            return df.reset_index()
        return df.copy()

    @staticmethod
    def _restore_index(df: pd.DataFrame, table_name: str, store_index_tables: List[str]) -> pd.DataFrame:
        if table_name in store_index_tables and "Time_[s]" in df.columns:
            df = df.set_index("Time_[s]")
        return df

    def _table_path(self, table_name: str) -> Path:
        return self.storage_dir / f"{table_name}.csv"

    def _list_tables(self) -> List[str]:
        return sorted(path.stem for path in self.storage_dir.glob("*.csv"))

    @log_function
    def store_to_csv(
        self,
        dataframes: Optional[Dict[str, pd.DataFrame]] = None,
        store_index_tables: Optional[List[str]] = None,
    ) -> None:
        """Save multiple DataFrames as CSV files inside the storage directory."""
        if not dataframes:
            print("No dataframes provided. Nothing to store.")
            return

        if store_index_tables is None:
            store_index_tables = []

        for table_name, df in dataframes.items():
            if not isinstance(df, pd.DataFrame):
                print(f"Skipped {table_name}: not a valid DataFrame.")
                continue

            include_index = table_name in store_index_tables
            output_df = self._prepare_frame_for_write(df, include_index=include_index)
            output_path = self._table_path(table_name)
            output_df.to_csv(output_path, index=False)
            print(f"Saved table '{table_name}' with shape {df.shape} to CSV: {output_path}")

    @log_function
    def load_all_data_from_csv(
        self,
        store_index_tables: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Load all CSV tables from the storage directory into a dataframe dict."""
        if store_index_tables is None:
            store_index_tables = DEFAULT_INDEX_TABLES

        dataframes = {}
        for table in self._list_tables():
            df = pd.read_csv(self._table_path(table))
            dataframes[table] = self._restore_index(df, table, store_index_tables)

        return dataframes

    @log_function
    def load_data_by_experiment_from_csv(
        self,
        experiment_id: int,
        store_index_tables: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Load all CSV tables and keep only rows for the requested experiment."""
        if store_index_tables is None:
            store_index_tables = DEFAULT_INDEX_TABLES

        dataframes = {}
        for table in self._list_tables():
            df = pd.read_csv(self._table_path(table))
            if "Experiment_ID" in df.columns:
                df = df[df["Experiment_ID"].astype(str) == str(experiment_id)].copy()
            dataframes[table] = self._restore_index(df, table, store_index_tables)

        return dataframes

    @log_function
    def load_experiment_ids_from_csv(self):
        """Load unique experiment IDs from the machine_and_movement CSV."""
        table_path = self._table_path("machine_and_movement")
        df = pd.read_csv(table_path)
        return df["Experiment_ID"].unique()