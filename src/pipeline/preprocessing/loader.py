import sqlite3
import pandas as pd
from typing import Dict

from src.logging.log_utils import log_function


class DataLoader:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    @log_function
    def store_to_sqlite(self, dataframes: dict, store_index_tables: list) -> None:
        """
        Save multiple DataFrames to a SQLite database and create indexes on all columns.

        Each DataFrame in `dataframes` is saved to a table named after its key.
        If a table is listed in `store_index_tables`, its index is stored as a column.
        Indexes are created on all columns to optimize queries. Existing tables are replaced.

        Args:
            dataframes (dict): Dictionary where keys are table names and values are DataFrames to save.
            store_index_tables (list): List of table names for which the DataFrame index should be stored as a column.

        Returns:
            None: Saves the DataFrames to the SQLite database specified by `self.db_path` and creates indexes.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if store_index_tables is None:
            store_index_tables = []

        for table_name, df in dataframes.items():
            # Case Distinction:
            #           If table should store index, reset index as a column named "index"
            if table_name in store_index_tables:
                index = True
            else:
                index = False

            # Save DataFrame to SQLite
            df.to_sql(table_name, conn, index=index, if_exists="replace")
            print(f"Saved table '{table_name}' with shape {df.shape} to SQLite.")

            # Create indexes on all columns to access faster to the data
            for col in df.columns:
                index_name = f"idx_{table_name}_{col}"
                try:
                    cursor.execute(
                        f'CREATE INDEX IF NOT EXISTS "{index_name}" ON "{table_name}"("{col}")'
                    )
                    print(f"Created index on {table_name}({col})")
                except sqlite3.OperationalError as e:
                    print(f"Failed to create index on {table_name}({col}): {e}")

        conn.commit()
        conn.close()

    @log_function
    def load_all_data_from_sqlite(self, store_index_tables=[
            "df_machine",
            "df_sensor",
            "df_machine_and_movement",
            "df_movements",
        ]) -> Dict[str, pd.DataFrame]:
        """
        Load all tables from the SQLite database into a dictionary of DataFrames.

        Tables listed in `store_index_tables` will have 'Time_[s]' set as the index
        if that column exists. Each table name is used as the key in the returned dictionary.

        Args:
            store_index_tables (list): List of table names for which 'Time_[s]' should be used as index.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping table names to their respective DataFrames.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        dataframes = {}
        for table in tables:
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)

            if table in store_index_tables and "Time_[s]" in df.columns:
                df.set_index("Time_[s]", inplace=True)

            dataframes[table] = df

        conn.close()
        return dataframes

    @log_function
    def load_data_by_experiment_from_sqlite(
        self,
        experiment_id,
        store_index_tables=[
            "df_machine",
            "df_sensor",
            "df_machine_and_movement",
            "df_movements",
        ],
    ) -> Dict[str, pd.DataFrame]:
        """
        Load tables from the SQLite database filtered by a specific Experiment_ID.

        Only rows with the specified `experiment_id` are loaded. For tables listed
        in `store_index_tables`, 'Time_[s]' is set as the index if present.
        Returns a dictionary mapping table names to their filtered DataFrames.

        Args:
            experiment_id (int): The Experiment_ID to filter rows by.
            store_index_tables (list): List of table names for which 'Time_[s]' should be used as index.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping table names to their filtered DataFrames.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        dataframes = {}
        for table in tables:
            df = pd.read_sql_query(
                f"SELECT * FROM {table} WHERE Experiment_ID = ?",
                conn,
                params=(experiment_id,),
            )
            if table in store_index_tables and "Time_[s]" in df.columns:
                df.set_index("Time_[s]", inplace=True)
            dataframes[table] = df

        conn.close()
        return dataframes

    def store_to_csv(self,
        cols_to_match: list[str],
        load_setup: pd.DataFrame | None,
        selected_dfs_features: list[pd.DataFrame],
        selected_dfs_target: list[pd.DataFrame],
        feature_file: str = "features.csv", 
        target_file: str = "targets.csv"
    ):
        """
        Store filtered machine and arc data to CSV files.

        Args:
            cols_to_match (list[str]): Columns to group by for duplicate detection.
            load_setup (pd.DataFrame | None): Setup DataFrame containing experiment information.
            selected_dfs_features (list[pd.DataFrame]): List of feature DataFrames to filter.
            selected_dfs_target (list[pd.DataFrame]): List of target DataFrames to filter.
            feature_file (str): Path to save the features CSV.
            target_file (str): Path to save the targets CSV.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame] | None:
                (df_machine_filtered, df_arc_filtered) – the filtered DataFrames, or None if no filtering is done.
        """

        # If no setup data is provided, return immediately
        if load_setup is None or load_setup.empty:
            print("load_setup is None or empty — skipping filtering.")
            return

        # If no matching columns provided, skip filtering and save as-is
        if len(cols_to_match) == 0:
            print("cols_to_match is empty — saving DataFrames without filtering.")
            for df in selected_dfs_features:
                df.to_csv(feature_file, index=False)
            for df in selected_dfs_target:
                df.to_csv(target_file, index=False)
            return

        # Group by specified columns and collect Experiment_IDs
        grouped = load_setup.groupby(cols_to_match)['Experiment_ID'].apply(list)

        # Find duplicate experiment groups
        duplicates_list = [exp_list for exp_list in grouped if len(exp_list) > 1]
        print("Duplicate experiment groups:", duplicates_list)

        # Take only the first experiment ID from each group
        first_experiments = [exp_list[0] for exp_list in grouped]

        # Filter and save
        for df in selected_dfs_features:
            df_filtered = df[df['Experiment_ID'].isin(first_experiments)]
            df_filtered.to_csv(feature_file, index=False)

        for df in selected_dfs_target:
            df_filtered = df[df['Experiment_ID'].isin(first_experiments)]
            df_filtered.to_csv(target_file, index=False)


        