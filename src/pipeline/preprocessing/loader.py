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
            "df_movement",
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
            "df_movement",
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
    
    @log_function
    def load_experiment_ids_from_sqlite(
        self):
        """
        Load Experiment_ID from the SQLite database.

        Returns:
            pd.DataFrame: panda dataframe of experiment ids.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        df = pd.read_sql_query(f"SELECT * FROM machine_and_movement", conn)
        conn.close()
        experiment_ids = df["Experiment_ID"].unique()
        return experiment_ids