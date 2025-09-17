import sqlite3
import pandas as pd
from typing import Dict

from src.logging.log_utils import log_function

class DataLoader:
    def __init__(self, db_path: str):
        self.db_path = db_path

    @log_function
    def save_to_sqlite(self, dataframes: Dict[str, pd.DataFrame]):
        """
        Save multiple pandas DataFrames to SQLite database and create indexes on all columns.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for table_name, df in dataframes.items():
            df.to_sql(table_name, conn, index=False, if_exists="replace")
            print(f"Saved table '{table_name}' with shape {df.shape} to SQLite.")

            for col in df.columns:
                index_name = f"idx_{table_name}_{col}"
                # Wrap table and column names in double quotes to handle special characters
                try:
                    cursor.execute(f'CREATE INDEX IF NOT EXISTS "{index_name}" ON "{table_name}"("{col}")')
                    print(f"Created index on {table_name}({col})")
                except sqlite3.OperationalError as e:
                    print(f"Failed to create index on {table_name}({col}): {e}")


        conn.commit()
        conn.close()

    @log_function
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all tables from the SQLite database into a dictionary of DataFrames.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        dataframes = {}
        for table in tables:
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            dataframes[table] = df

        conn.close()
        return dataframes
    
    @log_function
    def load_data_by_experiment(self, experiment_id) -> Dict[str, pd.DataFrame]:
        """
        Load all tables from the SQLite database into a dictionary of DataFrames.
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
                params=(experiment_id,)
            )
            dataframes[table] = df

        conn.close()
        return dataframes