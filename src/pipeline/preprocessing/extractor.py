import pickle
import pandas as pd
import sqlite3

from src.logging.log_utils import log_function


class DataExtractor:
    def __init__(self) -> None:
        with open("data/experiments_process_and_results.pkl", "rb") as f:
            self.loaded_dict = pickle.load(f)

    def _take_bending_setups(
        self, experiments_process_and_results, machine_part
    ) -> pd.DataFrame:
        """
        Extracts all bending setups to a Pandas DataFrame and adds an 'Experiment_ID' column
        if not already present.
        """
        if machine_part != "bending_setups":
            df_list = []
            for (
                experiment_name,
                experiment_data,
            ) in experiments_process_and_results.items():
                df = experiment_data[machine_part].copy()
                # Always move or insert Experiment_ID as the first column
                if "Experiment_ID" in df.columns:
                    cols = ["Experiment_ID"] + [
                        col for col in df.columns if col != "Experiment_ID"
                    ]
                    df = df[cols]
                else:
                    # Insert new column at position 0
                    df.insert(0, "Experiment_ID", experiment_name)

                df_list.append(df)

            result = pd.concat(df_list, ignore_index=True)
            result["Experiment_ID"] = result["Experiment_ID"].str.replace("Exp_", "")
            return result

        result = pd.concat(
            [
                experiments_process_and_results[experiment_name]["bending_setups"]
                for experiment_name in experiments_process_and_results.keys()
            ],
            ignore_index=True,
        )
        result.columns = result.columns.str.replace("Experiment", "Experiment_ID")
        return result

    def load_dict_getter(self):
        """Getter for self.load_dict
        obj.load_dict_getter"""
        return self.loaded_dict

    @log_function
    def get_part_bending_setups(self, machine_part):
        """Return all bending setups from the loaded dictionary."""
        return self._take_bending_setups(self.loaded_dict, machine_part)

    @log_function
    def get_all_bending_setups(self):
        """Return all bending setups from the loaded dictionary."""
        return {
        "df_arc": self._take_bending_setups(self.loaded_dict, "geometry_data_key_characteristics_arc"),
        "df_lin1": self._take_bending_setups(self.loaded_dict, "geometry_data_key_characteristics_linear_1"),
        "df_lin2": self._take_bending_setups(self.loaded_dict, "geometry_data_key_characteristics_linear_2"),
        "df_stl_arc": self._take_bending_setups(self.loaded_dict, "geometry_data_stl_suitable_arc"),
        "df_stl_lin1": self._take_bending_setups(self.loaded_dict, "geometry_data_stl_suitable_linear_1"),
        "df_stl_lin2": self._take_bending_setups(self.loaded_dict, "geometry_data_stl_suitable_linear_2"),
        "df_machine": self._take_bending_setups(self.loaded_dict, "process_parameters_loads_machine"),
        "df_sensor": self._take_bending_setups(self.loaded_dict, "process_parameters_loads_sensor"),
        "df_movements": self._take_bending_setups(self.loaded_dict, "process_parameters_movements"),
        "df_bending": self._take_bending_setups(self.loaded_dict, "bending_setups"),
    }

    @log_function
    def get_experiment_by_number(self, experiment_number):
        """Return a specific experiment as a dictionary given its number."""
        return self.loaded_dict.get(f"Exp_{experiment_number}")

    @log_function
    def get_experiment_keys(self):
        """Print keys of an experiment dictionary with numbering."""
        for key in list(self.loaded_dict.get(f"Exp_{2}")):
            print(key)
            
    @log_function
    def save_to_sqlite(self, db_path="data/experiments.db"):
        """
        Stores all bending setups and machine part data into an SQLite database.
        Each machine part gets its own table.
        """
        all_data = self.get_all_bending_setups()  # get dictionary of DataFrames
        
        # Connect to SQLite database (creates it if it doesn't exist)
        conn = sqlite3.connect(db_path)
        
        try:
            for table_name, df in all_data.items():
                # Ensure column names are valid for SQLite
                df.columns = [col.replace(" ", "_") for col in df.columns]
                
                # Store each DataFrame as a table, replace if table exists
                df.to_sql(table_name, conn, if_exists="replace", index=False)
                print(f"Stored '{table_name}' in SQLite database.")
        finally:
            conn.close()
            print(f"All data saved to {db_path}.")
            
