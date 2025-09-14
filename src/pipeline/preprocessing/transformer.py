import pandas as pd
from src.logging.log_utils import log_function, logger


class DataTransformer:
    def __init__(
        self,
        df_arc: pd.DataFrame,
        df_lin1: pd.DataFrame,
        df_lin2: pd.DataFrame,
        df_stl_arc: pd.DataFrame,
        df_stl_lin1: pd.DataFrame,
        df_stl_lin2: pd.DataFrame,
        df_machine: pd.DataFrame,
        df_sensor: pd.DataFrame,
        df_movements: pd.DataFrame,
        df_bending: pd.DataFrame,
    ) -> None:

        # Save as attributes
        self.df_arc = df_arc
        self.df_lin1 = df_lin1
        self.df_lin2 = df_lin2
        self.df_stl_arc = df_stl_arc
        self.df_stl_lin1 = df_stl_lin1
        self.df_stl_lin2 = df_stl_lin2
        self.df_machine = df_machine
        self.df_sensor = df_sensor
        self.df_movements = df_movements
        self.df_bending = df_bending

    @log_function
    def get_geometry_data(self, experiment_ids: list[int] | None = None):
        if experiment_ids is not None:
            df_arc = self.df_arc[self.df_arc["Experiment_ID"].isin(experiment_ids)]
            df_lin1 = self.df_lin1[self.df_lin1["Experiment_ID"].isin(experiment_ids)]
            df_lin2 = self.df_lin2[self.df_lin2["Experiment_ID"].isin(experiment_ids)]
        else:
            df_arc, df_lin1, df_lin2 = self.df_arc, self.df_lin1, self.df_lin2

        linear_df = pd.concat([df_lin1, df_lin2], axis=1)
        all_geometry_data = pd.concat([df_arc, df_lin1, df_lin2], axis=1)

        return df_arc, df_lin1, df_lin2, linear_df, all_geometry_data

    @log_function
    def get_process_data(self, experiment_ids: list[int] | None = None):
        if experiment_ids is not None:
            df_machine = self.df_machine[
                self.df_machine["Experiment_ID"].isin(experiment_ids)
            ]
            df_movements = self.df_movements[
                self.df_movements["Experiment_ID"].isin(experiment_ids)
            ]
            df_sensor = self.df_sensor[
                self.df_sensor["Experiment_ID"].isin(experiment_ids)
            ]
        else:
            df_machine, df_movements, df_sensor = (
                self.df_machine,
                self.df_movements,
                self.df_sensor,
            )

        df_movements_wo_experiment = df_movements.drop(columns=["Experiment_ID"])
        df_machine_and_movement = pd.concat(
            [df_machine, df_movements_wo_experiment], axis=1
        )

        outputs = [
            ("df_machine_and_movement", df_machine_and_movement),
            ("df_sensor", df_sensor),
            ("df_machine", df_machine),
            ("df_movements", df_movements),
        ]

        for df_name, df in outputs:
            nan_counts = {col: df[col].isna().sum() for col in df.columns}
            total_nans = df.isna().sum().sum()
            log_text = (
                f"\nHeaders of {df_name}: {list(df.columns)}\n"
                f"NaN counts per column in {df_name}: {nan_counts}\n"
                f"Total NaNs in {df_name}: {total_nans}"
            )
            logger.info(log_text)

        return df_machine_and_movement, df_sensor, df_machine, df_movements

    @log_function
    def delete_failed_experiment(self, failed_experiment: list[int]) -> None:
        """
        Delete rows corresponding to failed experiments from all DataFrames.

        Parameters
        ----------
        failed_experiment : list[int]
            List of row indices (experiment IDs) to remove.
        """
        for attr in [
            "df_arc",
            "df_lin1",
            "df_lin2",
            "df_stl_arc",
            "df_stl_lin1",
            "df_stl_lin2",
            "df_machine",
            "df_sensor",
            "df_movements",
            "df_bending",
        ]:
            df = getattr(self, attr)
            # Filter out rows where Experiment_ID is in failed_experiment
            df = df.loc[~df["Experiment_ID"].isin(failed_experiment)]
            # Set the filtered DataFrame back to the attribute
            setattr(self, attr, df)

    @log_function
    def eliminate_column(self, df_name: str, column_name: str) -> None:
        """
        Delete a column from one of the DataFrames.

        Parameters
        ----------
        df_name : str
            The attribute name of the DataFrame (e.g., 'df_machine', 'df_sensor').
        column_name : str
            The column to be deleted from the DataFrame.
        """
        if not hasattr(self, df_name):
            logger.warning(f"DataFrame '{df_name}' not found in DataTransformer.")
            return

        df = getattr(self, df_name)

        if column_name not in df.columns:
            logger.warning(f"Column '{column_name}' not found in {df_name}.")
            return

        df = df.drop(columns=[column_name])
        setattr(self, df_name, df)

        logger.info(f"Deleted column '{column_name}' from DataFrame '{df_name}'.")

    @log_function
    def check_quality(self):
        for df_name in [
            "df_machine",
            "df_sensor",
            "df_movements",
            "df_arc",
            "df_lin1",
            "df_lin2",
            "df_stl_arc",
            "df_stl_lin1",
            "df_stl_lin2",
        ]:
            df = getattr(self, df_name)

            # Check if the column exists
            if "Time_[s]" in df.columns:
                df["Time_[s]"] = df["Time_[s]"].astype(float)
                df.set_index("Time_[s]", inplace=True)

            df["Experiment_ID"] = pd.to_numeric(df["Experiment_ID"], errors="coerce")

            # Columns to plot (exclude Experiment_ID and Time_[s])
            sensor_cols = [
                col for col in df.columns if col not in ["Experiment_ID", "Time_[s]"]
            ]

            numeric_cols = []
            for col in sensor_cols:
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    if df[col].notna().any():
                        numeric_cols.append(col)
                except Exception:
                    continue

            # Update the DataFrame back to self
            setattr(self, df_name, df)

    @log_function
    def normalize_data(self):
        """
        Normalize all numeric columns in all DataFrames to [0, 1],
        excluding 'Experiment_ID'.
        """
        for attr_name in [
            "df_arc",
            "df_lin1",
            "df_lin2",
            "df_stl_arc",
            "df_stl_lin1",
            "df_stl_lin2",
            "df_machine",
            "df_sensor",
            "df_movements",
            "df_bending",
        ]:
            df = getattr(self, attr_name)

            # Select numeric columns excluding 'Experiment_ID'
            numeric_cols = df.select_dtypes(include="number").columns.difference(
                ["Experiment_ID", "Angle[degree]ORDistance[mm]"]
            )

            if len(numeric_cols) == 0:
                logger.info(f"No numeric columns to normalize in '{attr_name}'.")
                continue

            df.loc[:, numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (
                df[numeric_cols].max() - df[numeric_cols].min()
            )

            setattr(self, attr_name, df)
            logger.info(
                f"Normalized {len(numeric_cols)} numeric columns in '{attr_name}' (excluding 'Experiment_ID')."
            )
