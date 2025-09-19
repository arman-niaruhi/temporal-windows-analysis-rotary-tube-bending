import os
import matplotlib.pyplot as plt
import seaborn as sns
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
    def get_bending_setup(self, experiment_ids: list[int] | None = None):
        """
        Retrieve bending setup information from the stored DataFrame.

        Args:
            experiment_ids (list[int] | None): Optional list of Experiment_IDs to filter.
                                            If None, returns all bending setup data.

        Returns:
            pd.DataFrame: Filtered bending setup data for the specified experiments.
        """
        if experiment_ids is None:
            return self.df_bending

        return self.df_bending[self.df_bending["Experiment_ID"].isin(experiment_ids)]

    @log_function
    def get_geometry_data(self, experiment_ids: list[int] | None = None):
        """
        Retrieve geometry data, optionally filtered by experiment IDs.

        Args:
            experiment_ids (list[int] | None): List of Experiment_IDs to filter.
                                            If None, returns all data.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                df_arc, df_lin1, df_lin2, linear_df (lin1+lin2), all_geometry_data (arc+lin1+lin2)
        """
        if experiment_ids is None:
            df_arc, df_lin1, df_lin2 = self.df_arc, self.df_lin1, self.df_lin2
        else:
            # Filter each DataFrame by experiment_ids
            df_arc = self.df_arc[self.df_arc["Experiment_ID"].isin(experiment_ids)]
            df_lin1 = self.df_lin1[self.df_lin1["Experiment_ID"].isin(experiment_ids)]
            df_lin2 = self.df_lin2[self.df_lin2["Experiment_ID"].isin(experiment_ids)]

        # Combine lin1 and lin2
        linear_df = pd.concat([df_lin1, df_lin2], axis=1)

        # Combine arc, lin1, and lin2
        all_geometry_data = pd.concat([df_arc, df_lin1, df_lin2], axis=1)

        return df_arc, df_lin1, df_lin2, linear_df, all_geometry_data

    @log_function
    def get_process_data(self, experiment_ids: list[int] | None = None):
        """
        Retrieve machine, movements, and sensor data, optionally filtered by experiment IDs.

        Args:
            experiment_ids (list[int] | None): List of Experiment_IDs to filter.
                                            If None, returns all data.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                df_machine_and_movement, df_sensor, df_machine, df_movements
        """
        if experiment_ids is None:
            df_machine, df_movements, df_sensor = (
                self.df_machine,
                self.df_movements,
                self.df_sensor,
            )
        else:
            # Filter each DataFrame by experiment_ids
            df_machine = self.df_machine[
                self.df_machine["Experiment_ID"].isin(experiment_ids)
            ]
            df_movements = self.df_movements[
                self.df_movements["Experiment_ID"].isin(experiment_ids)
            ]
            df_sensor = self.df_sensor[
                self.df_sensor["Experiment_ID"].isin(experiment_ids)
            ]

        # Remove 'Experiment_ID' from movements
        df_movements_wo_experiment = df_movements.drop(columns=["Experiment_ID"])

        # Combine machine and movements
        df_machine_and_movement = pd.concat(
            [df_machine, df_movements_wo_experiment], axis=1
        )

        return df_machine_and_movement, df_sensor, df_machine, df_movements

    @log_function
    def delete_failed_experiment(self, failed_experiment: list[int]) -> None:
        """
        Remove rows corresponding to failed experiments from all relevant DataFrames.

        Args:
            failed_experiment (list[int]): List of Experiment_IDs to remove from the DataFrames.

        Returns:
            None: The method updates the DataFrame attributes in place and does not return anything.
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
            df = df.loc[~df["Experiment_ID"].isin(failed_experiment)]
            setattr(self, attr, df)

    @log_function
    def eliminate_column(self, df_name: str, column_name: str) -> None:
        """
        Delete a specific column from a specified DataFrame attribute.

        Args:
            df_name (str): Name of the DataFrame attribute in the class.
            column_name (str): Name of the column to remove from the DataFrame.

        Returns:
            None: Updates the DataFrame attribute in place and logs the operation.
                If the DataFrame or column does not exist, a warning is logged.
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
        """
        Ensure proper data types and indexing for all relevant DataFrames.

        Converts 'Time_[s]' to float and sets it as the index if present.
        Converts 'Experiment_ID' and other columns to numeric types, coercing errors.
        Updates the DataFrame attributes in place.

        Args:
            None

        Returns:
            None: The method modifies the DataFrame attributes of the class directly.
        """
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

            # Columns to convert (exclude Experiment_ID and Time_[s])
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
        Normalize numeric columns in all relevant DataFrames to the range [0, 1].

        Numeric columns are scaled using min-max normalization, excluding
        'Experiment_ID' and 'Angle[degree]ORDistance[mm]'. Updates the DataFrames
        in place and logs the number of columns normalized for each DataFrame.

        Args:
            None

        Returns:
            None: The method modifies the DataFrame attributes of the class directly.
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

            # Select numeric columns excluding 'Experiment_ID' and 'Angle[degree]ORDistance[mm]'
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

    @log_function
    def nan_handler(self):
        """
        Handle NaN values in all relevant DataFrames by dropping columns that are completely NaN.

        Numeric columns are identified (excluding 'Experiment_ID' and 'Angle[degree]ORDistance[mm]')
        and all-NaN columns are removed. Updates the DataFrames in place and logs the actions
        performed for each DataFrame.

        Args:
            None

        Returns:
            None: The method modifies the DataFrame attributes of the class directly.
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

            # Select numeric columns excluding 'Experiment_ID' and 'Angle[degree]ORDistance[mm]'
            numeric_cols = df.select_dtypes(include="number").columns.difference(
                ["Experiment_ID", "Angle[degree]ORDistance[mm]"]
            )

            if len(numeric_cols) == 0:
                logger.info(f"No numeric columns to process in '{attr_name}'.")
                continue

            # Find columns that are completely NaN
            nan_cols = df.columns[df.isna().all()].tolist()

            if nan_cols:
                df = df.drop(columns=nan_cols)
                logger.info(
                    f"Dropped {len(nan_cols)} all-NaN columns from '{attr_name}': {nan_cols}"
                )
            else:
                logger.info(f"No all-NaN columns found in '{attr_name}'.")

            setattr(self, attr_name, df)
            logger.info(
                f"Processed {len(numeric_cols)} numeric columns in '{attr_name}' (excluding 'Experiment_ID')."
            )

    @log_function
    def save_correlation_matrices(
        self, tables: list[str], output_dir: str = "results/correlations"
    ) -> None:
        """
        Compute and save correlation matrices for the specified DataFrames.

        Correlation matrices are computed for numeric columns (excluding 'Experiment_ID'
        and 'Angle[degree]ORDistance[mm]'), saved as CSV files, and visualized
        as heatmap plots using Matplotlib/Seaborn.

        Args:
            tables (list[str]): List of DataFrame attribute names to compute correlations for.
            output_dir (str): Directory where correlation matrices and plots will be saved. Defaults to 'correlations'.

        Returns:
            None: Saves correlation matrices as CSV files and heatmap plots, logs the process.
        """
        os.makedirs(output_dir, exist_ok=True)

        for attr_name in tables:
            if not hasattr(self, attr_name):
                logger.warning(f"DataFrame '{attr_name}' not found in DataTransformer.")
                continue

            df = getattr(self, attr_name)

            # Select numeric columns excluding 'Experiment_ID' and 'Angle[degree]ORDistance[mm]'
            numeric_cols = df.select_dtypes(include="number").columns.difference(
                ["Experiment_ID", "Angle[degree]ORDistance[mm]"]
            )

            if len(numeric_cols) == 0:
                logger.info(f"No numeric columns to compute correlation in '{attr_name}'.")
                continue

            corr_matrix = df[numeric_cols].corr()

            # Save correlation matrix as CSV
            csv_path = os.path.join(output_dir, f"{attr_name}_correlation.csv")
            corr_matrix.to_csv(csv_path)
            logger.info(f"Saved correlation matrix for '{attr_name}' to '{csv_path}'.")

            # Compute average absolute correlation per column
            abs_corr = corr_matrix.abs()
            # Ignore self-correlation
            abs_corr.values[[range(len(abs_corr))]*2] = 0
            avg_corr = abs_corr.mean().sort_values()

            # Save least correlated columns to CSV
            least_corr_path = os.path.join(output_dir, f"{attr_name}_least_correlated.csv")
            avg_corr.to_csv(least_corr_path, header=["avg_abs_correlation"])
            logger.info(
                f"Saved least correlated columns for '{attr_name}' to '{least_corr_path}'."
            )
            logger.info(f"Top 5 least correlated columns in '{attr_name}':\n{avg_corr.head()}")

            # Save heatmap plot
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
            plt.title(f"Correlation Matrix: {attr_name}", fontsize=14)
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()

            plot_path = os.path.join(output_dir, f"{attr_name}_correlation.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()
            logger.info(f"Saved correlation heatmap for '{attr_name}' to '{plot_path}'.")