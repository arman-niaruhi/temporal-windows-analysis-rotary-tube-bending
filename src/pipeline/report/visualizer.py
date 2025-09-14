from src.logging.log_utils import log_function, logger
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import re
import ipywidgets as widgets
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, clear_output


class DataVisulizer:
    def __init__(self) -> None:
        pass

    @log_function
    def multi_sensor_experiment(
        self,
        dfs: list,
        experiment_id: int,
        df_names: list,
        x_axes: list,
        save_fig=True,
        base_path="results/plot_data",
        part_name=None,
    ):
        """
        Plot all sensors for a specific Experiment_ID for multiple DataFrames.
        Each subplot has its own grouped legend inside the global legend box.

        Parameters
        ----------
        dfs : list
            List of DataFrames to plot.
        experiment_id : int
            The Experiment_ID to filter data.
        df_names : list, optional
            Names for each DataFrame (used in subplot titles).
        x_axes : list, optional
            List specifying which column or 'index' to use as x-axis for each DataFrame.
            Defaults to "Time_[s]" if present, otherwise "index".
        save_fig : bool
            Whether to save the plot as an HTML file.
        base_path : str
            Directory to save the figure.
        part_name : str
            Prefix for the saved figure file name.
        """

        def extract_trace_name(col_name: str) -> str:
            parts = col_name.split("_")
            for i in range(len(parts)):
                if re.match(r"^[A-Za-z0-9\-]+\_\[[^]]+\]$", "_".join(parts[i:])):
                    return " ".join(parts[:i]).replace("_", " ")
            return " ".join(parts[:-2]).replace("_", " ")

        # --- Validate df_names ---
        if df_names is None:
            df_names = [f"Dataset {i+1}" for i in range(len(dfs))]
        elif len(df_names) != len(dfs):
            raise ValueError(
                "Number of DataFrame names must match number of DataFrames"
            )

        # --- Validate or auto-generate x_axes ---
        if x_axes is None:
            x_axes = []
            for df in dfs:
                x_axes.append("Time_[s]" if "Time_[s]" in df.columns else "index")
        elif len(x_axes) != len(dfs):
            raise ValueError("Number of x_axes entries must match number of DataFrames")

        # --- Create subplots ---
        fig = make_subplots(
            rows=len(dfs),
            cols=1,
            subplot_titles=[
                f"{name} - Experiment {experiment_id}" for name in df_names
            ],
            vertical_spacing=0.15,
        )

        # --- Process each DataFrame ---
        for i, (df, x_axis_choice, df_name) in enumerate(
            zip(dfs, x_axes, df_names), start=1
        ):
            experiment_df = df[df["Experiment_ID"] == experiment_id]
            if experiment_df.empty:
                raise ValueError(
                    f"No data found for Experiment_ID {experiment_id} in DataFrame {i}"
                )

            # Pick x-axis
            if x_axis_choice == "index":
                x_axis = experiment_df.index
                x_label = "Time"
            else:
                if x_axis_choice not in experiment_df.columns:
                    raise ValueError(f"{x_axis_choice} not found in DataFrame {i}")
                x_axis = experiment_df[x_axis_choice]
                x_label = x_axis_choice

            # Numeric columns to plot
            numeric_cols = [
                col
                for col in experiment_df.columns
                if col not in ["Experiment_ID", x_axis_choice]
            ]

            # Add traces for a given subplot (i)
            for col_idx, col in enumerate(numeric_cols):
                legend_name = extract_trace_name(col)
                if not legend_name.strip():
                    legend_name = col

                fig.add_trace(
                    go.Scatter(
                        x=x_axis,
                        y=experiment_df[col],
                        mode="lines",
                        name=legend_name,
                        showlegend=True,  # allow independent toggling
                        legendgrouptitle_text=(
                            df_name if col_idx == 0 else None
                        ),  # only first trace shows group title
                    ),
                    row=i,
                    col=1,
                )

            # Update axes
            fig.update_yaxes(title_text="Sensor Values", row=i, col=1)
            fig.update_xaxes(title_text=x_label, row=i, col=1)

        # --- Layout ---
        fig.update_layout(
            height=350 * len(dfs),
            width=1400,
            title_text=f"Experiment {experiment_id}: Multiple Datasets Comparison",
            hovermode="x unified",
            legend=dict(
                bordercolor="black",
                borderwidth=1,
                bgcolor="white",
                tracegroupgap=5,
            ),
        )

        # --- Save or show ---
        if save_fig:
            saving_path = (
                f"{base_path}/{part_name}_experiment_plot_{experiment_id}.html"
            )
            fig.write_html(saving_path)
            print(f"Interactive plot with {len(dfs)} subplots saved to {saving_path}")
        else:
            fig.show()

    @log_function
    def interactive_plot(self, dfs, df_names=None, x_axes=None, min_id=2, max_id=300):
        """
        Display an interactive widget to change Experiment_ID and update plot.
        Allows selection from min_id to max_id.
        """
        if df_names is None:
            df_names = [f"Dataset {i+1}" for i in range(len(dfs))]
        if x_axes is None:
            x_axes = []
            for df in dfs:
                x_axes.append("Time_[s]" if "Time_[s]" in df.columns else "index")

        # Use a range of experiment IDs instead of extracting from data
        all_ids = list(range(min_id, max_id + 1))
        print(f"Experiment_ID options: {min_id} to {max_id}")

        # Create dropdown widget
        dropdown = widgets.Dropdown(
            options=all_ids,
            value=min_id,
            description="Experiment_ID:",
        )

        def update(exp_id):
            clear_output(wait=True)  # Clear previous plot
            display(dropdown)  # Keep dropdown visible
            # Call your plotting function without saving
            self.multi_sensor_experiment(
                dfs=dfs,
                experiment_id=exp_id,
                df_names=df_names,
                x_axes=x_axes,
                save_fig=False,
            )

        # Trigger update when dropdown value changes
        dropdown.observe(lambda change: update(change["new"]), names="value")

        # Display dropdown and initial plot
        display(dropdown)
        update(min_id)
