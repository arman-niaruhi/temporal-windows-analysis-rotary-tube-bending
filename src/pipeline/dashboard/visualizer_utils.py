from src.pipeline.preprocessing.loader import DataLoader
from src.logging.log_utils import log_function
import matplotlib

matplotlib.use("Agg")
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.express as px
import os

class DataVisualizer:
    def __init__(self):
        self.loader = DataLoader("data/processed")

    @log_function
    def multi_sensor_experiment(
        self,
        dfs,
        experiment_id,
        df_names,
        x_axes=None
    ):
        if df_names is None:
            df_names = [f"Dataset {i+1}" for i in range(len(dfs))]
        if x_axes is None:
            x_axes = ["Time_[s]" if "Time_[s]" in df.columns else "index" for df in dfs]

        fig = make_subplots(
            rows=len(dfs),
            cols=1,
            subplot_titles=[
                f"{name} - Experiment {experiment_id}" for name in df_names
            ],
            vertical_spacing=0.15,
        )
        colors = px.colors.qualitative.Plotly

        for i, (df, x_axis_choice, df_name) in enumerate(
            zip(dfs, x_axes, df_names), start=1
        ):
            experiment_df = df[df["Experiment_ID"] == experiment_id]
            if experiment_df.empty:
                continue
            x_axis = (
                experiment_df.index
                if x_axis_choice == "index"
                else experiment_df[x_axis_choice]
            )
            numeric_cols = [
                col
                for col in experiment_df.columns
                if col not in ["Experiment_ID", x_axis_choice]
            ]

            for col_idx, col in enumerate(numeric_cols):
                data_series = experiment_df[col]
                legend_name = f"{df_name}: {col} (min={data_series.min():.2f}, max={data_series.max():.2f})"
                color = colors[col_idx % len(colors)]
                fig.add_trace(
                    go.Scatter(
                        x=x_axis,
                        y=data_series,
                        mode="lines",
                        name=legend_name,
                        line=dict(color=color),
                    ),
                    row=i,
                    col=1,
                )

            fig.update_yaxes(title_text="Sensor Values", row=i, col=1)
            fig.update_xaxes(title_text=x_axis_choice, row=i, col=1)

        fig.update_layout(height=350 * len(dfs), width=1400, hovermode="x unified")
        return fig

    def load_experiment_data(self, experiment_id, selected_df_names):
        loaded_dfs = self.loader.load_data_by_experiment_from_csv(experiment_id)
        dfs = [loaded_dfs[name] for name in selected_df_names if name in loaded_dfs]
        return dfs, loaded_dfs