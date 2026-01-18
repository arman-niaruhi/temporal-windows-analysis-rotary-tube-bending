import streamlit as st
from src.pipeline.preprocessing.loader import DataLoader
from src.logging.log_utils import log_function
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.express as px
import os

class DataVisualizer:
    def __init__(self):
        self.loader = DataLoader("data/processed/tube_geometry.db")

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
        loaded_dfs = self.loader.load_data_by_experiment_from_sqlite(experiment_id)
        dfs = [loaded_dfs[name] for name in selected_df_names if name in loaded_dfs]
        return dfs, loaded_dfs

    def matplotlib_plot(self, experiment_df, df_name, experiment_id):
        """
        Full-featured Matplotlib plotting for Streamlit with:
        - Sensor selection
        - X-axis zoom
        - Custom labels & title
        - Paper-ready styling
        - Save plot button
        """
        if experiment_df.empty:
            st.write(f"No data available for {df_name}")
            return

        numeric_cols = [col for col in experiment_df.columns if col != "Experiment_ID"]

        selected_sensors = st.multiselect(
            f"Select sensors for {df_name}",
            options=numeric_cols,
            default=numeric_cols,
            key=f"{df_name}_sensors",
        )

        if "Time_[s]" in experiment_df.columns:
            x_axis = experiment_df["Time_[s]"]
        elif "Angle[degree]ORDistance[mm]" in experiment_df.columns:
            x_axis = experiment_df["Angle[degree]ORDistance[mm]"]
        else:
            x_axis = experiment_df.index  

        x_min, x_max = float(x_axis.min()), float(x_axis.max())
        x_start, x_end = st.slider(
            f"X-axis range ({df_name})",
            min_value=float(x_min),
            max_value=float(x_max),
            value=(float(x_min), float(x_max)),
            step=(float(x_max) - float(x_min)) / 100,
            key=f"{df_name}_xrange",
        )

        mask = (x_axis >= x_start) & (x_axis <= x_end)
        filtered_df = experiment_df.loc[mask]
        filtered_x = x_axis[mask]

        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            x_label = st.text_input(
                f"X-axis label ({df_name})", value=x_axis.name, key=f"{df_name}_xlabel"
            )
        with col2:
            y_label = st.text_input(
                f"Y-axis label ({df_name})", value="Value", key=f"{df_name}_ylabel"
            )
        with col3:
            title = st.text_input(
                f"Title ({df_name})",
                value=f"{df_name} - Experiment {experiment_id}",
                key=f"{df_name}_title",
            )

        plt.rcParams.update(
            {
                "figure.figsize": (12, 5),
                "axes.titlesize": 16,
                "axes.labelsize": 14,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "legend.fontsize": 9,
                "lines.linewidth": 2,
                "lines.markersize": 6,
                "font.family": "serif",
            }
        )
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(12, 5))
        palette = sns.color_palette("tab10", n_colors=len(selected_sensors))

        for i, col in enumerate(selected_sensors):
            ax.plot(
                filtered_x, filtered_df[col], label=col, color=palette[i], linewidth=2
            )

        ax.set_xlabel(x_label, fontsize=14, fontweight="bold")
        ax.set_ylabel(y_label, fontsize=14, fontweight="bold")
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)

        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=9,
            frameon=False,
            ncol=1,
        )

        plt.tight_layout()

        st.pyplot(fig, dpi=300)

        save_folder = "results/saved_plots"
        os.makedirs(save_folder, exist_ok=True)
        if st.button(f"Save Plot as PDF: {df_name}"):
            filename = f"{df_name}_Experiment{experiment_id}_{x_start}_{x_end}.pdf"
            filepath = os.path.join(save_folder, filename)
            fig.savefig(filepath, dpi=300, format='pdf')
            st.success(f"Plot saved as {filepath}")