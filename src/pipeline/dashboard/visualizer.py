from src.pipeline.ml.classification.inference import predict_activity
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
import json


# ------------------------------
# Data Visualizer Class
# ------------------------------
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

        # --- User selects sensors ---
        selected_sensors = st.multiselect(
            f"Select sensors for {df_name}",
            options=numeric_cols,
            default=numeric_cols,
            key=f"{df_name}_sensors",
        )

        # --- Determine proper x-axis ---
        if "Time_[s]" in experiment_df.columns:
            x_axis = experiment_df["Time_[s]"]
        elif "Angle[degree]ORDistance[mm]" in experiment_df.columns:
            x_axis = experiment_df["Angle[degree]ORDistance[mm]"]
        else:
            x_axis = experiment_df.index  # fallback if no proper column

        # --- X-axis zoom slider ---
        x_min, x_max = float(x_axis.min()), float(x_axis.max())
        x_start, x_end = st.slider(
            f"X-axis range ({df_name})",
            min_value=float(x_min),
            max_value=float(x_max),
            value=(float(x_min), float(x_max)),
            step=(float(x_max) - float(x_min)) / 100,
            key=f"{df_name}_xrange",
        )

        # --- Filter dataframe for selected x-axis range ---
        mask = (x_axis >= x_start) & (x_axis <= x_end)
        filtered_df = experiment_df.loc[mask]
        filtered_x = x_axis[mask]

        # --- Optional title/labels ---
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

        # --- Plotting ---
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

        # --- Axes and title ---
        ax.set_xlabel(x_label, fontsize=14, fontweight="bold")
        ax.set_ylabel(y_label, fontsize=14, fontweight="bold")
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)

        # --- Legend outside plot with smaller font ---
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=9,
            frameon=False,
            ncol=1,
        )

        # --- Tight layout to prevent clipping ---
        plt.tight_layout()

        # --- Display plot ---
        st.pyplot(fig, dpi=300)

        # --- Save plot button ---
        save_folder = "results/saved_plots"
        os.makedirs(save_folder, exist_ok=True)
        if st.button(f"Save Plot as PDF: {df_name}"):
            filename = f"{df_name}_Experiment{experiment_id}_{x_start}_{x_end}.pdf"
            filepath = os.path.join(save_folder, filename)
            fig.savefig(filepath, dpi=300, format='pdf')
            st.success(f"Plot saved as {filepath}")


    def matplotlib_mar_plot(self, experiment_df, labels, df_name, experiment_id):
        """
        Plot MAR (Machine Activity Recognition) predictions using Matplotlib with axvspan.
        Shaded regions indicate activity labels.

        Features:
            - Select which sensors to plot
            - Activity labels in legend (except 'idle')
            - Paper-ready styling
            - Save plot button
        """
        if experiment_df.empty:
            st.warning(f"No data for {df_name} / Experiment {experiment_id}")
            return
        experiment_df = experiment_df.iloc[:, 1:]
        print(experiment_df)
        numeric_cols = list(experiment_df.columns)
        selected_sensors = st.multiselect(
            f"Select sensors for {df_name}",
            options=numeric_cols,
            default=numeric_cols,
            key=f"{df_name}_mar_sensors",
        )
        if not selected_sensors:
            st.warning("No sensors selected for plotting.")
            return

        x_axis = experiment_df.index
        t_min, t_max = float(x_axis.min()), float(x_axis.max())
        t_start, t_end = st.slider(
            f"Select time range for {df_name}",
            min_value=t_min,
            max_value=t_max,
            value=(t_min, t_max),
            step=(t_max - t_min) / 100,
            key=f"{df_name}_mar_time",
        )
        # --- Filter dataframe and labels by time range ---
        mask = (x_axis >= t_start) & (x_axis <= t_end)
        filtered_df = experiment_df.loc[mask]
        filtered_x = x_axis[mask]
        filtered_labels = [lbl for lbl, m in zip(labels, mask) if m]

        if filtered_df.empty:
            st.warning("No data in the selected time range.")
            return

        # --- Matplotlib settings ---
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

        # --- Plot sensor values ---
        for i, col in enumerate(selected_sensors):
            ax.plot(
                filtered_x, filtered_df[col], label=col, color=palette[i], linewidth=2
            )

        # --- Color mapping for activity labels ---
        unique_labels = [lbl for lbl in set(filtered_labels) if lbl.lower() != "idle"]
        cmap = sns.color_palette("Set2", n_colors=len(unique_labels))
        label_color_map = {lbl: cmap[i] for i, lbl in enumerate(unique_labels)}

        # --- Draw axvspan for each activity interval ---
        start_idx = filtered_x[0]
        prev_label = filtered_labels[0]
        for idx, lbl in zip(filtered_x, filtered_labels):
            if lbl != prev_label:
                if prev_label.lower() != "idle":
                    ax.axvspan(
                        start_idx,
                        idx,
                        color=label_color_map[prev_label],
                        alpha=0.5,
                        label=prev_label,
                    )
                start_idx = idx
                prev_label = lbl
        # Last interval
        if prev_label.lower() != "idle":
            ax.axvspan(
                start_idx,
                filtered_x[-1],
                color=label_color_map[prev_label],
                alpha=0.5,
                label=prev_label,
            )

        # --- Remove duplicate labels in legend ---
        handles, legend_labels = ax.get_legend_handles_labels()
        by_label = dict(zip(legend_labels, handles))
        ax.legend(
            by_label.values(),
            by_label.keys(),
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
            ncol=1,
        )

        # --- Axes and title ---
        ax.set_xlabel("Time [s]", fontsize=14, fontweight="bold")
        ax.set_ylabel("Sensor Values", fontsize=14, fontweight="bold")
        ax.set_title(
            f"MAR Predictions - Experiment {experiment_id} ({df_name})",
            fontsize=16,
            fontweight="bold",
        )
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)
        plt.tight_layout()

        # --- Display plot ---
        st.pyplot(fig, dpi=300)

        # --- Save plot button ---
        save_folder = "results/saved_plots"
        os.makedirs(save_folder, exist_ok=True)
        if st.button(f"Save MAR Plot: {df_name}"):
            filename = (
                f"{df_name}_MAR_Experiment{experiment_id}_{t_start:.1f}_{t_end:.1f}.png"
            )
            filepath = os.path.join(save_folder, filename)
            fig.savefig(filepath, dpi=300)
            st.success(f"MAR plot saved as {filepath}")


# ------------------------------
# Streamlit App Class
# ------------------------------
class StreamlitApp:
    def __init__(self):
        self.visualizer = DataVisualizer()
        st.set_page_config(layout="wide")
        self.experiment_ids = self.visualizer.loader.load_experiment_ids_from_sqlite()

    def run(self):
        page = st.sidebar.selectbox(
            "Choose Page", ["Plots", "Tables", "Matplotlib", "MAR"]
        )
        
        use_test_experiments = st.sidebar.checkbox("Use Test Experiments", value=False)

        # Load test experiment IDs from JSON
        json_path = "data/ml/test_experiment_ids.json"
        with open(json_path, "r") as f:
            data = json.load(f)

        # Handle both list or dictionary structure
        if isinstance(data, dict):
            test_experiment_ids = data.get("experiment_ids", [])
        else:
            test_experiment_ids = data
            
        experiment_ids = test_experiment_ids if use_test_experiments else self.experiment_ids
            
        experiment_id = st.sidebar.selectbox("Select Experiment ID", experiment_ids, index=1)
        df_names = ["arc", "machine_and_movement", "sensor", "machine", "movements"]
        selected_df_names = st.sidebar.multiselect(
            "Select Datasets", df_names, default=df_names
        )

        # --- Load data ---
        dfs, loaded_dfs = self.visualizer.load_experiment_data(
            int(experiment_id), selected_df_names
        )

        # --- Pages ---
        if page == "Plots":
            st.title("Interactive Plotly Plots")
            if dfs:
                fig = self.visualizer.multi_sensor_experiment(
                    dfs, int(experiment_id), selected_df_names, x_axes=None
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available for selected Experiment ID and datasets.")

        elif page == "Tables":
            st.title("Experiment Data Tables")
            if loaded_dfs:
                for df_name in selected_df_names:
                    if df_name in loaded_dfs:
                        st.subheader(f"Table: {df_name}")
                        st.dataframe(loaded_dfs[df_name])
            else:
                st.warning("No data available for selected Experiment ID and datasets.")

        elif page == "Matplotlib":
            st.title("Matplotlib Sensor Plots (Paper-Ready)")
            if dfs:
                for df, df_name in zip(dfs, selected_df_names):
                    experiment_df = df[df["Experiment_ID"] == int(experiment_id)]
                    self.visualizer.matplotlib_plot(
                        experiment_df, df_name, int(experiment_id)
                    )
            else:
                st.warning("No data available for selected Experiment ID and datasets.")

        elif page == "MAR":
            st.title("MAR: Machine Activity Recognition")

            mar_options = [
                           "machine_and_movement", 
                           "movements", 
                           "machine", 
                           "Single Machine and Movement",
                           "Single Movement",
                           "Single Machine"
                           ]
            mar_dataset_name = st.selectbox("Select dataset for MAR", mar_options)
            if mar_dataset_name == "Single Machine and Movement":
                dataset_name = "machine_and_movement"
                experiment_df = loaded_dfs[dataset_name]
                experiment_df = experiment_df[
                    experiment_df["Experiment_ID"] == int(experiment_id)
                ]
                experiment_df.drop(columns=['MACHINE_PRESSURE-DIE_AXIAL_Max_Torque_[%]'], inplace=True)
                if "Time_[s]" in experiment_df.columns:
                    experiment_df = experiment_df.set_index("Time_[s]")

                model_path_declamping = f"models/classifier/{dataset_name}/De-Clamping"
                declamping_labels = predict_activity(sensors_df=experiment_df, model_path=model_path_declamping, num_classes=2)
                self.visualizer.matplotlib_mar_plot(
                    experiment_df, declamping_labels, mar_dataset_name, int(experiment_id)
                )


vizualiser = StreamlitApp()
vizualiser.run()
# streamlit run main.py visualize