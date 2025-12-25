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
from pathlib import Path
from src.pipeline.ml.classification.utils.plot_utils import (
    plot_predictions_vs_true_annot,
)
from src.pipeline.ml.classification.utils.training_utils import (
    analyze_features,
    training_pipeline,
)
from src.pipeline.ml.classification.utils.inference_one_label import (
    get_all_predictions,
    inference_one_label_in_one,
)

import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "mlruns"  # or your MLflow/experiment tracking path


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
            "Choose Page", ["Plots", "Tables", "Matplotlib",
                            "Activity Recognition Inference", "Context Extraction Inference" ]
        )
        
        use_test_experiments = st.sidebar.checkbox("Use Test Experiments", value=False)
    
        test_experiment_ids = [
            2, 3, 22, 23, 40, 54, 83, 85, 110, 112, 119, 120, 121, 122, 123,
            178, 179, 182, 183, 211, 212, 213, 255, 258, 261, 271, 272, 273,
            302, 303, 304, 317, 318
        ]

            
        experiment_ids = test_experiment_ids if use_test_experiments else self.experiment_ids
            
        experiment_id = st.sidebar.selectbox("Select Experiment ID", experiment_ids, index=1)
        df_names = ["arc", "machine_and_movement", "sensor", "machine", "movement"]
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
                st.plotly_chart(fig, width='stretch')
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
                
        elif page == "Activity Recognition Inference":
            st.title(page)
            # User selections
            dataset = st.selectbox(
                "Select Dataset",
                options=["movement", "machine_and_movement"]
            )

            machine_part = st.selectbox(
                "Select Machine Part",
                options=["Clamping", "Bending", "Mandrel Extraction", "De-Clamping", "All"]
            )

            experiment_id_inference = st.selectbox("Select Experiment ID", test_experiment_ids, index=1)
            
            image_path = f"results/activity_recognition/{dataset}/{machine_part}/labeled_timestamps_{experiment_id_inference}.png"
            st.image(image_path, caption="Label Predictions")
            
            analyze_image_dir_path = f"results/analyze_features/{dataset}/{machine_part}"
            
            if not Path(analyze_image_dir_path).exists():
                os.makedirs(analyze_image_dir_path, exist_ok=True)
                st.info("The results are not generated. They are generating now and it could take a while...")
                model, sensors_df, test_loader, device, feature_cols = training_pipeline(
                            "models/classifier", "data/processed/tube_geometry.db",
                                "data/ml/machine-and-movement_complete.json", "data/ml/unique_experiment_ids.json",
                                dataset,[
                                    "PRESSURE-DIE_LEFT_AXIAL_Movement_[mm]",
                                    "COLLET_ROTATING_Movement_[mm]",
                                    "BEND-DIE_VERTICAL_Movement_[mm]",
                                    "PRESSURE-DIE_LATERAL_Movement_[mm]",
                                ],machine_part,
                                {
                                    "dataloader_config": {"batch_size": 8},
                                    "model_config": {"hidden_size": 64, "num_layers": 2},
                                    "training_config": {
                                        "training": False,
                                        "num_epochs": 1,
                                        "learning_rate": 1e-5,
                                        "patience": 3,
                                    },
                                },
                            )
                
                analyze_features(analyze_image_dir_path, model, sensors_df, test_loader, device
                )
            analyze_image_dir_path = Path(analyze_image_dir_path)
            
            image_files = sorted(
                analyze_image_dir_path.glob("*.png")
            )  # adjust extension if needed

            if image_files:
                st.image(
                    image_files,
                    caption=[img.name for img in image_files]
                )
            else:
                st.warning("No images found in the selected directory.")
            
        
        # Updated Context Extraction Inference page section
        elif page == "Context Extraction Inference":
            st.title(page)
            
            # Sidebar for experiment and run selection
            with st.sidebar:
                st.header("Experiment Configuration")
                

                # Browse runs for the experiment using MLflow
                run_info_list = self.get_experiment_runs()
                
                if run_info_list:  
                    # Create selectbox with display names
                    display_names = [r['display'] for r in run_info_list]
                    selected_display = st.selectbox("Select Run", display_names)
                    
                    # Get the corresponding run ID and name
                    selected_idx = display_names.index(selected_display)
                    run_id = run_info_list[selected_idx]['id']
                    run_name = run_info_list[selected_idx]['name']
                    
                    # Display run details
                    with st.expander("Run Details"):
                        st.text(f"Run ID: {run_id}")
                        st.text(f"Run Name: {run_name}")
                        if run_info_list[selected_idx]['start_time']:
                            from datetime import datetime
                            start_time = datetime.fromtimestamp(run_info_list[selected_idx]['start_time'] / 1000)
                            st.text(f"Start Time: {start_time}")
                    
                else:
                    run_id = None
                    run_name = None
                
                # Video options
                st.header("Video Options")
                fps = st.slider("Frames per second (FPS)", 1, 30, 5)
            
            # Main content area
            if run_name and run_id:
                st.header(f"Experiment: 665463947744551178 | Run: {run_name}")
                
                # Construct the run path using run_id (MLflow stores by run_id)
                run_path = os.path.join(MLFLOW_TRACKING_URI, "665463947744551178", run_id, "artifacts")
                summary_txt_path = os.path.join(run_path, "experiment_description.txt")
    
                if os.path.exists(summary_txt_path):
                    with open(summary_txt_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        st.text_area("Exact file content:", content, height=400, disabled=True)
                        
                if os.path.exists(run_path):
                    with st.spinner(f"Searching for PNG images in {run_path}..."):
                        # Find PNG images with special rules
                        all_pngs, attention_line_images = self.find_png_images_with_rules(run_path)
                        
                        # Display regular images
                        if all_pngs: 
                            for img_path in sorted(all_pngs):
                                try:
                                    from PIL import Image
                                    img = Image.open(img_path)
                                    img_name = os.path.basename(img_path)
                                    folder_name = os.path.basename(os.path.dirname(img_path))
                                    
                                    st.subheader(f"{folder_name}/{img_name}")
                                    st.image(img, width='stretch')
                                    
                                except Exception as e:
                                    st.error(f"Error loading {img_path}: {e}")
                        
                        # Handle attention lines video
                        if attention_line_images:
                            st.subheader("🎬 Attention Lines Animation")
                            
                            # Sort images by angle number
                            sorted_angles = sorted(attention_line_images.items(), key=lambda x: x[0])
                            image_paths = [path for _, path in sorted_angles]
                        
                            
                            # Create video from images
                            video_path = self.create_video_from_images(image_paths, run_path, fps)
                            
                            if video_path and os.path.exists(video_path):
                                # Display video
                                video_file = open(video_path, 'rb')
                                video_bytes = video_file.read()
                                
                                # Download video button
                                st.download_button(
                                    label="📥 Download Video",
                                    data=video_bytes,
                                    file_name=f"attention_lines_animation_{run_name}.mp4",
                                    mime="video/mp4"
                                )
                                
                                # Clean up temp file
                                os.remove(video_path)
                            else:
                                st.error("Failed to create video")
                            
                            # Show individual images as well
                            with st.expander("Show Individual Angle Images"):
                                cols = st.columns(4)
                                for idx, (angle_num, img_path) in enumerate(sorted_angles):
                                    with cols[idx % 4]:
                                        try:
                                            img = Image.open(img_path)
                                            st.image(img, caption=f"Angle {angle_num:02d}", width='stretch')
                                        except:
                                            pass
                        else:
                            st.info("No attention lines images found")
                        
                        # Download option for all regular images
                        if all_pngs and st.button("Download All Regular Images as ZIP"):
                            import zipfile
                            from io import BytesIO
                            
                            zip_buffer = BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                for img_path in all_pngs:
                                    rel_path = os.path.relpath(img_path, run_path)
                                    zip_file.write(img_path, rel_path)
                            
                            zip_buffer.seek(0)
                            st.download_button(
                                label="Click to Download ZIP",
                                data=zip_buffer,
                                file_name=f"experiment_665463947744551178_run_{run_name}_images.zip",
                                mime="application/zip"
                            )
                        
                        # Show all image paths
                        if all_pngs or attention_line_images:
                            with st.expander("Show All Image Paths"):
                                if all_pngs:
                                    st.write("**Regular Images:**")
                                    for img_path in sorted(all_pngs):
                                        st.code(img_path)
                                if attention_line_images:
                                    st.write("**Attention Line Images:**")
                                    for angle_num, img_path in sorted(attention_line_images.items()):
                                        st.code(f"Angle {angle_num:02d}: {img_path}")
                        else:
                            st.info(f"No PNG images found in {run_path}")
                else:
                    st.error(f"Run path does not exist: {run_path}")
                    
    
    def get_experiment_runs(self):
        """Get all runs for a given experiment ID using MLflow, optionally filtered by search term"""
        try:
            # Initialize MLflow client
            client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
            
            # Get all runs for the experiment
            experiment_id = "665463947744551178"
            runs = client.search_runs(
                experiment_ids=[experiment_id],
                order_by=["start_time DESC"]  # Most recent first
            )
            
            # Extract run names and IDs
            run_info = []
            for run in runs:
                # Get run name from tags (this is the proper way to get custom run names)
                run_name = run.data.tags.get("mlflow.runName", "")
                
                # Fallback to run_id if no run name is set
                if not run_name:
                    run_name = run.info.run_id
                
                run_id = run.info.run_id
                
                run_info.append({
                        'name': run_name,
                        'id': run_id,
                        'display': f"{run_name} ({run_id[:8]})",
                        'start_time': run.info.start_time
                    })
            
            return run_info
            
        except Exception as e:
            st.error(f"Error fetching runs from MLflow: {e}")
            import traceback
            st.error(traceback.format_exc())
            # Fallback to directory listing
            return self.get_experiment_runs_fallback()


    def get_experiment_runs_fallback(self):
        """Fallback method: Get runs by listing directories"""
        runs = []
        experiment_path = os.path.join(MLFLOW_TRACKING_URI, "665463947744551178")
        
        if os.path.exists(experiment_path):
            all_runs = [d for d in os.listdir(experiment_path)
                        if os.path.isdir(os.path.join(experiment_path, d))]
            
            filtered = all_runs
            # Format as list of dicts for consistency
            runs = [{'name': r, 'id': r, 'display': r, 'start_time': None} for r in sorted(filtered)]
        
        return runs


    def get_all_run_names(self):
        """Get a list of all unique run names from the experiment"""
        try:
            client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
            experiment_id = "665463947744551178"
            runs = client.search_runs(
                experiment_ids=[experiment_id],
                order_by=["start_time DESC"]
            )
            
            # Extract unique run names
            run_names = []
            for run in runs:
                run_name = run.data.tags.get("mlflow.runName", "")
                if run_name and run_name not in run_names:
                    run_names.append(run_name)
            
            return sorted(run_names)
            
        except Exception as e:
            st.error(f"Error fetching run names: {e}")
            return []


    def find_png_images_with_rules(self, root_path):
        """Find PNG images with rules: 
        - For special folders, take only last epoch
        - For attention_lines, collect all angle images"""
        import glob
        from pathlib import Path
        import re
        
        all_images = []
        attention_line_images = {}  # dict: angle_number -> image_path
        
        root_path = Path(root_path)
        
        # First, find all PNGs recursively
        pattern = str(root_path / "**" / "*.png")
        all_pngs = glob.glob(pattern, recursive=True)
        
        # Group images by their directory
        images_by_dir = {}
        for png_path in all_pngs:
            dir_path = os.path.dirname(png_path)
            if dir_path not in images_by_dir:
                images_by_dir[dir_path] = []
            images_by_dir[dir_path].append(png_path)
        
        # Process each directory
        for dir_path, images in images_by_dir.items():
            dir_name = os.path.basename(dir_path)
            
            # Check for attention_lines folder
            if "attention_lines" in dir_name.lower() or "04_attention_lines" in dir_name:
                # Special handling for attention_lines - extract angle numbers
                for img_path in images:
                    img_name = os.path.basename(img_path)
                    
                    # Look for angle pattern: attention_angle_01.png, attention_angle_02.png, etc.
                    match = re.search(r'attention_angle_(\d+)\.png$', img_name, re.IGNORECASE)
                    if match:
                        angle_num = int(match.group(1))
                        attention_line_images[angle_num] = img_path
                    else:
                        # If no angle pattern, just add to regular images
                        all_images.append(img_path)
            
            # Check if this is one of the other special folders (predictions, loss, attention)
            elif any(special in dir_name.lower() 
                    for special in ["predictions", "loss", "attention"]):
                # Skip if it's the attention folder (not attention_lines)
                if "attention_lines" not in dir_name.lower():
                    # For special folders, find images with epoch pattern and take last one
                    epoch_images = {}
                    
                    for img_path in images:
                        img_name = os.path.basename(img_path)
                        
                        # Look for epoch pattern: _epoch_0001, _epoch_0002, etc.
                        match = re.search(r'_epoch_(\d+)\.png$', img_name, re.IGNORECASE)
                        if match:
                            epoch_num = int(match.group(1))
                            epoch_images[epoch_num] = img_path
                        else:
                            # If no epoch pattern found, just add it
                            all_images.append(img_path)
                    
                    # If we found epoch images, take the one with highest epoch number
                    if epoch_images:
                        max_epoch = max(epoch_images.keys())
                        all_images.append(epoch_images[max_epoch])
                    else:
                        # If no epoch pattern, take all images
                        all_images.extend(images)
            else:
                # For non-special folders, take all images
                all_images.extend(images)
        
        return list(set(all_images)), attention_line_images

        
    def create_video_from_images(self, image_paths, output_dir, fps=5):
        """Create a video from a list of image paths with better codec and looping"""
        try:
            import cv2
            import numpy as np
            from PIL import Image
            import tempfile
            
            if not image_paths:
                st.error("No image paths provided")
                return None
            
            st.info(f"Creating video from {len(image_paths)} images at {fps} FPS...")
            
            # Read and store all images first
            frames = []
            for img_path in image_paths:
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        # Try with PIL if OpenCV fails
                        pil_img = Image.open(img_path)
                        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    frames.append(img)
                except Exception as e:
                    st.warning(f"Failed to load image {img_path}: {e}")
                    continue
            
            if not frames:
                st.error("No frames could be loaded")
                return None
            
            # Get dimensions from first frame
            height, width = frames[0].shape[:2]
            
            # Resize all frames to match first frame dimensions
            resized_frames = []
            for img in frames:
                if img.shape[:2] != (height, width):
                    img = cv2.resize(img, (width, height))
                resized_frames.append(img)
            
            # Create temporary video file
            temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False, dir=output_dir)
            temp_video_path = temp_video.name
            temp_video.close()
            
            # Try multiple codecs in order of preference
            codecs_to_try = [
                ('mp4v', 'MPEG-4'),  # Fallback
                ('XVID', 'Xvid'),    # Another fallback
            ]
            
            video = None
            for codec_code, codec_name in codecs_to_try:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec_code)
                    video = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
                    
                    if video.isOpened():
                        st.success(f"Using {codec_name} codec")
                        break
                    else:
                        video.release()
                        video = None
                except:
                    continue
            
            if video is None or not video.isOpened():
                st.error("Failed to create video with any codec")
                return None
            
            # Write frames multiple times for looping effect
            num_loops = 3  # Number of times to loop through images
            
            for loop in range(num_loops):
                for frame in resized_frames:
                    video.write(frame)
                
                # Hold last frame briefly between loops
                if loop < num_loops - 1:
                    for _ in range(fps // 2):  # Half second pause
                        video.write(resized_frames[-1])
            
            # Hold final frame longer
            for _ in range(fps * 2):  # 2 second hold at end
                video.write(resized_frames[-1])
            
            video.release()
            
            # Verify the video file was created and has size
            if os.path.exists(temp_video_path) and os.path.getsize(temp_video_path) > 0:
                st.success(f"Video created successfully: {os.path.getsize(temp_video_path)} bytes")
                return temp_video_path
            else:
                st.error("Video file is empty or wasn't created")
                return None
                
        except Exception as e:
            st.error(f"Error creating video: {e}")
            import traceback
            st.error(traceback.format_exc())
            return None