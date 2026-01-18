import os
from pathlib import Path
import streamlit as st
import shutil
from src.pipeline.dashboard.visualizer_utils import DataVisualizer
from src.pipeline.ml.classification.utils.plot_utils import (
    plot_predictions_vs_true_annot,
)
from src.pipeline.ml.classification.utils.training_utils import (
    analyze_features, training_pipeline,
)
from src.pipeline.ml.classification.utils.inference_one_label import (
    get_all_predictions, inference_one_label_in_one,
)
from src.pipeline.dashboard.context_extractor_utils import (
    get_experiment_runs, create_video_from_images, find_png_images_with_rules)

MLFLOW_TRACKING_URI = "mlruns"


class StreamlitApp:
    def __init__(self):
        self.visualizer = DataVisualizer()
        st.set_page_config(layout="wide")
        self.experiment_ids = self.visualizer.loader.load_experiment_ids_from_sqlite()

    def run(self):
        if "run_refresh_counter" not in st.session_state:
            st.session_state.run_refresh_counter = 0
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
        df_names = ["arc", "machine_and_movement", "movement"]
        selected_df_names = st.sidebar.multiselect(
            "Select Datasets", df_names, default=df_names
        )

        dfs, loaded_dfs = self.visualizer.load_experiment_data(
            int(experiment_id), selected_df_names
        )

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
            )  

            if image_files:
                st.image(
                    image_files,
                    caption=[img.name for img in image_files]
                )
            else:
                st.warning("No images found in the selected directory.")
            
        
        elif page == "Context Extraction Inference":
            st.title(page)
            
            with st.sidebar:
                st.header("Experiment Configuration")
                

                run_info_list = get_experiment_runs()
                
                if run_info_list:  
                    display_names = [r['display'] for r in run_info_list]
                    selected_display = st.selectbox("Select Run", display_names)
                    
                    selected_idx = display_names.index(selected_display)
                    run_id = run_info_list[selected_idx]['id']
                    run_name = run_info_list[selected_idx]['name']
                    run_path = os.path.join(
                        MLFLOW_TRACKING_URI,
                        "665463947744551178",
                        run_id
                    )

                    with st.expander("Run Details"):
                        st.text(f"Run ID: {run_id}")
                        st.text(f"Run Name: {run_name}")

                        st.divider()

                        confirm_delete = st.checkbox(
                            "I understand this will permanently delete the run folder",
                            key=f"confirm_delete_{run_id}"
                        )

                        if st.button("🗑️ Delete Run Folder", disabled=not confirm_delete):
                            try:
                                if os.path.exists(run_path):
                                    shutil.rmtree(run_path)
                                    st.success(f"Run folder deleted:\n{run_path}")
                                else:
                                    st.warning("Run folder does not exist")

                                st.session_state.run_refresh_counter = (
                                    st.session_state.get("run_refresh_counter", 0) + 1
                                )

                            except Exception as e:
                                st.error(f"Failed to delete run folder: {e}")

                    
                else:
                    run_id = None
                    run_name = None
                
                st.header("Video Options")
                fps = st.slider("Frames per second (FPS)", 1, 30, 5)
            
            if run_name and run_id:
                st.header(f"Experiment: 665463947744551178 | Run: {run_name}")
                
                run_path = os.path.join(MLFLOW_TRACKING_URI, "665463947744551178", run_id, "artifacts")
                summary_txt_path = os.path.join(run_path, "experiment_description.txt")
    
                if os.path.exists(summary_txt_path):
                    with open(summary_txt_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        st.text_area("Exact file content:", content, height=400, disabled=True)
                        
                if os.path.exists(run_path):
                    with st.spinner(f"Searching for PNG images in {run_path}..."):
                        all_pngs, attention_line_images = find_png_images_with_rules(run_path)
                        
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
                        
                        if attention_line_images:
                            st.subheader("🎬 Attention Lines Animation")
                            
                            sorted_angles = sorted(attention_line_images.items(), key=lambda x: x[0])
                            image_paths = [path for _, path in sorted_angles]
                        
                            
                            video_path = create_video_from_images(image_paths, run_path, fps)
                            
                            if video_path and os.path.exists(video_path):
                                video_file = open(video_path, 'rb')
                                video_bytes = video_file.read()
                                
                                st.download_button(
                                    label="📥 Download Video",
                                    data=video_bytes,
                                    file_name=f"attention_lines_animation_{run_name}.mp4",
                                    mime="video/mp4"
                                )
                                
                                os.remove(video_path)
                            else:
                                st.error("Failed to create video")
                            
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
                    
                    
if __name__ == "__main__":              
    vizualiser = StreamlitApp()
    vizualiser.run()

