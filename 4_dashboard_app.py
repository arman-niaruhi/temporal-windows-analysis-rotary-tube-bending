import streamlit as st
from src.pipeline.dashboard.visualizer_utils import DataVisualizer


class StreamlitApp:
    def __init__(self):
        self.visualizer = DataVisualizer()
        st.set_page_config(layout="wide")
        self.experiment_ids = self.visualizer.loader.load_experiment_ids_from_csv()

    def run(self):
        if "run_refresh_counter" not in st.session_state:
            st.session_state.run_refresh_counter = 0
        
        # Initialize video cache in session state
        if "video_cache" not in st.session_state:
            st.session_state.video_cache = {}
            
        page = st.sidebar.selectbox(
            "Choose Page", ["Plots", "Tables"]
        )
    
        experiment_id = st.sidebar.selectbox("Select Experiment ID", self.experiment_ids, index=1)
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

                    
if __name__ == "__main__":              
    vizualiser = StreamlitApp()
    vizualiser.run()
