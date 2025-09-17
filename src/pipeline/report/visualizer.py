from src.logging.log_utils import log_function, logger
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.express as px
import streamlit as st
import re
import os
from src.pipeline.preprocessing.loader import DataLoader
# IPython widgets for Jupyter interactivity
from IPython.display import display, clear_output
import ipywidgets as widgets


st.set_page_config(layout="wide")

class DataVisulizer:
    def __init__(self) -> None:
        self.loader = DataLoader("data/tube_geometry.db")

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
        def extract_trace_name(col_name: str) -> str:
            parts = col_name.split("_")
            for i in range(len(parts)):
                if re.match(r"^[A-Za-z0-9\-]+\_\[[^]]+\]$", "_".join(parts[i:])):
                    return " ".join(parts[:i]).replace("_", " ")
            return " ".join(parts[:-2]).replace("_", " ")

        if df_names is None:
            df_names = [f"Dataset {i+1}" for i in range(len(dfs))]
        elif len(df_names) != len(dfs):
            raise ValueError("Number of DataFrame names must match number of DataFrames")

        if x_axes is None:
            x_axes = ["Time_[s]" if "Time_[s]" in df.columns else "index" for df in dfs]
        elif len(x_axes) != len(dfs):
            raise ValueError("Number of x_axes entries must match number of DataFrames")

        fig = make_subplots(
            rows=len(dfs),
            cols=1,
            subplot_titles=[f"{name} - Experiment {experiment_id}" for name in df_names],
            vertical_spacing=0.15,
        )

        colors = px.colors.qualitative.Plotly

        for i, (df, x_axis_choice, df_name) in enumerate(zip(dfs, x_axes, df_names), start=1):
            experiment_df = df[df["Experiment_ID"] == experiment_id]
            if experiment_df.empty:
                raise ValueError(f"No data found for Experiment_ID {experiment_id} in DataFrame {i}")

            x_axis = experiment_df.index if x_axis_choice == "index" else experiment_df[x_axis_choice]
            x_label = "Time" if x_axis_choice == "index" else x_axis_choice

            numeric_cols = [col for col in experiment_df.columns if col not in ["Experiment_ID", x_axis_choice]]

            for col_idx, col in enumerate(numeric_cols):
                data_series = experiment_df[col]
                min_val = data_series.min()
                max_val = data_series.max()
                mean_val = data_series.mean()
                std_val = data_series.std()
                
                # Format legend with statistics
                legend_name = (
                    f"{df_name}: {extract_trace_name(col) or col} "
                    f"(min={min_val:.2f}, max={max_val:.2f}, mean={mean_val:.2f}, std={std_val:.2f})"
                )
                
                color = colors[col_idx % len(colors)]
                fig.add_trace(
                    go.Scatter(
                        x=x_axis,
                        y=data_series,
                        mode="lines",
                        name=legend_name,
                        line=dict(color=color),
                        showlegend=True,
                    ),
                    row=i,
                    col=1,
                )


            fig.update_yaxes(title_text="Sensor Values", row=i, col=1)
            fig.update_xaxes(title_text=x_label, row=i, col=1)

        fig.update_layout(
            height=350 * len(dfs),
            width=1400,
            title_text=f"Experiment {experiment_id}: Multiple Datasets Comparison",
            hovermode="x unified",
            legend=dict(
                orientation="v",      # vertical legend
                yanchor="top",
                y=1,                  # align with top
                xanchor="left",
                x=1.02,               # slightly outside the right side
                bordercolor="black",
                borderwidth=1,
                bgcolor="rgba(0,0,0,0)",
                tracegroupgap=5,
            ),
            margin=dict(r=200)        # add extra right margin for the legend
        )

        if save_fig:
            os.makedirs(base_path, exist_ok=True)
            saving_path = f"{base_path}/{part_name}_experiment_plot_{experiment_id}.html"
            fig.write_html(saving_path)
            print(f"Interactive plot with {len(dfs)} subplots saved to {saving_path}")

        return fig

    @log_function
    def interactive_plot_streamlit(self, min_id=2):
        import streamlit as st

        # Corrected lists (remove trailing commas)
        df_names = ["df_arc", "df_machine_and_movement", "df_sensor", "df_machine", "df_movements"]
        x_axes = ["Angle[degree]ORDistance[mm]", "index", "index", "index", "index"]

        st.title("Tube Geometry Sensors")

        # Experiment ID input
        experiment_id = st.text_input("Enter Experiment ID", value=str(min_id))

        # Multiselect for datasets
        selected_df_names = st.multiselect(
            "Select Datasets",
            options=df_names,
            default=df_names
        )

        # Load data
        loaded_dfs = self.loader.load_data_by_experiment(experiment_id)
        dfs = [loaded_dfs[name] for name in selected_df_names if name in loaded_dfs]

        if dfs:
            fig = self.multi_sensor_experiment(
                dfs=dfs,
                experiment_id=int(experiment_id),
                df_names=selected_df_names,
                x_axes=x_axes[:len(selected_df_names)],  # make sure x_axes matches dfs
                save_fig=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for selected Experiment ID and datasets.")


    @log_function
    def interactive_plot_jupyter(self, df_names: list, x_axes: list, min_id=2, max_id=318):
        """
        Create an interactive widget to select Experiment_ID and plot data loaded from SQLite via DataLoader.
        
        Parameters
        ----------
        df_names : list
            Names for each DataFrame (used in subplot titles).
        x_axes : list
            List of x-axis columns for each DataFrame.
        min_id : int
            Minimum Experiment_ID.
        max_id : int
            Maximum Experiment_ID.
        """
        # Create the slider widget for Experiment_ID
        experiment_selector = widgets.IntSlider(
            value=min_id,
            min=min_id,
            max=max_id,
            step=1,
            description='Experiment ID:',
            continuous_update=False
        )
        
        output = widgets.Output()

        # Callback to load data and update plot
        def update_plot(change):
            with output:
                clear_output(wait=True)
                # Load data for the selected Experiment_ID
                loaded_dfs = self.loader.load_data_by_experiment(experiment_selector.value)
                dfs = [loaded_dfs[name] for name in df_names if name in loaded_dfs]
                # Generate the plot
                self.multi_sensor_experiment(
                    dfs=dfs,
                    experiment_id=experiment_selector.value,
                    df_names=df_names,
                    x_axes=x_axes,
                    save_fig=False  # show interactive plot
                )
        
        # Observe changes in slider
        experiment_selector.observe(update_plot, names='value')

        # Display the widgets
        display(experiment_selector, output)

        # Initial plot
        update_plot(None)
        
    