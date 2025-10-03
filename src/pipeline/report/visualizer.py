import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.logging.log_utils import log_function
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
import seaborn as sns


st.set_page_config(layout="wide")


class DataVisulizer:
    def __init__(self) -> None:
        """
        Initialize the DataVisualizer with a DataLoader for accessing experiment data.

        Attributes:
            loader (DataLoader): Instance of DataLoader connected to tube geometry database.
        """
        self.loader = DataLoader("data/processed/tube_geometry.db")

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
        Plot multiple sensor datasets for a single experiment using Plotly subplots.

        Args:
            dfs (list): List of DataFrames to plot.
            experiment_id (int): ID of the experiment to visualize.
            df_names (list): Names of the datasets corresponding to dfs.
            x_axes (list): List of x-axis column names ("Time_[s]" or "index") per dataset.
            save_fig (bool): Whether to save the plot as an interactive HTML file.
            base_path (str): Directory to save plots.
            part_name (str | None): Optional prefix for saved plot filenames.

        Returns:
            plotly.graph_objs._figure.Figure: Interactive Plotly figure with subplots.
        """

        def extract_trace_name(col_name: str) -> str:
            """
            Extract a clean trace name from a column name by parsing underscores and units.

            Args:
                col_name (str): Original column name.

            Returns:
                str: Clean, human-readable name for plotting.
            """
            parts = col_name.split("_")
            for i in range(len(parts)):
                if re.match(r"^[A-Za-z0-9\-]+\_\[[^]]+\]$", "_".join(parts[i:])):
                    return " ".join(parts[:i]).replace("_", " ")
            return " ".join(parts[:-2]).replace("_", " ")

        if df_names is None:
            df_names = [f"Dataset {i+1}" for i in range(len(dfs))]
        elif len(df_names) != len(dfs):
            raise ValueError(
                "Number of DataFrame names must match number of DataFrames"
            )

        if x_axes is None:
            x_axes = ["Time_[s]" if "Time_[s]" in df.columns else "index" for df in dfs]
        elif len(x_axes) != len(dfs):
            raise ValueError("Number of x_axes entries must match number of DataFrames")

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
                raise ValueError(
                    f"No data found for Experiment_ID {experiment_id} in DataFrame {i}"
                )

            x_axis = (
                experiment_df.index
                if x_axis_choice == "index"
                else experiment_df[x_axis_choice]
            )
            x_label = "Time" if x_axis_choice == "index" else x_axis_choice

            numeric_cols = [
                col
                for col in experiment_df.columns
                if col not in ["Experiment_ID", x_axis_choice]
            ]

            for col_idx, col in enumerate(numeric_cols):
                data_series = experiment_df[col]
                min_val, max_val, mean_val, std_val, count_val = (
                    data_series.min(),
                    data_series.max(),
                    data_series.mean(),
                    data_series.std(),
                    data_series.nunique(),
                )

                # Legend entry includes descriptive stats
                legend_name = (
                    f"{df_name}: {extract_trace_name(col) or col} "
                    f"(min={min_val:.2f}, max={max_val:.2f}, mean={mean_val:.2f}, std={std_val:.2f}, unique count={count_val})"
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
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                bordercolor="black",
                borderwidth=1,
                bgcolor="rgba(0,0,0,0)",
                tracegroupgap=5,
            ),
            margin=dict(r=200),
        )

        if save_fig:
            os.makedirs(base_path, exist_ok=True)
            saving_path = (
                f"{base_path}/{part_name}_experiment_plot_{experiment_id}.html"
            )
            fig.write_html(saving_path)
            print(f"Interactive plot with {len(dfs)} subplots saved to {saving_path}")

        return fig

    @log_function
    def interactive_plot_streamlit(self, min_id=2):
        """
        Build an interactive Streamlit app for exploring tube geometry and sensor data.

        Args:
            min_id (int): Default Experiment_ID to display.

        Returns:
            None: Displays interactive UI and plots in Streamlit.
        """

        # Dataset names and default x_axes
        df_names = [
            "arc",
            "machine_and_movement",
            "sensor",
            "machine",
            "movements",
        ]
        x_axes = ["Angle[degree]ORDistance[mm]", "Time_[s]", "Time_[s]", "Time_[s]", "Time_[s]"]

        st.title("Tube Geometry Sensors")

        # Experiment ID input
        experiment_id = st.text_input("Enter Experiment ID", value=str(min_id))

        # Reset Matplotlib state when experiment ID changes
        if "prev_experiment_id" not in st.session_state:
            st.session_state.prev_experiment_id = experiment_id
        if st.session_state.prev_experiment_id != experiment_id:
            st.session_state.show_matplotlib = False
            st.session_state.prev_experiment_id = experiment_id

        if "show_matplotlib" not in st.session_state:
            st.session_state.show_matplotlib = False

        # Dataset selection
        selected_df_names = st.multiselect(
            "Select Datasets", options=df_names, default=df_names
        )

        # Make order irrelevant by sorting alphabetically (or by your original df_names order)
        selected_df_names = sorted(selected_df_names, key=lambda x: df_names.index(x))

        # Load data
        loaded_dfs = self.loader.load_data_by_experiment_from_sqlite(experiment_id)
        dfs = [loaded_dfs[name] for name in selected_df_names if name in loaded_dfs]
        df_bending_setup = loaded_dfs.get("bending", None)

        # Show setup info
        if df_bending_setup is not None and not df_bending_setup.empty:
            st.subheader("Setup Information")
            st.dataframe(df_bending_setup)

        if dfs:
            # Determine x-axis dynamically per dataset
            x_axes_selected = []
            for df in dfs:
                if "Angle[degree]ORDistance[mm]" in df.columns:
                    x_axes_selected.append("Angle[degree]ORDistance[mm]")
                elif "Time_[s]" in df.columns:
                    x_axes_selected.append("Time_[s]")
                else:
                    x_axes_selected.append("index")

            fig = self.multi_sensor_experiment(
                dfs=dfs,
                experiment_id=int(experiment_id),
                df_names=selected_df_names,
                x_axes=x_axes_selected,
                save_fig=False,
            )

            st.plotly_chart(fig, use_container_width=True)
            # --- Show raw DataFrames for selected datasets ---
            if st.checkbox("Show Selected Tables"):
                for df_name in selected_df_names:
                    if df_name in loaded_dfs:
                        st.subheader(f"Table: {df_name}")
                        st.dataframe(loaded_dfs[df_name])

            # Folder for saved plots
            save_folder = "results/saved_plots"
            os.makedirs(save_folder, exist_ok=True)

            # Button for Matplotlib plots
            if st.button("Show Matplotlib Plot"):
                st.session_state.show_matplotlib = True

            if st.session_state.show_matplotlib:
                # Dynamically determine x-axis per dataset
                x_axes_selected = []
                for df in dfs:
                    if "Angle[degree]ORDistance[mm]" in df.columns:
                        x_axes_selected.append("Angle[degree]ORDistance[mm]")
                    elif "Time_[s]" in df.columns:
                        x_axes_selected.append("Time_[s]")
                    else:
                        x_axes_selected.append("index")

                for df, x_axis_choice, df_name in zip(dfs, x_axes_selected, selected_df_names):
                    experiment_df = df[df["Experiment_ID"] == int(experiment_id)]
                    if experiment_df.empty:
                        st.write(f"No data available for {df_name}")
                        continue

                    # --- Settings per dataset ---
                    st.markdown(f"### Settings for {df_name}")
                    col1, col2, col3 = st.columns([2, 2, 2])
                    with col1:
                        x_label = st.text_input(
                            f"X-axis label",
                            value=x_axis_choice,
                            key=f"{df_name}_xlabel",
                        )
                    with col2:
                        y_label = st.text_input(
                            f"Y-axis label", value="Values", key=f"{df_name}_ylabel"
                        )
                    with col3:
                        title = st.text_input(
                            f"Title for {df_name}",
                            value=f"{df_name} - Experiment {experiment_id}",
                        )

                    # --- Sensor selection ---
                    numeric_cols = [
                        col
                        for col in experiment_df.columns
                        if col not in ["Experiment_ID", x_axis_choice]
                    ]
                    selected_sensors = st.multiselect(
                        f"Select sensors to plot",
                        options=numeric_cols,
                        default=numeric_cols,
                        key=f"{df_name}_sensors",
                    )

                    # --- X-axis zoom slider ---
                    x_axis = experiment_df.index if x_axis_choice == "index" else experiment_df[x_axis_choice]
                    x_min, x_max = int(x_axis.min()), int(x_axis.max())
                    x_start, x_end = st.slider(
                        f"Select X-axis range for {df_name}",
                        min_value=x_min,
                        max_value=x_max,
                        value=(x_min, x_max),
                        step=1,
                        key=f"{df_name}_xarea",
                    )

                    # Filter dataframe for selected x-axis area
                    mask = (x_axis >= x_start) & (x_axis <= x_end)
                    filtered_df = experiment_df.loc[mask]
                    filtered_x = x_axis[mask]

                    # --- Plotting ---
                    sns.set_style("white")
                    plt.rcParams.update({
                        "figure.figsize": (12, 5),
                        "axes.titlesize": 16,
                        "axes.labelsize": 10,
                        "xtick.labelsize": 10,
                        "ytick.labelsize": 10,
                        "legend.fontsize": 6,
                        "lines.linewidth": 2,
                        "lines.markersize": 6,
                        "font.family": "serif",
                    })

                    palette = sns.color_palette("tab10", n_colors=len(selected_sensors))
                    fig, ax = plt.subplots()
                    for i, col in enumerate(selected_sensors):
                        ax.plot(filtered_x, filtered_df[col], label=col, color=palette[i], linestyle="-", marker=None)

                    ax.set_xlabel(x_label, fontsize=14)
                    ax.set_ylabel(y_label, fontsize=14)
                    ax.set_title(title, fontsize=16, fontweight="bold")
                    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
                    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False)
                    plt.tight_layout()
                    st.pyplot(fig)

                    # --- Save plot button ---
                    if st.button(f"Save Plot: {df_name}"):
                        filename = f"{df_name}_Experiment{experiment_id}_{x_start}_{x_end}.png"
                        filepath = os.path.join(save_folder, filename)
                        fig.savefig(filepath, dpi=300)
                        st.success(f"Plot saved as {filepath}")

        else:
            st.warning("No data available for selected Experiment ID and datasets.")

    @log_function
    def interactive_plot_jupyter(
        self, df_names: list, x_axes: list, min_id=2, max_id=318
    ):
        """
        Create an interactive widget for Jupyter to visualize experiment data.

        Args:
            df_names (list): Names of datasets to plot.
            x_axes (list): X-axis column names for each dataset.
            min_id (int): Minimum Experiment_ID for the slider.
            max_id (int): Maximum Experiment_ID for the slider.

        Returns:
            None: Displays interactive widgets and plots inline in Jupyter.
        """
        experiment_selector = widgets.IntSlider(
            value=min_id,
            min=min_id,
            max=max_id,
            step=1,
            description="Experiment ID:",
            continuous_update=False,
        )

        output = widgets.Output()

        def update_plot(change):
            with output:
                clear_output(wait=True)
                loaded_dfs = self.loader.load_data_by_experiment_from_sqlite(
                    experiment_selector.value
                )
                dfs = [loaded_dfs[name] for name in df_names if name in loaded_dfs]
                self.multi_sensor_experiment(
                    dfs=dfs,
                    experiment_id=experiment_selector.value,
                    df_names=df_names,
                    x_axes=x_axes,
                    save_fig=False,
                )

        experiment_selector.observe(update_plot, names="value")
        display(experiment_selector, output)
        update_plot(None)
