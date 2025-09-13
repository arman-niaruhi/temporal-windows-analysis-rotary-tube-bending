#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 09:12:01 2025

@author: zoz
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import re

import warnings
warnings.filterwarnings('ignore')


def take_all_bending_setups(experiments_process_and_results, machine_part) -> pd.DataFrame:
    """
    Extracts all bending setups to a Pandas DataFrame and adds an 'Experiment_ID' column
    if not already present.
    """
    if machine_part != "bending_setups":
        df_list = []
        for experiment_name, experiment_data in experiments_process_and_results.items():
            df = experiment_data[machine_part].copy()
            # Always move or insert Experiment_ID as the first column
            if 'Experiment_ID' in df.columns:
                cols = ['Experiment_ID'] + [col for col in df.columns if col != 'Experiment_ID']
                df = df[cols]
            else:
                # Insert new column at position 0
                df.insert(0, 'Experiment_ID', experiment_name)
            
            df_list.append(df)
            
        result = pd.concat(df_list, ignore_index=True)
        result['Experiment_ID'] = result['Experiment_ID'].str.replace("Exp_","")
        return result
    
    result = pd.concat([experiments_process_and_results[experiment_name]['bending_setups'] for experiment_name in experiments_process_and_results.keys()], ignore_index=True)
    result.columns = result.columns.str.replace("Experiment", "Experiment_ID")
    return result


def multi_sensor_subplots(df: pd.DataFrame, 
                          save_fig=False, 
                          output_path="machine_process_flow.html") -> go.Figure:  
    
    
    def extract_y_label(col_name: str) -> str:
        """
        Extracts signal type and unit for y-axis label.
        E.g., "MANDREL_AXIAL_Load_[kN]" -> "Load [kN]"
        """
        match = re.search(r'([A-Za-z0-9\- ]+_\[[^]]+\])$', col_name)
        return match.group(1).replace('_', ' ') if match else col_name.replace('_', ' ')

    def extract_trace_name(col_name: str) -> str:
        """
        Extracts the part before the measurement and unit.
        E.g., "SENSOR_MANDREL_AXIAL_Load_[kN]" -> "SENSOR MANDREL AXIAL"
        """
        parts = col_name.split('_')
        for i in range(len(parts)):
            if re.match(r'^[A-Za-z0-9\-]+\_\[[^]]+\]$', '_'.join(parts[i:])):
                return ' '.join(parts[:i]).replace('_', ' ')
        return ' '.join(parts[:-2]).replace('_', ' ') 
    
    
    time = df.iloc[:, 0]
    sensor_cols = df.columns[1:]

    numeric_cols = []
    for col in sensor_cols:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].notna().any():
                numeric_cols.append(col)
        except Exception:
            continue

    num_sensors = len(numeric_cols)

    # Subplot titles = descriptive sensor names (not units)
    subplot_titles = [extract_trace_name(col) for col in numeric_cols]

    fig = make_subplots(
        rows=num_sensors, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=subplot_titles
    )

    for i, col in enumerate(numeric_cols, start=1):
        y_values = df[col]
        y_min = y_values.min()
        y_max = y_values.max()
        margin = (y_max - y_min) * 0.1 if y_max != y_min else 1

        fig.add_trace(
            go.Scatter(
                x=time,
                y=y_values,
                mode='lines',
                name=extract_trace_name(col),  # for tooltip only
                line=dict(width=1)
            ),
            row=i, col=1
        )

        fig.update_yaxes(
            title_text=extract_y_label(col),  # this is the y-axis label on left
            range=[y_min - margin, y_max + margin],
            row=i, col=1
        )

    fig.update_xaxes(title_text='Time [s]', row=num_sensors, col=1)

    fig.update_layout(
        # title="Sensor Time Series - One per Subplot",
        height=300 * num_sensors,
        width=1400,
        showlegend=False,
        hovermode='x unified'
    )

    if save_fig:
        fig.write_html(output_path)
        print(f"Interactive subplot saved to {output_path}")

    return fig