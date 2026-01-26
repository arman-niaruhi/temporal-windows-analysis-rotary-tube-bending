# Tube Geometry Prediction

End-to-end pipeline for tube bending data: ETL from raw experiments, activity recognition labeling, context extraction, and springback prediction, plus a Streamlit dashboard for visualization and inference.

## What is in this repo

- ETL pipeline that reads a pickled dataset and builds a SQLite database.
- Annotation GUI for machine activity labels.
- Activity recognition training and inference.
- Context extraction models and interpretability plots.
- Springback prediction (random forest and LSTM).
- Streamlit dashboard for plots, tables, and inference views.

## Quickstart

1) Create and activate a virtual environment, then install dependencies:

```bash
python -m venv tube-venv
source tube-venv/bin/activate
pip install -r requirements.txt
```

2) Place the raw dataset pickle here:

```
data/raw/experiments_process_and_results.pkl
```

The dataset can be downloaded from:

```
https://github.com/zeyneddinoz/tubebend
```

3) Run the ETL pipeline to build the SQLite database:

```bash
python 0_data_etl.py
```

4) (Optional) Launch the annotator to create or edit activity labels:

```bash
python 1_annotator.py
```

5) Train activity recognition models:

```bash
python 2_activtiy_recognition.py
```

6) Train context extraction models:

```bash
python 4_context_extractor.py
```

7) Train springback prediction models:

```bash
python 6_springback_predictor.py
```

8) Run the Streamlit dashboard:

```bash
streamlit run 5_dashboard_app.py
```

## Data layout

- `data/raw/experiments_process_and_results.pkl` raw dataset input
- `data/processed/tube_geometry.db` SQLite database produced by ETL
- `data/ml/` annotations and experiment ID lists used by ML pipelines

## Main pipelines and scripts

- `0_data_etl.py` runs the ETL pipeline using `config/preprocessing/preprocessing_config.json`.
- `1_annotator.py` opens a PyQt-based labeling GUI and saves/loads annotation JSON.
- `2_activtiy_recognition.py` trains classifiers and (optionally) runs analysis and plots.
- `3_split_data.py` generates grouped train/test splits and writes configs to `config/data-split-config/`.
- `4_context_extractor.py` trains context extraction models and logs MLflow runs.
- `5_dashboard_app.py` launches the Streamlit dashboard (plots, tables, inference).
- `6_springback_predictor.py` trains random forest and LSTM springback models.

## Configuration

- `config/preprocessing/preprocessing_config.json` controls ETL, normalization, and column filtering.
- `config/machine-activity-recognition/machine-activity-recognition-config.json` controls activity recognition training and inference.
- `config/context-extraction/context-extraction-config.json` controls context extraction and interpretability settings.
- `config/springback-prediction/springback-prediction-config.json` controls springback prediction training.
- `config/data-split-config/` contains split rules and generated files for experiment grouping.

## Outputs

- `data/processed/` SQLite database from ETL.
- `models/` trained model artifacts.
- `results/` plots, predictions, and feature analyses.
- `mlruns/` MLflow tracking runs (used by the dashboard).

## Notes

- The annotator and some plots require a GUI backend (PyQt5 or PySide6).
- If you change dataset paths or labels, update the matching config JSON files.
- For reproducibility, the context extraction and springback pipelines use the configured seed values.
