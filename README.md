# Tube Geometry Prediction

End-to-end pipeline for tube bending experiments: preprocessing raw production data, annotating process phases, training machine-learning models for activity recognition and context extraction, predicting springback, and inspecting results in a Streamlit dashboard.

## Project Scope

This repository covers five main workflows:

- Data ETL from the raw experiment pickle into structured CSV tables
- Manual annotation of machine phases with a PyQt GUI
- Activity recognition for process-phase labeling
- Context extraction for predicting setup-dependent target features
- Springback prediction with random forest and TCN-LSTM models

## Repository Layout

```text
.
|-- 0_0_data_etl.py
|-- 1_0_annotator.py
|-- 1_1_activtiy_recognition.py
|-- 2_1_split_data.py
|-- 2_2_context_extractor.py
|-- 3_springback_predictor.py
|-- 4_dashboard_app.py
|-- config/
|-- data/
|-- models/
|-- results/
|-- src/
```

## Setup

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The project uses:

- `PyQt5` or `PySide6` for the annotator
- `PyTorch` and `scikit-learn` for modeling
- `MLflow` for context-extraction experiment tracking
- `Streamlit` for the dashboard

## Data Requirements

Place the raw dataset pickle at:

```text
data/raw/experiments_process_and_results.pkl
```

The raw source dataset is referenced from:

```text
https://github.com/zeyneddinoz/tubebend
```

Some ML steps also expect these repository-local files to exist:

- `data/ml/machine-and-movement_complete.json`
- `data/ml/unique_bending_setups.csv`

## Run Order

### 1. Preprocess the raw dataset

Generates normalized CSV tables under `data/processed`.

```bash
python 0_0_data_etl.py
```

Configuration:

- `config/preprocessing/preprocessing_config.json`

### 2. Create or update annotations

Launches the desktop annotation tool for labeling process phases such as `Clamping`, `Bending`, `Mandrel Extraction`, and `De-Clamping`.

```bash
python 1_0_annotator.py
```

Notes:

- Requires a GUI-capable environment
- Reads processed CSV data from `data/processed`

### 3. Train activity-recognition models

Trains sequence models for process-phase classification and can also generate plots and per-experiment inference outputs.

```bash
python 1_1_activtiy_recognition.py
```

Configuration:

- `config/machine-activity-recognition/machine-activity-recognition-config.json`

Default behavior from the current config:

- Uses `data/processed` as input
- Reads annotations from `data/ml/machine-and-movement_complete.json`
- Reads experiment group definitions from `data/ml/unique_bending_setups.csv`
- Stores classifier artifacts in `models/classifier`
- Stores plots in `results/activity_recognition`

### 4. Generate grouped train/test splits

Builds split files from setup metadata and writes them to `config/data-split-config/`.

```bash
python 2_1_split_data.py
```

Input:

- `data/ml/unique_bending_setups.csv`

Outputs include:

- `train_test_split_each_setup_80.json`
- `train_test_split_randomly.json`
- `train_test_split_based_on_column_gp*.json`
- `normalization_mappings.json`

### 5. Train the context-extraction model

Trains the context-extraction pipeline and logs runs to MLflow.

```bash
python 2_2_context_extractor.py
```

Configuration:

- `config/context-extraction/context-extraction-config.json`

Current config highlights:

- Input process part: `Bending`
- Split file: `config/data-split-config/train_test_split_each_setup_80.json`
- Target feature indices: `[1, 3]`
- Model type: `tcn_lstm`
- Model artifacts path: `models/context_extraction`

### 6. Train springback prediction models

Runs both:

- a random forest baseline
- a TCN-LSTM springback regressor

```bash
python 3_springback_predictor.py
```

Configuration:

- `config/springback-prediction/springback-prediction-config.json`

Current config highlights:

- Input process part: `All`
- Split file: `config/data-split-config/train_test_split_each_setup_80.json`
- Target window count: `400`
- Model artifacts path: `models/spring_back`

### 7. Launch the dashboard

Starts the Streamlit app for browsing plots, tables, activity-recognition outputs, and context-extraction artifacts.

```bash
streamlit run 4_dashboard_app.py
```

The dashboard expects:

- processed data in `data/processed`
- MLflow runs in `mlruns/`
- generated results in `results/`

## Important Paths

### Config

- `config/preprocessing/preprocessing_config.json`
- `config/machine-activity-recognition/machine-activity-recognition-config.json`
- `config/context-extraction/context-extraction-config.json`
- `config/springback-prediction/springback-prediction-config.json`
- `config/data-split-config/`

### Data

- `data/raw/experiments_process_and_results.pkl`
- `data/processed/`
- `data/ml/machine-and-movement_complete.json`
- `data/ml/unique_bending_setups.csv`

### Outputs

- `models/`
- `results/`
- `mlruns/`

## Notes

- The annotator and some plotting workflows require a desktop/GUI environment.
- Several scripts depend on relative paths, so run them from the repository root.
- The file name `1_1_activtiy_recognition.py` is intentionally spelled that way in the repository.
