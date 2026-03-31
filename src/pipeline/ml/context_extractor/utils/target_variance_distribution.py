from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker


DEFAULT_INPUT_CSV = Path("data/ml/unique_bending_setups.csv")
DEFAULT_ARC_CSV = Path("data/processed/arc.csv")
DEFAULT_OUTPUT_DIR = Path("results/target_variance_distribution")
DEFAULT_SPLIT_CONFIG = Path("config/data-split-config/train_test_split_each_setup_80.json")
EXPERIMENT_COL = "Experiment_Number"
ANGLE_COL = "Angle[degree]ORDistance[mm]"
GREY_AREA_COLOR = "grey"
GREY_AREA_ALPHA = 0.15


def _parse_groups(values: Iterable) -> list[list[int]]:
    groups = []
    for value in values:
        if isinstance(value, str):
            value = ast.literal_eval(value)
        if isinstance(value, (int, np.integer)):
            value = [int(value)]
        groups.append([int(v) for v in value])
    return groups


def _explode_groups(df: pd.DataFrame, group_col: str = EXPERIMENT_COL) -> pd.DataFrame:
    out = df.copy()
    out[group_col] = _parse_groups(out[group_col])
    return out.explode(group_col).rename(columns={group_col: "Experiment_ID"}).reset_index(drop=True)


def _target_columns(df: pd.DataFrame) -> list[str]:
    ignore = {"Experiment_ID", EXPERIMENT_COL}
    numeric = df.select_dtypes(include=[np.number]).columns
    return [c for c in numeric if c not in ignore]


def _feature_columns(df: pd.DataFrame, angle_col: str = ANGLE_COL) -> list[str]:
    return [c for c in _target_columns(df) if c != angle_col]


def _load_split_ids(split_config_path: str | Path) -> tuple[list[int], list[int]]:
    with open(split_config_path, "r", encoding="utf-8") as f:
        split_cfg = json.load(f)
    return (
        sorted({eid for group in split_cfg["train_groups"] for eid in group}),
        sorted({eid for group in split_cfg["test_groups"] for eid in group}),
    )


def _load_split_groups(split_config_path: str | Path, key: str = "test_groups") -> list[list[int]]:
    with open(split_config_path, "r", encoding="utf-8") as f:
        split_cfg = json.load(f)
    return [[int(eid) for eid in group] for group in split_cfg.get(key, [])]


def _interp_to_grid(dfg: pd.DataFrame, x_grid: np.ndarray, feature: str, angle_col: str) -> np.ndarray:
    dfg = dfg[[angle_col, feature]].dropna().sort_values(angle_col)
    x = dfg[angle_col].to_numpy()
    y = dfg[feature].to_numpy()
    if len(x) < 2:
        return np.full_like(x_grid, np.nan, dtype=float)
    y_interp = np.interp(x_grid, x, y)
    y_interp[x_grid < x.min()] = np.nan
    y_interp[x_grid > x.max()] = np.nan
    return y_interp


def compute_target_variance_distribution(
    input_csv: str | Path = DEFAULT_INPUT_CSV,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> pd.DataFrame:
    df = _explode_groups(pd.read_csv(input_csv))
    targets = _target_columns(df)
    summary = pd.DataFrame(
        {
            "target": targets,
            "variance": [float(df[c].var()) for c in targets],
            "std": [float(df[c].std()) for c in targets],
            "mean": [float(df[c].mean()) for c in targets],
            "min": [float(df[c].min()) for c in targets],
            "max": [float(df[c].max()) for c in targets],
        }
    ).sort_values("variance", ascending=False, ignore_index=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "target_variance_distribution.csv", index=False)

    plt.figure(figsize=(max(8, len(summary) * 0.6), 5))
    plt.bar(summary["target"], summary["variance"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Variance")
    plt.title("Target Variance Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "target_variance_distribution.png", dpi=300)
    plt.close()

    return summary


def plot_mean_std_by_angle_train_test(
    split_config_path: str | Path = DEFAULT_SPLIT_CONFIG,
    arc_csv_path: str | Path = DEFAULT_ARC_CSV,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    angle_col: str = ANGLE_COL,
    experiment_col: str = "Experiment_ID",
    feature_cols: list[str] | None = None,
) -> list[Path]:
    df = pd.read_csv(arc_csv_path)
    feature_cols = feature_cols or _feature_columns(df, angle_col)
    train_ids, test_ids = _load_split_ids(split_config_path)
    df_train = df[df[experiment_col].isin(train_ids)]
    df_test = df[df[experiment_col].isin(test_ids)]
    output_dir = Path(output_dir) / Path(split_config_path).stem
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for col in feature_cols:
        train_stats = df_train.groupby(angle_col)[col].agg(["mean", "std"]).reset_index()
        test_stats = df_test.groupby(angle_col)[col].agg(["mean", "std"]).reset_index()
        fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharey="row")
        axes[0, 0].scatter(df_train[angle_col], df_train[col], s=12, alpha=0.3, color="#1f77b4")
        axes[0, 0].plot(train_stats[angle_col], train_stats["mean"], color="#1f77b4", linewidth=2)
        axes[0, 0].set_title("Train")
        axes[0, 1].scatter(df_test[angle_col], df_test[col], s=12, alpha=0.3, color="#d62728")
        axes[0, 1].plot(test_stats[angle_col], test_stats["mean"], color="#d62728", linewidth=2)
        axes[0, 1].set_title("Test")
        axes[0, 2].plot(train_stats[angle_col], train_stats["mean"], color="#1f77b4", linewidth=2, label="Train")
        axes[0, 2].plot(test_stats[angle_col], test_stats["mean"], color="#d62728", linewidth=2, label="Test")
        axes[0, 2].set_title("Train vs Test")
        for ax, stats, color in ((axes[1, 0], train_stats, "#1f77b4"), (axes[1, 1], test_stats, "#d62728")):
            ax.plot(stats[angle_col], stats["mean"], color=color, linewidth=2)
            ax.fill_between(stats[angle_col], stats["mean"] - stats["std"], stats["mean"] + stats["std"], color=color, alpha=0.2)
        axes[1, 2].plot(train_stats[angle_col], train_stats["mean"], color="#1f77b4", linewidth=2, label="Train")
        axes[1, 2].fill_between(train_stats[angle_col], train_stats["mean"] - train_stats["std"], train_stats["mean"] + train_stats["std"], color="#1f77b4", alpha=0.2)
        axes[1, 2].plot(test_stats[angle_col], test_stats["mean"], color="#d62728", linewidth=2, label="Test")
        axes[1, 2].fill_between(test_stats[angle_col], test_stats["mean"] - test_stats["std"], test_stats["mean"] + test_stats["std"], color="#d62728", alpha=0.2)
        for ax in axes.flat:
            ax.grid(True, alpha=0.2)
        fig.tight_layout()
        save_path = output_dir / f"mean_std_by_angle_train_test_{col}.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(save_path)

    return saved_paths


def plot_group_variations_vs_baseline(
    split_config_path: str | Path = DEFAULT_SPLIT_CONFIG,
    arc_csv_path: str | Path = DEFAULT_ARC_CSV,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    angle_col: str = ANGLE_COL,
    experiment_col: str = "Experiment_ID",
    feature_cols: list[str] | None = None,
    groups_key: str = "test_groups",
) -> list[Path]:
    df = pd.read_csv(arc_csv_path)
    feature_cols = feature_cols or _feature_columns(df, angle_col)
    output_dir = Path(output_dir) / "variations" / Path(split_config_path).stem
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for experiment_group in _load_split_groups(split_config_path, key=groups_key):
        df_selected = df[df[experiment_col].isin(experiment_group)].copy()
        if df_selected.empty:
            continue

        x_grid = np.sort(df_selected[angle_col].dropna().unique())
        if len(x_grid) == 0:
            continue

        exp_groups = {
            exp_id: dfg for exp_id, dfg in df_selected.groupby(experiment_col)
        }

        selected_features = df.columns.tolist()[:4]
        print(selected_features)
        available_features = [f for f in selected_features if f in feature_cols]
        if not available_features:
            continue

        fig, axs = plt.subplots(1, len(available_features), figsize=(12, 4), sharex=True)
        axs = np.atleast_1d(axs).ravel()

        for i, feature in enumerate(available_features):
            ax = axs[i]
            y_stack = np.vstack(
                [_interp_to_grid(dfg, x_grid, feature, angle_col) for dfg in exp_groups.values()]
            )
            y_base = np.nanmean(y_stack, axis=0)
            ax.plot(x_grid, y_base, linewidth=2)

            for _, dfg in exp_groups.items():
                y_exp = _interp_to_grid(dfg, x_grid, feature, angle_col)
                mask = np.isfinite(y_exp) & np.isfinite(y_base)
                ax.plot(x_grid, y_exp)
                ax.fill_between(
                    x_grid[mask],
                    y_exp[mask],
                    y_base[mask],
                    color=GREY_AREA_COLOR,
                    alpha=GREY_AREA_ALPHA,
                    linewidth=0,
                )

            ax.set_xlabel("Angle", fontsize=14)
            ax.tick_params(axis='x', labelsize=12, rotation=0)
            ax.tick_params(axis='y', labelsize=12)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(6))  # max ~6 ticks
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.grid(True)

        for ax in axs[len(available_features):]:
            ax.axis("off")

        fig.tight_layout()
        save_path = output_dir / f"variation_{'_'.join(map(str, experiment_group))}.pdf"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(save_path)

    return saved_paths
