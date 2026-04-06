from __future__ import annotations

import ast
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from collections import defaultdict


# DOE split reference used by `split_based_on_column_value()`:
# - `gp=1..5`: fixed collet boost, aggregated across the listed mandrel timings
# - `gp=6..9`: fixed mandrel retraction timing, aggregated across the listed
#   collet boosts
#
# Group summary:
# - `1`: timing `[0, 2, 5, 10]`, collet `0.85`, 41 distinct configurations, 108 experiments
# - `2`: timing `[0, 5, 10]`, collet `0.87`, 12 distinct configurations, 41 experiments
# - `3`: timing `[0, 2, 5, 10]`, collet `0.90`, 21 distinct configurations, 72 experiments
# - `4`: timing `[0, 5, 10]`, collet `0.92`, 12 distinct configurations, 42 experiments
# - `5`: timing `[0, 2, 5, 10]`, collet `0.95`, 14 distinct configurations, 52 experiments
# - `6`: timing `0`, collet `[0.85, 0.87, 0.90, 0.92, 0.95]`, 20 distinct configurations, 71 experiments
# - `7`: timing `2`, collet `[0.85, 0.90, 0.95]`, 40 distinct configurations, 104 experiments
# - `8`: timing `5`, collet `[0.85, 0.87, 0.90, 0.92, 0.95]`, 20 distinct configurations, 70 experiments
# - `9`: timing `10`, collet `[0.85, 0.87, 0.90, 0.92, 0.95]`, 20 distinct configurations, 70 experiments


# -----------------------------
# 1) Configuration
# -----------------------------

SAVE_DIR = Path("config") / "data-split-config"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = Path("data/ml/unique_bending_setups.csv")

RANDOM_SEED = 42

experiments_df = pd.read_csv(INPUT_CSV)

GROUPS = (
    experiments_df["Experiment_Number"]
    .apply(ast.literal_eval)
    .tolist()
)


DROP_COLUMNS = [
    "Outer-diameter",
    "Wall-thickness",
    "Target-angle",
    "Wiper-die shortening",
    "Mandrel position",
    "Tube_numbers",
    "Pressure-die lateral position",
    "Clamp-die lateral position",
]

EXPERIMENT_COL = "Experiment_Number"

# Columns used by split rules
COLLET_COL = "Collet boost"
TIMING_COL = "Mandrel retraction timing"


# -----------------------------
# 2) Helpers
# -----------------------------


def _ensure_required_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns: {missing}. Available: {list(df.columns)}"
        )


def _parse_list_cell(x) -> List[int]:
    """Parse a cell that should represent a list of ints (or a scalar int)."""
    if isinstance(x, list):
        return [int(v) for v in x]
    if isinstance(x, (int, np.integer)):
        return [int(x)]
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, str):
        parsed = ast.literal_eval(x)
        if isinstance(parsed, (list, tuple, np.ndarray)):
            return [int(v) for v in parsed]
        return [int(parsed)]
    if isinstance(x, (float, np.floating)):
        return [int(x)]
    raise ValueError(f"Unexpected value for experiment list: {x!r} ({type(x)})")


def _categorical_normalize(series: pd.Series) -> Tuple[pd.Series, Dict]:
    """Map discrete classes to ordinal values in [0,1]."""
    classes = sorted(series.dropna().unique())
    n = len(classes)
    if n <= 1:
        mapping = {classes[0]: 0.0} if n == 1 else {}
        return series.map(mapping), mapping

    mapping = {cls: i / (n - 1) for i, cls in enumerate(classes)}
    return series.map(mapping), mapping


def _minmax_normalize(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    mn, mx = s.min(), s.max()
    if mn == mx:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)


@dataclass(frozen=True)
class SplitResult:
    train_groups: List[List[int]]
    test_groups: List[List[int]]
    meta: Dict

    def to_dict(self) -> Dict:
        return {
            "train_groups": self.train_groups,
            "test_groups": self.test_groups,
            **self.meta,
        }


def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_exp_to_group(groups: Sequence[Sequence[int]]) -> Dict[int, int]:
    exp_to_group: Dict[int, int] = {}
    for gi, grp in enumerate(groups):
        for exp_id in grp:
            exp_to_group[int(exp_id)] = gi
    return exp_to_group


# -----------------------------
# 3) Data preparation
# -----------------------------


def load_and_prepare_setup_df(
    input_csv: Path = INPUT_CSV,
    normalize: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Returns:
      df_expanded: one row per Experiment_ID
      mappings: dict of categorical normalization maps (if any)
    """
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv.resolve()}")

    df = pd.read_csv(input_csv)

    # Basic schema checks early
    _ensure_required_columns(df, [EXPERIMENT_COL, COLLET_COL, TIMING_COL])

    # Drop columns if present (ignore missing gracefully)
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    mappings: Dict[str, Dict] = {}

    if normalize:
        exclude = {EXPERIMENT_COL}
        cols_to_normalize = [c for c in df.columns if c not in exclude]

        for col in cols_to_normalize:
            # Decide categorical vs numeric
            nunique = df[col].nunique(dropna=True)
            if nunique <= 10:
                df[col], mappings[col] = _categorical_normalize(df[col])
            else:
                df[col] = _minmax_normalize(df[col])

    # Expand Experiment_Number to Experiment_ID
    df[EXPERIMENT_COL] = df[EXPERIMENT_COL].apply(_parse_list_cell)
    df_expanded = (
        df.explode(EXPERIMENT_COL)
        .rename(columns={EXPERIMENT_COL: "Experiment_ID"})
        .reset_index(drop=True)
    )
    df_expanded["Experiment_ID"] = df_expanded["Experiment_ID"].astype(int)

    return df_expanded, mappings


# -----------------------------
# 4) Split strategies
# -----------------------------


def split_random_groups(
    groups: Sequence[Sequence[int]],
    test_ratio: float = 0.15,
    seed: int = RANDOM_SEED,
    keep_if: callable = lambda x: True,
) -> SplitResult:
    """
    Randomly assigns whole groups to test until reaching ~test_ratio of total samples.
    """
    rng = random.Random(seed)

    filtered = [[int(n) for n in grp if keep_if(int(n))] for grp in groups]
    filtered = [grp for grp in filtered if grp]

    rng.shuffle(filtered)

    total = sum(len(g) for g in filtered)
    target = int(round(total * test_ratio))

    train, test = [], []
    current = 0

    for grp in filtered:
        if current < target:
            test.append(grp)
            current += len(grp)
        else:
            train.append(grp)

    return SplitResult(
        train_groups=train,
        test_groups=test,
        meta={
            "split_rule": f"random whole groups until ~{test_ratio:.0%} samples",
            "seed": seed,
        },
    )


def split_based_on_column_value(
    groups: Sequence[Sequence[int]],
    df_expanded: pd.DataFrame,
    gp: int,
) -> SplitResult:
    """
    Any group that contains an Experiment_ID meeting the selected DOE slice goes
    to test.
    """
    group_rules = {
        1: (COLLET_COL, 0.85),
        2: (COLLET_COL, 0.87),
        3: (COLLET_COL, 0.90),
        4: (COLLET_COL, 0.92),
        5: (COLLET_COL, 0.95),
        6: (TIMING_COL, 0.0),
        7: (TIMING_COL, 2.0),
        8: (TIMING_COL, 5.0),
        9: (TIMING_COL, 10.0),
    }

    if gp not in group_rules:
        raise ValueError(f"Unknown gp={gp}. Expected 1..9.")

    column, value = group_rules[gp]
    _ensure_required_columns(df_expanded, ["Experiment_ID", column])

    test_ids: Set[int] = set(
        df_expanded.loc[df_expanded[column] == value, "Experiment_ID"]
        .astype(int)
        .tolist()
    )

    train_groups, test_groups = [], []
    for grp in groups:
        grp_set = set(map(int, grp))
        if grp_set & test_ids:
            test_groups.append(list(map(int, grp)))
        else:
            train_groups.append(list(map(int, grp)))

    all_ids = [experiment_id for gp in test_groups for experiment_id in gp]
    print(f"The group {gp} has the length of:{len(all_ids)}")

    return SplitResult(
        train_groups=train_groups,
        test_groups=test_groups,
        meta={"split_rule": f"test if any Experiment_ID has {column} == {value}"},
    )


def split_80_20_per_cell(
    groups: Sequence[Sequence[int]],
    df_expanded: pd.DataFrame,
    cell_cols: Tuple[str, str] = (COLLET_COL, TIMING_COL),
    test_ratio: float = 0.20,
    seed: int = RANDOM_SEED,
) -> SplitResult:
    """
    For each (collet_boost, timing) cell, split the *group-ids present in that cell* 80/20.
    Train wins on overlaps.
    """
    c1, c2 = cell_cols
    _ensure_required_columns(df_expanded, ["Experiment_ID", c1, c2])

    exp_to_group = build_exp_to_group(groups)

    # cell -> set(group_id)
    cell_to_group_ids: Dict[Tuple[float, float], Set[int]] = defaultdict(set)

    for _, row in df_expanded.iterrows():
        exp_id = int(row["Experiment_ID"])
        gid = exp_to_group.get(exp_id)
        if gid is None:
            continue
        cell = (row[c1], row[c2])
        cell_to_group_ids[cell].add(gid)

    rng = np.random.RandomState(seed)

    train_ids: Set[int] = set()
    test_ids: Set[int] = set()

    for _, group_ids in cell_to_group_ids.items():
        group_ids = list(group_ids)
        rng.shuffle(group_ids)

        n_total = len(group_ids)
        if n_total == 0:
            continue

        n_test = max(1, int(test_ratio * n_total))
        test_ids.update(group_ids[:n_test])
        train_ids.update(group_ids[n_test:])

    # resolve overlap: train wins
    test_ids -= train_ids

    train_groups = [list(map(int, groups[i])) for i in sorted(train_ids)]
    test_groups = [list(map(int, groups[i])) for i in sorted(test_ids)]

    return SplitResult(
        train_groups=train_groups,
        test_groups=test_groups,
        meta={
            "split_rule": f"{int((1-test_ratio)*100)}/{int(test_ratio*100)} per ({c1} × {c2}) cell, train wins overlaps",
            "seed": seed,
            "cell_cols": [c1, c2],
        },
    )


# -----------------------------
# 5) Runner / Output
# -----------------------------


def run_and_save(
    strategy: str = "each_setup",
    gp: int = 1,
    save_dir: Path = SAVE_DIR,
) -> None:
    """
    strategy:
      - 'each_setup'      -> 80/20 per DOE cell (collet × timing)
      - 'randomly'        -> random whole groups to ~15% samples
      - 'based_on_column' -> groups to test by collet boost (gp 1-5) or timing (gp 6-9)
    """
    normalize = strategy != "based_on_column"
    df_expanded, mappings = load_and_prepare_setup_df(INPUT_CSV, normalize=normalize)

    # Always save expanded setups (useful for debugging)
    save_csv(df_expanded, save_dir / "experiment_setups.csv")

    if mappings:
        save_json(mappings, save_dir / "normalization_mappings.json")

    if strategy == "based_on_column":
        out_name = f"train_test_split_{strategy}_gp{gp}.json"
    else:
        out_name = f"train_test_split_{strategy}.json"
    if strategy == "each_setup_80":
        split = split_80_20_per_cell(
            GROUPS, df_expanded, test_ratio=0.20, seed=RANDOM_SEED
        )

    elif strategy == "randomly":
        split = split_random_groups(GROUPS, test_ratio=0.15, seed=RANDOM_SEED)

    elif strategy == "based_on_column":
        split = split_based_on_column_value(
            GROUPS, df_expanded, gp
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    out_path = save_dir / out_name
    save_json(split.to_dict(), out_path)

    # Small, useful console summary
    print("Saved:")
    print(" -", (save_dir / "experiment_setups.csv").resolve())
    if mappings:
        print(" -", (save_dir / "normalization_mappings.json").resolve())
    print(" -", out_path.resolve())
    print()
    print("Split summary:")
    print(" - train_groups:", len(split.train_groups))
    print(" - test_groups :", len(split.test_groups))
    print(" - rule        :", split.meta.get("split_rule"))


if __name__ == "__main__":
    # Run all strategies (based_on_column runs gp=1..9)
    run_and_save(strategy="each_setup_80")
    run_and_save(strategy="randomly")
    for gp in range(1, 10):
        run_and_save(strategy="based_on_column", gp=gp)
