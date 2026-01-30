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


# -----------------------------
# 1) Configuration
# -----------------------------

SAVE_DIR = Path("config") / "data-split-config"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = Path("data/ml/unique_bending_setups.csv")

RANDOM_SEED = 42

# Original groups
GROUPS: List[List[int]] = [
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9],
    [10, 11],
    [12, 13],
    [14, 15, 50],
    [16, 17],
    [18, 19],
    [20, 21, 51, 52, 53],
    [22, 23, 54],
    [55],
    [24, 25, 43, 45, 46, 47],
    [56],
    [26, 27],
    [28, 29],
    [30, 31, 44],
    [32, 33],
    [34, 35],
    [36, 37],
    [38],
    [39],
    [40],
    [41],
    [42],
    [57],
    [49],
    [193, 194, 195],
    [268, 269, 270],
    [241, 242, 243],
    [253, 257, 259],
    [196, 197, 198],
    [223, 224, 225],
    [265, 266, 267],
    [238, 239, 240],
    [254, 256, 260],
    [220, 221, 222],
    [208, 209, 210],
    [262, 263, 264],
    [235, 236, 237],
    [255, 258, 261],
    [205, 206, 207],
    [190, 191, 192],
    [280, 281, 282, 283, 284],
    [214, 215, 216],
    [211, 212, 213],
    [271, 272, 273],
    [274, 275, 276],
    [277, 278, 279],
    [290, 291, 292],
    [293, 294, 295],
    [296, 297, 298],
    [299, 300, 301],
    [226, 227, 228],
    [302, 303, 304, 317, 318],
    [229, 230, 231],
    [232, 233, 234],
    [305, 306, 307],
    [308, 309, 310],
    [311, 312, 313],
    [314, 315, 316],
    [244, 247, 250],
    [245, 248, 251],
    [246, 249, 252],
    [199, 200, 201],
    [285, 286, 287, 288, 289],
    [217, 218, 219],
    [202, 203, 204],
    [59, 90, 129, 152],
    [88, 115, 184, 185],
    [69, 103, 134, 154],
    [83, 110, 182, 183],
    [70, 92, 135, 165],
    [60, 99, 130, 151],
    [87, 114, 186, 187],
    [68, 102, 133, 155],
    [84, 111, 180, 181],
    [71, 93, 136, 164],
    [61, 100, 131, 150, 153],
    [86, 113, 188, 189],
    [67, 101, 132, 156],
    [85, 112, 178, 179],
    [72, 94, 137, 163],
    [58, 89, 146, 147],
    [119, 120, 121, 122, 123],
    [62, 98, 145, 148],
    [63, 97, 144, 149],
    [77, 116, 168, 174],
    [78, 117, 167, 173],
    [79, 118, 172],
    [64, 76, 104, 141, 159],
    [65, 105, 142, 158],
    [66, 106, 143, 157],
    [82, 109, 169, 175],
    [81, 108, 170, 176],
    [80, 107, 171, 177],
    [75, 91, 140, 160],
    [124, 125, 126, 127, 128],
    [74, 96, 139, 161],
    [73, 95, 138, 162],
]


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
    Any group that contains an Experiment_ID meeting condition goes to test.
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



'''

------------------Collet boost-------------------------

    Mandrel retraction timing  Collet boost

Group 1:    
                    0.0          0.85
                    2.0          0.85
                    5.0          0.85
                    10.0         0.85


Group 2:  
                    0.0          0.87
                    5.0          0.87
                    10.0         0.87


Group 3:  
                    0.0          0.90
                    2.0          0.90
                    5.0          0.90
                    10.0         0.90


Group 4:  
                    0.0          0.92
                    5.0          0.92
                    10.0         0.92


Group 5:  
                    0.0          0.95
                    2.0          0.95
                    5.0          0.95
                    10.0         0.95

------------------Manderl timing-------------------------

Group 6:      
                    0.0          0.00
                    0.0          0.25
                    0.0          0.50
                    0.0          0.75
                    0.0          1.00


Group 7:  
                    2.0          0.00
                    2.0          0.50
                    2.0          1.00


Group 8:  
                    5.0          0.00
                    5.0          0.25
                    5.0          0.50
                    5.0          0.75
                    5.0          1.00

Group 9:  
                    10.0         0.00
                    10.0         0.25
                    10.0         0.50
                    10.0         0.75
                    10.0         1.00
'''
