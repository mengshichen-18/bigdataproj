from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

DATA_DIR = Path("dataset/data_format1")
DEFAULT_OUT_DIR = Path("baselines/output")
RANDOM_STATE = 42

ACTION_TYPES = {
    0: "click",
    1: "cart",
    2: "buy",
    3: "fav",
}

CATEGORICAL_FEATURES = ["age_range", "gender"]

# Use day windows measured by days before 11/11 (inclusive with day 0).
# "all" is a catch-all to include the entire 6-month log span.
WINDOW_SPECS: List[Tuple[str, int]] = [
    ("1", 1),
    ("3", 3),
    ("7", 7),
    ("14", 14),
    ("30", 30),
    ("60", 60),
    ("90", 90),
    ("180", 180),
    ("all", 9999),
]

WINDOW_LABELS: List[str] = [label for label, _ in WINDOW_SPECS]
BIN_EDGES: List[int] = [value for _, value in WINDOW_SPECS]
N_BINS: int = len(BIN_EDGES) + 1

RECENCY_DEFAULT = np.int16(999)

_MONTH_CUM_DAYS = np.array(
    [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], dtype=np.int16
)
_DOY_1111 = int(_MONTH_CUM_DAYS[10] + 11)  # Nov 11, non-leap year


def mmdd_to_days_before_1111(mmdd: np.ndarray) -> np.ndarray:
    month = (mmdd // 100).astype(np.int16)
    day = (mmdd % 100).astype(np.int16)
    doy = _MONTH_CUM_DAYS[month - 1] + day
    return (_DOY_1111 - doy).astype(np.int16)


def expected_count_columns(prefix: str) -> List[str]:
    cols = []
    for label in WINDOW_LABELS:
        for action_name in ACTION_TYPES.values():
            cols.append(f"{prefix}_{action_name}_w{label}")
    return cols


def compute_window_counts(
    df: pd.DataFrame, group_cols: Sequence[str], prefix: str
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=expected_count_columns(prefix))

    bins = np.digitize(df["days_before"].to_numpy(), BIN_EDGES, right=False).astype(
        np.int8
    )
    df = df.assign(_bin=bins)

    grouped = (
        df.groupby([*group_cols, "_bin", "action_type"], sort=False)
        .size()
        .unstack(["_bin", "action_type"], fill_value=0)
    )

    expected = pd.MultiIndex.from_product(
        [range(N_BINS), sorted(ACTION_TYPES)], names=["_bin", "action_type"]
    )
    grouped = grouped.reindex(columns=expected, fill_value=0)

    out = {}
    for action_type, action_name in ACTION_TYPES.items():
        per_bin = grouped.xs(action_type, level="action_type", axis=1)
        per_bin = per_bin.reindex(columns=range(N_BINS), fill_value=0).to_numpy(
            dtype=np.int32
        )
        cum = np.cumsum(per_bin[:, : len(BIN_EDGES)], axis=1)
        for i, (label, _) in enumerate(WINDOW_SPECS):
            out[f"{prefix}_{action_name}_w{label}"] = cum[:, i]

    return pd.DataFrame(out, index=grouped.index)


def compute_recency(df: pd.DataFrame, group_cols: Sequence[str], prefix: str) -> pd.DataFrame:
    if df.empty:
        cols = [f"{prefix}_recency_{name}" for name in ACTION_TYPES.values()] + [
            f"{prefix}_recency_any"
        ]
        return pd.DataFrame(columns=cols)

    rec = (
        df.groupby([*group_cols, "action_type"], sort=False)["days_before"]
        .min()
        .unstack(fill_value=int(RECENCY_DEFAULT))
    )
    rec = rec.reindex(columns=sorted(ACTION_TYPES), fill_value=int(RECENCY_DEFAULT))
    rec.columns = [f"{prefix}_recency_{ACTION_TYPES[a]}" for a in rec.columns]
    rec[f"{prefix}_recency_any"] = rec.min(axis=1)
    return rec.astype(np.int16)


def update_sum(agg: pd.DataFrame, part: pd.DataFrame) -> None:
    if part.empty:
        return
    cols = list(part.columns)
    idx = part.index
    agg.loc[idx, cols] = agg.loc[idx, cols].to_numpy() + part[cols].to_numpy()


def update_min(agg: pd.DataFrame, part: pd.DataFrame) -> None:
    if part.empty:
        return
    cols = list(part.columns)
    idx = part.index
    agg.loc[idx, cols] = np.minimum(agg.loc[idx, cols].to_numpy(), part[cols].to_numpy())


def safe_div(numer: pd.Series, denom: pd.Series, eps: float = 1.0) -> pd.Series:
    return numer / (denom + eps)

