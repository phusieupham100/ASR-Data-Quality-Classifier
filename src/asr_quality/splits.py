from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from .config import RANDOM_STATE, TARGET_COLUMN


@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = RANDOM_STATE
    group_column: str | None = "folder"


def make_splits(df: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    """Add a split column using group-aware splitting when possible."""

    output = df.copy()
    output["split"] = "train"

    if config.group_column and config.group_column in output.columns:
        return _make_group_splits(output, config)

    return _make_stratified_splits(output, config)


def _make_stratified_splits(df: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    train_val_idx, test_idx = train_test_split(
        df.index,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=df[TARGET_COLUMN],
    )
    val_fraction_of_train_val = config.val_size / (1.0 - config.test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_fraction_of_train_val,
        random_state=config.random_state,
        stratify=df.loc[train_val_idx, TARGET_COLUMN],
    )
    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    return df


def _make_group_splits(df: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    groups = df[config.group_column].astype(str)
    y = df[TARGET_COLUMN]

    outer = StratifiedGroupKFold(n_splits=max(round(1 / config.test_size), 2), shuffle=True, random_state=config.random_state)
    train_val_pos, test_pos = next(outer.split(df, y, groups))
    train_val_idx = df.index[train_val_pos]
    test_idx = df.index[test_pos]

    train_val = df.loc[train_val_idx]
    inner_groups = train_val[config.group_column].astype(str)
    inner_y = train_val[TARGET_COLUMN]
    val_fraction_of_train_val = config.val_size / (1.0 - config.test_size)
    inner_splits = max(round(1 / val_fraction_of_train_val), 2)
    inner = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=config.random_state)
    train_pos, val_pos = next(inner.split(train_val, inner_y, inner_groups))

    train_idx = train_val.index[train_pos]
    val_idx = train_val.index[val_pos]
    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    return df


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["split", TARGET_COLUMN])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["split", TARGET_COLUMN])
    )
