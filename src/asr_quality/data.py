from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .config import TARGET_COLUMN


USABLE_LABEL = 1
UNUSABLE_LABEL = 0


def normalize_text(text: Any) -> str:
    """Normalize transcript text for duplicate and similarity checks."""

    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_training_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load metadata and create a canonical binary target column.

    The provided local CSV uses label_text as the clearest source of truth:
    usable -> 1, unusable -> 0. If label_text is unavailable, numeric labels
    are mapped with the observed convention 1 -> usable and 2/0 -> unusable.
    """

    df = pd.read_csv(csv_path)
    if "label_text" in df.columns:
        label_text = df["label_text"].astype(str).str.lower().str.strip()
        df[TARGET_COLUMN] = label_text.map({"usable": USABLE_LABEL, "unusable": UNUSABLE_LABEL})
    elif "label" in df.columns:
        df[TARGET_COLUMN] = df["label"].map({1: USABLE_LABEL, 2: UNUSABLE_LABEL, 0: UNUSABLE_LABEL})
    else:
        raise ValueError("CSV must contain either label_text or label column.")

    if df[TARGET_COLUMN].isna().any():
        bad_rows = int(df[TARGET_COLUMN].isna().sum())
        raise ValueError(f"Could not map labels for {bad_rows} rows.")

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    df["normalized_transcript"] = df.get("transcript", "").map(normalize_text)
    df["transcript_word_count"] = df["normalized_transcript"].str.split().map(len)
    df["transcript_char_count"] = df["normalized_transcript"].str.len()
    return df


def audit_labels(df: pd.DataFrame) -> dict[str, Any]:
    """Return dataset audit statistics useful for the report."""

    label_counts = df[TARGET_COLUMN].value_counts().sort_index().to_dict()
    label_text_counts = (
        df["label_text"].value_counts().to_dict() if "label_text" in df.columns else {}
    )
    transcript_label_counts = df.groupby("normalized_transcript")[TARGET_COLUMN].nunique()
    mixed_transcripts = transcript_label_counts[transcript_label_counts > 1]
    folder_label_counts = (
        df.groupby("folder")[TARGET_COLUMN].nunique() if "folder" in df.columns else pd.Series(dtype=int)
    )
    mixed_folders = folder_label_counts[folder_label_counts > 1]

    transcript_length_by_class = (
        df.groupby(TARGET_COLUMN)["transcript_word_count"]
        .describe()
        .round(3)
        .to_dict()
    )

    return {
        "rows": int(len(df)),
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "label_text_counts": {str(k): int(v) for k, v in label_text_counts.items()},
        "unique_file_paths": int(df["file_path"].nunique()) if "file_path" in df.columns else None,
        "unique_folders": int(df["folder"].nunique()) if "folder" in df.columns else None,
        "duplicate_transcript_rows": int(df.duplicated("normalized_transcript").sum()),
        "mixed_label_transcripts": int(len(mixed_transcripts)),
        "rows_in_mixed_label_transcripts": int(
            df[df["normalized_transcript"].isin(mixed_transcripts.index)].shape[0]
        ),
        "mixed_label_folders": int(len(mixed_folders)),
        "transcript_word_count_by_target": transcript_length_by_class,
    }


def write_json(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
