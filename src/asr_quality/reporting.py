from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PROB_COLUMN, TARGET_COLUMN


def write_evaluation_report(
    output_path: str | Path,
    audit: dict[str, Any],
    split_summary: pd.DataFrame,
    model_comparison: pd.DataFrame,
    selected_model: str,
    selected_thresholds: dict[str, float],
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    threshold_sweep: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    false_positive = _examples(predictions, true_label=0, pred_label=1)
    false_negative = _examples(predictions, true_label=1, pred_label=0)

    content = f"""# ASR Quality Classifier Evaluation

## Summary

This project trains a binary classifier that scores whether an audio-transcript pair is usable for ASR training. The model is evaluated as a probabilistic filter, so threshold selection is performed on the validation set instead of assuming `0.5` is always optimal.

- Selected model: `{selected_model}`
- Default threshold: `{selected_thresholds['default_threshold']:.2f}`
- Best validation macro-F1 threshold: `{selected_thresholds['best_macro_f1_threshold']:.2f}`
- High-precision usable threshold: `{selected_thresholds['high_precision_usable_threshold']:.2f}`

## Data Audit

```json
{json.dumps(audit, indent=2, ensure_ascii=False)}
```

## Split Summary

{_table_text(split_summary)}

## Model Comparison On Validation

{_table_text(_model_comparison_preview(model_comparison))}

## Validation Metrics At Selected Threshold

```json
{json.dumps(validation_metrics, indent=2, ensure_ascii=False)}
```

## Test Metrics At Selected Threshold

```json
{json.dumps(test_metrics, indent=2, ensure_ascii=False)}
```

## Threshold Trade-Off

The table below shows selected operating points. The full sweep is saved as `threshold_sweep.csv`.

{_table_text(_threshold_preview(threshold_sweep))}

## Error Analysis Samples

False positives are unusable samples predicted as usable. These are risky because they may enter ASR training data.

{_table_text(false_positive)}

False negatives are usable samples predicted as unusable. These reduce retained data volume but are usually less harmful for a high-quality filter.

{_table_text(false_negative)}

## Interpretation

If macro-F1 remains below the original target, the audit results should be considered when interpreting the score. The dataset contains duplicated transcripts and mixed labels for identical transcripts, so audio quality and annotation noise can limit text-only and weak audio models. ROC-AUC and PR-AUC are reported to show ranking quality, and threshold tuning is used to choose an operating point aligned with ASR data filtering.
"""
    path.write_text(content, encoding="utf-8")


def _threshold_preview(sweep: pd.DataFrame) -> pd.DataFrame:
    desired = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    rows = []
    for threshold in desired:
        idx = (sweep["threshold"] - threshold).abs().idxmin()
        rows.append(sweep.loc[idx])
    return pd.DataFrame(rows)[
        [
            "threshold",
            "macro_f1",
            "precision_usable",
            "recall_usable",
            "precision_unusable",
            "recall_unusable",
        ]
    ].round(4)


def _model_comparison_preview(model_comparison: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model",
        "best_val_macro_f1",
        "best_val_threshold",
        "macro_f1",
        "roc_auc",
        "pr_auc",
        "precision_unusable",
        "recall_unusable",
        "precision_usable",
        "recall_usable",
    ]
    available = [column for column in columns if column in model_comparison.columns]
    return model_comparison[available].round(4)


def _table_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    display = df.copy()
    for column in display.columns:
        if display[column].dtype == object:
            display[column] = display[column].map(_truncate_cell)
    return _to_markdown_table(display)


def _to_markdown_table(df: pd.DataFrame) -> str:
    headers = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = [_escape_markdown_cell(row[column]) for column in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _escape_markdown_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _truncate_cell(value: object, max_length: int = 100) -> str:
    text = "" if pd.isna(value) else str(value).replace("\n", " ")
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def _examples(predictions: pd.DataFrame, true_label: int, pred_label: int, n: int = 8) -> pd.DataFrame:
    if PROB_COLUMN not in predictions.columns:
        return pd.DataFrame()
    subset = predictions[
        (predictions[TARGET_COLUMN] == true_label) & (predictions["prediction"] == pred_label)
    ].copy()
    if subset.empty:
        return pd.DataFrame(columns=["file_path", "prob_usable", "transcript"])
    subset["distance_from_boundary"] = (subset[PROB_COLUMN] - 0.5).abs()
    columns = [
        column
        for column in ["file_path", PROB_COLUMN, TARGET_COLUMN, "prediction", "transcript"]
        if column in subset.columns
    ]
    return subset.sort_values("distance_from_boundary", ascending=False)[columns].head(n)
