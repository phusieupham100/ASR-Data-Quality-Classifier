from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from asr_quality.config import RANDOM_STATE, TARGET_COLUMN
from asr_quality.data import audit_labels, load_training_csv, write_json
from asr_quality.features import build_feature_table
from asr_quality.modeling import (
    attach_predictions,
    choose_thresholds,
    evaluate_probabilities,
    fit_and_select_model,
    prepare_model_input,
    predict_proba_usable,
    threshold_sweep,
)
from asr_quality.reporting import write_evaluation_report
from asr_quality.splits import SplitConfig, make_splits, split_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ASR data quality classifier.")
    parser.add_argument("--csv", default="training.csv", help="Path to metadata CSV.")
    parser.add_argument(
        "--audio-root",
        default="training_audio",
        help="Root directory that contains audio files.",
    )
    parser.add_argument(
        "--asr-cache",
        default=None,
        help="Optional CSV with file_path,asr_transcript columns for cross-modal features.",
    )
    parser.add_argument("--artifacts-dir", default="artifacts", help="Output artifact directory.")
    parser.add_argument("--reports-dir", default="reports", help="Output report directory.")
    parser.add_argument(
        "--split-mode",
        choices=["group", "stratified"],
        default="group",
        help="Use group split by folder or standard stratified split.",
    )
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = (ROOT / args.csv).resolve()
    audio_root = (ROOT / args.audio_root).resolve() if args.audio_root else None
    artifacts_dir = (ROOT / args.artifacts_dir).resolve()
    reports_dir = (ROOT / args.reports_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_training_csv(csv_path)
    audit = audit_labels(metadata)
    write_json(audit, reports_dir / "data_audit.json")

    group_column = "folder" if args.split_mode == "group" else None
    metadata = make_splits(
        metadata,
        SplitConfig(random_state=args.random_state, group_column=group_column),
    )
    metadata.to_csv(artifacts_dir / "metadata_with_splits.csv", index=False)
    split_info = split_summary(metadata)
    split_info.to_csv(reports_dir / "split_summary.csv", index=False)

    features = build_feature_table(
        metadata,
        audio_root=audio_root,
        audio_cache_path=artifacts_dir / "audio_features.csv",
        asr_cache_path=args.asr_cache,
    )
    features.to_csv(artifacts_dir / "features.csv", index=False)

    selected_name, model, comparison = fit_and_select_model(
        features,
        metadata,
        random_state=args.random_state,
    )
    comparison.to_csv(reports_dir / "model_comparison.csv", index=False)

    val_mask = metadata["split"] == "val"
    test_mask = metadata["split"] == "test"
    model_input = prepare_model_input(features, metadata)
    val_prob = predict_proba_usable(model, model_input.loc[val_mask])
    test_prob = predict_proba_usable(model, model_input.loc[test_mask])

    sweep = threshold_sweep(metadata.loc[val_mask, TARGET_COLUMN], val_prob)
    sweep.to_csv(reports_dir / "threshold_sweep.csv", index=False)
    thresholds = choose_thresholds(sweep)
    selected_threshold = thresholds["best_macro_f1_threshold"]

    validation_metrics = evaluate_probabilities(
        metadata.loc[val_mask, TARGET_COLUMN],
        val_prob,
        threshold=selected_threshold,
    )
    test_metrics = evaluate_probabilities(
        metadata.loc[test_mask, TARGET_COLUMN],
        test_prob,
        threshold=selected_threshold,
    )

    predictions = attach_predictions(
        metadata.loc[test_mask].reset_index(drop=True),
        test_prob,
        threshold=selected_threshold,
    )
    predictions.to_csv(reports_dir / "test_predictions.csv", index=False)

    joblib.dump(
        {
            "model": model,
            "numeric_feature_columns": list(features.columns),
            "feature_columns": list(features.columns),
            "thresholds": thresholds,
            "selected_threshold": selected_threshold,
            "selected_model": selected_name,
        },
        artifacts_dir / "quality_classifier.joblib",
    )
    (reports_dir / "metrics.json").write_text(
        json.dumps(
            {
                "selected_model": selected_name,
                "thresholds": thresholds,
                "validation": validation_metrics,
                "test": test_metrics,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    write_evaluation_report(
        output_path=reports_dir / "evaluation.md",
        audit=audit,
        split_summary=split_info,
        model_comparison=comparison,
        selected_model=selected_name,
        selected_thresholds=thresholds,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        threshold_sweep=sweep,
        predictions=predictions,
    )

    print(f"Selected model: {selected_name}")
    print(f"Validation macro-F1: {validation_metrics['macro_f1']:.4f}")
    print(f"Validation ROC-AUC: {validation_metrics['roc_auc']:.4f}")
    print(f"Test macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print("Saved model to: artifacts/quality_classifier.joblib")
    print("Saved report to: reports/evaluation.md")


if __name__ == "__main__":
    main()
