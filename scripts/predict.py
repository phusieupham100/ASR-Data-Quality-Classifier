from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE_CANDIDATES = [
    ROOT / "src",
    ROOT / "1_source_code" / "src",
    ROOT.parent / "1_source_code" / "src",
]
for source_path in SOURCE_CANDIDATES:
    if source_path.exists():
        sys.path.insert(0, str(source_path))
        break

from asr_quality.data import normalize_text
from asr_quality.features import build_feature_table
from asr_quality.modeling import prepare_model_input, predict_proba_usable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify one audio-transcript pair.")
    parser.add_argument("--model", default="artifacts/quality_classifier.joblib")
    parser.add_argument("--audio", required=True, help="Path to a wav/mp3 file.")
    parser.add_argument("--transcript", required=True, help="Provided transcript text.")
    parser.add_argument("--audio-root", default=".", help="Base path for resolving audio.")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--low-threshold", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = joblib.load(_resolve_path(args.model))
    model = artifact["model"]
    feature_columns = artifact.get("numeric_feature_columns", artifact["feature_columns"])
    thresholds = artifact.get("thresholds", {})
    threshold = args.threshold
    if threshold is None:
        threshold = thresholds.get("high_precision_usable_threshold", artifact.get("selected_threshold", 0.5))

    row = pd.DataFrame(
        [
            {
                "file_path": str(Path(args.audio).resolve()),
                "transcript": args.transcript,
                "normalized_transcript": normalize_text(args.transcript),
            }
        ]
    )
    features = build_feature_table(row, audio_root=args.audio_root)
    features = features.reindex(columns=feature_columns, fill_value=0.0)
    model_input = prepare_model_input(features, row)
    prob_usable = float(predict_proba_usable(model, model_input)[0])
    prediction = int(prob_usable >= threshold)

    if prob_usable >= threshold:
        decision = "accept"
    elif prob_usable <= args.low_threshold:
        decision = "reject"
    else:
        decision = "manual_review"

    print(
        json.dumps(
            {
                "prob_usable": prob_usable,
                "threshold_used": threshold,
                "prediction": prediction,
                "decision": decision,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.exists():
        return path.resolve()
    root_path = ROOT / path
    if root_path.exists():
        return root_path.resolve()
    sibling_path = ROOT.parent / path
    if sibling_path.exists():
        return sibling_path.resolve()
    return root_path.resolve()


if __name__ == "__main__":
    main()
