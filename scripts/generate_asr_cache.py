from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asr_quality.data import load_training_csv
from asr_quality.features import resolve_audio_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ASR hypotheses cache for cross-modal features.")
    parser.add_argument("--csv", default="training.csv", help="Metadata CSV.")
    parser.add_argument("--audio-root", default="training_audio", help="Audio root directory.")
    parser.add_argument("--output", default="artifacts/asr_cache.csv", help="Output ASR cache CSV.")
    parser.add_argument(
        "--model",
        default="openai/whisper-tiny",
        help="Hugging Face ASR model. For better Vietnamese quality try openai/whisper-small if compute allows.",
    )
    parser.add_argument("--language", default="vi", help="Language hint for Whisper models.")
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    parser.add_argument("--limit", type=int, default=None, help="Optional max new files to transcribe.")
    parser.add_argument(
        "--max-duration-sec",
        type=float,
        default=None,
        help="Optional max audio duration for pilot runs.",
    )
    parser.add_argument(
        "--sort-by-duration",
        action="store_true",
        help="Process shorter clips first; useful for quick pilots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = load_training_csv(ROOT / args.csv)
    audio_root = (ROOT / args.audio_root).resolve()
    output_path = (ROOT / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.max_duration_sec is not None:
        audio_features_path = ROOT / "artifacts" / "audio_features.csv"
        if audio_features_path.exists():
            durations = pd.read_csv(audio_features_path)[["file_path", "audio_duration_sec"]]
            metadata = metadata.merge(durations, on="file_path", how="left")
            metadata = metadata[metadata["audio_duration_sec"] <= args.max_duration_sec]

    if args.sort_by_duration and "audio_duration_sec" in metadata.columns:
        metadata = metadata.sort_values("audio_duration_sec")

    done = _read_done(output_path)
    todo = metadata[~metadata["file_path"].isin(done)].copy()
    if args.limit is not None:
        todo = todo.head(args.limit)

    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    asr = pipeline(
        "automatic-speech-recognition",
        model=args.model,
        device=device,
        torch_dtype=dtype,
    )

    needs_header = not output_path.exists()
    with output_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["file_path", "asr_transcript", "asr_model"],
        )
        if needs_header:
            writer.writeheader()

        for _, row in tqdm(todo.iterrows(), total=len(todo), desc="ASR"):
            file_path = row["file_path"]
            audio_path = resolve_audio_path(file_path, audio_root)
            if audio_path is None:
                transcript = ""
            else:
                result = asr(
                    str(audio_path),
                    generate_kwargs={"language": args.language, "task": args.task},
                )
                transcript = result["text"] if isinstance(result, dict) else str(result)

            writer.writerow(
                {
                    "file_path": file_path,
                    "asr_transcript": transcript.strip(),
                    "asr_model": args.model,
                }
            )
            handle.flush()

    print(f"Wrote ASR cache: {output_path.relative_to(ROOT)}")


def _read_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    cache = pd.read_csv(path)
    if "file_path" not in cache.columns:
        return set()
    return set(cache["file_path"].astype(str))


if __name__ == "__main__":
    main()
