from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import normalize_text


TEXT_FILLERS = {
    "ừ",
    "ừm",
    "à",
    "ạ",
    "dạ",
    "vâng",
    "alo",
    "a-lô",
}


def resolve_audio_path(file_path: str, audio_root: str | Path | None) -> Path | None:
    """Resolve CSV relative file_path against common local audio layouts."""

    if audio_root is None or not isinstance(file_path, str) or not file_path:
        return None

    root = Path(audio_root)
    rel = Path(file_path)
    candidates = [
        rel,
        root / rel,
        root / "data2" / rel,
        root / "training_audio" / "data2" / rel,
        root / "training_audio" / rel,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def build_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create transcript-only features that are available for every row."""

    transcripts = df.get("transcript", pd.Series([""] * len(df), index=df.index)).map(normalize_text)
    features = pd.DataFrame(index=df.index)
    tokens = transcripts.str.split()
    word_counts = tokens.map(len).astype(float)
    char_counts = transcripts.str.len().astype(float)

    features["text_char_count"] = char_counts
    features["text_word_count"] = word_counts
    features["text_avg_word_len"] = char_counts / word_counts.replace(0, np.nan)
    features["text_avg_word_len"] = features["text_avg_word_len"].fillna(0.0)
    features["text_question_marks"] = df.get("transcript", "").astype(str).str.count(r"\?")
    features["text_commas"] = df.get("transcript", "").astype(str).str.count(",")
    features["text_digit_count"] = transcripts.str.count(r"\d")
    features["text_special_ratio"] = transcripts.map(_special_char_ratio)
    features["text_unique_ratio"] = tokens.map(_unique_ratio)
    features["text_repetition_ratio"] = tokens.map(_repetition_ratio)
    features["text_filler_ratio"] = tokens.map(_filler_ratio)
    features["text_short_flag"] = (word_counts <= 8).astype(float)
    features["text_long_flag"] = (word_counts >= 120).astype(float)
    return features.fillna(0.0)


def build_audio_features(
    df: pd.DataFrame,
    audio_root: str | Path | None,
    cache_path: str | Path | None = None,
    sample_rate: int = 16_000,
) -> pd.DataFrame:
    """Extract lightweight audio features, using a cache if present."""

    if cache_path is not None:
        cache = Path(cache_path)
        if cache.exists():
            cached = pd.read_csv(cache)
            if "file_path" in cached.columns:
                return cached.set_index("file_path").reindex(df["file_path"]).reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        file_path = row.get("file_path", "")
        audio_path = resolve_audio_path(file_path, audio_root)
        row_features = {"file_path": file_path, "audio_found": float(audio_path is not None)}
        if audio_path is None:
            row_features.update(_empty_audio_features())
        else:
            row_features.update(_extract_audio_features(audio_path, sample_rate=sample_rate))
        rows.append(row_features)

    features = pd.DataFrame(rows)
    if cache_path is not None:
        cache = Path(cache_path)
        cache.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(cache, index=False)

    return features.drop(columns=["file_path"], errors="ignore").fillna(0.0)


def build_cross_modal_features(
    df: pd.DataFrame,
    asr_cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build ASR-vs-transcript similarity features if an ASR cache is available.

    The cache should contain file_path and asr_transcript columns. It can be
    produced offline with Whisper or another ASR model. If absent, neutral
    feature values are returned so the pipeline still runs.
    """

    features = pd.DataFrame(index=df.index)
    features["cross_has_asr"] = 0.0
    features["cross_word_error_rate"] = 1.0
    features["cross_char_error_rate"] = 1.0
    features["cross_token_jaccard"] = 0.0

    if asr_cache_path is None or not Path(asr_cache_path).exists():
        return features

    asr_df = pd.read_csv(asr_cache_path)
    if not {"file_path", "asr_transcript"}.issubset(asr_df.columns):
        return features

    merged = df[["file_path", "transcript"]].merge(asr_df, on="file_path", how="left")
    features["cross_has_asr"] = merged["asr_transcript"].notna().astype(float).values
    reference = merged["transcript"].map(normalize_text)
    hypothesis = merged["asr_transcript"].fillna("").map(normalize_text)
    features["cross_word_error_rate"] = [
        _safe_wer(ref, hyp) for ref, hyp in zip(reference, hypothesis, strict=False)
    ]
    features["cross_char_error_rate"] = [
        _char_error_rate(ref, hyp) for ref, hyp in zip(reference, hypothesis, strict=False)
    ]
    features["cross_token_jaccard"] = [
        _token_jaccard(ref, hyp) for ref, hyp in zip(reference, hypothesis, strict=False)
    ]
    return features.fillna(0.0)


def build_feature_table(
    df: pd.DataFrame,
    audio_root: str | Path | None = None,
    audio_cache_path: str | Path | None = None,
    asr_cache_path: str | Path | None = None,
) -> pd.DataFrame:
    text_features = build_text_features(df)
    audio_features = build_audio_features(df, audio_root=audio_root, cache_path=audio_cache_path)
    cross_features = build_cross_modal_features(df, asr_cache_path=asr_cache_path)
    return pd.concat(
        [
            text_features.reset_index(drop=True),
            audio_features.reset_index(drop=True),
            cross_features.reset_index(drop=True),
        ],
        axis=1,
    ).fillna(0.0)


def _extract_audio_features(path: Path, sample_rate: int) -> dict[str, float]:
    try:
        import librosa

        y, sr = librosa.load(path, sr=sample_rate, mono=True)
        if y.size == 0:
            return _empty_audio_features()

        duration = float(librosa.get_duration(y=y, sr=sr))
        rms = librosa.feature.rms(y=y)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        silence_threshold = max(float(np.percentile(rms, 20)), 1e-6)
        silence_ratio = float(np.mean(rms <= silence_threshold))

        values: dict[str, float] = {
            "audio_duration_sec": duration,
            "audio_rms_mean": float(np.mean(rms)),
            "audio_rms_std": float(np.std(rms)),
            "audio_zcr_mean": float(np.mean(zcr)),
            "audio_zcr_std": float(np.std(zcr)),
            "audio_centroid_mean": float(np.mean(centroid)),
            "audio_bandwidth_mean": float(np.mean(bandwidth)),
            "audio_silence_ratio": silence_ratio,
        }
        for idx in range(mfcc.shape[0]):
            values[f"audio_mfcc_{idx + 1}_mean"] = float(np.mean(mfcc[idx]))
            values[f"audio_mfcc_{idx + 1}_std"] = float(np.std(mfcc[idx]))
        return values
    except Exception:
        values = _empty_audio_features()
        values["audio_read_error"] = 1.0
        return values


def _empty_audio_features() -> dict[str, float]:
    values = {
        "audio_duration_sec": 0.0,
        "audio_rms_mean": 0.0,
        "audio_rms_std": 0.0,
        "audio_zcr_mean": 0.0,
        "audio_zcr_std": 0.0,
        "audio_centroid_mean": 0.0,
        "audio_bandwidth_mean": 0.0,
        "audio_silence_ratio": 1.0,
        "audio_read_error": 0.0,
    }
    for idx in range(13):
        values[f"audio_mfcc_{idx + 1}_mean"] = 0.0
        values[f"audio_mfcc_{idx + 1}_std"] = 0.0
    return values


def _special_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    special = len(re.findall(r"[^0-9a-zA-ZÀ-ỹ\s]", text))
    return special / max(len(text), 1)


def _unique_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _repetition_ratio(tokens: list[str]) -> float:
    if len(tokens) < 2:
        return 0.0
    repeated = sum(1 for left, right in zip(tokens, tokens[1:], strict=False) if left == right)
    return repeated / (len(tokens) - 1)


def _filler_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return sum(1 for token in tokens if token in TEXT_FILLERS) / len(tokens)


def _safe_wer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 1.0 if hypothesis else 0.0
    try:
        from jiwer import wer

        return float(min(wer(reference, hypothesis), 3.0))
    except Exception:
        return 1.0


def _char_error_rate(reference: str, hypothesis: str) -> float:
    if not reference:
        return 1.0 if hypothesis else 0.0
    return min(_levenshtein(reference, hypothesis) / max(len(reference), 1), 3.0)


def _token_jaccard(reference: str, hypothesis: str) -> float:
    ref_tokens = set(reference.split())
    hyp_tokens = set(hypothesis.split())
    if not ref_tokens and not hyp_tokens:
        return 1.0
    return len(ref_tokens & hyp_tokens) / max(len(ref_tokens | hyp_tokens), 1)


def _levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (left_char != right_char)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]
