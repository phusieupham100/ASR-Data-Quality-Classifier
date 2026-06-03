from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import PROB_COLUMN, RANDOM_STATE, TARGET_COLUMN


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    macro_f1: float
    precision_usable: float
    recall_usable: float
    precision_unusable: float
    recall_unusable: float


TEXT_COLUMN = "model_transcript"


def prepare_model_input(features: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Combine numeric features with transcript text for sklearn pipelines."""

    model_input = features.copy()
    transcript = metadata.get("normalized_transcript", metadata.get("transcript", ""))
    model_input[TEXT_COLUMN] = transcript.fillna("").astype(str).values
    return model_input


def candidate_models(
    numeric_columns: list[str],
    random_state: int = RANDOM_STATE,
) -> dict[str, Pipeline]:
    numeric_preprocess = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    numeric_only = ColumnTransformer(
        transformers=[("numeric", numeric_preprocess, numeric_columns)],
        remainder="drop",
        sparse_threshold=0.0,
    )
    numeric_and_text = ColumnTransformer(
        transformers=[
            ("numeric", numeric_preprocess, numeric_columns),
            (
                "text",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=20_000,
                    sublinear_tf=True,
                ),
                TEXT_COLUMN,
            ),
        ],
        remainder="drop",
    )
    return {
        "text_audio_logistic": Pipeline(
            steps=[
                ("preprocess", numeric_and_text),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=random_state,
                        solver="liblinear",
                    ),
                ),
            ]
        ),
        "logistic_regression": Pipeline(
            steps=[
                ("preprocess", numeric_only),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", numeric_only),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        min_samples_leaf=3,
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("preprocess", numeric_only),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=250,
                        l2_regularization=0.1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def fit_and_select_model(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> tuple[str, Pipeline, pd.DataFrame]:
    model_input = prepare_model_input(features, metadata)
    train_mask = metadata["split"] == "train"
    val_mask = metadata["split"] == "val"
    x_train = model_input.loc[train_mask]
    y_train = metadata.loc[train_mask, TARGET_COLUMN]
    x_val = model_input.loc[val_mask]
    y_val = metadata.loc[val_mask, TARGET_COLUMN]

    rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}
    for name, model in candidate_models(list(features.columns), random_state=random_state).items():
        model.fit(x_train, y_train)
        prob = predict_proba_usable(model, x_val)
        sweep = threshold_sweep(y_val, prob)
        best_row = sweep.sort_values(["macro_f1", "roc_auc"], ascending=False).iloc[0]
        metrics = evaluate_probabilities(y_val, prob, threshold=0.5)
        rows.append(
            {
                "model": name,
                "best_val_macro_f1": float(best_row["macro_f1"]),
                "best_val_threshold": float(best_row["threshold"]),
                **metrics,
            }
        )
        fitted_models[name] = model

    comparison = pd.DataFrame(rows).sort_values(
        ["best_val_macro_f1", "roc_auc"], ascending=False, na_position="last"
    )
    best_name = str(comparison.iloc[0]["model"])
    return best_name, fitted_models[best_name], comparison


def predict_proba_usable(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)
        classes = list(model.classes_) if hasattr(model, "classes_") else list(model[-1].classes_)
        usable_idx = classes.index(1)
        return proba[:, usable_idx]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(features)
        return 1 / (1 + np.exp(-scores))

    return model.predict(features).astype(float)


def evaluate_probabilities(
    y_true: pd.Series | np.ndarray,
    prob_usable: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    pred = (prob_usable >= threshold).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        pred,
        labels=[0, 1],
        zero_division=0,
    )
    try:
        roc_auc = float(roc_auc_score(y_true, prob_usable))
    except ValueError:
        roc_auc = float("nan")
    try:
        pr_auc = float(average_precision_score(y_true, prob_usable))
    except ValueError:
        pr_auc = float("nan")

    return {
        "threshold": float(threshold),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision_unusable": float(precision[0]),
        "recall_unusable": float(recall[0]),
        "f1_unusable": float(f1[0]),
        "support_unusable": int(support[0]),
        "precision_usable": float(precision[1]),
        "recall_usable": float(recall[1]),
        "f1_usable": float(f1[1]),
        "support_usable": int(support[1]),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }


def threshold_sweep(
    y_true: pd.Series | np.ndarray,
    prob_usable: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)

    rows = []
    for threshold in thresholds:
        metrics = evaluate_probabilities(y_true, prob_usable, threshold=float(threshold))
        metrics.pop("confusion_matrix", None)
        rows.append(metrics)
    return pd.DataFrame(rows)


def choose_thresholds(sweep: pd.DataFrame, min_precision_usable: float = 0.9) -> dict[str, float]:
    best_macro_row = sweep.sort_values(["macro_f1", "roc_auc"], ascending=False).iloc[0]
    candidates = sweep[sweep["precision_usable"] >= min_precision_usable]
    if candidates.empty:
        high_precision_row = sweep.sort_values(
            ["precision_usable", "recall_usable"], ascending=False
        ).iloc[0]
    else:
        high_precision_row = candidates.sort_values(
            ["recall_usable", "macro_f1"], ascending=False
        ).iloc[0]

    return {
        "default_threshold": 0.5,
        "best_macro_f1_threshold": float(best_macro_row["threshold"]),
        "high_precision_usable_threshold": float(high_precision_row["threshold"]),
    }


def attach_predictions(
    metadata: pd.DataFrame,
    prob_usable: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    output = metadata.copy()
    output[PROB_COLUMN] = prob_usable
    output["prediction"] = (output[PROB_COLUMN] >= threshold).astype(int)
    return output
