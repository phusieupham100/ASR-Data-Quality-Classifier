from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


RANDOM_STATE = 42
TARGET_COLUMN = "target"
PROB_COLUMN = "prob_usable"


@dataclass(frozen=True)
class ProjectPaths:
    """Centralized project paths so scripts avoid hard-coded scattered paths."""

    root: Path
    data_csv: Path
    audio_root: Path | None
    artifacts_dir: Path
    reports_dir: Path

    @classmethod
    def from_root(
        cls,
        root: str | Path,
        data_csv: str | Path = "training.csv",
        audio_root: str | Path | None = None,
        artifacts_dir: str | Path = "artifacts",
        reports_dir: str | Path = "reports",
    ) -> "ProjectPaths":
        root_path = Path(root).resolve()
        audio_path = None if audio_root is None else (root_path / audio_root).resolve()
        return cls(
            root=root_path,
            data_csv=(root_path / data_csv).resolve(),
            audio_root=audio_path,
            artifacts_dir=(root_path / artifacts_dir).resolve(),
            reports_dir=(root_path / reports_dir).resolve(),
        )
