from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare task deliverables folder.")
    parser.add_argument("--name", default="your_name", help="Submission folder name.")
    parser.add_argument("--output-root", default="submission", help="Parent output directory.")
    return parser.parse_args()


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )


def main() -> None:
    args = parse_args()
    out = ROOT / args.output_root / args.name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Source code
    source_dir = out / "1_source_code"
    copy_tree(ROOT / "src", source_dir / "src")
    copy_tree(ROOT / "scripts", source_dir / "scripts")
    copy_file(ROOT / "README.md", source_dir / "README.md")
    copy_file(ROOT / "requirements.txt", source_dir / "requirements.txt")

    # 2. Trained model
    copy_file(
        ROOT / "artifacts" / "quality_classifier.joblib",
        out / "2_trained_model" / "quality_classifier.joblib",
    )

    # 3. Evaluation report
    report_dir = out / "3_evaluation_report"
    for filename in [
        "evaluation.md",
        "metrics.json",
        "model_comparison.csv",
        "threshold_sweep.csv",
        "split_summary.csv",
        "data_audit.json",
        "test_predictions.csv",
    ]:
        copy_file(ROOT / "reports" / filename, report_dir / filename)

    # 4. Exact requirements
    copy_file(ROOT / "requirements.txt", out / "4_requirements.txt")

    # 5. Standalone inference script
    copy_file(ROOT / "scripts" / "predict.py", out / "5_inference_script" / "predict.py")

    manifest = out / "SUBMISSION_MANIFEST.md"
    manifest.write_text(
        """# Submission Manifest

This folder follows the task deliverable format.

1. `1_source_code/`: Python project source code and README.
2. `2_trained_model/quality_classifier.joblib`: serialized trained model pipeline.
3. `3_evaluation_report/`: evaluation report, metrics, threshold sweep, confusion matrix in metrics JSON, and error-analysis predictions.
4. `4_requirements.txt`: exact direct package versions for reproduction.
5. `5_inference_script/predict.py`: standalone inference script.

Recommended inference command from the project root:

```powershell
python scripts/predict.py --model artifacts/quality_classifier.joblib --audio path\\to\\sample.wav --transcript "..."
```
""",
        encoding="utf-8",
    )
    print(f"Prepared submission folder: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
