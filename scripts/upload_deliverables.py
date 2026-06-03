from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asr_quality.azure_io import upload_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload deliverables to Azure Blob.")
    parser.add_argument("--container-url", required=True, help="Container URL without hard-coded SAS.")
    parser.add_argument("--blob-prefix", required=True, help="Destination folder, e.g. your_name/")
    parser.add_argument("--local-dir", default=".", help="Directory to upload.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    upload_directory(
        container_url=args.container_url,
        local_dir=ROOT / args.local_dir,
        blob_prefix=args.blob_prefix,
    )
    print(f"Uploaded {args.local_dir} to {args.blob_prefix}")


if __name__ == "__main__":
    main()
