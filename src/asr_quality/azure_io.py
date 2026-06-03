from __future__ import annotations

import os
from pathlib import Path

from azure.storage.blob import ContainerClient


def container_from_env(container_url: str) -> ContainerClient:
    """Create a blob container client with SAS token from AZURE_SAS_TOKEN."""

    token = os.environ.get("AZURE_SAS_TOKEN")
    if not token:
        raise RuntimeError("Missing AZURE_SAS_TOKEN environment variable.")
    separator = "&" if "?" in container_url else "?"
    return ContainerClient.from_container_url(f"{container_url}{separator}{token.lstrip('?')}")


def upload_directory(container_url: str, local_dir: str | Path, blob_prefix: str) -> None:
    client = container_from_env(container_url)
    root = Path(local_dir)
    if not root.exists():
        raise FileNotFoundError(root)

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        blob_name = f"{blob_prefix.rstrip('/')}/{rel}"
        with path.open("rb") as handle:
            client.upload_blob(blob_name, handle, overwrite=True)
