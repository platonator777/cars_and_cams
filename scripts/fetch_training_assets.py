"""Загружает и проверяет публичные исходные веса DINOv2 для обучения."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path


ASSETS = {
    "dinov2_vitb14_pretrain.pth": {
        "url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
        "sha256": "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73",
    },
    "dinov2_vits14_pretrain.pth": {
        "url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
        "sha256": "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9",
    },
}


def digest(path: Path) -> str:
    """Вычисляет SHA-256 файла."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name: str, root: Path) -> dict:
    """Находит или скачивает указанный checkpoint и проверяет его хеш."""
    meta = ASSETS[name]
    target = root / name
    if target.exists() and digest(target) == meta["sha256"]:
        return {"file": name, "status": "verified_cached", **meta, "bytes": target.stat().st_size}
    if target.exists():
        target.unlink()
    partial = target.with_suffix(target.suffix + ".partial")
    request = urllib.request.Request(meta["url"], headers={"User-Agent": "vehicle-reid-reproduction/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    actual = digest(partial)
    if actual != meta["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {name}: expected {meta['sha256']}, got {actual}")
    os.replace(partial, target)
    return {"file": name, "status": "downloaded_and_verified", **meta, "bytes": target.stat().st_size}


def main():
    """Выбирает модель и каталог исходных весов из аргументов командной строки."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", required=True)
    parser.add_argument("--model", choices=["b14", "s14", "both"], default="both")
    args = parser.parse_args()
    root = Path(args.assets).resolve()
    root.mkdir(parents=True, exist_ok=True)
    names = {"b14": ["dinov2_vitb14_pretrain.pth"], "s14": ["dinov2_vits14_pretrain.pth"],
             "both": list(ASSETS)}[args.model]
    records = [fetch(name, root) for name in names]
    result = {"repository": "https://github.com/facebookresearch/dinov2",
              "source_commit": "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
              "license": "DINOv2 code Apache-2.0; pretrained model weights FAIR Noncommercial Research License (see vendor/dinov2/LICENSE_CELL_DINO_MODELS)",
              "assets": records}
    (root / "assets_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
