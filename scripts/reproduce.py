"""Запускает воспроизводимое обучение и фиксирует состав входных данных и кода."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    """Вычисляет SHA-256 файла."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def image_inventory(data_root: Path, config: dict) -> tuple[dict, str]:
    """Проверяет обучающие кадры и собирает их хеши в манифест."""
    images_dir = data_root / config.get("images_dir", "images")
    csv_path = data_root / "train.csv"
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        image_ids = list(dict.fromkeys(row["image_id"] for row in csv.DictReader(stream)))
    rows = []
    for image_id in image_ids:
        path = images_dir / f"{image_id}.jpg"
        if not path.is_file():
            raise FileNotFoundError(f"dataset image is missing: {path}")
        rows.append({"image_id": image_id, "bytes": path.stat().st_size, "sha256": sha256(path)})
    aggregate = hashlib.sha256("".join(f"{row['image_id']}:{row['sha256']}\n" for row in rows).encode()).hexdigest()
    return {"images_directory": str(images_dir), "train_csv_sha256": sha256(csv_path),
            "count": len(rows), "aggregate_sha256": aggregate, "images": rows}, aggregate


def training_source_inventory(config_path: Path) -> dict:
    """Собирает хеши исходного кода и конфигурации обучения."""
    selected = [HERE / "train.py", Path(__file__).resolve(), HERE / "vehicle_reid/protocol.py",
                HERE / "vehicle_reid/scoring.py", HERE / "vehicle_reid/models/dino.py",
                HERE / "vehicle_reid/training/data.py", config_path,
                HERE / "configs/protocols/fold0/fold_0_manifest.json",
                HERE / "configs/protocols/fold0/query.csv", HERE / "configs/protocols/fold0/gallery.csv",
                HERE / "configs/protocols/fold0/ground_truth.csv", HERE / "official/evaluate.py",
                HERE / "requirements-train.lock", HERE / "pyproject.toml",
                HERE / "vendor/dinov2/LICENSE", HERE / "vendor/dinov2/LICENSE_CELL_DINO_MODELS"]
    selected.extend(sorted((HERE / "vendor/dinov2").rglob("*.py")))
    files = [{"path": path.relative_to(HERE).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size}
             for path in selected if path.is_file()]
    aggregate = hashlib.sha256("".join(f"{row['path']}:{row['sha256']}\n" for row in files).encode()).hexdigest()
    return {"files": files, "aggregate_sha256": aggregate}


def main():
    """Проверяет входы, создает или возобновляет прогон и запускает train.py."""
    parser = argparse.ArgumentParser(description="Reproduce the selected fold-0 training DAG from public initial weights")
    parser.add_argument("--config", required=True, help="DINOv2 model config JSON")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--output", required=True, help="new external run directory; must not exist")
    parser.add_argument("--resume")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_root, assets, output = Path(args.data_root).resolve(), Path(args.assets).resolve(), Path(args.output).resolve()
    fold_manifest = HERE / "configs/protocols/fold0/fold_0_manifest.json"
    protocol_dir = HERE / "configs/protocols/fold0"
    initial_weights = assets / config["initial_weights_filename"]
    if not initial_weights.exists() or sha256(initial_weights) != config["initial_weights_sha256"]:
        raise ValueError("Public initial weights are missing or do not match pinned SHA-256; run fetch_training_assets.py")
    image_manifest, image_manifest_hash = image_inventory(data_root, config)
    source_manifest = training_source_inventory(config_path)
    resuming = bool(args.resume)
    if output.exists():
        if not resuming:
            raise FileExistsError(f"Refusing to reuse existing training run: {output}")
        run_path = output / "run.json"
        if not run_path.is_file() or (output / "complete.json").exists():
            raise FileExistsError("resume requires an existing incomplete run directory with run.json")
        metadata = json.loads(run_path.read_text(encoding="utf-8"))
        expected = {
            "config_sha256": sha256(config_path),
            "initial_weights_sha256": sha256(initial_weights),
            "data_train_csv_sha256": sha256(data_root / "train.csv"),
            "fold_manifest_sha256": sha256(fold_manifest),
            "data_images_manifest_sha256": image_manifest_hash,
            "training_source_manifest_sha256": source_manifest["aggregate_sha256"],
        }
        mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
        if mismatches:
            raise ValueError(f"resume input hash mismatch: {mismatches}")
        resume_path = Path(args.resume).resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        if resume_path != (output / "resume.pt").resolve():
            raise ValueError("resume checkpoint must be the resume.pt inside the same incomplete run directory")
        metadata.update({"status": "running", "resume_from": str(resume_path),
                         "resume_checkpoint_sha256": sha256(resume_path), "resume_argv": sys.argv})
    else:
        if resuming:
            raise FileNotFoundError("--resume requires an existing incomplete output run directory")
        output.mkdir(parents=True)
        metadata = {
            "run_id": output.name, "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
            "uuid": str(uuid.uuid4()), "config": config, "config_sha256": sha256(config_path),
            "initial_weights_sha256": sha256(initial_weights), "data_train_csv_sha256": sha256(data_root / "train.csv"),
            "fold_manifest_sha256": sha256(fold_manifest), "data_images_manifest_sha256": image_manifest_hash,
            "training_source_manifest_sha256": source_manifest["aggregate_sha256"],
            "source_reproduce_sha256": sha256(Path(__file__)), "argv": sys.argv,
            "source_commit": config["source_commit"], "selection_metric": "official_csv_map_at_10",
            "output": str(output),
        }
    if not resuming:
        (output / "data_images_manifest.json").write_text(json.dumps(image_manifest, indent=2), encoding="utf-8")
        (output / "training_source_manifest.json").write_text(json.dumps(source_manifest, indent=2), encoding="utf-8")
    else:
        stored_manifest = output / "data_images_manifest.json"
        if not stored_manifest.is_file():
            raise FileNotFoundError("resume requires the original data_images_manifest.json")
        saved_inventory = json.loads(stored_manifest.read_text(encoding="utf-8"))
        if saved_inventory.get("aggregate_sha256") != image_manifest_hash:
            raise ValueError("dataset image hashes differ from the original run")
        saved_source = output / "training_source_manifest.json"
        if not saved_source.is_file():
            raise FileNotFoundError("resume requires the original training_source_manifest.json")
        if json.loads(saved_source.read_text(encoding="utf-8")).get("aggregate_sha256") != source_manifest["aggregate_sha256"]:
            raise ValueError("training source hashes differ from the original run")
    (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    command = [sys.executable, str(HERE / "train.py"), "--config", str(config_path),
               "--data-root", str(data_root), "--fold-manifest", str(fold_manifest),
               "--protocol-dir", str(protocol_dir), "--source-dir", str(HERE / "vendor/dinov2"),
               "--initial-weights", str(initial_weights), "--run-dir", str(output)]
    if args.resume:
        command += ["--resume", str(Path(args.resume).resolve())]
    start = time.perf_counter()
    try:
        with (output / "training_stdout.log").open("a" if resuming else "w", encoding="utf-8") as log:
            proc = subprocess.run(command, cwd=HERE.parent, stdout=log, stderr=subprocess.STDOUT, text=True)
        metadata["return_code"] = proc.returncode
        metadata["wall_seconds"] = time.perf_counter() - start
        if proc.returncode:
            metadata["status"] = "failed"
            (output / "failure.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            raise subprocess.CalledProcessError(proc.returncode, command)
        metadata["status"] = "complete"
        metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
        (output / "reproduce_wrapper.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    except Exception as exc:
        if not (output / "failure.json").exists():
            metadata.update({"status": "failed", "error": repr(exc), "wall_seconds": time.perf_counter() - start})
            (output / "failure.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        raise
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
