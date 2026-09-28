"""Читает CSV-протокол и формирует фиксированный локальный query/gallery split."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Record:
    """Запись кадра, BBox и необязательных меток автомобиля и камеры."""
    image_id: str
    x: int
    y: int
    w: int
    h: int
    vehicle_id: int | None = None
    camera_id: int | None = None


def load_records(path: str | Path) -> list[Record]:
    """Преобразует строки CSV с BBox и метками в записи Record."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    records = []
    for row in rows:
        records.append(Record(
            image_id=str(row["image_id"]), x=int(float(row["x"])),
            y=int(float(row["y"])), w=int(float(row["w"])), h=int(float(row["h"])),
            vehicle_id=int(row["vehicle_id"]) if row.get("vehicle_id") not in (None, "") else None,
            camera_id=int(row["camera_id"]) if row.get("camera_id") not in (None, "") else None,
        ))
    return records


def sha256_file(path: str | Path) -> str:
    """Вычисляет SHA-256 файла."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(seed: int, value: str) -> bytes:
    """Создает детерминированный ключ для выбора кадров."""
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()


def fixed_closed_split(records: list[Record], seed: int = 8510) -> tuple[list[Record], list[Record], dict]:
    """Choose exactly two fixed gallery frames per validation identity."""
    grouped: dict[int, list[Record]] = {}
    for row in records:
        if row.vehicle_id is None or row.camera_id is None:
            raise ValueError("fold-0 protocol needs vehicle_id and camera_id labels")
        grouped.setdefault(row.vehicle_id, []).append(row)
    gallery_ids: set[str] = set()
    selection = {}
    for vehicle_id in sorted(grouped):
        rows = sorted(grouped[vehicle_id], key=lambda row: (stable_key(seed, row.image_id), row.image_id))
        if len(rows) < 2:
            raise ValueError(f"vehicle {vehicle_id} has fewer than two images")
        first = rows[0]
        second = next((row for row in rows[1:] if row.camera_id != first.camera_id), rows[1])
        chosen = {first.image_id, second.image_id}
        gallery_ids.update(chosen)
        selection[str(vehicle_id)] = {
            "gallery_image_ids": [row.image_id for row in records if row.image_id in chosen],
            "gallery_cameras": [first.camera_id, second.camera_id],
            "cross_camera": first.camera_id != second.camera_id,
        }
    gallery = [row for row in records if row.image_id in gallery_ids]
    query = [row for row in records if row.image_id not in gallery_ids]
    if set(r.image_id for r in gallery) & set(r.image_id for r in query):
        raise AssertionError("query/gallery image overlap")
    gallery_pairs = {(row.vehicle_id, row.camera_id) for row in gallery}
    eligible = sum(any(vid == row.vehicle_id and cam != row.camera_id for vid, cam in gallery_pairs) for row in query)
    protocol = {
        "protocol_id": "cars-fold0-qg8510-v1",
        "seed": seed,
        "selection_rule": "two images per vehicle; deterministic SHA256 order; choose different cameras where available; keep original validation row order",
        "query_image_ids": [row.image_id for row in query],
        "gallery_image_ids": [row.image_id for row in gallery],
        "eligible_queries": eligible,
        "ineligible_queries": len(query) - eligible,
        "identities": len(grouped),
        "per_identity": selection,
    }
    return query, gallery, protocol


def write_query_csv(path: str | Path, rows: Iterable[Record]) -> None:
    """Записывает image_id и BBox в CSV протокола."""
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["image_id", "x", "y", "w", "h"])
        for row in rows:
            writer.writerow([row.image_id, row.x, row.y, row.w, row.h])


def write_ground_truth(path: str | Path, query: list[Record], gallery: list[Record]) -> None:
    """Записывает метки query и gallery для локальной оценки."""
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["image_id", "vehicle_id", "camera_id", "split"])
        for split, rows in (("query", query), ("gallery", gallery)):
            for row in rows:
                writer.writerow([row.image_id, row.vehicle_id, row.camera_id, split])


def build_protocol(project_root: str | Path, output_dir: str | Path, seed: int = 8510) -> dict:
    """Строит и сохраняет query/gallery протокол для фиксированного fold 0."""
    root = Path(project_root)
    manifest_path = root / "outputs/E002b/manifests/fold_0.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = load_records(root / "train.csv")
    validation_ids = set(manifest["validation_vehicle_ids"])
    val_records = [row for row in records if row.vehicle_id in validation_ids]
    if [row.image_id for row in val_records] != manifest["validation_image_ids"]:
        raise ValueError("fixed fold-0 manifest does not match train.csv row order")
    query, gallery, protocol = fixed_closed_split(val_records, seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    write_query_csv(out / "query.csv", query)
    write_query_csv(out / "gallery.csv", gallery)
    write_ground_truth(out / "ground_truth.csv", query, gallery)
    protocol.update({
        "fold_manifest_sha256": sha256_file(manifest_path),
        "train_csv_sha256": sha256_file(root / "train.csv"),
        "query_csv_sha256": sha256_file(out / "query.csv"),
        "gallery_csv_sha256": sha256_file(out / "gallery.csv"),
        "ground_truth_sha256": sha256_file(out / "ground_truth.csv"),
        "row_order": "query/gallery retain their original fold-0 train.csv order",
        "labels_used_only_for_protocol_and_official_local_evaluation": True,
    })
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    return protocol
