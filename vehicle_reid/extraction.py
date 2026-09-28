"""Извлекает нормированные признаки автомобиля из кадров и BBox."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from vehicle_reid.protocol import Record
from vehicle_reid.training.data import VehicleDataset, build_transform, crop_bbox, seed_worker


def extract_records(model, rows: list[Record], images_dir: str | Path, image_size: int,
                    batch_size: int = 16, workers: int = 4, device: torch.device | None = None) -> np.ndarray:
    """Обрабатывает список кадров батчами и возвращает матрицу признаков."""
    if not rows:
        raise ValueError("cannot extract an empty image list")
    device = device or next(model.parameters()).device
    dataset = VehicleDataset(rows, images_dir, build_transform(image_size, train=False))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        worker_init_fn=seed_worker if workers else None,
                        generator=torch.Generator().manual_seed(1907), persistent_workers=False,
                        pin_memory=device.type == "cuda")
    model.eval()
    chunks = []
    with torch.inference_mode():
        for images, _, _ in loader:
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                vectors = model.encode(images.to(device, non_blocking=True))
            chunks.append(vectors.float().cpu().numpy())
    output = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    if output.ndim != 2 or not np.isfinite(output).all() or np.any(np.linalg.norm(output, axis=1) < 1e-8):
        raise ValueError("extractor returned non-finite or zero embeddings")
    return output


def extract_one(model, row: Record, images_dir: str | Path, image_size: int,
                device: torch.device | None = None) -> np.ndarray:
    """Извлекает признак одного автомобиля для измерения задержки."""
    device = device or next(model.parameters()).device
    started = time.perf_counter()
    with Image.open(Path(images_dir) / f"{row.image_id}.jpg") as source:
        crop = crop_bbox(source.convert("RGB"), row)
    image = build_transform(image_size, train=False)(crop).unsqueeze(0).to(device)
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                               enabled=device.type == "cuda"):
        vector = model.encode(image).float()
    if not torch.isfinite(vector).all() or vector.norm().item() < 1e-8:
        raise ValueError(f"invalid descriptor for {row.image_id}")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return vector[0].cpu().numpy().astype(np.float32, copy=False)
