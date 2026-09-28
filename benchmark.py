"""Измеряет задержку и пропускную способность извлечения признаков из кадров."""

from __future__ import annotations

import argparse
import ctypes
import csv
import json
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from vehicle_reid.extraction import extract_one
from vehicle_reid.models.dino import DinoReID
from vehicle_reid.protocol import load_records, sha256_file
from vehicle_reid.training.data import VehicleDataset, build_transform, seed_worker


HERE = Path(__file__).resolve().parent


def rss_bytes() -> int | None:
    """Возвращает потребление оперативной памяти текущим процессом."""
    if os.name == "nt":
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        get_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process.restype = ctypes.c_void_p
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), ctypes.c_ulong]
        get_memory.restype = ctypes.c_int
        process = get_process()
        if get_memory(process, ctypes.byref(counters), counters.cb):
            return int(counters.WorkingSetSize)
        return None
    try:
        import resource
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if platform.system() == "Darwin" else value * 1024
    except Exception:
        return None


def load_model(config: dict, device: torch.device):
    """Проверяет хеш и загружает релизную модель для замеров."""
    checkpoint_path = HERE / config["weights"]
    expected_hash = config.get("weights_sha256")
    if not expected_hash or sha256_file(checkpoint_path) != expected_hash:
        raise RuntimeError("benchmark checkpoint is missing or does not match the frozen config")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    classes = int(checkpoint.get("num_classes", 1)) if isinstance(checkpoint, dict) else 1
    model = DinoReID(config["architecture"], HERE / "vendor/dinov2", None,
                     num_classes=classes, embedding_dim=int(config["embedding_dim"]))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), checkpoint_path


def sync(device):
    """Дожидается завершения операций CUDA перед измерением времени."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def throughput_profile(model, rows, images_dir, image_size, batch_size, duration_s, device, workers):
    """Измеряет FPS и память для заданного размера батча."""
    dataset = VehicleDataset(rows, images_dir, build_transform(image_size, train=False))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                        num_workers=workers, worker_init_fn=seed_worker if workers else None,
                        generator=torch.Generator().manual_seed(7181), persistent_workers=workers > 0,
                        pin_memory=device.type == "cuda")
    iterator = iter(loader)
    count, started = 0, time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    try:
        while time.perf_counter() - started < duration_s:
            try:
                images, _, _ = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                images, _, _ = next(iterator)
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                                        enabled=device.type == "cuda"):
                vectors = model.encode(images.to(device, non_blocking=True))
            count += len(images)
    except RuntimeError as exc:
        del iterator, loader
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if "out of memory" not in str(exc).lower():
            raise
        return {"batch_size": batch_size, "duration_seconds": time.perf_counter() - started,
                "images": count, "fps": None, "rss_bytes": rss_bytes(),
                "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                "status": "oom"}
    sync(device)
    elapsed = time.perf_counter() - started
    del iterator, loader
    return {"batch_size": batch_size, "duration_seconds": elapsed, "images": count,
            "fps": count / elapsed, "rss_bytes": rss_bytes(),
            "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}


def main():
    """Считывает параметры, проводит замеры и сохраняет отчет JSON."""
    parser = argparse.ArgumentParser(description="Measure the actual image-to-descriptor function")
    parser.add_argument("--config", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--warmups", type=int, default=50)
    parser.add_argument("--latency-samples", type=int, default=300)
    parser.add_argument("--throughput-seconds", type=float, default=10.0)
    parser.add_argument("--soak-seconds", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() and config.get("device", "cuda") != "cpu" else "cpu")
    model, checkpoint_path = load_model(config, device)
    rows = load_records(args.query)
    images_dir = Path(args.images).resolve()
    if len(rows) < 32:
        raise ValueError("benchmark needs at least 32 real query images")
    rng = random.Random(int(config.get("seed", 8514)))
    sample = [rows[rng.randrange(len(rows))] for _ in range(args.warmups + args.latency_samples)]
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    for row in sample[:args.warmups]:
        extract_one(model, row, images_dir, config["image_size"], device)
    latencies = []
    for row in sample[args.warmups:]:
        sync(device)
        started = time.perf_counter()
        extract_one(model, row, images_dir, config["image_size"], device)
        sync(device)
        latencies.append((time.perf_counter() - started) * 1000.0)
    throughput = [throughput_profile(model, rows, images_dir, config["image_size"], batch,
                                     args.throughput_seconds, device, args.workers)
                  for batch in (1, 8, 16, 32)]
    successes = [row for row in throughput if row.get("status") != "oom"]
    if not successes:
        raise RuntimeError("all throughput batch sizes failed")
    best = max(successes, key=lambda row: row["fps"])
    soak = []
    if args.soak_seconds > 0:
        until = time.monotonic() + args.soak_seconds
        start = time.monotonic()
        while time.monotonic() < until:
            period = min(60.0, until - time.monotonic())
            if period <= 0:
                break
            point = throughput_profile(model, rows, images_dir, config["image_size"],
                                       best["batch_size"], period, device, args.workers)
            point["elapsed_seconds"] = time.monotonic() - start
            soak.append(point)
    result = {
        "status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": config.get("model_id", config["architecture"]),
        "checkpoint_sha256": sha256_file(checkpoint_path), "checkpoint_bytes": checkpoint_path.stat().st_size,
        "extract_scope": "disk read + decode + 5% bbox context + preprocessing + forward + L2",
        "search_and_reranking_in_latency": False, "warmups": args.warmups, "latency_samples": len(latencies),
        "latency_batch1_ms": {"p50": float(np.percentile(latencies, 50)),
                              "p95": float(np.percentile(latencies, 95)),
                              "mean": float(np.mean(latencies))},
        "throughput": throughput, "best_throughput": best,
        "soak": {"target_seconds": args.soak_seconds, "actual_seconds": soak[-1]["elapsed_seconds"] if soak else 0,
                 "intervals": soak, "status": "completed" if args.soak_seconds and soak else "not_requested"},
        "hardware": {"gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                     "cuda_runtime": torch.version.cuda, "torch": torch.__version__,
                     "cpu": platform.processor(), "python": sys.version,
                     "cpu_threads": torch.get_num_threads(), "rss_bytes": rss_bytes(),
                     "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None},
        "protocol": {"images_directory": str(images_dir), "query_csv_sha256": sha256_file(args.query),
                     "query_rows": len(rows), "precision": "FP16 autocast on CUDA; float32 output"},
    }
    report = Path(args.report).resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
