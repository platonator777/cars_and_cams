"""Обучает DINOv2 на фиксированном fold 0 и выбирает checkpoint по локальной оценке."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("XFORMERS_DISABLED", "1")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from vehicle_reid.models.dino import DinoReID, model_state
from vehicle_reid.protocol import Record, load_records, sha256_file
from vehicle_reid.scoring import normalize, stream_rank, verify_submission, write_submission
from vehicle_reid.training.data import CameraPKSampler, VehicleDataset, build_transform, seed_worker


def now_utc() -> str:
    """Возвращает текущую дату и время в UTC."""
    return datetime.now(timezone.utc).isoformat()


def seed_all(seed: int) -> None:
    """Фиксирует генераторы случайных чисел и настройки детерминизма."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.use_deterministic_algorithms(True, warn_only=True)


def write_json(path: Path, value) -> None:
    """Создает родительский каталог и сохраняет значение в JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def read_config(path: Path) -> dict:
    """Загружает JSON-конфигурацию обучения."""
    return json.loads(path.read_text(encoding="utf-8"))


def split_fold(records: list[Record], manifest_path: Path) -> tuple[list[Record], list[Record], dict[int, int]]:
    """Разделяет размеченные записи по фиксированному списку ID fold 0."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    val_ids = set(map(int, manifest["validation_vehicle_ids"]))
    val_rows = [row for row in records if row.vehicle_id in val_ids]
    if [row.image_id for row in val_rows] != manifest["validation_image_ids"]:
        raise ValueError("fold-0 manifest row order or IDs do not match train.csv")
    train_rows = [row for row in records if row.vehicle_id not in val_ids]
    train_ids = sorted({row.vehicle_id for row in train_rows})
    if len(train_ids) != 1232 or len(val_ids) != 309 or set(train_ids) & val_ids:
        raise ValueError(f"unexpected fixed fold-0 split sizes: {len(train_ids)} train IDs, {len(val_ids)} val IDs")
    return train_rows, val_rows, {vid: i for i, vid in enumerate(train_ids)}


def triplet_loss(embedding: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    """Считает batch-hard triplet loss по самым сложным парам батча."""
    dist = (1.0 - embedding @ embedding.T).clamp_min(0.0)
    same = labels[:, None].eq(labels[None, :])
    same.fill_diagonal_(False)
    different = ~labels[:, None].eq(labels[None, :])
    if not bool(same.any(dim=1).all()) or not bool(different.any(dim=1).all()):
        raise ValueError("batch-hard triplet requires at least two samples per ID and two IDs")
    hardest_positive = dist.masked_fill(~same, float("-inf")).max(dim=1).values
    hardest_negative = dist.masked_fill(~different, float("inf")).min(dim=1).values
    return F.relu(hardest_positive - hardest_negative + margin).mean()


def set_lrs(optimizer, config: dict, epoch: int, batch_index: int, steps_per_epoch: int) -> None:
    """Обновляет learning rate головы и backbone с прогревом и спадом."""
    warm_epochs = int(config["head_warmup_epochs"])
    epoch_step = epoch * steps_per_epoch + batch_index
    if epoch < warm_epochs:
        ramp_steps = max(1, steps_per_epoch)
        head_factor = min(1.0, 0.1 + 0.9 * (epoch_step + 1) / ramp_steps)
        backbone_factor = 0.0
    else:
        adapt_step = (epoch - warm_epochs) * steps_per_epoch + batch_index
        total = max(1, int(config["max_adaptation_epochs"]) * steps_per_epoch)
        warm_steps = min(100, total // 10)
        warm = min(1.0, (adapt_step + 1) / max(1, warm_steps))
        progress = min(1.0, adapt_step / total)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        head_factor = warm * cosine
        backbone_factor = warm * cosine
    optimizer.param_groups[0]["lr"] = float(config["head_lr"]) * head_factor
    optimizer.param_groups[1]["lr"] = float(config["backbone_lr"]) * backbone_factor


def extract_rows(model: DinoReID, rows: list[Record], images_dir: Path, size: int,
                 device: torch.device, batch_size: int, workers: int) -> np.ndarray:
    """Извлекает признаки в исходном порядке строк для валидации."""
    dataset = VehicleDataset(rows, images_dir, build_transform(size, train=False))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        worker_init_fn=seed_worker if workers else None,
                        generator=torch.Generator().manual_seed(1907),
                        persistent_workers=False)
    chunks = []
    model.eval()
    with torch.inference_mode():
        for images, _, _ in loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                vectors = model.encode(images.to(device, non_blocking=True))
            chunks.append(vectors.float().cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def csv_ids(path: Path) -> list[str]:
    """Читает image_id из CSV с сохранением порядка строк."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [row["image_id"] for row in csv.DictReader(stream)]


def evaluate_export(run_dir: Path, epoch: int, kind: str, vectors: np.ndarray, val_rows: list[Record],
                    protocol_dir: Path, official_python: str, official_path: Path,
                    rerank_cfg: dict | None) -> tuple[dict, Path]:
    """Экспортирует валидационные файлы и запускает официальный оценщик."""
    query_ids, gallery_ids = csv_ids(protocol_dir / "query.csv"), csv_ids(protocol_dir / "gallery.csv")
    row_index = {row.image_id: i for i, row in enumerate(val_rows)}
    q_idx, g_idx = [row_index[x] for x in query_ids], [row_index[x] for x in gallery_ids]
    q_vectors, g_vectors = vectors[q_idx], vectors[g_idx]
    if rerank_cfg is None:
        from vehicle_reid.scoring import cosine_top10
        rankings, _ = cosine_top10(q_vectors, g_vectors, gallery_ids, query_ids)
    else:
        rankings, _ = stream_rank(q_vectors, g_vectors, gallery_ids, **rerank_cfg)
    out = run_dir / "validation" / f"epoch_{epoch:02d}" / kind
    out.mkdir(parents=True, exist_ok=False)
    submission = out / "submission.csv"
    embeddings_path = out / "embeddings.npy"
    write_submission(submission, query_ids, rankings)
    np.save(embeddings_path, np.concatenate([normalize(q_vectors), normalize(g_vectors)]).astype(np.float32))
    verify_submission(submission, query_ids, gallery_ids)
    metrics_path = out / "official_metrics.json"
    command = [official_python, str(official_path), "--gt", str(protocol_dir / "ground_truth.csv"),
               "--submission", str(submission), "--embeddings", str(embeddings_path),
               "--query", str(protocol_dir / "query.csv"), "--gallery", str(protocol_dir / "gallery.csv"),
               "--json", str(metrics_path)]
    proc = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (out / "official_stdout.txt").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode:
        raise RuntimeError(f"official evaluate.py failed at {epoch}/{kind}: {proc.stderr[-2000:]}")
    return json.loads(metrics_path.read_text(encoding="utf-8")), out


def save_training_state(path: Path, model, optimizer, scaler, epoch, global_step, best_metric,
                        best_epoch, patience, loader_generator, config, best_eval=None,
                        elapsed_seconds=0.0):
    """Сохраняет состояние модели, оптимизатора и генераторов для resume."""
    state = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
        "epoch_completed": epoch, "global_step": global_step, "best_metric": best_metric,
        "best_epoch": best_epoch, "patience": patience, "loader_generator": loader_generator.get_state(),
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
        "config": config, "best_eval": best_eval, "elapsed_seconds": float(elapsed_seconds),
        "resume_semantics": "epoch-boundary; next sampler epoch starts from its deterministic seed",
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def restore_training_state(checkpoint_data, model, optimizer, scaler, loader_generator):
    """Восстанавливает состояние незавершенного обучения из checkpoint."""
    model.load_state_dict(checkpoint_data["model"], strict=True)
    optimizer.load_state_dict(checkpoint_data["optimizer"])
    scaler.load_state_dict(checkpoint_data["scaler"])
    loader_generator.set_state(checkpoint_data["loader_generator"])
    random.setstate(checkpoint_data["python_rng"])
    np.random.set_state(checkpoint_data["numpy_rng"])
    torch.set_rng_state(checkpoint_data["torch_rng"])
    torch.cuda.set_rng_state_all(checkpoint_data["cuda_rng"])
    return {
        "epoch_start": int(checkpoint_data["epoch_completed"]),
        "global_step": int(checkpoint_data["global_step"]),
        "best_metric": float(checkpoint_data["best_metric"]),
        "best_epoch": int(checkpoint_data["best_epoch"]),
        "patience": int(checkpoint_data["patience"]),
        "best_eval": checkpoint_data.get("best_eval"),
        "elapsed_seconds": float(checkpoint_data.get("elapsed_seconds", 0.0)),
    }


def run_smoke(config: dict, args) -> dict:
    """Проверяет один шаг обучения и наличие конечных градиентов на CUDA."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("training smoke requires CUDA for the configured AMP recipe")
    seed_all(config["seed"])
    sample_ids = []
    if args.data_root and args.fold_manifest:
        all_rows = load_records(Path(args.data_root) / "train.csv")
        train_rows, _, label_map = split_fold(all_rows, Path(args.fold_manifest))
        smoke_num_classes = len(label_map)
        sampler = CameraPKSampler(train_rows, config["p"], config["k"], config["seed"])
        sample_indices = next(iter(sampler))
        sample_dataset = VehicleDataset(train_rows, Path(args.data_root) / config.get("images_dir", "images"),
                                        build_transform(config["image_size"], train=True), label_map)
        samples = [sample_dataset[i] for i in sample_indices]
        images = torch.stack([item[0] for item in samples]).to(device)
        labels = torch.tensor([item[1] for item in samples], dtype=torch.long, device=device)
        sample_ids = [train_rows[i].image_id for i in sample_indices]
    else:
        smoke_num_classes = 4
        images = torch.randn(16, 3, config["image_size"], config["image_size"], device=device)
        labels = torch.arange(4, device=device).repeat_interleave(4)
    model = DinoReID(config["architecture"], args.source_dir, args.initial_weights,
                     num_classes=smoke_num_classes, embedding_dim=config["embedding_dim"]).to(device)
    model.set_epoch(config["head_warmup_epochs"])
    model.train()
    head_before = model.embedding.weight.detach().clone()
    block_before = model.backbone.blocks[-1].attn.qkv.weight.detach().clone()
    optimizer = torch.optim.AdamW([
        {"params": list(model.embedding.parameters()) + list(model.bn.parameters()) + list(model.classifier.parameters()), "lr": config["head_lr"]},
        {"params": [p for block in model.backbone.blocks[-2:] for p in block.parameters()] + list(model.backbone.norm.parameters()), "lr": config["backbone_lr"]},
    ], weight_decay=config["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", init_scale=32768.0)
    torch.cuda.reset_peak_memory_stats()
    tic = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        emb, logits = model(images)
        loss = F.cross_entropy(logits, labels, label_smoothing=config["label_smoothing"]) + triplet_loss(emb.float(), labels, config["triplet_margin"])
    scaler.scale(loss).backward()
    head_grad = float(model.embedding.weight.grad.abs().sum()) if model.embedding.weight.grad is not None else 0.0
    block_grad = float(model.backbone.blocks[-1].attn.qkv.weight.grad.abs().sum()) if model.backbone.blocks[-1].attn.qkv.weight.grad is not None else 0.0
    finite_gradients = all(bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters() if parameter.grad is not None)
    scaler.step(optimizer)
    scaler.update()
    train_seconds = time.perf_counter() - tic
    head_delta = float((model.embedding.weight.detach() - head_before).abs().max())
    block_delta = float((model.backbone.blocks[-1].attn.qkv.weight.detach() - block_before).abs().max())
    model.eval()
    with torch.inference_mode():
        one = model.encode(images[:1])
    result = {"status": "completed", "batch_size": len(labels), "loss": float(loss.detach()), "finite_gradients": finite_gradients,
            "grad_scaler_value": float(scaler.get_scale()), "head_gradient_l1": head_grad,
            "last_block_gradient_l1": block_grad, "head_parameter_max_delta": head_delta,
            "last_block_parameter_max_delta": block_delta, "single_image_embedding_shape": list(one.shape),
            "single_image_embedding_norm": float(one.norm(dim=1).item()), "step_seconds": train_seconds,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(), "gpu": torch.cuda.get_device_name(0),
            "real_training_images": bool(sample_ids), "sample_image_ids": sample_ids}
    if not torch.isfinite(loss) or not finite_gradients or head_grad <= 0 or block_grad <= 0 or head_delta <= 0 or block_delta <= 0:
        result["status"] = "failed"
        print(json.dumps(result, indent=2), flush=True)
        raise RuntimeError("smoke failed finite loss/gradient/update invariants; metrics printed above")
    return result


def train_main(args) -> None:
    """Выполняет эпохи обучения, валидацию и сохранение лучшей модели."""
    config = read_config(Path(args.config).resolve())
    run_dir = Path(args.run_dir).resolve()
    if not (run_dir / "run.json").exists():
        raise FileNotFoundError("Create immutable run.json before starting a training run")
    images_dir = Path(args.data_root).resolve() / config.get("images_dir", "images")
    data_root = Path(args.data_root).resolve()
    manifest_path = Path(args.fold_manifest).resolve()
    protocol_dir = Path(args.protocol_dir).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required by the configured training recipe")
    seed_all(config["seed"])
    rows = load_records(data_root / "train.csv")
    train_rows, val_rows, label_map = split_fold(rows, manifest_path)
    source_dir, initial_weights = Path(args.source_dir).resolve(), Path(args.initial_weights).resolve()
    if sha256_file(initial_weights) != config["initial_weights_sha256"]:
        raise ValueError("initial checkpoint SHA256 does not match the frozen config")
    model = DinoReID(config["architecture"], source_dir, initial_weights,
                     num_classes=len(label_map), embedding_dim=config["embedding_dim"]).to(device)
    optimizer = torch.optim.AdamW([
        {"params": list(model.embedding.parameters()) + list(model.bn.parameters()) + list(model.classifier.parameters()), "lr": config["head_lr"]},
        {"params": [p for block in model.backbone.blocks[-2:] for p in block.parameters()] + list(model.backbone.norm.parameters()), "lr": config["backbone_lr"]},
    ], weight_decay=config["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", init_scale=32768.0)
    train_set = VehicleDataset(train_rows, images_dir, build_transform(config["image_size"], train=True), label_map)
    val_set = VehicleDataset(val_rows, images_dir, build_transform(config["image_size"], train=False))
    sampler = CameraPKSampler(train_rows, config["p"], config["k"], config["seed"])
    loader_generator = torch.Generator().manual_seed(config["seed"])
    loader = DataLoader(train_set, batch_sampler=sampler, num_workers=config["workers"],
                        worker_init_fn=seed_worker if config["workers"] else None,
                        generator=loader_generator, persistent_workers=False,
                        pin_memory=True)
    epoch_start, global_step, best_metric, best_epoch, patience = 0, 0, -1.0, -1, 0
    history_path = run_dir / "history.jsonl"
    base = run_dir / "artifacts"
    base.mkdir(parents=True, exist_ok=True)
    resume_path = Path(args.resume).resolve() if args.resume else None
    best_eval, elapsed_before_resume = None, 0.0
    if resume_path:
        checkpoint_data = torch.load(resume_path, map_location="cpu", weights_only=False)
        if checkpoint_data.get("config") != config:
            raise ValueError("resume checkpoint config differs from this training run")
        restored = restore_training_state(checkpoint_data, model, optimizer, scaler, loader_generator)
        epoch_start, global_step = restored["epoch_start"], restored["global_step"]
        best_metric, best_epoch, patience = restored["best_metric"], restored["best_epoch"], restored["patience"]
        best_eval, elapsed_before_resume = restored["best_eval"], restored["elapsed_seconds"]
        if best_eval is None and (base / "best_official_metrics.json").exists():
            best_eval = json.loads((base / "best_official_metrics.json").read_text(encoding="utf-8"))
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": "resumed", "epoch_completed": epoch_start, "time_utc": now_utc()}) + "\n")
    epochs_total = int(config["head_warmup_epochs"] + config["max_adaptation_epochs"])
    primary_rerank = config["selection_rerank"]
    official_python = config["official_python"]
    official_path = Path(config.get("official_evaluator", Path(__file__).resolve().parent / "official/evaluate.py")).resolve()
    started = time.perf_counter()
    for epoch in range(epoch_start, epochs_total):
        model.set_epoch(epoch, config["head_warmup_epochs"])
        model.train()
        sampler.set_epoch(epoch)
        losses, ce_values, triplet_values = [], [], []
        torch.cuda.reset_peak_memory_stats()
        epoch_started = time.perf_counter()
        for batch_index, (images, labels, _) in enumerate(loader):
            set_lrs(optimizer, config, epoch, batch_index, len(loader))
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                embedding, logits = model(images)
                ce = F.cross_entropy(logits, labels, label_smoothing=config["label_smoothing"])
                tri = triplet_loss(embedding.float(), labels, config["triplet_margin"])
                loss = ce + config["triplet_coefficient"] * tri
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
            ce_values.append(float(ce.detach()))
            triplet_values.append(float(tri.detach()))
            global_step += 1
        val_features = extract_rows(model, val_rows, images_dir, config["image_size"], device,
                                    config["eval_batch_size"], config["workers"])
        feature_path = run_dir / "validation" / f"epoch_{epoch:02d}" / "all_fold0_features.npy"
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(feature_path, val_features)
        raw_metrics, raw_dir = evaluate_export(run_dir, epoch, "raw_cosine", val_features, val_rows,
                                                protocol_dir, official_python, official_path, None)
        reranked_metrics, reranked_dir = evaluate_export(run_dir, epoch, "stream_rerank", val_features, val_rows,
                                                          protocol_dir, official_python, official_path, primary_rerank)
        value = float(reranked_metrics["ranking"]["mAP@10"])
        improved = value > best_metric + float(config["min_delta"])
        if improved:
            best_metric, best_epoch, patience, best_eval = value, epoch, 0, reranked_metrics
            torch.save({"model": model.state_dict(), "architecture": config["architecture"],
                        "embedding_dim": config["embedding_dim"], "num_classes": len(label_map),
                        "best_epoch": epoch, "selection_metric": value, "config": config}, base / "best_model.pt")
            shutil.copy2(feature_path, base / "best_val_features.npy")
            shutil.copy2(reranked_dir / "submission.csv", base / "best_submission.csv")
            shutil.copy2(reranked_dir / "embeddings.npy", base / "best_eval_embeddings.npy")
            shutil.copy2(reranked_dir / "official_metrics.json", base / "best_official_metrics.json")
        else:
            patience += 1
        result = {
            "epoch": epoch + 1, "epoch_zero_based": epoch, "loss": float(np.mean(losses)),
            "cross_entropy": float(np.mean(ce_values)), "triplet": float(np.mean(triplet_values)),
            "raw_mAP@10": float(raw_metrics["ranking"]["mAP@10"]),
            "reranked_mAP@10": value, "selection_metric": "official_csv_map_at_10",
            "best_epoch": best_epoch + 1, "best_mAP@10": best_metric, "improved": improved,
            "patience": patience, "train_seconds": time.perf_counter() - epoch_started,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(),
            "head_lr": optimizer.param_groups[0]["lr"], "backbone_lr": optimizer.param_groups[1]["lr"],
            "raw_metrics_file": str(raw_dir / "official_metrics.json"),
            "reranked_metrics_file": str(reranked_dir / "official_metrics.json"), "time_utc": now_utc(),
        }
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(json.dumps(result, ensure_ascii=False), flush=True)
        elapsed_so_far = elapsed_before_resume + time.perf_counter() - started
        save_training_state(run_dir / "resume.pt", model, optimizer, scaler, epoch + 1,
                            global_step, best_metric, best_epoch, patience, loader_generator, config,
                            best_eval=best_eval, elapsed_seconds=elapsed_so_far)
        if epoch + 1 >= config["head_warmup_epochs"] + config["min_adaptation_epochs"] and patience >= config["patience"]:
            break
    summary = {
        "status": "completed", "architecture": config["architecture"], "best_epoch": best_epoch + 1,
        "best_official_csv_map_at_10": best_metric, "best_official_metrics": best_eval,
        "train_images": len(train_rows), "train_ids": len(label_map), "val_images": len(val_rows),
        "fold_manifest_sha256": sha256_file(manifest_path), "train_csv_sha256": sha256_file(data_root / "train.csv"),
        "initial_weights_sha256": sha256_file(initial_weights), "source_commit": config["source_commit"],
        "runtime_checkpoint_sha256": sha256_file(base / "best_model.pt"),
        "runtime_checkpoint_bytes": (base / "best_model.pt").stat().st_size,
        "feature_cache_sha256": sha256_file(base / "best_val_features.npy"),
        "feature_cache_shape": list(np.load(base / "best_val_features.npy", mmap_mode="r").shape),
        "total_wall_seconds": elapsed_before_resume + time.perf_counter() - started, "finished_utc": now_utc(),
        "resume_checkpoint_sha256": sha256_file(run_dir / "resume.pt"),
    }
    write_json(run_dir / "result.json", summary)
    write_json(run_dir / "timing.json", {"total_wall_seconds": summary["total_wall_seconds"],
                "gpu_hours": summary["total_wall_seconds"] / 3600, "device": torch.cuda.get_device_name(0),
                "peak_vram_bytes": torch.cuda.max_memory_allocated()})
    run_record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    run_record.update({"status": "complete", "completed_utc": now_utc(), "result_sha256": sha256_file(run_dir / "result.json"),
                       "weights_initial_sha256": sha256_file(initial_weights), "vendor_source_dir": str(source_dir),
                       "val_protocol_id": json.loads((protocol_dir / "protocol.json").read_text(encoding="utf-8"))["protocol_id"]})
    write_json(run_dir / "run.json", run_record)
    write_json(run_dir / "complete.json", {"status": "complete", "result_sha256": sha256_file(run_dir / "result.json"),
                                             "checkpoint_sha256": sha256_file(base / "best_model.pt"), "completed_utc": now_utc()})
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def main():
    """Разбирает аргументы и выбирает полный прогон или smoke-проверку."""
    parser = argparse.ArgumentParser(description="Reproducible fold-0 DINOv2 vehicle retrieval trainer")
    parser.add_argument("--config")
    parser.add_argument("--data-root")
    parser.add_argument("--fold-manifest")
    parser.add_argument("--protocol-dir")
    parser.add_argument("--source-dir")
    parser.add_argument("--initial-weights")
    parser.add_argument("--run-dir")
    parser.add_argument("--resume")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        config = read_config(Path(args.config).resolve())
        result = run_smoke(config, args)
        print(json.dumps(result, indent=2))
    else:
        required = [args.config, args.data_root, args.fold_manifest, args.protocol_dir,
                    args.source_dir, args.initial_weights, args.run_dir]
        if not all(required):
            parser.error("training requires --config, --data-root, --fold-manifest, --protocol-dir, --source-dir, --initial-weights and --run-dir")
        train_main(args)


if __name__ == "__main__":
    main()
