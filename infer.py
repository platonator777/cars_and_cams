"""Выполняет пакетный поиск автомобилей и сохраняет три файла результата."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from vehicle_reid.extraction import extract_records
from vehicle_reid.models.dino import DinoReID
from vehicle_reid.protocol import load_records, sha256_file
from vehicle_reid.scoring import normalize, stream_rank, verify_submission, write_submission


HERE = Path(__file__).resolve().parent


def write_candidates(path: Path, query_ids: list[str], gallery_ids: list[str], rankings: list[list[str]],
                     confidence: np.ndarray, threshold: float) -> int:
    """Записывает принятые top-1 совпадения с учетом порога уверенности."""
    if not np.isfinite(confidence).all():
        raise ValueError("confidence scores must be finite")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["query_id", "gallery_id", "confidence"])
        accepted = 0
        for qid, ranked, score in zip(query_ids, rankings, confidence, strict=True):
            if float(score) >= threshold:
                writer.writerow([qid, ranked[0], format(float(score), ".9g")])
                accepted += 1
    return accepted


def main() -> None:
    """Загружает данные и модель, ранжирует галерею и проверяет файлы результата."""
    parser = argparse.ArgumentParser(description="Single-query-independent batch retrieval inference")
    parser.add_argument("--images", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--gallery", required=True)
    parser.add_argument("--output", required=True, help="new output directory")
    parser.add_argument("--config", default=str(HERE / "configs/inference.yaml"))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not config.get("release_frozen", False):
        raise RuntimeError("inference config is still a draft; freeze model, scorer and calibration threshold first")
    out = Path(args.output).resolve()
    if out.exists():
        if any(out.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty inference output: {out}")
    else:
        out.mkdir(parents=True)
    query_rows, gallery_rows = load_records(args.query), load_records(args.gallery)
    query_ids, gallery_ids = [r.image_id for r in query_rows], [r.image_id for r in gallery_rows]
    if not query_rows or not gallery_rows or len(gallery_ids) != len(set(gallery_ids)):
        raise ValueError("query and gallery must be non-empty; gallery IDs must be unique")
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("query IDs must be unique")
    device = torch.device("cuda" if torch.cuda.is_available() and config.get("device", "cuda") != "cpu" else "cpu")
    checkpoint_path = HERE / config["weights"]
    expected_weight_hash = config.get("weights_sha256")
    if not expected_weight_hash or sha256_file(checkpoint_path) != expected_weight_hash:
        raise RuntimeError("runtime checkpoint is missing or does not match configs/inference.yaml")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    num_classes = int(checkpoint.get("num_classes", 1)) if isinstance(checkpoint, dict) else 1
    model = DinoReID(config["architecture"], HERE / "vendor/dinov2", None,
                     num_classes=num_classes, embedding_dim=int(config["embedding_dim"]))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    q_vectors = extract_records(model, query_rows, args.images, config["image_size"],
                                int(config.get("batch_size", 16)), int(config.get("workers", 4)), device)
    g_vectors = extract_records(model, gallery_rows, args.images, config["image_size"],
                                int(config.get("batch_size", 16)), int(config.get("workers", 4)), device)
    scorer = config["scorer"]
    rankings, confidence = stream_rank(q_vectors, g_vectors, gallery_ids,
                                       top_k=int(scorer["top_k"]), k1=int(scorer["k1"]),
                                       k2=int(scorer["k2"]), lambda_value=float(scorer["lambda_value"]),
                                       export_k=min(10, len(gallery_ids)))
    threshold = float(config["confidence_threshold"])
    submission_path, embeddings_path, candidates_path = out / "submission.csv", out / "embeddings.npy", out / "candidates.csv"
    write_submission(submission_path, query_ids, rankings)
    embeddings = np.concatenate([normalize(q_vectors), normalize(g_vectors)]).astype(np.float32)
    np.save(embeddings_path, embeddings)
    accepted = write_candidates(candidates_path, query_ids, gallery_ids, rankings, confidence, threshold)
    format_result = verify_submission(submission_path, query_ids, gallery_ids)
    if embeddings.shape != (len(query_rows) + len(gallery_rows), int(config["embedding_dim"])):
        raise ValueError("embeddings.npy has the wrong query-then-gallery shape")
    if not np.isfinite(embeddings).all() or np.any(np.linalg.norm(embeddings, axis=1) < 1e-8):
        raise ValueError("embeddings.npy must contain finite non-zero float32 vectors")
    print(json.dumps({"output": str(out), "query": len(query_rows), "gallery": len(gallery_rows),
                      "accepted": accepted, "embeddings": list(embeddings.shape), "format": format_result}, indent=2))


if __name__ == "__main__":
    main()
