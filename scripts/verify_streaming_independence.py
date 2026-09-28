"""Проверяет независимость результата запроса от других запросов в батче."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from infer import load_records
from vehicle_reid.extraction import extract_records
from vehicle_reid.models.dino import DinoReID
from vehicle_reid.protocol import sha256_file
from vehicle_reid.scoring import stream_rank


def main() -> None:
    """Сравнивает признаки, ранжирование и уверенность при разных составах батча."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--gallery", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if not cfg.get("release_frozen"):
        raise ValueError("independence test requires a frozen release config")
    qrows, grows = load_records(args.query), load_records(args.gallery)
    if len(qrows) < 33 or len(grows) < 10:
        raise ValueError("independence test needs at least 33 queries and 10 gallery images")
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "cuda") != "cpu" else "cpu")
    checkpoint_path = HERE / cfg["weights"]
    expected_hash = cfg.get("weights_sha256")
    if not expected_hash or sha256_file(checkpoint_path) != expected_hash:
        raise RuntimeError("runtime checkpoint is missing or does not match the frozen config")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = DinoReID(cfg["architecture"], HERE / "vendor/dinov2", None,
                     num_classes=int(checkpoint["num_classes"]),
                     embedding_dim=int(cfg["embedding_dim"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    images = Path(args.images)
    gallery_ids = [r.image_id for r in grows]
    qids = [r.image_id for r in qrows]
    gallery_before = sha256_file(args.gallery)
    gallery_vec = extract_records(model, grows, images, int(cfg["image_size"]),
                                  batch_size=16, workers=2, device=device)
    target_index = 7
    target = qrows[target_index]
    top32 = qrows[:32]
    b1 = [extract_records(model, [target], images, int(cfg["image_size"]), 1, 0, device)[0]]
    b8_rows = [qrows[3], qrows[5], target, qrows[2], qrows[10], qrows[13], qrows[17], qrows[1]]
    b8 = extract_records(model, b8_rows, images, int(cfg["image_size"]), 8, 0, device)
    batch_target_index = [r.image_id for r in b8_rows].index(target.image_id)
    b32_rows = list(reversed(top32))
    b32 = extract_records(model, b32_rows, images, int(cfg["image_size"]), 32, 0, device)
    batch32_target_index = [r.image_id for r in b32_rows].index(target.image_id)
    distractors = [r for r in qrows if r.image_id != target.image_id][:7]
    augmented = [target, *distractors]
    b_added = extract_records(model, augmented, images, int(cfg["image_size"]), 8, 0, device)
    query_ids = [target.image_id]
    scorer = cfg["scorer"]
    rankings = []
    confs = []
    for vector in (b1[0], b8[batch_target_index], b32[batch32_target_index], b_added[0]):
        ranked, confidence = stream_rank(vector.reshape(1, -1), gallery_vec, gallery_ids,
                                         top_k=int(scorer["top_k"]), k1=int(scorer["k1"]),
                                         k2=int(scorer["k2"]), lambda_value=float(scorer["lambda_value"]),
                                         export_k=min(10, len(grows)))
        rankings.append(ranked[0])
        confs.append(float(confidence[0]))
    pairwise = [float(np.max(np.abs(b1[0] - vector))) for vector in
                (b8[batch_target_index], b32[batch32_target_index], b_added[0])]
    result = {
        "status": "passed" if all(rank == rankings[0] for rank in rankings[1:]) and
                  max(pairwise) <= 2e-3 and max(confs) - min(confs) <= 2e-3 else "failed",
        "query_id": target.image_id, "comparison_modes": ["single", "batch8_mixed", "batch32_reversed", "batch8_plus_distractors"],
        "max_abs_descriptor_delta_vs_single": pairwise,
        "confidence_values": confs,
        "ranking_exact_match": all(rank == rankings[0] for rank in rankings[1:]),
        "rankings": rankings, "gallery_csv_sha256_before": gallery_before,
        "gallery_csv_sha256_after": sha256_file(args.gallery),
        "gallery_file_unchanged": gallery_before == sha256_file(args.gallery),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "query_file_sha256": sha256_file(args.query), "gallery_file_sha256": sha256_file(args.gallery),
        "no_query_labels_loaded": True, "other_query_features_only_used_for_batching": True,
        "tolerance": {"descriptor_max_abs": 2e-3, "confidence_abs": 2e-3, "ranking": "exact"},
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
