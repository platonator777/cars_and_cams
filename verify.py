"""Проверяет формат и согласованность трех файлов результата инференса."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from vehicle_reid.protocol import load_records
from vehicle_reid.scoring import verify_submission


def verify_files(output: Path, query_path: Path, gallery_path: Path) -> dict:
    """Проверяет submission, embeddings и принятые кандидаты по входным CSV."""
    query_ids = [r.image_id for r in load_records(query_path)]
    gallery_ids = [r.image_id for r in load_records(gallery_path)]
    fmt = verify_submission(output / "submission.csv", query_ids, gallery_ids)
    embeddings = np.load(output / "embeddings.npy", allow_pickle=False)
    if embeddings.dtype != np.float32 or embeddings.ndim != 2 or embeddings.shape[0] != len(query_ids) + len(gallery_ids):
        raise ValueError("embeddings must be float32 2D, query rows then gallery rows")
    if not np.isfinite(embeddings).all() or np.any(np.linalg.norm(embeddings, axis=1) < 1e-8):
        raise ValueError("embeddings contain invalid or zero rows")
    with (output / "candidates.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["query_id", "gallery_id", "confidence"]:
            raise ValueError("candidates.csv header must be query_id,gallery_id,confidence")
        accepted = list(reader)
    qset, gset = set(query_ids), set(gallery_ids)
    accepted_qids = []
    submission_rows = {}
    with (output / "submission.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream):
            if row:
                submission_rows[row[0]] = row[1:]
    for row in accepted:
        if (row["query_id"] not in qset or row["gallery_id"] not in gset or
                not row["confidence"] or not np.isfinite(float(row["confidence"]))):
            raise ValueError("candidates.csv has invalid IDs or confidence")
        if submission_rows[row["query_id"]][0] != row["gallery_id"]:
            raise ValueError("accepted candidate must equal the query's submission top-1")
        accepted_qids.append(row["query_id"])
    if len(accepted_qids) != len(set(accepted_qids)):
        raise ValueError("candidates.csv must have at most one top-1 accepted answer per query")
    return {"status": "passed", "submission": fmt, "embeddings_shape": list(embeddings.shape),
            "accepted_candidate_rows": len(accepted),
            "query_count": len(query_ids), "gallery_count": len(gallery_ids)}


def main():
    """Разбирает пути к файлам и печатает итог проверки в JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--gallery", required=True)
    args = parser.parse_args()
    print(json.dumps(verify_files(Path(args.output), Path(args.query), Path(args.gallery)), indent=2))


if __name__ == "__main__":
    main()
