"""Ранжирует галерею по признакам и проверяет формат submission.csv."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Приводит матрицу признаков к float32 и нормирует строки по L2."""
    values = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def cosine_top10(query_vectors: np.ndarray, gallery_vectors: np.ndarray, gallery_ids: list[str], query_ids: list[str]) -> tuple[list[list[str]], np.ndarray]:
    """Возвращает top-10 галереи по косинусному сходству."""
    q = normalize(query_vectors)
    g = normalize(gallery_vectors)
    if q.shape[1] != g.shape[1]:
        raise ValueError(f"query/gallery dimensions differ: {q.shape[1]} vs {g.shape[1]}")
    scores = q @ g.T
    k = min(10, len(gallery_ids))
    order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    rankings = [[gallery_ids[j] for j in row] for row in order]
    return rankings, scores


def rerank_one(query_vector: np.ndarray, candidate_vectors: np.ndarray, *, k1: int = 15,
               k2: int = 3, lambda_value: float = 0.64) -> tuple[np.ndarray, np.ndarray]:
    """Re-rank a single query and its cosine shortlist; no other query is read."""
    q = normalize(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))
    g = normalize(candidate_vectors)
    if g.ndim != 2 or len(g) == 0 or g.shape[1] != q.shape[1]:
        raise ValueError("candidate shortlist must be a non-empty matrix matching the query dimension")
    if not 0 <= lambda_value <= 1:
        raise ValueError("lambda_value must be in [0, 1]")
    emb = np.concatenate([q, g], axis=0)
    n = len(emb)
    k1 = min(max(1, int(k1)), n - 1)
    k2 = min(max(1, int(k2)), n)
    original = (2.0 - 2.0 * np.clip(emb @ emb.T, -1.0, 1.0)).astype(np.float32)
    col_max = np.maximum(original.max(axis=0), 1e-12)
    original = (original / col_max).T
    initial_rank = np.argsort(original, axis=1, kind="stable").astype(np.int32)
    affinity = np.zeros_like(original, dtype=np.float32)
    half_k = max(1, round(k1 / 2))
    for i in range(n):
        forward = initial_rank[i, :k1 + 1]
        backward = initial_rank[forward, :k1 + 1]
        reciprocal = forward[np.any(backward == i, axis=1)]
        expanded = list(map(int, reciprocal))
        for candidate in reciprocal:
            candidate_forward = initial_rank[candidate, :half_k + 1]
            candidate_backward = initial_rank[candidate_forward, :half_k + 1]
            candidate_reciprocal = candidate_forward[
                np.any(candidate_backward == candidate, axis=1)]
            if len(candidate_reciprocal) and np.intersect1d(candidate_reciprocal, reciprocal).size > (2.0 / 3.0) * len(candidate_reciprocal):
                expanded.extend(map(int, candidate_reciprocal))
        expanded_idx = np.unique(np.asarray(expanded, dtype=np.int32))
        weights = np.exp(-original[i, expanded_idx])
        affinity[i, expanded_idx] = weights / max(float(weights.sum()), 1e-12)
    if k2 > 1:
        initial_affinity = affinity
        affinity = np.zeros_like(initial_affinity)
        for i in range(n):
            affinity[i] = initial_affinity[initial_rank[i, :k2]].mean(axis=0)
    inverted = [np.flatnonzero(affinity[:, j]) for j in range(n)]
    q_nonzero = np.flatnonzero(affinity[0])
    if len(q_nonzero):
        related = np.unique(np.concatenate([inverted[j] for j in q_nonzero]))
        overlap = np.zeros(n, dtype=np.float32)
        vals = affinity[np.ix_(related, q_nonzero)]
        overlap[related] = np.minimum(vals, affinity[0, q_nonzero]).sum(axis=1)
        jaccard = 1.0 - overlap / np.maximum(2.0 - overlap, 1e-12)
    else:
        jaccard = np.ones(n, dtype=np.float32)
    final_distance = (1.0 - lambda_value) * jaccard + lambda_value * original[0]
    return final_distance[1:], 1.0 - final_distance[1:]


def stream_rank(query_vectors: np.ndarray, gallery_vectors: np.ndarray, gallery_ids: list[str], *,
                top_k: int = 50, k1: int = 15, k2: int = 3, lambda_value: float = 0.64,
                export_k: int = 10) -> tuple[list[list[str]], np.ndarray]:
    """Cosine shortlist then independent current-query reranking against fixed gallery."""
    q, g = normalize(query_vectors), normalize(gallery_vectors)
    if q.shape[1] != g.shape[1]:
        raise ValueError("query/gallery dimensions differ")
    cosine = q @ g.T
    shortlist_size = min(max(1, int(top_k)), len(gallery_ids))
    export_size = min(int(export_k), len(gallery_ids))
    rankings: list[list[str]] = []
    confidence = np.empty(len(q), dtype=np.float32)
    for i in range(len(q)):
        shortlist = np.argsort(-cosine[i], kind="stable")[:shortlist_size]
        _, conf = rerank_one(q[i], g[shortlist], k1=k1, k2=k2, lambda_value=lambda_value)
        reranked = shortlist[np.argsort(-conf, kind="stable")]
        if shortlist_size < len(gallery_ids):
            mask = np.ones(len(gallery_ids), dtype=bool)
            mask[shortlist] = False
            tail = np.flatnonzero(mask)
            tail = tail[np.argsort(-cosine[i, tail], kind="stable")]
            reranked = np.concatenate([reranked, tail])
        rankings.append([gallery_ids[j] for j in reranked[:export_size]])
        confidence[i] = float(conf[np.argmax(conf)])
    return rankings, confidence


def write_submission(path: str | Path, query_ids: list[str], rankings: list[list[str]]) -> None:
    """Записывает ранжирование без заголовка и повторов кандидатов."""
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        for qid, ranked in zip(query_ids, rankings, strict=True):
            if len(set(ranked)) != len(ranked):
                raise ValueError(f"duplicate candidate for query {qid}")
            writer.writerow([qid, *ranked])


def verify_submission(path: str | Path, query_ids: list[str], gallery_ids: list[str], top_k: int = 10) -> dict:
    """Проверяет порядок запросов, число кандидатов и допустимость ID."""
    allowed = set(gallery_ids)
    seen_queries: list[str] = []
    lengths = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream):
            if not row:
                continue
            if row[0] in {"query_id", "query"} or len(row) < 2:
                raise ValueError("submission must be headerless and contain candidates")
            qid, preds = row[0], row[1:]
            if qid in seen_queries:
                raise ValueError(f"duplicate query {qid}")
            if len(preds) != min(top_k, len(gallery_ids)):
                raise ValueError(f"query {qid} has {len(preds)} candidates; expected {min(top_k, len(gallery_ids))}")
            if len(set(preds)) != len(preds) or not set(preds) <= allowed:
                raise ValueError(f"query {qid} has duplicate or unknown gallery ids")
            seen_queries.append(qid)
            lengths.append(len(preds))
    if seen_queries != query_ids:
        raise ValueError("submission query order or membership does not match query.csv")
    return {"rows": len(seen_queries), "candidate_count_min": min(lengths), "candidate_count_max": max(lengths), "headerless": True}
