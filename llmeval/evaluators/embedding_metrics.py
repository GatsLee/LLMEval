"""
Embedding evaluation metrics: cosine similarity, Spearman correlation, retrieval ranking.
Pure Python implementation (no numpy/scipy dependency).
"""
import math
from typing import List, Dict, Any


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """두 벡터의 코사인 유사도."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _rank(values: List[float]) -> List[float]:
    """타이 처리 포함 순위 계산 (평균 순위 방식)."""
    n = len(values)
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman_rank_correlation(predicted: List[float], human: List[float]) -> float:
    """Spearman 순위 상관계수 (ρ)."""
    n = len(predicted)
    if n < 2:
        return 0.0
    rank_p = _rank(predicted)
    rank_h = _rank(human)

    mean_p = sum(rank_p) / n
    mean_h = sum(rank_h) / n

    cov = sum((rp - mean_p) * (rh - mean_h) for rp, rh in zip(rank_p, rank_h))
    std_p = math.sqrt(sum((rp - mean_p) ** 2 for rp in rank_p))
    std_h = math.sqrt(sum((rh - mean_h) ** 2 for rh in rank_h))

    if std_p == 0 or std_h == 0:
        return 0.0
    return cov / (std_p * std_h)


def evaluate_sts(predicted_sim: float, human_score: float) -> Dict[str, Any]:
    """STS 개별 쌍 평가 결과."""
    return {
        "score": predicted_sim,
        "predicted_sim": round(predicted_sim, 4),
        "human_score": human_score,
        "error": round(abs(predicted_sim - human_score), 4),
    }


def evaluate_retrieval(
    query_embedding: List[float],
    candidate_embeddings: List[List[float]],
    correct_idx: int,
) -> Dict[str, Any]:
    """검색 평가: 쿼리-후보 유사도 기반 랭킹, Recall@K, MRR 계산."""
    sims = [cosine_similarity(query_embedding, c) for c in candidate_embeddings]

    ranked_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
    correct_rank = ranked_indices.index(correct_idx) + 1  # 1-based

    recall_at_1 = 1.0 if correct_rank <= 1 else 0.0
    recall_at_3 = 1.0 if correct_rank <= 3 else 0.0
    mrr = 1.0 / correct_rank

    return {
        "score": recall_at_1,
        "ranking": ranked_indices,
        "correct_rank": correct_rank,
        "recall_at_1": recall_at_1,
        "recall_at_3": recall_at_3,
        "mrr": round(mrr, 4),
        "similarities": [round(s, 4) for s in sims],
    }
