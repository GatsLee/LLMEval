from rouge_score import rouge_scorer

_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)


def evaluate(response: str, reference: str, **kwargs) -> dict:
    """ROUGE-1/2/L 평균 점수 반환."""
    if not response.strip() or not reference.strip():
        return {"score": 0.0, "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    scores = _scorer.score(reference, response)
    avg = (
        scores["rouge1"].fmeasure
        + scores["rouge2"].fmeasure
        + scores["rougeL"].fmeasure
    ) / 3
    return {
        "score": round(avg, 3),
        "rouge1": round(scores["rouge1"].fmeasure, 3),
        "rouge2": round(scores["rouge2"].fmeasure, 3),
        "rougeL": round(scores["rougeL"].fmeasure, 3),
    }
