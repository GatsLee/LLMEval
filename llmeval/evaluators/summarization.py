"""
요약 하이브리드 평가기.
ROUGE (lexical overlap) + LLM Judge (semantic quality) 결합.
언어 감지 → ROUGE → LLM Judge 순서로 평가.
비한국어 응답은 fluency=1 강제 + 페널티.
"""
import json
import re
import statistics

from rouge_score import rouge_scorer

from ..judge import judge_call, get_default

_rouge_scorer_inst = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"], use_stemmer=False
)

_JUDGE_PROMPT = """\
당신은 한국어 요약 품질을 평가하는 엄격한 전문 평가자입니다.
반드시 아래 채점 기준을 정확히 적용하세요.

## 원본 텍스트
{source_text}

## 참조 요약 (정답 예시)
{reference}

## 모델 생성 요약 (평가 대상)
{response}

## 채점 기준 (각 1~5점, 엄격하게 평가)

### completeness (완전성)
- 5점: 참조 요약의 핵심 3가지 포인트를 모두 포함
- 4점: 핵심 2가지 포인트 포함, 1가지 누락
- 3점: 핵심 1가지만 포함
- 2점: 핵심 내용 대부분 누락, 부차적 내용만 언급
- 1점: 원문과 무관한 내용

### accuracy (정확성)
- 5점: 사실 오류 없음
- 4점: 사소한 부정확함 1건 (예: 연도, 수치 약간 다름)
- 3점: 의미를 왜곡하는 오류 1건
- 2점: 여러 사실 오류 또는 원문에 없는 내용 추가
- 1점: 전체적으로 부정확하거나 지어낸 내용

### fluency (유창성)
- 5점: 자연스러운 한국어, 문법 오류 없음
- 4점: 사소한 어색함 1건
- 3점: 문법 오류 2-3건 또는 어색한 직역체
- 2점: 한국어와 다른 언어가 섞여 있음
- 1점: 영어/중국어/일본어 등 다른 언어로 응답, 또는 의미 없는 텍스트

### conciseness (간결성)
- 5점: 정확히 3문장 이내, 군더더기 없음
- 4점: 3문장이지만 약간의 불필요한 정보 포함
- 3점: 4-5문장 또는 불필요한 반복
- 2점: 6문장 이상 또는 상당한 장황함
- 1점: 요약이 아닌 전문 재서술

다음 JSON 형식으로만 응답하세요 (다른 텍스트 없이):
{{"completeness": <1-5>, "accuracy": <1-5>, "fluency": <1-5>, "conciseness": <1-5>, "reason": "<한 문장 총평>"}}"""


def _korean_ratio(text: str) -> float:
    """텍스트에서 한국어(Hangul) 문자 비율 계산."""
    if not text.strip():
        return 0.0
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    korean = sum(1 for c in chars
                 if '\uAC00' <= c <= '\uD7A3'      # Hangul syllables
                 or '\u3131' <= c <= '\u318E'       # Hangul Jamo
                 or '\u1100' <= c <= '\u11FF')      # Hangul Jamo extended
    return korean / len(chars)


def _parse_judge(raw: str) -> dict:
    """Judge 응답에서 JSON 파싱."""
    match = re.search(r'\{.*?\}', raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            dims = {}
            for k in ("completeness", "accuracy", "fluency", "conciseness"):
                v = data.get(k, 3)
                dims[k] = max(1, min(5, int(v)))
            dims["reason"] = data.get("reason", "")
            return dims
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return {"completeness": 3, "accuracy": 3, "fluency": 3, "conciseness": 3, "reason": ""}


def evaluate(
    response: str,
    reference: str,
    source_text: str = "",
    n_trials: int = 3,
    **kwargs,
) -> dict:
    """ROUGE + LLM Judge 하이브리드 요약 평가."""
    # ── 언어 감지 ──
    kr_ratio = _korean_ratio(response)
    lang_penalty = kr_ratio < 0.3  # 한국어 비율 30% 미만이면 페널티

    # ── ROUGE (보충 지표) ──
    if response.strip() and reference.strip():
        rouge = _rouge_scorer_inst.score(reference, response)
        rouge_detail = {
            "rouge1": round(rouge["rouge1"].fmeasure, 3),
            "rouge2": round(rouge["rouge2"].fmeasure, 3),
            "rougeL": round(rouge["rougeL"].fmeasure, 3),
        }
    else:
        rouge_detail = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    # ── LLM Judge (3회 → 중앙값) ──
    prompt = _JUDGE_PROMPT.format(
        source_text=source_text,
        reference=reference,
        response=response,
    )

    all_scores = []
    all_dims = []
    reasons = []

    for _ in range(n_trials):
        try:
            raw = judge_call(prompt)
            dims = _parse_judge(raw)

            # 언어 페널티 강제 적용 (judge가 놓치는 경우 보정)
            if lang_penalty:
                dims["fluency"] = 1
                if not dims["reason"]:
                    dims["reason"] = "비한국어 응답 감지 (한국어 비율 {:.0%})".format(kr_ratio)

            avg = (dims["completeness"] + dims["accuracy"]
                   + dims["fluency"] + dims["conciseness"]) / 4
            all_scores.append(avg)
            all_dims.append(dims)
            if dims["reason"]:
                reasons.append(dims["reason"])
        except Exception:
            all_scores.append(3.0)
            all_dims.append({
                "completeness": 3, "accuracy": 3,
                "fluency": 3, "conciseness": 3, "reason": "",
            })

    # 중앙값 trial 선택
    median_score = statistics.median(all_scores)
    median_idx = all_scores.index(
        min(all_scores, key=lambda x: abs(x - median_score))
    )
    best_dims = all_dims[median_idx]

    return {
        "score": round(median_score, 3),
        "completeness": best_dims["completeness"],
        "accuracy": best_dims["accuracy"],
        "fluency": best_dims["fluency"],
        "conciseness": best_dims["conciseness"],
        "korean_ratio": round(kr_ratio, 3),
        "reason": best_dims.get("reason", reasons[0] if reasons else ""),
        "scores_raw": [round(s, 3) for s in all_scores],
        "judge_model": get_default().label(),
        **rouge_detail,
    }
