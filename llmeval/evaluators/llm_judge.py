"""
LLM-as-Judge 평가기.
로컬 Ollama 모델을 judge로 사용 (외부 API 불필요).
같은 입력을 3회 실행 후 중앙값을 최종 점수로 사용.
"""
import json
import re
import statistics

import ollama

JUDGE_MODEL = "llama3.1:8b"

_PROMPTS = {
    "faithfulness": """\
아래 답변이 주어진 컨텍스트에만 근거하는지 평가하세요.
컨텍스트에 없는 내용을 지어냈으면 낮은 점수를 주세요.

컨텍스트: {context}
질문: {question}
답변: {response}

다음 JSON 형식으로만 응답하세요 (다른 텍스트 없이):
{{"score": <1~5 정수>, "reason": "<한 문장 이유>"}}""",

    "fluency": """\
아래 한국어 텍스트의 자연스러움을 평가하세요.
문법 오류, 어색한 표현, 외국어 직역체가 있으면 낮은 점수를 주세요.

텍스트: {response}

다음 JSON 형식으로만 응답하세요:
{{"score": <1~5 정수>, "reason": "<한 문장 이유>"}}""",

    "correctness": """\
아래 답변이 참조 정답과 의미적으로 일치하는지 평가하세요.

참조 정답: {reference}
답변: {response}

다음 JSON 형식으로만 응답하세요:
{{"score": <1~5 정수>, "reason": "<한 문장 이유>"}}""",

    "compliance": """\
아래 답변이 지시사항을 정확히 따랐는지 평가하세요.
(형식, 길이, 언어 등)

지시사항: {instruction}
답변: {response}

다음 JSON 형식으로만 응답하세요:
{{"score": <1~5 정수>, "reason": "<한 문장 이유>"}}""",
}


def _parse_score(raw: str) -> float:
    raw = raw.strip()
    # JSON 블록 추출
    match = re.search(r'\{.*?\}', raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return float(data.get("score", 3))
        except (json.JSONDecodeError, ValueError):
            pass
    # 숫자만 추출 (fallback)
    nums = re.findall(r'\b([1-5])\b', raw)
    if nums:
        return float(nums[0])
    return 3.0


def evaluate(
    response: str,
    criteria: str,
    context: str = "",
    question: str = "",
    reference: str = "",
    instruction: str = "",
    judge_model: str = JUDGE_MODEL,
    n_trials: int = 3,
    **kwargs,
) -> dict:
    if criteria not in _PROMPTS:
        raise ValueError(f"Unknown criteria: {criteria}. Available: {list(_PROMPTS)}")

    prompt = _PROMPTS[criteria].format(
        context=context,
        question=question,
        response=response,
        reference=reference,
        instruction=instruction,
    )

    scores = []
    reasons = []
    for _ in range(n_trials):
        try:
            result = ollama.chat(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0},
            )
            raw = result["message"]["content"]
            score = _parse_score(raw)
            scores.append(score)
            match = re.search(r'"reason"\s*:\s*"([^"]+)"', raw)
            if match:
                reasons.append(match.group(1))
        except Exception:
            scores.append(3.0)

    median = statistics.median(scores)
    return {
        "score": median,
        "scores_raw": scores,
        "reason": reasons[0] if reasons else "",
        "criteria": criteria,
        "judge_model": judge_model,
    }
