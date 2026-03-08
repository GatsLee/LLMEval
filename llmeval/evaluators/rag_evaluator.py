"""
RAG 평가기: claim-level faithfulness + context relevance + answer relevance.
"""
import json
import re

from ..judge import judge_call, get_default

_DECOMPOSE_PROMPT = """\
다음 답변을 독립적인 사실 주장(claim)으로 분해하세요.
각 claim은 한 문장으로 표현하세요.

답변: {response}

다음 JSON 형식으로만 응답하세요:
{{"claims": ["주장1", "주장2", ...]}}"""

_VERIFY_CLAIM_PROMPT = """\
다음 주장이 주어진 컨텍스트에 의해 뒷받침되는지 판단하세요.

컨텍스트: {context}
주장: {claim}

다음 JSON 형식으로만 응답하세요:
{{"supported": true, "reason": "<한 문장 이유>"}}
또는
{{"supported": false, "reason": "<한 문장 이유>"}}"""

_CONTEXT_RELEVANCE_PROMPT = """\
주어진 컨텍스트가 질문에 답하기에 얼마나 관련이 있는지 평가하세요.

질문: {question}
컨텍스트: {context}

다음 JSON 형식으로만 응답하세요:
{{"score": <1~5 정수>, "reason": "<한 문장 이유>"}}"""

_ANSWER_RELEVANCE_PROMPT = """\
답변이 질문에 얼마나 적절하게 답하고 있는지 평가하세요.
컨텍스트 정확성이 아닌, 질문에 대한 응답 적합성을 평가합니다.

질문: {question}
답변: {response}

다음 JSON 형식으로만 응답하세요:
{{"score": <1~5 정수>, "reason": "<한 문장 이유>"}}"""


def _parse_json(raw: str) -> dict:
    match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def _parse_json_with_list(raw: str) -> dict:
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def _llm_call(prompt: str, judge_model: str = "") -> str:
    return judge_call(prompt)


def decompose_claims(response: str) -> list:
    raw = _llm_call(_DECOMPOSE_PROMPT.format(response=response))
    data = _parse_json_with_list(raw)
    return data.get("claims", [response])


def verify_claims(claims: list, context: str) -> list:
    results = []
    for claim in claims:
        raw = _llm_call(_VERIFY_CLAIM_PROMPT.format(context=context, claim=claim))
        data = _parse_json(raw)
        results.append({
            "claim": claim,
            "supported": data.get("supported", False),
            "reason": data.get("reason", ""),
        })
    return results


def evaluate(
    response: str,
    context: str = "",
    question: str = "",
    reference: str = "",
    **kwargs,
) -> dict:
    # 1. Decompose into claims
    claims = decompose_claims(response)

    # 2. Verify each claim
    claim_results = verify_claims(claims, context)
    grounded = sum(1 for c in claim_results if c["supported"])
    total = len(claim_results)
    faithfulness_score = grounded / total if total > 0 else 0.0

    # 3. Context relevance
    raw_cr = _llm_call(_CONTEXT_RELEVANCE_PROMPT.format(question=question, context=context))
    cr_data = _parse_json(raw_cr)
    context_relevance = float(cr_data.get("score", 3)) / 5.0

    # 4. Answer relevance
    raw_ar = _llm_call(_ANSWER_RELEVANCE_PROMPT.format(question=question, response=response))
    ar_data = _parse_json(raw_ar)
    answer_relevance = float(ar_data.get("score", 3)) / 5.0

    # Composite: weighted average
    composite = (faithfulness_score * 0.5 + context_relevance * 0.25 + answer_relevance * 0.25) * 5.0

    return {
        "score": round(composite, 2),
        "faithfulness_score": round(faithfulness_score, 4),
        "context_relevance": round(context_relevance, 4),
        "answer_relevance": round(answer_relevance, 4),
        "total_claims": total,
        "grounded_claims": grounded,
        "ungrounded_claims": total - grounded,
        "claim_details": claim_results,
        "judge_model": get_default().label(),
    }
