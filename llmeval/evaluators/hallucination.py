"""
Hallucination Detection 평가기.
answer를 atomic claims로 분해 후, known_facts/context 대비 검증.
"""
import json
import re

from ..judge import judge_call, get_default

_DECOMPOSE_PROMPT = """\
다음 텍스트에서 사실적 주장(factual claim)을 모두 추출하세요.
의견이나 주관적 표현은 제외하고, 검증 가능한 사실만 추출하세요.

텍스트: {response}

다음 JSON 형식으로만 응답하세요:
{{"claims": ["사실1", "사실2", ...]}}"""

_VERIFY_PROMPT = """\
다음 주장이 주어진 사실(known facts)에 의해 뒷받침되는지,
모순되는지, 판단할 수 없는지 분류하세요.

알려진 사실:
{known_facts}

주장: {claim}

다음 JSON 형식으로만 응답하세요 (verdict는 반드시 "supported", "contradicted", "unverifiable" 중 하나):
{{"verdict": "supported", "reason": "<한 문장>"}}"""


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


def evaluate(
    response: str,
    context: str = "",
    known_facts: str = "",
    question: str = "",
    **kwargs,
) -> dict:
    facts = known_facts or context

    # Step 1: Decompose
    raw = judge_call(_DECOMPOSE_PROMPT.format(response=response))
    data = _parse_json_with_list(raw)
    claims = data.get("claims", [response])

    # Step 2: Verify each claim
    claim_results = []
    for claim in claims:
        raw = judge_call(_VERIFY_PROMPT.format(known_facts=facts, claim=claim))
        vdata = _parse_json(raw)
        verdict = vdata.get("verdict", "unverifiable")
        if verdict not in ("supported", "contradicted", "unverifiable"):
            verdict = "unverifiable"
        claim_results.append({
            "claim": claim,
            "verdict": verdict,
            "reason": vdata.get("reason", ""),
        })

    total = len(claim_results)
    supported = sum(1 for c in claim_results if c["verdict"] == "supported")
    contradicted = sum(1 for c in claim_results if c["verdict"] == "contradicted")
    unverifiable = sum(1 for c in claim_results if c["verdict"] == "unverifiable")

    hallucination_rate = (contradicted + unverifiable) / total if total > 0 else 0.0
    score = (1.0 - hallucination_rate) * 5.0

    return {
        "score": round(score, 2),
        "hallucination_rate": round(hallucination_rate, 4),
        "total_claims": total,
        "supported": supported,
        "contradicted": contradicted,
        "unverifiable": unverifiable,
        "claim_details": claim_results,
        "judge_model": get_default().label(),
    }
