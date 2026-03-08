"""
Multi-turn coherence 평가기.
instruction drift, consistency, context retention 세 축으로 평가.
"""
import json
import re

from ..judge import judge_call, get_default

_COHERENCE_PROMPT = """\
다음은 멀티턴 대화에서 모델의 응답 전체입니다.

시스템 프롬프트: {system_prompt}

대화:
{conversation_text}

다음 세 기준으로 평가하세요:

1. instruction_drift (지시 이탈): 시스템 프롬프트의 지시를 마지막 턴까지 유지하는가? (1~5)
2. consistency (일관성): 이전 턴에서 말한 내용과 모순되지 않는가? (1~5)
3. context_retention (문맥 유지): 이전 턴의 정보를 기억하고 활용하는가? (1~5)

다음 JSON 형식으로만 응답하세요:
{{"instruction_drift": 4, "consistency": 3, "context_retention": 5, "reason": "<한 문장>"}}"""


def evaluate(
    turns: list,
    system_prompt: str = "",
    **kwargs,
) -> dict:
    # Format conversation for judge
    conv_lines = []
    for t in turns:
        role = "사용자" if t["role"] == "user" else "모델"
        conv_lines.append(f"[{role}] {t['content']}")
    conversation_text = "\n".join(conv_lines)

    raw = judge_call(_COHERENCE_PROMPT.format(
        system_prompt=system_prompt or "(없음)",
        conversation_text=conversation_text,
    ))

    match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
    data = {}
    if match:
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            pass

    drift = float(data.get("instruction_drift", 3))
    consistency = float(data.get("consistency", 3))
    retention = float(data.get("context_retention", 3))
    reason = data.get("reason", "")

    composite = (drift + consistency + retention) / 3.0

    return {
        "score": round(composite, 2),
        "instruction_drift": drift,
        "consistency": consistency,
        "context_retention": retention,
        "reason": reason,
        "judge_model": get_default().label(),
        "num_turns": len(turns),
    }
