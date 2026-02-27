def evaluate(response: str, reference: str, **kwargs) -> dict:
    """정답 문자열이 응답에 포함되는지 확인 (대소문자 무시)."""
    matched = reference.lower().strip() in response.lower()
    return {
        "score": 1.0 if matched else 0.0,
        "matched": matched,
        "reference": reference,
    }
