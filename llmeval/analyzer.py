"""
Auto Analyzer: 실험 결과를 로컬 LLM으로 한국어 자동 분석 텍스트 생성.
"""
import ollama

from .store import get_run, get_run_results

ANALYST_MODEL = "llama3.1:8b"


def _build_comparison_table(run: dict, results: list) -> str:
    """결과를 분석용 텍스트 테이블로 변환."""
    models = list(dict.fromkeys(r["model"] for r in results))
    lines = []
    lines.append(f"태스크: {run['task_name']}")
    lines.append(f"설명: {run['description']}")
    lines.append(f"날짜: {run['created_at']}")
    lines.append("")
    lines.append("| 모델 | Avg Score | Avg t/s | TTFT(ms) | VRAM 피크(MB) | GPU 온도(°C) | 전력(W) | 스필오버 | 쓰로틀링 |")
    lines.append("|------|-----------|---------|----------|---------------|-------------|---------|---------|---------|")

    for model in models:
        rows = [r for r in results if r["model"] == model]
        scores = [r["score"] for r in rows if r["score"] is not None]
        tps_vals = [r["tps"] for r in rows if r["tps"]]
        ttft_vals = [r["ttft_ms"] for r in rows if r["ttft_ms"]]
        hw = rows[0].get("hw_summary") or {} if rows else {}

        avg_score = sum(scores) / len(scores) if scores else 0
        avg_tps = sum(tps_vals) / len(tps_vals) if tps_vals else 0
        avg_ttft = sum(ttft_vals) / len(ttft_vals) if ttft_vals else 0

        lines.append(
            f"| {model} | {avg_score:.2f} | {avg_tps:.1f} | {avg_ttft:.0f} "
            f"| {hw.get('vram_peak_mb', 0):,} | {hw.get('gpu_temp_peak_c', 0)} "
            f"| {hw.get('gpu_power_avg_w', 0):.1f} "
            f"| {'Yes' if hw.get('spillover_detected') else 'No'} "
            f"| {'Yes' if hw.get('throttle_detected') else 'No'} |"
        )

    return "\n".join(lines)


def analyze(run_id: str, analyst_model: str = ANALYST_MODEL) -> str:
    """실험 결과를 분석하여 한국어 텍스트를 반환."""
    run = get_run(run_id)
    if not run:
        return f"run_id '{run_id}' 없음"

    results = get_run_results(run_id)
    if not results:
        return "결과 없음"

    table = _build_comparison_table(run, results)

    prompt = f"""\
다음은 로컬 LLM 평가 실험 결과입니다.
타겟 하드웨어: RTX 4070 Super (12GB VRAM), DDR5 64GB, i5-14600K.

{table}

다음 항목을 한국어로 분석하세요:
1. 품질/속도/VRAM 트레이드오프 요약
2. 이 하드웨어에서 최적 모델 권장 (용도별)
3. 이상 징후 (스필오버/쓰로틀링) 해석
4. 다음 실험 제안

간결하게, 수치를 근거로 작성하세요."""

    result = ollama.chat(
        model=analyst_model,
        messages=[{"role": "user", "content": prompt}],
    )
    return result["message"]["content"]
