"""
Runner: YAML 태스크 로드 → Ollama 추론 + HW 프로파일링 동시 실행 → 평가 → 저장.
"""
import time
import uuid
from pathlib import Path
from typing import List

import ollama
import yaml
from jinja2 import Template

from .evaluators import exact_match, rouge, llm_judge
from .models import InferenceResult, HWSample, RunConfig
from .profiler import HWProfiler
from .store import init_db, save_run, save_result, save_hw_samples, update_hw_summary


# ── 프롬프트 빌더 ─────────────────────────────────────────────────────────────

_PROMPT_TEMPLATES = {
    "qa_with_context": """\
다음 컨텍스트를 바탕으로 질문에 답하세요. 컨텍스트에 없는 내용은 답하지 마세요.

컨텍스트: {{ context }}

질문: {{ question }}

답변:""",

    "summarization": """\
다음 텍스트를 3문장 이내로 핵심만 요약하세요.

텍스트: {{ text }}

요약:""",

    "structured_output": """\
다음 요청을 처리하고 JSON 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요.

요청: {{ request }}

JSON:""",

    "instruction_following": "{{ instruction }}",
}


def _build_prompt(task_type: str, item: dict) -> str:
    tmpl = _PROMPT_TEMPLATES.get(task_type, "{{ prompt }}")
    return Template(tmpl).render(**item)


# ── 추론 실행 ─────────────────────────────────────────────────────────────────

def _run_inference(model: str, prompt: str) -> tuple:
    """Returns: (response, tps, ttft_ms, total_ms, token_count)"""
    start = time.perf_counter()
    first_token_ts = None
    chunks = []
    eval_count = 0

    stream = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    for chunk in stream:
        content = chunk["message"]["content"]
        if content:
            if first_token_ts is None:
                first_token_ts = time.perf_counter()
            chunks.append(content)
        if chunk.get("done"):
            eval_count = chunk.get("eval_count", len(chunks))

    total_time = time.perf_counter() - start
    response = "".join(chunks)
    token_count = eval_count if eval_count else len(chunks)
    tps = token_count / total_time if total_time > 0 else 0
    ttft_ms = ((first_token_ts or start) - start) * 1000
    total_ms = total_time * 1000

    return response, tps, ttft_ms, total_ms, token_count


# ── 평가기 디스패치 ───────────────────────────────────────────────────────────

def _evaluate(response: str, task: dict, item: dict) -> tuple[float, dict]:
    evaluator = task.get("evaluator", "none")
    detail = {}

    if evaluator == "exact_match":
        detail = exact_match.evaluate(response, item.get("reference", ""))
    elif evaluator == "rouge":
        detail = rouge.evaluate(response, item.get("reference", ""))
    elif evaluator == "llm_judge":
        detail = llm_judge.evaluate(
            response=response,
            criteria=task.get("judge_criteria", "faithfulness"),
            context=item.get("context", ""),
            question=item.get("question", ""),
            reference=item.get("reference", ""),
            instruction=item.get("instruction", ""),
        )
    else:
        return 0.0, {}

    return float(detail.get("score", 0.0)), detail


# ── 메인 실험 실행 ────────────────────────────────────────────────────────────

def run_experiment(
    task_path: str,
    models: List[str],
    description: str = "",
    console=None,
) -> str:
    """실험 실행 후 run_id 반환."""
    init_db()

    with open(task_path) as f:
        task = yaml.safe_load(f)

    run_id = str(uuid.uuid4())[:8]
    config = RunConfig(
        run_id=run_id,
        description=description or task["name"],
        task_name=task["name"],
        task_type=task["type"],
        models=models,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    save_run(config)

    inputs = task.get("inputs", [])

    for model in models:
        if console:
            console.print(f"\n  [bold cyan]모델:[/] {model}")

        profiler = HWProfiler(sample_interval_ms=50)
        profiler.start()

        tps_timeline = []

        for idx, item in enumerate(inputs):
            prompt = _build_prompt(task["type"], item)

            if console:
                console.print(f"    입력 {idx + 1}/{len(inputs)} ...", end=" ")

            response, tps, ttft_ms, total_ms, token_count = _run_inference(model, prompt)
            tps_timeline.append(tps)

            score, score_detail = _evaluate(response, task, item)

            if console:
                console.print(f"score={score:.2f} | {tps:.1f} t/s")

            result = InferenceResult(
                model=model,
                input_idx=idx,
                prompt=prompt,
                response=response,
                tps=tps,
                ttft_ms=ttft_ms,
                total_ms=total_ms,
                token_count=token_count,
                score=score,
                score_detail=score_detail,
            )

            # hw_summary는 모델 완료 후 채울 예정
            save_result(run_id, result)

        hw_summary_dict = profiler.stop()

        # 쓰로틀링 감지: 전반 vs 후반 t/s
        if len(tps_timeline) >= 4:
            split = max(1, len(tps_timeline) // 5)
            early_avg = sum(tps_timeline[:split]) / split
            late_avg = sum(tps_timeline[-split:]) / split
            if early_avg > 0 and late_avg < early_avg * 0.85:
                hw_summary_dict["throttle_detected"] = True

        update_hw_summary(run_id, model, hw_summary_dict)

        # hw_samples 저장
        raw = profiler.get_raw_samples()
        hw_samples = [
            HWSample(
                ts_ms=s["ts_ms"],
                vram_mb=s["vram_mb"],
                gpu_util=s["gpu_util"],
                gpu_temp=s["gpu_temp"],
                gpu_power_w=s["gpu_power_w"],
                ram_mb=s["ram_mb"],
                cpu_util=s["cpu_util"],
            )
            for s in raw
        ]
        save_hw_samples(run_id, model, hw_samples)

    return run_id
