"""
Runner: YAML 태스크 로드 → Ollama 추론 + HW 프로파일링 동시 실행 → 평가 → 저장.
"""
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any

import ollama
import yaml
from jinja2 import Template

from .evaluators import exact_match, rouge, llm_judge, embedding_metrics
from .evaluators import rag_evaluator, hallucination
from .evaluators import structured_output as structured_output_eval
from .evaluators import multi_turn, safety, summarization
from .models import InferenceResult, HWSample, RunConfig
from .profiler import HWProfiler
from .judge import JudgeConfig, set_default, check_backend_available
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

    "safety": "{{ prompt }}",

    "hallucination_detection": """\
다음 컨텍스트와 알려진 사실을 바탕으로 질문에 답하세요.

컨텍스트: {{ context }}
알려진 사실: {{ known_facts }}

질문: {{ question }}

답변:""",
}


def _build_prompt(task_type: str, item: dict) -> str:
    tmpl = _PROMPT_TEMPLATES.get(task_type, "{{ prompt }}")
    return Template(tmpl).render(**item)


# ── 추론 실행 ─────────────────────────────────────────────────────────────────

def _run_inference(
    model: str,
    prompt: str,
    options: Optional[Dict[str, Any]] = None,
    messages_override: Optional[List[Dict]] = None,
) -> tuple:
    """Returns: (response, tps, ttft_ms, total_ms, token_count)"""
    start = time.perf_counter()
    first_token_ts = None
    chunks = []
    eval_count = 0

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages_override or [{"role": "user", "content": prompt}],
        "stream": True,
    }
    if options:
        kwargs["options"] = options

    stream = ollama.chat(**kwargs)

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


# ── 임베딩 실행 ──────────────────────────────────────────────────────────────

def _run_embedding(model: str, texts: List[str]) -> tuple:
    """Returns: (embeddings, latency_ms, embeds_per_sec)"""
    start = time.perf_counter()
    result = ollama.embed(model=model, input=texts)
    elapsed = time.perf_counter() - start
    embeddings = result["embeddings"]
    latency_ms = elapsed * 1000
    eps = len(texts) / elapsed if elapsed > 0 else 0
    return embeddings, latency_ms, eps


def _run_embedding_experiment(
    task: dict,
    task_type: str,
    models: List[str],
    run_id: str,
    inputs: list,
    console=None,
) -> None:
    """임베딩 모델 실험 (STS 또는 Retrieval)."""
    import json as _json

    for model in models:
        if console:
            console.print(f"\n  [bold cyan]모델:[/] {model}")

        profiler = HWProfiler(sample_interval_ms=50)
        profiler.start()

        for idx, item in enumerate(inputs):
            if console:
                console.print(f"    입력 {idx + 1}/{len(inputs)} ...", end=" ")

            if task_type == "embedding_sts":
                texts = [item["text_a"], item["text_b"]]
                embeddings, latency_ms, eps = _run_embedding(model, texts)
                sim = embedding_metrics.cosine_similarity(embeddings[0], embeddings[1])
                detail = embedding_metrics.evaluate_sts(sim, item["human_score"])
                detail["dimensions"] = len(embeddings[0])
                score = sim
                prompt_str = _json.dumps({"text_a": item["text_a"], "text_b": item["text_b"]}, ensure_ascii=False)
                response_str = f"cosine_sim={sim:.4f}"

            elif task_type == "embedding_retrieval":
                all_texts = [item["query"]] + item["candidates"]
                embeddings, latency_ms, eps = _run_embedding(model, all_texts)
                detail = embedding_metrics.evaluate_retrieval(
                    embeddings[0], embeddings[1:], item["correct_idx"]
                )
                detail["dimensions"] = len(embeddings[0])
                score = detail["recall_at_1"]
                prompt_str = _json.dumps({"query": item["query"]}, ensure_ascii=False)
                response_str = f"rank={detail['correct_rank']},recall@1={detail['recall_at_1']}"

            else:
                continue

            if console:
                console.print(f"score={score:.4f} | {eps:.1f} emb/s | {latency_ms:.0f}ms")

            result = InferenceResult(
                model=model,
                input_idx=idx,
                prompt=prompt_str,
                response=response_str,
                tps=eps,
                ttft_ms=latency_ms,
                total_ms=latency_ms,
                token_count=0,
                score=score,
                score_detail=detail,
            )
            save_result(run_id, result)

        hw_summary_dict = profiler.stop()
        update_hw_summary(run_id, model, hw_summary_dict)

        raw = profiler.get_raw_samples()
        hw_samples = [
            HWSample(
                ts_ms=s["ts_ms"], vram_mb=s["vram_mb"], gpu_util=s["gpu_util"],
                gpu_temp=s["gpu_temp"], gpu_power_w=s["gpu_power_w"],
                ram_mb=s["ram_mb"], cpu_util=s["cpu_util"],
            )
            for s in raw
        ]
        save_hw_samples(run_id, model, hw_samples)


# ── 평가기 디스패치 ───────────────────────────────────────────────────────────

def _evaluate(response: str, task: dict, item: dict) -> tuple[float, dict]:
    evaluator = task.get("evaluator", "none")
    detail = {}

    if evaluator == "exact_match":
        detail = exact_match.evaluate(response, item.get("reference", ""))
    elif evaluator == "rouge":
        detail = rouge.evaluate(response, item.get("reference", ""))
    elif evaluator == "summarization":
        detail = summarization.evaluate(
            response=response,
            reference=item.get("reference", ""),
            source_text=item.get("text", ""),
        )
    elif evaluator == "llm_judge":
        detail = llm_judge.evaluate(
            response=response,
            criteria=task.get("judge_criteria", "faithfulness"),
            context=item.get("context", ""),
            question=item.get("question", ""),
            reference=item.get("reference", ""),
            instruction=item.get("instruction", ""),
        )
    elif evaluator == "rag_detailed":
        detail = rag_evaluator.evaluate(
            response=response,
            context=item.get("context", ""),
            question=item.get("question", ""),
            reference=item.get("reference", ""),
        )
    elif evaluator == "hallucination":
        detail = hallucination.evaluate(
            response=response,
            context=item.get("context", ""),
            known_facts=item.get("known_facts", ""),
            question=item.get("question", ""),
        )
    elif evaluator == "structured_output":
        detail = structured_output_eval.evaluate(
            response=response,
            reference=item.get("reference", ""),
            schema=item.get("schema"),
            required_fields=item.get("required_fields"),
        )
    elif evaluator == "safety":
        detail = safety.evaluate(
            response=response,
            prompt=item.get("prompt", ""),
            expected_behavior=item.get("expected_behavior", ""),
        )
    else:
        return 0.0, {}

    return float(detail.get("score", 0.0)), detail


# ── 멀티턴 실험 ──────────────────────────────────────────────────────────────

def _run_multi_turn_experiment(
    task: dict,
    models: List[str],
    run_id: str,
    inputs: list,
    ollama_options: Optional[Dict[str, Any]] = None,
    console=None,
) -> None:
    """멀티턴 대화 실험 (여러 턴 순차 추론 → 일관성 평가)."""
    import json as _json

    for model in models:
        if console:
            console.print(f"\n  [bold cyan]모델:[/] {model}")

        profiler = HWProfiler(sample_interval_ms=50)
        profiler.start()

        for idx, item in enumerate(inputs):
            if console:
                console.print(f"    대화 {idx + 1}/{len(inputs)} ...", end=" ")

            system_prompt = item.get("system_prompt", "")
            user_turns = item.get("turns", [])

            messages: List[Dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            all_turns: List[Dict[str, str]] = []
            total_ms_sum = 0.0
            total_tokens = 0

            for user_msg in user_turns:
                messages.append({"role": "user", "content": user_msg})
                all_turns.append({"role": "user", "content": user_msg})

                response, tps, ttft_ms, turn_ms, token_count = _run_inference(
                    model, "", options=ollama_options, messages_override=messages,
                )
                total_ms_sum += turn_ms
                total_tokens += token_count

                messages.append({"role": "assistant", "content": response})
                all_turns.append({"role": "assistant", "content": response})

            eval_result = multi_turn.evaluate(
                turns=all_turns,
                system_prompt=system_prompt,
            )
            score = eval_result.get("score", 0.0)

            if console:
                console.print(f"score={score:.2f} | {len(user_turns)} turns")

            result = InferenceResult(
                model=model,
                input_idx=idx,
                prompt=_json.dumps({"system": system_prompt, "turns": user_turns}, ensure_ascii=False),
                response=_json.dumps(all_turns, ensure_ascii=False),
                tps=total_tokens / (total_ms_sum / 1000) if total_ms_sum > 0 else 0,
                ttft_ms=0,
                total_ms=total_ms_sum,
                token_count=total_tokens,
                score=score,
                score_detail=eval_result,
            )
            save_result(run_id, result)

        hw_summary_dict = profiler.stop()
        update_hw_summary(run_id, model, hw_summary_dict)

        raw = profiler.get_raw_samples()
        hw_samples = [
            HWSample(
                ts_ms=s["ts_ms"], vram_mb=s["vram_mb"], gpu_util=s["gpu_util"],
                gpu_temp=s["gpu_temp"], gpu_power_w=s["gpu_power_w"],
                ram_mb=s["ram_mb"], cpu_util=s["cpu_util"],
            )
            for s in raw
        ]
        save_hw_samples(run_id, model, hw_samples)


# ── 메인 실험 실행 ────────────────────────────────────────────────────────────

def run_experiment(
    task_path: str,
    models: List[str],
    description: str = "",
    console=None,
    cli_options_override: Optional[Dict[str, Any]] = None,
    judge_spec: Optional[str] = None,
) -> str:
    """실험 실행 후 run_id 반환."""
    init_db()

    with open(task_path) as f:
        task = yaml.safe_load(f)

    # Judge 백엔드 설정: CLI > YAML > default
    raw_judge = judge_spec or task.get("judge") or "ollama:llama3.1:8b"
    judge_config = JudgeConfig.parse(raw_judge)
    ok, msg = check_backend_available(judge_config)
    if not ok:
        if console:
            console.print(f"[bold red]Judge 오류:[/] {msg}")
        raise SystemExit(1)
    set_default(judge_config)
    if console and judge_config.backend != "ollama":
        console.print(f"  [bold]Judge:[/] {judge_config.label()}")

    # Ollama options: YAML 태스크 설정 + CLI 오버라이드 병합
    ollama_options = task.get("ollama_options", None)
    if cli_options_override:
        if ollama_options is None:
            ollama_options = {}
        ollama_options.update(cli_options_override)

    run_id = str(uuid.uuid4())[:8]
    config = RunConfig(
        run_id=run_id,
        description=description or task["name"],
        task_name=task["name"],
        task_type=task["type"],
        models=models,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        ollama_options=ollama_options,
        judge=judge_config.label(),
    )
    save_run(config)

    inputs = task.get("inputs", [])

    # 멀티턴 대화 태스크
    if task["type"] == "multi_turn":
        _run_multi_turn_experiment(task, models, run_id, inputs, ollama_options, console)
        return run_id

    # 임베딩 태스크는 별도 실험 루프로 처리
    if task["type"] in ("embedding_sts", "embedding_retrieval"):
        _run_embedding_experiment(task, task["type"], models, run_id, inputs, console)
        return run_id

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

            response, tps, ttft_ms, total_ms, token_count = _run_inference(model, prompt, options=ollama_options)
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
