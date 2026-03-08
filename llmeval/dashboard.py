"""Web dashboard: FastAPI + Jinja2 + HTMX."""
import asyncio
import io
import json
import subprocess
from pathlib import Path
from typing import Optional

import yaml

import httpx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .store import (
    init_db, get_all_runs, get_run, get_run_results,
    get_hw_samples, get_leaderboard, get_embedding_summary,
    get_all_models, get_model_profile, get_score_matrix,
    get_available_judges,
)

BASE = Path(__file__).parent.parent
app = FastAPI(title="LLMEval Dashboard")
templates = Jinja2Templates(directory=str(BASE / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")

_chart_cache: dict[str, bytes] = {}


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────

def _model_stats(results: list) -> dict:
    """결과 리스트에서 모델별 집계."""
    models = list(dict.fromkeys(r["model"] for r in results))
    stats = {}
    for model in models:
        rows = [r for r in results if r["model"] == model]
        scores = [r["score"] for r in rows if r["score"] is not None]
        tps_vals = [r["tps"] for r in rows if r["tps"]]
        ttft_vals = [r["ttft_ms"] for r in rows if r["ttft_ms"]]
        hw = rows[0].get("hw_summary") or {} if rows else {}
        stats[model] = {
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "n": len(rows),
            "avg_tps": sum(tps_vals) / len(tps_vals) if tps_vals else 0,
            "avg_ttft_ms": sum(ttft_vals) / len(ttft_vals) if ttft_vals else 0,
            "vram_peak_mb": hw.get("vram_peak_mb", 0),
            "vram_budget_mb": hw.get("vram_budget_mb", 12282),
            "gpu_util_avg": hw.get("gpu_util_avg_pct", 0),
            "gpu_temp_peak": hw.get("gpu_temp_peak_c", 0),
            "gpu_power_avg": hw.get("gpu_power_avg_w", 0),
            "ram_peak_mb": hw.get("ram_peak_mb", 0),
            "spillover": hw.get("spillover_detected", False),
            "throttle": hw.get("throttle_detected", False),
        }
    return stats


def _score_class(score: float) -> str:
    if score >= 4.0:
        return "good"
    if score >= 2.5:
        return "warn"
    return "bad"


# ── 태스크 카탈로그 ──────────────────────────────────────────────────────────

_EVALUATOR_METRICS = {
    "exact_match": ["score (0 or 1)", "matched", "reference"],
    "rouge": ["score (avg F1)", "rouge1", "rouge2", "rougeL"],
    "llm_judge": ["score (1-5)", "reason", "judge_model"],
    "rag_detailed": [
        "score (composite 0-5)", "faithfulness_score", "context_relevance",
        "answer_relevance", "total_claims", "grounded_claims",
    ],
    "hallucination": [
        "score (0-5)", "hallucination_rate", "total_claims",
        "supported", "contradicted", "unverifiable",
    ],
    "structured_output": [
        "score (0-5)", "parse_success", "schema_valid",
        "field_accuracy", "reference_match",
    ],
    "multi_turn": [
        "score (avg 1-5)", "instruction_drift", "consistency",
        "context_retention", "num_turns",
    ],
    "safety": [
        "score (1-5)", "harmfulness", "refusal",
        "info_safety", "reason",
    ],
    "summarization": [
        "score (1-5 judge avg)", "completeness", "accuracy",
        "fluency", "conciseness", "rouge1", "rouge2", "rougeL", "reason",
    ],
    "embedding_sts": [
        "score (cosine_sim)", "Spearman rho", "human_score",
        "error", "dimensions",
    ],
    "embedding_retrieval": [
        "score (recall@1)", "recall@3", "MRR",
        "correct_rank", "dimensions",
    ],
}

_EMBEDDING_NAME_PATTERNS = {"embed", "nomic", "mxbai", "bge-", "bge_", "e5-", "e5_", "gte-", "gte_", "minilm"}


def _classify_task_category(task_data: dict, filename: str) -> str:
    task_type = task_data.get("type", "")
    if task_type in ("embedding_sts", "embedding_retrieval"):
        return "Embedding"
    if task_type == "multi_turn":
        return "Multi-turn"
    if task_type == "hallucination_detection":
        return "QA/Reasoning"
    if task_type == "summarization":
        return "Korean"
    if task_type == "safety":
        return "Safety"
    if task_type in ("instruction_following", "structured_output"):
        return "Instruction"

    fname = filename.replace(".yaml", "")
    if "korean" in fname:
        return "Korean"
    if "code" in fname:
        return "Code"
    if any(k in fname for k in ("rag", "multihop", "context_length", "gpu_offload", "reasoning", "hallucination")):
        return "QA/Reasoning"
    if any(k in fname for k in ("thermal", "quantization")):
        return "Hardware"
    return "Other"


def _load_task_catalog() -> list:
    tasks_dir = BASE / "tasks"
    catalog = []
    for yaml_path in sorted(tasks_dir.glob("*.yaml")):
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        evaluator = data.get("evaluator", "none")
        inputs = data.get("inputs", [])
        catalog.append({
            "filename": yaml_path.name,
            "name": data.get("name", yaml_path.stem),
            "description": data.get("description", "").strip(),
            "type": data.get("type", ""),
            "evaluator": evaluator,
            "judge_criteria": data.get("judge_criteria", ""),
            "category": _classify_task_category(data, yaml_path.name),
            "metrics": _EVALUATOR_METRICS.get(evaluator, []),
            "input_fields": list(inputs[0].keys()) if inputs else [],
            "input_count": len(inputs),
        })
    return catalog


def _is_embedding_model(model_data: dict) -> bool:
    details = model_data.get("details", {})
    families = details.get("families", []) or []
    if "bert" in families:
        return True
    name_lower = model_data.get("name", "").lower()
    return any(p in name_lower for p in _EMBEDDING_NAME_PATTERNS)


# ── 차트 생성 ────────────────────────────────────────────────────────────────

def _build_chart(run_id: str, chart_name: str) -> Optional[bytes]:
    cache_key = f"{run_id}_{chart_name}"
    if cache_key in _chart_cache:
        return _chart_cache[cache_key]

    results = get_run_results(run_id)
    if not results:
        return None

    stats = _model_stats(results)
    models = list(stats.keys())

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="#e0e0e0")
    ax.xaxis.label.set_color("#e0e0e0")
    ax.yaxis.label.set_color("#e0e0e0")
    ax.title.set_color("#e0e0e0")
    for spine in ax.spines.values():
        spine.set_color("#333")

    colors = ["#4fc3f7", "#81c784", "#ffb74d", "#e57373", "#ba68c8"]

    if chart_name == "scores":
        vals = [stats[m]["avg_score"] for m in models]
        bars = ax.bar(models, vals, color=colors[:len(models)])
        ax.set_ylabel("Avg Score")
        ax.set_title("Quality Scores")
        ax.set_ylim(0, 5.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.1, f"{v:.2f}",
                    ha="center", color="#e0e0e0", fontsize=10)

    elif chart_name == "speed":
        vals = [stats[m]["avg_tps"] for m in models]
        bars = ax.bar(models, vals, color=colors[:len(models)])
        ax.set_ylabel("Tokens/sec (or Embed/sec)")
        ax.set_title("Inference Speed")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f"{v:.1f}",
                    ha="center", color="#e0e0e0", fontsize=10)

    elif chart_name == "vram":
        vals = [stats[m]["vram_peak_mb"] for m in models]
        budget = stats[models[0]]["vram_budget_mb"] if models else 12282
        bars = ax.bar(models, vals, color=colors[:len(models)])
        ax.axhline(y=budget, color="#e57373", linestyle="--", label=f"Budget {budget}MB")
        ax.set_ylabel("VRAM (MB)")
        ax.set_title("VRAM Peak Usage")
        ax.legend(facecolor="#16213e", edgecolor="#333", labelcolor="#e0e0e0")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 50, f"{v:,}",
                    ha="center", color="#e0e0e0", fontsize=9)

    elif chart_name == "pareto_vram":
        scores_vals = [stats[m]["avg_score"] for m in models]
        vram_vals = [stats[m]["vram_peak_mb"] for m in models]
        ax.scatter(vram_vals, scores_vals, c=colors[:len(models)], s=100, zorder=5)
        for i, m in enumerate(models):
            ax.annotate(m, (vram_vals[i], scores_vals[i]), textcoords="offset points",
                        xytext=(6, 4), fontsize=9, color="#e0e0e0")
        points = sorted(zip(vram_vals, scores_vals))
        pareto_x, pareto_y = [], []
        max_score = -1
        for x, y in points:
            if y > max_score:
                pareto_x.append(x)
                pareto_y.append(y)
                max_score = y
        if len(pareto_x) >= 2:
            ax.plot(pareto_x, pareto_y, 'r--', linewidth=1, alpha=0.7, label="Pareto Frontier")
            ax.legend(facecolor="#16213e", edgecolor="#333", labelcolor="#e0e0e0")
        ax.set_xlabel("VRAM Peak (MB)")
        ax.set_ylabel("Avg Score")
        ax.set_title("Quality vs VRAM (Pareto)")

    elif chart_name == "pareto_speed":
        scores_vals = [stats[m]["avg_score"] for m in models]
        tps_vals = [stats[m]["avg_tps"] for m in models]
        ax.scatter(tps_vals, scores_vals, c=colors[:len(models)], s=100, zorder=5)
        for i, m in enumerate(models):
            ax.annotate(m, (tps_vals[i], scores_vals[i]), textcoords="offset points",
                        xytext=(6, 4), fontsize=9, color="#e0e0e0")
        ax.set_xlabel("Avg Tokens/sec")
        ax.set_ylabel("Avg Score")
        ax.set_title("Quality vs Speed")

    else:
        plt.close(fig)
        return None

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    data = buf.getvalue()
    _chart_cache[cache_key] = data
    return data


# ── 라우트 ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    init_db()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    runs = get_all_runs()
    total_models = len(set(m for r in runs for m in r.get("models", [])))
    return templates.TemplateResponse("index.html", {
        "request": request,
        "runs": runs,
        "total_runs": len(runs),
        "total_models": total_models,
    })


@app.get("/run/{run_id}", response_class=HTMLResponse)
async def run_detail(request: Request, run_id: str):
    run = get_run(run_id)
    if not run:
        return HTMLResponse("<h1>Run not found</h1>", status_code=404)
    return templates.TemplateResponse("run_detail.html", {
        "request": request,
        "run": run,
    })


@app.get("/api/run/{run_id}/quality", response_class=HTMLResponse)
async def quality_partial(request: Request, run_id: str):
    run = get_run(run_id)
    results = get_run_results(run_id)
    task_type = run.get("task_type", "")

    if task_type == "embedding_sts":
        summary = get_embedding_summary(run_id)
        return templates.TemplateResponse("_partials/embedding_sts.html", {
            "request": request, "run": run, "summary": summary,
        })
    elif task_type == "embedding_retrieval":
        summary = get_embedding_summary(run_id)
        return templates.TemplateResponse("_partials/embedding_retrieval.html", {
            "request": request, "run": run, "summary": summary,
        })
    else:
        stats = _model_stats(results)
        return templates.TemplateResponse("_partials/quality_table.html", {
            "request": request, "stats": stats, "score_class": _score_class,
        })


@app.get("/api/run/{run_id}/details", response_class=HTMLResponse)
async def details_partial(request: Request, run_id: str):
    run = get_run(run_id)
    results = get_run_results(run_id)
    models = list(dict.fromkeys(r["model"] for r in results))
    judge = run.get("judge", "") if run else ""
    return templates.TemplateResponse("_partials/detail_results.html", {
        "request": request,
        "results": results,
        "models": models,
        "judge": judge,
    })


@app.get("/api/run/{run_id}/hardware", response_class=HTMLResponse)
async def hw_partial(request: Request, run_id: str):
    results = get_run_results(run_id)
    stats = _model_stats(results)
    return templates.TemplateResponse("_partials/hw_table.html", {
        "request": request, "stats": stats,
    })


@app.get("/api/run/{run_id}/charts", response_class=HTMLResponse)
async def charts_partial(request: Request, run_id: str):
    results = get_run_results(run_id)
    models = list(dict.fromkeys(r["model"] for r in results))
    return templates.TemplateResponse("_partials/charts.html", {
        "request": request, "run_id": run_id, "models": models,
    })


@app.get("/api/run/{run_id}/chart/{chart_name}.png")
async def chart_png(run_id: str, chart_name: str):
    data = _build_chart(run_id, chart_name)
    if not data:
        return Response(status_code=404)
    return Response(content=data, media_type="image/png")


@app.get("/api/run/{run_id}/hw_timeline_data")
async def hw_timeline_data(run_id: str):
    results = get_run_results(run_id)
    models = list(dict.fromkeys(r["model"] for r in results))
    data = {}
    for model in models:
        samples = get_hw_samples(run_id, model)
        if samples:
            t0 = samples[0]["ts_ms"]
            data[model] = {
                "ts": [(s["ts_ms"] - t0) / 1000 for s in samples],
                "vram": [s["vram_mb"] for s in samples],
                "gpu_util": [s["gpu_util"] for s in samples],
                "temp": [s["gpu_temp"] for s in samples],
                "power": [s["gpu_power_w"] for s in samples],
                "ram": [s["ram_mb"] for s in samples],
            }
    return JSONResponse(data)


@app.get("/leaderboard", response_class=HTMLResponse)
async def leaderboard_page(request: Request):
    runs = get_all_runs()
    tasks = sorted(set(r["task_name"] for r in runs))
    rows = get_leaderboard()
    return templates.TemplateResponse("leaderboard.html", {
        "request": request, "rows": rows, "tasks": tasks, "selected_task": "",
    })


@app.get("/api/leaderboard", response_class=HTMLResponse)
async def leaderboard_partial(request: Request, task: str = ""):
    rows = get_leaderboard(task or None)
    return templates.TemplateResponse("_partials/leaderboard_body.html", {
        "request": request, "rows": rows,
    })


@app.get("/tasks", response_class=HTMLResponse)
async def tasks_page(request: Request):
    catalog = _load_task_catalog()
    categories: dict[str, list] = {}
    for task in catalog:
        categories.setdefault(task["category"], []).append(task)
    cat_order = ["Embedding", "QA/Reasoning", "Korean", "Code",
                 "Instruction", "Multi-turn", "Safety", "Hardware", "Other"]
    ordered = [(c, categories[c]) for c in cat_order if c in categories]
    return templates.TemplateResponse("tasks.html", {
        "request": request,
        "categories": ordered,
        "total_tasks": len(catalog),
    })


def _task_name_to_category(task_name: str) -> str:
    name = task_name.lower()
    if any(k in name for k in ("embed", "sts", "retrieval", "임베딩")):
        return "Embedding"
    # Hardware must come before QA/Reasoning to catch "내구성" before "추론"
    if any(k in name for k in ("thermal", "quant", "gpu", "offload", "vram",
                                "오프로딩", "열화", "양자화", "내구성")):
        return "Hardware"
    if any(k in name for k in ("rag", "multihop", "reasoning", "hallucin", "context",
                                "충실도", "환각", "추론", "할루시네이션", "컨텍스트")):
        return "QA/Reasoning"
    if any(k in name for k in ("korean", "한국", "요약")):
        return "Korean"
    if any(k in name for k in ("code", "코드")):
        return "Code"
    if any(k in name for k in ("instruct", "structured", "json", "구조화", "지시")):
        return "Instruction"
    if ("multi" in name and "turn" in name) or "멀티턴" in name:
        return "Multi-turn"
    if any(k in name for k in ("safety", "toxicity", "안전")):
        return "Safety"
    return "Other"


@app.get("/matrix", response_class=HTMLResponse)
async def matrix_page(request: Request, judge: str = ""):
    data = get_score_matrix(judge_filter=judge or None)
    models = data["models"]
    tasks = data["tasks"]
    model_stats = data["model_stats"]

    # Available judges for filter
    available_judges = get_available_judges()

    # Separate LLM vs embedding models
    llm_models = [m for m in models if not any(p in m.lower() for p in _EMBEDDING_NAME_PATTERNS)]
    emb_models = [m for m in models if any(p in m.lower() for p in _EMBEDDING_NAME_PATTERNS)]

    # Rank models by score
    models_ranked = sorted(models, key=lambda m: model_stats[m]["avg"], reverse=True)

    # Chart data: bar chart (model ranking)
    chart_colors = ["#4fc3f7", "#81c784", "#ffb74d", "#e57373", "#ba68c8",
                    "#4dd0e1", "#aed581", "#ff8a65", "#f06292", "#7986cb"]
    bar_chart = {
        "labels": [m for m in models_ranked],
        "scores": [model_stats[m]["avg"] for m in models_ranked],
        "ci_low": [model_stats[m]["ci_low"] for m in models_ranked],
        "ci_high": [model_stats[m]["ci_high"] for m in models_ranked],
        "colors": [chart_colors[i % len(chart_colors)] for i in range(len(models_ranked))],
    }

    # Radar chart: category averages for top 5 LLM models
    cat_scores: dict[str, dict[str, list]] = {}
    for model in models:
        cat_scores[model] = {}
        for task in tasks:
            key = f"{model}|{task}"
            if key in data["cells"]:
                cat = _task_name_to_category(task)
                cat_scores[model].setdefault(cat, []).append(data["cells"][key]["avg"])

    all_categories = sorted(set(
        cat for m in cat_scores for cat in cat_scores[m]
    ))

    top_llm = [m for m in models_ranked if m in llm_models][:5]
    radar_chart = {
        "categories": all_categories,
        "datasets": [],
    }
    for i, model in enumerate(top_llm):
        values = []
        for cat in all_categories:
            scores = cat_scores[model].get(cat, [])
            values.append(round(sum(scores) / len(scores), 2) if scores else 0)
        radar_chart["datasets"].append({
            "label": model,
            "data": values,
            "borderColor": chart_colors[i % len(chart_colors)],
            "backgroundColor": chart_colors[i % len(chart_colors)] + "22",
            "pointBackgroundColor": chart_colors[i % len(chart_colors)],
        })

    # Task category mapping for filter
    task_cat_map = {task: _task_name_to_category(task) for task in tasks}
    task_categories = sorted(set(task_cat_map.values()))

    return templates.TemplateResponse("matrix.html", {
        "request": request,
        **data,
        "models_ranked": models_ranked,
        "llm_models": llm_models,
        "emb_models": emb_models,
        "task_cat_map": task_cat_map,
        "task_categories": task_categories,
        "bar_chart_json": json.dumps(bar_chart),
        "radar_chart_json": json.dumps(radar_chart),
        "available_judges": available_judges,
        "selected_judge": judge,
    })


@app.get("/compare", response_class=HTMLResponse)
async def compare_page(request: Request, a: str = "", b: str = ""):
    all_models = get_all_models()
    llm_models = [m for m in all_models if not any(p in m.lower() for p in _EMBEDDING_NAME_PATTERNS)]
    emb_models = [m for m in all_models if any(p in m.lower() for p in _EMBEDDING_NAME_PATTERNS)]

    compare_data = None
    if a and b:
        profile_a = get_model_profile(a)
        profile_b = get_model_profile(b)
        # Build per-task comparison rows
        all_tasks = list(dict.fromkeys(
            [t["task_name"] for t in profile_a["tasks"]] +
            [t["task_name"] for t in profile_b["tasks"]]
        ))
        task_a = {t["task_name"]: t for t in profile_a["tasks"]}
        task_b = {t["task_name"]: t for t in profile_b["tasks"]}
        task_rows = []
        for task_name in all_tasks:
            ta = task_a.get(task_name)
            tb = task_b.get(task_name)
            sa = ta["avg_score"] if ta and ta["avg_score"] is not None else None
            sb = tb["avg_score"] if tb and tb["avg_score"] is not None else None
            delta = (sa - sb) if sa is not None and sb is not None else None
            # Determine winner for this task
            winner = None
            if sa is not None and sb is not None:
                if sa > sb + 0.01:
                    winner = "a"
                elif sb > sa + 0.01:
                    winner = "b"
            task_rows.append({
                "task_name": task_name,
                "score_a": sa, "score_b": sb,
                "tps_a": ta["avg_tps"] if ta else 0,
                "tps_b": tb["avg_tps"] if tb else 0,
                "n_a": ta["n"] if ta else 0,
                "n_b": tb["n"] if tb else 0,
                "delta": delta,
                "winner": winner,
            })
        # Count wins
        wins_a = sum(1 for r in task_rows if r["winner"] == "a")
        wins_b = sum(1 for r in task_rows if r["winner"] == "b")
        ties = sum(1 for r in task_rows if r["winner"] is None and r["score_a"] is not None)
        compare_data = {
            "profile_a": profile_a,
            "profile_b": profile_b,
            "task_rows": task_rows,
            "wins_a": wins_a,
            "wins_b": wins_b,
            "ties": ties,
        }

    return templates.TemplateResponse("compare.html", {
        "request": request,
        "llm_models": llm_models,
        "emb_models": emb_models,
        "a": a, "b": b,
        "compare_data": compare_data,
    })


# ── Efficiency ────────────────────────────────────────────────────────────────

def _efficiency_data(results: list) -> list:
    stats = _model_stats(results)
    rows = []
    for model, s in stats.items():
        vram_gb = s["vram_peak_mb"] / 1024 if s["vram_peak_mb"] else 1
        avg_latency_s = (s["avg_ttft_ms"] / 1000) if s["avg_ttft_ms"] else 1
        quality = s["avg_score"]
        efficiency = quality / (vram_gb * avg_latency_s) if (vram_gb * avg_latency_s) > 0 else 0
        rows.append({
            "model": model,
            "avg_score": s["avg_score"],
            "avg_tps": s["avg_tps"],
            "vram_peak_mb": s["vram_peak_mb"],
            "vram_gb": round(vram_gb, 2),
            "avg_ttft_ms": s["avg_ttft_ms"],
            "efficiency_score": round(efficiency, 4),
        })
    rows.sort(key=lambda r: r["efficiency_score"], reverse=True)
    return rows


@app.get("/api/run/{run_id}/efficiency", response_class=HTMLResponse)
async def efficiency_partial(request: Request, run_id: str):
    results = get_run_results(run_id)
    rows = _efficiency_data(results)
    return templates.TemplateResponse("_partials/efficiency.html", {
        "request": request, "run_id": run_id, "rows": rows,
    })


# ── Ollama Model Management ──────────────────────────────────────────────────

OLLAMA_BASE = "http://localhost:11434"


@app.get("/models", response_class=HTMLResponse)
async def models_page(request: Request):
    return templates.TemplateResponse("models.html", {"request": request})


@app.get("/api/models/status")
async def models_status():
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{OLLAMA_BASE}/api/tags")
            return JSONResponse({"running": resp.status_code == 200})
    except Exception:
        return JSONResponse({"running": False})


@app.post("/api/models/start")
async def models_start():
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        await asyncio.sleep(2)
        return JSONResponse({"started": True})
    except Exception as e:
        return JSONResponse({"started": False, "error": str(e)}, status_code=500)


@app.get("/api/models/list", response_class=HTMLResponse)
async def models_list(request: Request):
    llm_models = []
    embedding_models = []
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{OLLAMA_BASE}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                for m in data.get("models", []):
                    size_bytes = m.get("size", 0)
                    size_gb = size_bytes / (1024**3)
                    size_display = f"{size_gb:.1f}GB" if size_gb >= 1 else f"{size_bytes / (1024**2):.0f}MB"
                    details = m.get("details", {})
                    info = {
                        "name": m.get("name", ""),
                        "size_display": size_display,
                        "modified": m.get("modified_at", "")[:16] if m.get("modified_at") else "",
                        "format": details.get("format", ""),
                        "parameter_size": details.get("parameter_size", ""),
                    }
                    if _is_embedding_model(m):
                        embedding_models.append(info)
                    else:
                        llm_models.append(info)
    except Exception:
        pass
    return templates.TemplateResponse("_partials/models_list.html", {
        "request": request,
        "llm_models": llm_models,
        "embedding_models": embedding_models,
        "models": llm_models + embedding_models,
    })


@app.get("/api/models/pull/{model_name:path}")
async def models_pull(model_name: str):
    async def event_stream():
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_BASE}/api/pull",
                    json={"name": model_name},
                    timeout=None,
                ) as resp:
                    async for line in resp.aiter_lines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                is_done = data.get("status", "").lower() == "success"
                                event_data = {
                                    "status": data.get("status", ""),
                                    "completed": data.get("completed", 0),
                                    "total": data.get("total", 0),
                                    "done": is_done,
                                }
                                yield f"data: {json.dumps(event_data)}\n\n"
                            except json.JSONDecodeError:
                                pass
            yield f"data: {json.dumps({'status': 'Complete', 'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'status': f'Error: {str(e)}', 'done': True})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.delete("/api/models/delete/{model_name:path}")
async def models_delete(model_name: str):
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.request(
                "DELETE",
                f"{OLLAMA_BASE}/api/delete",
                json={"name": model_name},
            )
            return JSONResponse({"deleted": resp.status_code == 200})
    except Exception as e:
        return JSONResponse({"deleted": False, "error": str(e)}, status_code=500)
