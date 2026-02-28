"""Web dashboard: FastAPI + Jinja2 + HTMX."""
import io
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .store import (
    init_db, get_all_runs, get_run, get_run_results,
    get_hw_samples, get_leaderboard, get_embedding_summary,
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


@app.get("/compare", response_class=HTMLResponse)
async def compare_page(request: Request, a: str = "", b: str = ""):
    runs = get_all_runs()
    compare_data = None

    if a and b:
        run_a = get_run(a)
        run_b = get_run(b)
        res_a = get_run_results(a)
        res_b = get_run_results(b)

        if run_a and run_b:
            all_models = list(dict.fromkeys(
                [r["model"] for r in res_a] + [r["model"] for r in res_b]
            ))
            rows = []
            for model in all_models:
                sa = [r["score"] for r in res_a if r["model"] == model and r["score"] is not None]
                sb = [r["score"] for r in res_b if r["model"] == model and r["score"] is not None]
                ta = [r["tps"] for r in res_a if r["model"] == model and r["tps"]]
                tb = [r["tps"] for r in res_b if r["model"] == model and r["tps"]]
                avg_sa = sum(sa) / len(sa) if sa else None
                avg_sb = sum(sb) / len(sb) if sb else None
                delta = (avg_sb - avg_sa) if avg_sa is not None and avg_sb is not None else None
                rows.append({
                    "model": model,
                    "score_a": avg_sa, "score_b": avg_sb, "delta": delta,
                    "tps_a": sum(ta) / len(ta) if ta else 0,
                    "tps_b": sum(tb) / len(tb) if tb else 0,
                })
            compare_data = {"run_a": run_a, "run_b": run_b, "rows": rows}

    return templates.TemplateResponse("compare.html", {
        "request": request, "runs": runs, "a": a, "b": b,
        "compare_data": compare_data,
    })
