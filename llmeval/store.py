import sqlite3
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Dict, Any

from .models import RunConfig, InferenceResult, HWSample

DB_PATH = Path(__file__).parent.parent / "results" / "llmeval.db"


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS runs (
        run_id          TEXT PRIMARY KEY,
        created_at      TEXT NOT NULL,
        description     TEXT,
        task_name       TEXT,
        task_type       TEXT,
        models          TEXT,
        ollama_options  TEXT
    );

    CREATE TABLE IF NOT EXISTS results (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id       TEXT REFERENCES runs(run_id),
        model        TEXT,
        input_idx    INTEGER,
        prompt       TEXT,
        response     TEXT,
        score        REAL,
        score_detail TEXT,
        tps          REAL,
        ttft_ms      REAL,
        total_ms     REAL,
        token_count  INTEGER,
        hw_summary   TEXT
    );

    CREATE TABLE IF NOT EXISTS hw_samples (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id       TEXT REFERENCES runs(run_id),
        model        TEXT,
        ts_ms        INTEGER,
        vram_mb      INTEGER,
        gpu_util     INTEGER,
        gpu_temp     INTEGER,
        gpu_power_w  REAL,
        ram_mb       INTEGER,
        cpu_util     REAL
    );
    """)
    # 기존 DB 마이그레이션
    for col in ("ollama_options TEXT", "judge TEXT"):
        try:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass  # 이미 존재
    conn.commit()
    conn.close()


def save_run(config: RunConfig) -> None:
    conn = _conn()
    conn.execute(
        "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            config.run_id,
            config.created_at,
            config.description,
            config.task_name,
            config.task_type,
            json.dumps(config.models),
            json.dumps(config.ollama_options) if config.ollama_options else None,
            config.judge,
        ),
    )
    conn.commit()
    conn.close()


def save_result(run_id: str, result: InferenceResult) -> None:
    conn = _conn()
    hw_json = result.hw_summary.model_dump_json() if result.hw_summary else None
    conn.execute(
        """INSERT INTO results
           (run_id, model, input_idx, prompt, response,
            score, score_detail, tps, ttft_ms, total_ms, token_count, hw_summary)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id,
            result.model,
            result.input_idx,
            result.prompt,
            result.response,
            result.score,
            json.dumps(result.score_detail),
            result.tps,
            result.ttft_ms,
            result.total_ms,
            result.token_count,
            hw_json,
        ),
    )
    conn.commit()
    conn.close()


def save_hw_samples(run_id: str, model: str, samples: List[HWSample]) -> None:
    conn = _conn()
    conn.executemany(
        """INSERT INTO hw_samples
           (run_id, model, ts_ms, vram_mb, gpu_util, gpu_temp, gpu_power_w, ram_mb, cpu_util)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (run_id, model, s.ts_ms, s.vram_mb, s.gpu_util,
             s.gpu_temp, s.gpu_power_w, s.ram_mb, s.cpu_util)
            for s in samples
        ],
    )
    conn.commit()
    conn.close()


def update_hw_summary(run_id: str, model: str, hw_summary: dict) -> None:
    conn = _conn()
    conn.execute(
        "UPDATE results SET hw_summary = ? WHERE run_id = ? AND model = ?",
        (json.dumps(hw_summary), run_id, model),
    )
    conn.commit()
    conn.close()


def get_latest_run_id() -> Optional[str]:
    conn = _conn()
    row = conn.execute(
        "SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return row["run_id"] if row else None


def get_run(run_id: str) -> Optional[Dict]:
    conn = _conn()
    row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d["models"] = json.loads(d["models"])
    if d.get("ollama_options"):
        d["ollama_options"] = json.loads(d["ollama_options"])
    return d


def get_run_results(run_id: str) -> List[Dict]:
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM results WHERE run_id = ? ORDER BY model, input_idx",
        (run_id,),
    ).fetchall()
    conn.close()
    results = []
    for r in rows:
        d = dict(r)
        if d["score_detail"]:
            d["score_detail"] = json.loads(d["score_detail"])
        if d["hw_summary"]:
            d["hw_summary"] = json.loads(d["hw_summary"])
        results.append(d)
    return results


def get_hw_samples(run_id: str, model: str) -> List[Dict]:
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM hw_samples WHERE run_id = ? AND model = ? ORDER BY ts_ms",
        (run_id, model),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_runs() -> List[Dict]:
    conn = _conn()
    rows = conn.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
    conn.close()
    results = []
    for r in rows:
        d = dict(r)
        d["models"] = json.loads(d["models"])
        results.append(d)
    return results


def get_embedding_summary(run_id: str) -> Dict[str, Dict]:
    """임베딩 실험의 모델별 요약 통계 (Spearman, Recall, MRR 등)."""
    from .evaluators.embedding_metrics import spearman_rank_correlation

    results = get_run_results(run_id)
    models = list(dict.fromkeys(r["model"] for r in results))
    summary = {}

    for model in models:
        rows = [r for r in results if r["model"] == model]
        scores = [r["score"] for r in rows if r["score"] is not None]
        latencies = [r["ttft_ms"] for r in rows if r["ttft_ms"]]
        eps_vals = [r["tps"] for r in rows if r["tps"]]
        details = [r.get("score_detail") or {} for r in rows]

        dims = details[0].get("dimensions", 0) if details else 0

        s = {
            "n": len(rows),
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "avg_eps": sum(eps_vals) / len(eps_vals) if eps_vals else 0,
            "dimensions": dims,
        }

        # STS: Spearman 상관계수
        predicted = [d.get("predicted_sim") for d in details if d.get("predicted_sim") is not None]
        human = [d.get("human_score") for d in details if d.get("human_score") is not None]
        if predicted and human and len(predicted) == len(human):
            s["spearman"] = spearman_rank_correlation(predicted, human)
            errors = [d.get("error", 0) for d in details]
            s["avg_error"] = sum(errors) / len(errors) if errors else 0

        # Retrieval: Recall, MRR
        recall1 = [d.get("recall_at_1") for d in details if d.get("recall_at_1") is not None]
        recall3 = [d.get("recall_at_3") for d in details if d.get("recall_at_3") is not None]
        mrr_vals = [d.get("mrr") for d in details if d.get("mrr") is not None]
        if recall1:
            s["avg_recall_at_1"] = sum(recall1) / len(recall1)
        if recall3:
            s["avg_recall_at_3"] = sum(recall3) / len(recall3)
        if mrr_vals:
            s["avg_mrr"] = sum(mrr_vals) / len(mrr_vals)

        summary[model] = s

    return summary


def get_leaderboard(task_name: Optional[str] = None) -> List[Dict]:
    """모든 run에서 모델별 평균 점수/속도 집계."""
    conn = _conn()
    query = """
        SELECT r.model,
               runs.task_name,
               COUNT(*) as n,
               AVG(r.score) as avg_score,
               AVG(r.tps) as avg_tps,
               AVG(r.ttft_ms) as avg_ttft_ms
        FROM results r
        JOIN runs ON runs.run_id = r.run_id
    """
    params = ()
    if task_name:
        query += " WHERE runs.task_name = ?"
        params = (task_name,)
    query += " GROUP BY r.model, runs.task_name ORDER BY avg_score DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_models() -> List[str]:
    """DB에 결과가 있는 모든 모델 목록."""
    conn = _conn()
    rows = conn.execute(
        "SELECT DISTINCT model FROM results ORDER BY model"
    ).fetchall()
    conn.close()
    return [r["model"] for r in rows]


# t-distribution critical values for 95% CI (two-tailed, alpha=0.05)
_T_CRITICAL = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042, 50: 2.009,
    100: 1.984,
}


def _t_critical(df: int) -> float:
    if df <= 0:
        return 0.0
    if df in _T_CRITICAL:
        return _T_CRITICAL[df]
    keys = sorted(_T_CRITICAL.keys())
    if df > keys[-1]:
        return 1.96
    for i, k in enumerate(keys):
        if k > df:
            prev_k = keys[i - 1] if i > 0 else keys[0]
            frac = (df - prev_k) / (k - prev_k) if k != prev_k else 0
            return _T_CRITICAL[prev_k] + frac * (_T_CRITICAL[k] - _T_CRITICAL[prev_k])
    return 1.96


def _compute_ci(scores: List[float]) -> Dict:
    n = len(scores)
    if n == 0:
        return {"avg": 0, "std": 0, "n": 0, "ci_low": 0, "ci_high": 0}
    mean = sum(scores) / n
    if n >= 2:
        variance = sum((s - mean) ** 2 for s in scores) / (n - 1)
        std = math.sqrt(variance)
        ci_half = _t_critical(n - 1) * std / math.sqrt(n)
    else:
        std = 0.0
        ci_half = 0.0
    return {
        "avg": round(mean, 3),
        "std": round(std, 3),
        "n": n,
        "ci_low": round(max(0, mean - ci_half), 3),
        "ci_high": round(min(5, mean + ci_half), 3),
    }


# Tasks archived due to zero differentiation (all models score identically)
_ARCHIVED_TASKS = {"구조화 출력 준수 평가"}


def get_available_judges() -> List[str]:
    """DB에 기록된 모든 judge 목록."""
    conn = _conn()
    rows = conn.execute(
        "SELECT DISTINCT judge FROM runs WHERE judge IS NOT NULL ORDER BY judge"
    ).fetchall()
    conn.close()
    return [r["judge"] for r in rows]


def get_score_matrix(judge_filter: Optional[str] = None) -> Dict:
    """Cross-tabulation of models × tasks with 95% CI.

    judge_filter: None=all, 'claude'=claude:* runs only, 'ollama'=ollama:* runs only,
                  or exact label like 'claude:sonnet'.
    """
    conn = _conn()
    if judge_filter == "claude":
        rows = conn.execute("""
            SELECT r.model, runs.task_name, r.score
            FROM results r
            JOIN runs ON runs.run_id = r.run_id
            WHERE r.score IS NOT NULL AND runs.judge LIKE 'claude:%'
            ORDER BY r.model, runs.task_name
        """).fetchall()
    elif judge_filter == "ollama":
        rows = conn.execute("""
            SELECT r.model, runs.task_name, r.score
            FROM results r
            JOIN runs ON runs.run_id = r.run_id
            WHERE r.score IS NOT NULL AND (runs.judge LIKE 'ollama:%' OR runs.judge IS NULL)
            ORDER BY r.model, runs.task_name
        """).fetchall()
    elif judge_filter:
        rows = conn.execute("""
            SELECT r.model, runs.task_name, r.score
            FROM results r
            JOIN runs ON runs.run_id = r.run_id
            WHERE r.score IS NOT NULL AND runs.judge = ?
            ORDER BY r.model, runs.task_name
        """, (judge_filter,)).fetchall()
    else:
        rows = conn.execute("""
            SELECT r.model, runs.task_name, r.score
            FROM results r
            JOIN runs ON runs.run_id = r.run_id
            WHERE r.score IS NOT NULL
            ORDER BY r.model, runs.task_name
        """).fetchall()
    conn.close()

    raw_scores: Dict[tuple, list] = defaultdict(list)
    for r in rows:
        if r["task_name"] in _ARCHIVED_TASKS:
            continue
        raw_scores[(r["model"], r["task_name"])].append(r["score"])

    models = sorted(set(k[0] for k in raw_scores))
    tasks = sorted(set(k[1] for k in raw_scores))

    # Per-cell stats with CI (string keys for Jinja2)
    cells = {}
    for (model, task), score_list in raw_scores.items():
        cells[f"{model}|{task}"] = _compute_ci(score_list)

    # Per-model overall stats
    model_stats = {}
    for model in models:
        all_scores = []
        for task in tasks:
            if (model, task) in raw_scores:
                all_scores.extend(raw_scores[(model, task)])
        model_stats[model] = _compute_ci(all_scores)

    # Per-task averages
    task_avgs = {}
    for task in tasks:
        task_scores = []
        for model in models:
            if (model, task) in raw_scores:
                task_scores.extend(raw_scores[(model, task)])
        task_avgs[task] = round(sum(task_scores) / len(task_scores), 3) if task_scores else 0

    return {
        "models": models,
        "tasks": tasks,
        "cells": cells,
        "model_stats": model_stats,
        "task_avgs": task_avgs,
    }


def get_model_profile(model: str) -> Dict:
    """모델 하나의 전체 태스크별 성능 프로필."""
    conn = _conn()
    rows = conn.execute("""
        SELECT runs.task_name, runs.task_type,
               COUNT(*) as n,
               AVG(r.score) as avg_score,
               AVG(r.tps) as avg_tps,
               AVG(r.ttft_ms) as avg_ttft_ms
        FROM results r
        JOIN runs ON runs.run_id = r.run_id
        WHERE r.model = ?
        GROUP BY runs.task_name
        ORDER BY runs.task_name
    """, (model,)).fetchall()

    hw_row = conn.execute("""
        SELECT AVG(CAST(json_extract(r.hw_summary, '$.vram_peak_mb') AS REAL)) as avg_vram,
               AVG(CAST(json_extract(r.hw_summary, '$.gpu_util_avg_pct') AS REAL)) as avg_gpu_util,
               AVG(CAST(json_extract(r.hw_summary, '$.gpu_temp_peak_c') AS REAL)) as avg_temp,
               AVG(CAST(json_extract(r.hw_summary, '$.gpu_power_avg_w') AS REAL)) as avg_power,
               AVG(CAST(json_extract(r.hw_summary, '$.ram_peak_mb') AS REAL)) as avg_ram
        FROM results r
        WHERE r.model = ? AND r.hw_summary IS NOT NULL
    """, (model,)).fetchone()
    conn.close()

    tasks = [dict(r) for r in rows]
    all_scores = [t["avg_score"] for t in tasks if t["avg_score"] is not None]
    all_tps = [t["avg_tps"] for t in tasks if t["avg_tps"]]
    total_n = sum(t["n"] for t in tasks)

    hw = dict(hw_row) if hw_row else {}

    return {
        "model": model,
        "tasks": tasks,
        "overall_avg_score": sum(all_scores) / len(all_scores) if all_scores else 0,
        "overall_avg_tps": sum(all_tps) / len(all_tps) if all_tps else 0,
        "total_evaluations": total_n,
        "task_count": len(tasks),
        "hw": {
            "vram_peak_mb": hw.get("avg_vram", 0) or 0,
            "gpu_util": hw.get("avg_gpu_util", 0) or 0,
            "gpu_temp": hw.get("avg_temp", 0) or 0,
            "gpu_power": hw.get("avg_power", 0) or 0,
            "ram_peak_mb": hw.get("avg_ram", 0) or 0,
        },
    }
