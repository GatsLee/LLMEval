import sqlite3
import json
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
    # 기존 DB 마이그레이션: ollama_options 컬럼 추가
    try:
        conn.execute("ALTER TABLE runs ADD COLUMN ollama_options TEXT")
    except sqlite3.OperationalError:
        pass  # 이미 존재
    conn.commit()
    conn.close()


def save_run(config: RunConfig) -> None:
    conn = _conn()
    conn.execute(
        "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            config.run_id,
            config.created_at,
            config.description,
            config.task_name,
            config.task_type,
            json.dumps(config.models),
            json.dumps(config.ollama_options) if config.ollama_options else None,
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
