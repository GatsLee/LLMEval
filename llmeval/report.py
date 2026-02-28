"""
Rich 기반 터미널 리포트.
임계값 기반 색상 경고, 모델 비교 테이블 출력.
"""
import json
from typing import List, Dict, Optional

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box

from .store import get_run, get_run_results, get_all_runs, get_leaderboard, get_embedding_summary

console = Console()

# ── 임계값 ────────────────────────────────────────────────────────────────────
VRAM_WARN_PCT = 0.85   # VRAM 85% 이상 경고
TEMP_WARN_C   = 78     # 78°C 이상 경고
TEMP_CRIT_C   = 83     # 83°C 이상 위험
SCORE_LOW     = 2.5    # 3점 미만 낮은 품질
SCORE_HIGH    = 4.0    # 4점 이상 양호


def _score_color(score: Optional[float]) -> str:
    if score is None:
        return "dim"
    if score >= SCORE_HIGH:
        return "green"
    if score >= SCORE_LOW:
        return "yellow"
    return "red"


def _temp_color(temp: int) -> str:
    if temp >= TEMP_CRIT_C:
        return "bold red"
    if temp >= TEMP_WARN_C:
        return "yellow"
    return "green"


def _vram_color(used: int, budget: int) -> str:
    pct = used / budget if budget else 0
    if pct >= VRAM_WARN_PCT:
        return "yellow"
    return "cyan"


def _fmt_score(score: Optional[float]) -> Text:
    if score is None:
        return Text("—", style="dim")
    txt = f"{score:.2f}"
    return Text(txt, style=_score_color(score))


def _warn(flag: bool, label: str) -> Text:
    if flag:
        return Text(f"⚠ {label}", style="bold red")
    return Text("—", style="dim")


# ── 임베딩 리포트 ─────────────────────────────────────────────────────────────

def _show_embedding_report(run: dict, results: list) -> None:
    """임베딩 태스크 결과 리포트."""
    task_type = run["task_type"]
    models = list(dict.fromkeys(r["model"] for r in results))
    summary = get_embedding_summary(run["run_id"])

    if task_type == "embedding_sts":
        t = Table(title="STS 임베딩 품질", box=box.ROUNDED, show_lines=True)
        t.add_column("모델", style="bold white")
        t.add_column("Avg Cosine Sim", justify="center")
        t.add_column("Spearman ρ", justify="center")
        t.add_column("Avg Error", justify="center")
        t.add_column("Dimensions", justify="center", style="dim")
        t.add_column("Avg Latency", justify="right")
        t.add_column("Embed/s", justify="right")
        t.add_column("N", justify="center", style="dim")

        for model in models:
            s = summary.get(model, {})
            rho = s.get("spearman", 0)
            rho_color = "green" if rho >= 0.7 else ("yellow" if rho >= 0.4 else "red")
            t.add_row(
                model,
                f"{s.get('avg_score', 0):.4f}",
                Text(f"{rho:.4f}", style=rho_color),
                f"{s.get('avg_error', 0):.4f}",
                str(s.get("dimensions", "?")),
                f"{s.get('avg_latency_ms', 0):.0f}ms",
                f"{s.get('avg_eps', 0):.1f}",
                str(s.get("n", 0)),
            )
        console.print(t)

    elif task_type == "embedding_retrieval":
        t = Table(title="검색 임베딩 품질", box=box.ROUNDED, show_lines=True)
        t.add_column("모델", style="bold white")
        t.add_column("Recall@1", justify="center")
        t.add_column("Recall@3", justify="center")
        t.add_column("MRR", justify="center")
        t.add_column("Dimensions", justify="center", style="dim")
        t.add_column("Avg Latency", justify="right")
        t.add_column("Embed/s", justify="right")
        t.add_column("N", justify="center", style="dim")

        for model in models:
            s = summary.get(model, {})
            r1 = s.get("avg_recall_at_1", 0)
            r1_color = "green" if r1 >= 0.8 else ("yellow" if r1 >= 0.5 else "red")
            t.add_row(
                model,
                Text(f"{r1:.2f}", style=r1_color),
                f"{s.get('avg_recall_at_3', 0):.2f}",
                f"{s.get('avg_mrr', 0):.4f}",
                str(s.get("dimensions", "?")),
                f"{s.get('avg_latency_ms', 0):.0f}ms",
                f"{s.get('avg_eps', 0):.1f}",
                str(s.get("n", 0)),
            )
        console.print(t)


# ── 메인 리포트 ───────────────────────────────────────────────────────────────

def show_report(run_id: str) -> None:
    run = get_run(run_id)
    if not run:
        console.print(f"[red]run_id '{run_id}' 없음[/]")
        return

    results = get_run_results(run_id)
    if not results:
        console.print("[yellow]결과 없음[/]")
        return

    console.rule(f"[bold]Run: {run['description']}  │  {run['created_at']}  │  Task: {run['task_name']}")

    # 임베딩 태스크는 별도 리포트
    if run.get("task_type") in ("embedding_sts", "embedding_retrieval"):
        _show_embedding_report(run, results)
        return

    # 모델별 집계
    models = list(dict.fromkeys(r["model"] for r in results))
    stats: Dict[str, dict] = {}
    for model in models:
        rows = [r for r in results if r["model"] == model]
        scores = [r["score"] for r in rows if r["score"] is not None]
        tps_vals = [r["tps"] for r in rows if r["tps"]]
        ttft_vals = [r["ttft_ms"] for r in rows if r["ttft_ms"]]

        hw = rows[0].get("hw_summary") or {}

        stats[model] = {
            "avg_score": sum(scores) / len(scores) if scores else None,
            "n": len(rows),
            "avg_tps": sum(tps_vals) / len(tps_vals) if tps_vals else 0,
            "avg_ttft_ms": sum(ttft_vals) / len(ttft_vals) if ttft_vals else 0,
            "vram_peak_mb": hw.get("vram_peak_mb", 0),
            "vram_budget_mb": hw.get("vram_budget_mb", 12282),
            "vram_headroom_mb": hw.get("vram_headroom_mb", 0),
            "gpu_util_avg": hw.get("gpu_util_avg_pct", 0),
            "gpu_temp_peak": hw.get("gpu_temp_peak_c", 0),
            "gpu_power_avg": hw.get("gpu_power_avg_w", 0),
            "ram_peak_mb": hw.get("ram_peak_mb", 0),
            "spillover": hw.get("spillover_detected", False),
            "throttle": hw.get("throttle_detected", False),
            "nvml": hw.get("nvml_available", False),
        }

    # ── 품질 테이블 ──────────────────────────────────────────────────────────
    qt = Table(title="품질 지표", box=box.ROUNDED, show_lines=True)
    qt.add_column("모델", style="bold white")
    qt.add_column("Avg Score", justify="center")
    qt.add_column("N", justify="center", style="dim")
    qt.add_column("Avg t/s", justify="right")
    qt.add_column("Avg TTFT", justify="right")

    for model in models:
        s = stats[model]
        qt.add_row(
            model,
            _fmt_score(s["avg_score"]),
            str(s["n"]),
            f"{s['avg_tps']:.1f}",
            f"{s['avg_ttft_ms']:.0f}ms",
        )
    console.print(qt)

    # ── 하드웨어 테이블 ──────────────────────────────────────────────────────
    ht = Table(title="하드웨어 지표 (RTX 4070 Super 12282MB)", box=box.ROUNDED, show_lines=True)
    ht.add_column("모델", style="bold white")
    ht.add_column("VRAM 피크", justify="right")
    ht.add_column("여유", justify="right")
    ht.add_column("GPU Util", justify="right")
    ht.add_column("GPU Temp", justify="right")
    ht.add_column("전력", justify="right")
    ht.add_column("RAM 피크", justify="right")
    ht.add_column("⚠", justify="center")

    for model in models:
        s = stats[model]
        vram_txt = Text(
            f"{s['vram_peak_mb']:,}MB",
            style=_vram_color(s["vram_peak_mb"], s["vram_budget_mb"]),
        )
        temp_txt = Text(
            f"{s['gpu_temp_peak']}°C",
            style=_temp_color(s["gpu_temp_peak"]),
        )
        warnings = []
        if s["spillover"]:
            warnings.append("SPILL")
        if s["throttle"]:
            warnings.append("THRTL")
        warn_txt = Text(", ".join(warnings) if warnings else "—",
                        style="bold red" if warnings else "dim")

        ht.add_row(
            model,
            vram_txt,
            f"{s['vram_headroom_mb']:,}MB",
            f"{s['gpu_util_avg']:.0f}%",
            temp_txt,
            f"{s['gpu_power_avg']:.1f}W",
            f"{s['ram_peak_mb']:,}MB",
            warn_txt,
        )

    if stats and list(stats.values())[0]["nvml"]:
        console.print(ht)
    else:
        console.print("[dim]하드웨어 메트릭 없음 (nvidia-ml-py 미연결)[/]")

    # ── 경고 요약 ────────────────────────────────────────────────────────────
    alerts = []
    for model, s in stats.items():
        if s["spillover"]:
            alerts.append(f"[bold red]⚠ VRAM 스필오버:[/] {model} — VRAM 초과 → RAM 오프로드 발생, 레이턴시 증가 가능")
        if s["throttle"]:
            alerts.append(f"[bold red]⚠ 쓰로틀링:[/] {model} — 추론 후반 t/s 저하 감지 (온도/전력 확인)")
        if s["avg_score"] is not None and s["avg_score"] < SCORE_LOW:
            alerts.append(f"[yellow]⚠ 낮은 품질:[/] {model} — 평균 점수 {s['avg_score']:.2f} (기준: {SCORE_LOW})")

    if alerts:
        console.rule("[bold red]경고")
        for a in alerts:
            console.print(f"  {a}")


# ── 리스트 ────────────────────────────────────────────────────────────────────

def show_runs() -> None:
    runs = get_all_runs()
    if not runs:
        console.print("[yellow]저장된 실험 없음[/]")
        return

    t = Table(title="저장된 실험 목록", box=box.SIMPLE_HEAVY)
    t.add_column("run_id", style="cyan")
    t.add_column("날짜", style="dim")
    t.add_column("태스크")
    t.add_column("모델")
    t.add_column("설명")

    for r in runs:
        t.add_row(
            r["run_id"],
            r["created_at"],
            r["task_name"],
            ", ".join(r["models"]),
            r["description"] or "—",
        )
    console.print(t)


# ── 리더보드 ─────────────────────────────────────────────────────────────────

def show_leaderboard(task_name: Optional[str] = None) -> None:
    rows = get_leaderboard(task_name)
    if not rows:
        console.print("[yellow]리더보드 데이터 없음[/]")
        return

    title = f"리더보드 — {task_name}" if task_name else "전체 리더보드"
    t = Table(title=title, box=box.ROUNDED, show_lines=True)
    t.add_column("순위", justify="center", style="dim")
    t.add_column("모델", style="bold white")
    t.add_column("태스크")
    t.add_column("실험 수", justify="center")
    t.add_column("Avg Score", justify="center")
    t.add_column("Avg t/s", justify="right")
    t.add_column("Avg TTFT", justify="right")

    for i, r in enumerate(rows, 1):
        t.add_row(
            str(i),
            r["model"],
            r["task_name"],
            str(r["n"]),
            _fmt_score(r["avg_score"]),
            f"{r['avg_tps']:.1f}" if r["avg_tps"] else "—",
            f"{r['avg_ttft_ms']:.0f}ms" if r["avg_ttft_ms"] else "—",
        )
    console.print(t)


# ── compare ──────────────────────────────────────────────────────────────────

def show_compare(run_id_a: str, run_id_b: str) -> None:
    run_a = get_run(run_id_a)
    run_b = get_run(run_id_b)
    if not run_a or not run_b:
        console.print("[red]run_id를 찾을 수 없습니다.[/]")
        return

    res_a = get_run_results(run_id_a)
    res_b = get_run_results(run_id_b)

    def agg(results, model):
        rows = [r for r in results if r["model"] == model]
        scores = [r["score"] for r in rows if r["score"] is not None]
        tps_vals = [r["tps"] for r in rows if r["tps"]]
        return {
            "avg_score": sum(scores) / len(scores) if scores else None,
            "avg_tps": sum(tps_vals) / len(tps_vals) if tps_vals else 0,
        }

    all_models = list(dict.fromkeys(
        [r["model"] for r in res_a] + [r["model"] for r in res_b]
    ))

    console.rule(f"[bold]비교: {run_id_a} vs {run_id_b}")
    console.print(f"  A: {run_a['description']}  ({run_a['created_at']})")
    console.print(f"  B: {run_b['description']}  ({run_b['created_at']})")

    t = Table(box=box.ROUNDED, show_lines=True)
    t.add_column("모델", style="bold white")
    t.add_column("Score A", justify="center")
    t.add_column("Score B", justify="center")
    t.add_column("Δ Score", justify="center")
    t.add_column("t/s A", justify="right")
    t.add_column("t/s B", justify="right")

    for model in all_models:
        a = agg(res_a, model)
        b = agg(res_b, model)

        sa = a["avg_score"]
        sb = b["avg_score"]

        if sa is not None and sb is not None:
            delta = sb - sa
            delta_txt = Text(
                f"{delta:+.2f}",
                style="green" if delta > 0 else ("red" if delta < 0 else "dim"),
            )
        else:
            delta_txt = Text("—", style="dim")

        t.add_row(
            model,
            _fmt_score(sa),
            _fmt_score(sb),
            delta_txt,
            f"{a['avg_tps']:.1f}" if a["avg_tps"] else "—",
            f"{b['avg_tps']:.1f}" if b["avg_tps"] else "—",
        )
    console.print(t)
