"""
그래프 엔진:
- plotext: 터미널 내 ASCII 그래프 (빠른 확인)
- matplotlib: PNG 파일 저장 (블로그/README용)
"""
from pathlib import Path
from typing import Optional
import json

EXPORTS_DIR = Path(__file__).parent.parent / "exports"


def _get_model_stats(run_id: str) -> dict:
    from .store import get_run_results, get_hw_samples
    results = get_run_results(run_id)
    models = list(dict.fromkeys(r["model"] for r in results))

    stats = {}
    for model in models:
        rows = [r for r in results if r["model"] == model]
        scores = [r["score"] for r in rows if r["score"] is not None]
        tps_vals = [r["tps"] for r in rows if r["tps"]]
        ttft_vals = [r["ttft_ms"] for r in rows if r["ttft_ms"]]
        hw = rows[0].get("hw_summary") or {} if rows else {}

        stats[model] = {
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "avg_tps": sum(tps_vals) / len(tps_vals) if tps_vals else 0.0,
            "avg_ttft_ms": sum(ttft_vals) / len(ttft_vals) if ttft_vals else 0.0,
            "vram_peak_mb": hw.get("vram_peak_mb", 0),
            "gpu_temp_peak_c": hw.get("gpu_temp_peak_c", 0),
            "gpu_power_avg_w": hw.get("gpu_power_avg_w", 0.0),
            "gpu_util_avg_pct": hw.get("gpu_util_avg_pct", 0.0),
            "vram_budget_mb": hw.get("vram_budget_mb", 12282),
            "hw_samples": get_hw_samples(run_id, model),
        }
    return stats


# ── 터미널 그래프 (plotext) ───────────────────────────────────────────────────

def plot_terminal(run_id: str) -> None:
    try:
        import plotext as plt
    except ImportError:
        print("plotext 미설치. pip install plotext")
        return

    stats = _get_model_stats(run_id)
    if not stats:
        print("데이터 없음")
        return

    models = list(stats.keys())
    short = [m.split(":")[0] for m in models]

    scores   = [stats[m]["avg_score"] for m in models]
    tps      = [stats[m]["avg_tps"] for m in models]
    vram     = [stats[m]["vram_peak_mb"] for m in models]
    temp     = [stats[m]["gpu_temp_peak_c"] for m in models]
    power    = [stats[m]["gpu_power_avg_w"] for m in models]

    # 1. 품질 점수
    plt.clear_figure()
    plt.bar(short, scores, color="green")
    plt.title(f"품질 점수 (Avg Score) — run: {run_id}")
    plt.ylim(0, 5)
    plt.ylabel("Score (1–5)")
    plt.show()

    # 2. 생성 속도
    plt.clear_figure()
    plt.bar(short, tps, color="blue")
    plt.title(f"생성 속도 (tokens/sec) — run: {run_id}")
    plt.ylabel("t/s")
    plt.show()

    # 3. VRAM 사용량
    budget = list(stats.values())[0]["vram_budget_mb"]
    plt.clear_figure()
    plt.bar(short, vram, color="magenta")
    plt.hline(budget, "red")
    plt.title(f"VRAM 피크 사용량 (MB) │ 빨간선={budget}MB 한계 — run: {run_id}")
    plt.ylabel("MB")
    plt.show()

    # 4. GPU 온도
    plt.clear_figure()
    plt.bar(short, temp, color="orange")
    plt.hline(83, "red")
    plt.hline(78, "yellow")
    plt.title(f"GPU 온도 피크 (°C) │ 빨간=83°C 위험 — run: {run_id}")
    plt.ylabel("°C")
    plt.show()

    # 5. Score vs VRAM 산점도 (트레이드오프)
    if len(models) >= 2:
        plt.clear_figure()
        plt.scatter(vram, scores)
        for i, m in enumerate(short):
            plt.text(m, x=vram[i], y=scores[i])
        plt.title(f"품질 vs VRAM — 왼쪽 위가 최적 — run: {run_id}")
        plt.xlabel("VRAM 피크 (MB)")
        plt.ylabel("Score")
        plt.show()

    # 6. 온도 타임라인 (첫 번째 모델)
    first_model = models[0]
    samples = stats[first_model]["hw_samples"]
    if samples:
        ts = [s["ts_ms"] / 1000 for s in samples]
        temps_line = [s["gpu_temp"] for s in samples]
        vram_line = [s["vram_mb"] for s in samples]

        plt.clear_figure()
        plt.plot(ts, temps_line, color="orange", label="온도(°C)")
        plt.title(f"GPU 온도 타임라인 — {first_model}")
        plt.xlabel("시간 (s)")
        plt.ylabel("°C")
        plt.show()

        plt.clear_figure()
        plt.plot(ts, vram_line, color="cyan", label="VRAM(MB)")
        plt.title(f"VRAM 사용량 타임라인 — {first_model}")
        plt.xlabel("시간 (s)")
        plt.ylabel("MB")
        plt.show()


# ── PNG 내보내기 (matplotlib) ─────────────────────────────────────────────────

def export_png(run_id: str) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as mplt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib 미설치.")
        return []

    EXPORTS_DIR.mkdir(exist_ok=True)
    stats = _get_model_stats(run_id)
    if not stats:
        return []

    models = list(stats.keys())
    short = [m.replace(":latest", "").split("/")[-1] for m in models]
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336"][:len(models)]
    saved = []

    def _save(fig, name):
        path = EXPORTS_DIR / f"{run_id}_{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        mplt.close(fig)
        saved.append(str(path))
        return str(path)

    scores  = [stats[m]["avg_score"] for m in models]
    tps     = [stats[m]["avg_tps"] for m in models]
    vram    = [stats[m]["vram_peak_mb"] for m in models]
    temp    = [stats[m]["gpu_temp_peak_c"] for m in models]
    power   = [stats[m]["gpu_power_avg_w"] for m in models]
    budget  = list(stats.values())[0]["vram_budget_mb"]

    # 1. 4-in-1 summary
    fig, axes = mplt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"LLMEval — run: {run_id}", fontsize=14, fontweight="bold")

    ax = axes[0][0]
    bars = ax.bar(short, scores, color=colors)
    ax.set_title("품질 점수 (Avg Score)")
    ax.set_ylim(0, 5)
    ax.axhline(4.0, color="green", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(2.5, color="red",   linestyle="--", linewidth=0.8, alpha=0.6)
    for bar, v in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    ax = axes[0][1]
    bars = ax.bar(short, tps, color=colors)
    ax.set_title("생성 속도 (tokens/sec)")
    for bar, v in zip(bars, tps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    ax = axes[1][0]
    bars = ax.bar(short, vram, color=colors)
    ax.axhline(budget, color="red", linestyle="--", linewidth=1, label=f"{budget}MB 한계")
    ax.set_title("VRAM 피크 사용량 (MB)")
    ax.legend(fontsize=8)
    for bar, v in zip(bars, vram):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f"{v:,}", ha="center", va="bottom", fontsize=9)

    ax = axes[1][1]
    bars = ax.bar(short, temp, color=colors)
    ax.axhline(83, color="red",    linestyle="--", linewidth=1, label="83°C 위험")
    ax.axhline(78, color="orange", linestyle="--", linewidth=1, label="78°C 경고")
    ax.set_title("GPU 온도 피크 (°C)")
    ax.legend(fontsize=8)
    for bar, v in zip(bars, temp):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{v}°C", ha="center", va="bottom", fontsize=9)

    mplt.tight_layout()
    _save(fig, "summary")

    # 2. Score vs VRAM 산점도
    fig, ax = mplt.subplots(figsize=(7, 5))
    ax.scatter(vram, scores, c=colors, s=120, zorder=5)
    for i, m in enumerate(short):
        ax.annotate(m, (vram[i], scores[i]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.axvline(budget, color="red", linestyle="--", linewidth=0.8, alpha=0.6,
               label=f"{budget}MB 한계")
    ax.set_xlabel("VRAM 피크 (MB)")
    ax.set_ylabel("Avg Score (1–5)")
    ax.set_title(f"품질 vs VRAM 트레이드오프 — run: {run_id}")
    ax.set_ylim(0, 5)
    ax.legend(fontsize=8)
    mplt.tight_layout()
    _save(fig, "score_vs_vram")

    # 3. 타임라인 (모델별 VRAM + 온도)
    for model in models:
        samples = stats[model]["hw_samples"]
        if not samples:
            continue
        ts       = [s["ts_ms"] / 1000 for s in samples]
        vram_tl  = [s["vram_mb"] for s in samples]
        temp_tl  = [s["gpu_temp"] for s in samples]
        power_tl = [s["gpu_power_w"] for s in samples]

        fig, axes = mplt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(f"하드웨어 타임라인 — {model} — run: {run_id}")

        axes[0].plot(ts, vram_tl, color="#2196F3")
        axes[0].axhline(budget, color="red", linestyle="--", linewidth=0.8)
        axes[0].set_ylabel("VRAM (MB)")
        axes[0].fill_between(ts, vram_tl, alpha=0.15, color="#2196F3")

        axes[1].plot(ts, temp_tl, color="#FF9800")
        axes[1].axhline(83, color="red",    linestyle="--", linewidth=0.8)
        axes[1].axhline(78, color="orange", linestyle="--", linewidth=0.8)
        axes[1].set_ylabel("온도 (°C)")

        axes[2].plot(ts, power_tl, color="#9C27B0")
        axes[2].set_ylabel("전력 (W)")
        axes[2].set_xlabel("시간 (s)")

        mplt.tight_layout()
        safe_name = model.replace(":", "_").replace("/", "_")
        _save(fig, f"timeline_{safe_name}")

    return saved
