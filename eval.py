#!/usr/bin/env python3
"""
LLMEval CLI
사용법:
  python eval.py run   --task tasks/rag_faithfulness.yaml --models llama3.1:8b
  python eval.py report
  python eval.py compare <run_a> <run_b>
  python eval.py graph
  python eval.py hardware
  python eval.py analyze
  python eval.py leaderboard
  python eval.py runs
"""
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich import box
from rich.table import Table

app = typer.Typer(add_completion=False, help="로컬 LLM 하드웨어 인식 평가 파이프라인")
console = Console()


# ── run ───────────────────────────────────────────────────────────────────────

@app.command()
def run(
    task: str = typer.Option(..., "--task", "-t", help="YAML 태스크 파일 경로"),
    models: str = typer.Option(..., "--models", "-m", help="모델 목록 (콤마 구분, 예: llama3.1:8b,mistral:7b)"),
    description: str = typer.Option("", "--description", "-d", help="실험 설명"),
    num_gpu: int = typer.Option(-1, "--num-gpu", help="GPU 레이어 수 (-1=자동, 0=CPU, 99=전체 GPU)"),
    judge: str = typer.Option("", "--judge", "-j", help="Judge 백엔드 (예: claude:sonnet, ollama:llama3.1:8b)"),
):
    """태스크 실험 실행 (추론 + HW 프로파일링 + 평가)."""
    from llmeval.runner import run_experiment
    from llmeval.report import show_report

    model_list = [m.strip() for m in models.split(",")]

    info = f"[bold]태스크:[/] {task}\n[bold]모델:[/] {', '.join(model_list)}"
    if num_gpu >= 0:
        info += f"\n[bold]num_gpu:[/] {num_gpu}"
    if judge:
        info += f"\n[bold]judge:[/] {judge}"

    console.print(Panel(info, title="[bold cyan]LLMEval 실험 시작", box=box.ROUNDED))

    cli_options = {}
    if num_gpu >= 0:
        cli_options["num_gpu"] = num_gpu

    run_id = run_experiment(task, model_list, description, console=console,
                            cli_options_override=cli_options or None,
                            judge_spec=judge or None)
    console.print(f"\n[bold green]완료! run_id:[/] [cyan]{run_id}[/]")
    console.print()
    show_report(run_id)


# ── report ────────────────────────────────────────────────────────────────────

@app.command()
def report(
    run_id: str = typer.Argument("latest", help="run_id 또는 'latest'"),
):
    """실험 결과 리포트 출력."""
    from llmeval.store import get_latest_run_id
    from llmeval.report import show_report

    if run_id == "latest":
        run_id = get_latest_run_id()
        if not run_id:
            console.print("[red]저장된 실험 없음[/]")
            raise typer.Exit(1)

    show_report(run_id)


# ── compare ───────────────────────────────────────────────────────────────────

@app.command()
def compare(
    run_a: str = typer.Argument(..., help="비교 기준 run_id"),
    run_b: str = typer.Argument(..., help="비교 대상 run_id"),
):
    """두 실험 결과 비교."""
    from llmeval.report import show_compare
    show_compare(run_a, run_b)


# ── graph ─────────────────────────────────────────────────────────────────────

@app.command()
def graph(
    run_id: str = typer.Argument("latest", help="run_id 또는 'latest'"),
    export: bool = typer.Option(False, "--export", "-e", help="PNG 파일로 저장"),
):
    """실험 결과 그래프 출력 (터미널 + 선택적 PNG 저장)."""
    from llmeval.store import get_latest_run_id
    from llmeval.graph import plot_terminal, export_png

    if run_id == "latest":
        run_id = get_latest_run_id()
        if not run_id:
            console.print("[red]저장된 실험 없음[/]")
            raise typer.Exit(1)

    console.print(f"[bold cyan]터미널 그래프 — run: {run_id}[/]\n")
    plot_terminal(run_id)

    if export:
        saved = export_png(run_id)
        console.print(f"\n[green]PNG 저장 완료:[/]")
        for path in saved:
            console.print(f"  {path}")


# ── hardware ──────────────────────────────────────────────────────────────────

@app.command()
def hardware():
    """현재 하드웨어 상태 스냅샷."""
    from llmeval.profiler import snapshot, NVML_AVAILABLE
    import psutil

    snap = snapshot()
    vm = psutil.virtual_memory()
    cpu_count = psutil.cpu_count()

    t = Table(title="하드웨어 현재 상태", box=box.ROUNDED)
    t.add_column("항목", style="bold")
    t.add_column("값", justify="right")

    if NVML_AVAILABLE:
        t.add_row("GPU", snap.get("gpu_name", "N/A"))
        t.add_row("VRAM 전체", f"{snap['vram_budget_mb']:,} MB")
        t.add_row("VRAM 사용 중", f"{snap['vram_mb']:,} MB")
        t.add_row("VRAM 여유", f"{snap['vram_budget_mb'] - snap['vram_mb']:,} MB")
        t.add_row("GPU 사용률", f"{snap['gpu_util']} %")
        t.add_row("GPU 온도", f"{snap['gpu_temp']} °C")
        t.add_row("GPU 전력", f"{snap['gpu_power_w']:.1f} W")
    else:
        t.add_row("GPU", "[dim]nvidia-ml-py 미연결[/]")

    t.add_row("", "")
    t.add_row("RAM 전체", f"{vm.total // 1024 // 1024:,} MB")
    t.add_row("RAM 사용 중", f"{vm.used // 1024 // 1024:,} MB")
    t.add_row("RAM 여유", f"{vm.available // 1024 // 1024:,} MB")
    t.add_row("RAM 사용률", f"{vm.percent} %")
    t.add_row("CPU 코어 수", str(cpu_count))
    t.add_row("CPU 사용률", f"{psutil.cpu_percent(interval=0.5)} %")

    console.print(t)


# ── leaderboard ───────────────────────────────────────────────────────────────

@app.command()
def leaderboard(
    task: Optional[str] = typer.Option(None, "--task", "-t", help="특정 태스크 필터"),
):
    """전체 실험에서 모델별 누적 성능 리더보드."""
    from llmeval.report import show_leaderboard
    show_leaderboard(task)


# ── analyze ────────────────────────────────────────────────────────────────────

@app.command()
def analyze(
    run_id: str = typer.Argument("latest", help="run_id 또는 'latest'"),
    model: str = typer.Option("llama3.1:8b", "--model", "-m", help="분석에 사용할 모델"),
):
    """실험 결과 자동 분석 (LLM 기반 한국어 분석 텍스트 생성)."""
    from llmeval.store import get_latest_run_id
    from llmeval.analyzer import analyze as run_analyze

    if run_id == "latest":
        run_id = get_latest_run_id()
        if not run_id:
            console.print("[red]저장된 실험 없음[/]")
            raise typer.Exit(1)

    console.print(f"[bold cyan]자동 분석 생성 중...[/] (run: {run_id}, model: {model})")
    text = run_analyze(run_id, analyst_model=model)
    console.print()
    console.print(Panel(text, title="[bold]자동 분석", box=box.ROUNDED))


# ── runs ──────────────────────────────────────────────────────────────────────

@app.command()
def runs():
    """저장된 실험 목록 조회."""
    from llmeval.report import show_runs
    show_runs()


# ── dashboard ────────────────────────────────────────────────────────────────

@app.command()
def dashboard(
    host: str = typer.Option("127.0.0.1", "--host", help="바인드 주소"),
    port: int = typer.Option(8501, "--port", "-p", help="포트"),
):
    """웹 대시보드 실행 (FastAPI + HTMX)."""
    import uvicorn
    from llmeval.dashboard import app as dash_app
    console.print(f"[bold cyan]대시보드 시작:[/] http://{host}:{port}")
    uvicorn.run(dash_app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    app()
