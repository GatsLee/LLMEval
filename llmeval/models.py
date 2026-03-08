from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class HWSummary(BaseModel):
    vram_peak_mb: int
    vram_avg_mb: int
    vram_budget_mb: int
    vram_headroom_mb: int
    gpu_util_avg_pct: float
    gpu_temp_peak_c: int
    gpu_power_avg_w: float
    ram_peak_mb: int
    cpu_util_avg_pct: float
    spillover_detected: bool
    throttle_detected: bool
    sample_count: int
    nvml_available: bool


class InferenceResult(BaseModel):
    model: str
    input_idx: int
    prompt: str
    response: str
    tps: float
    ttft_ms: float
    total_ms: float
    token_count: int
    hw_summary: Optional[HWSummary] = None
    score: Optional[float] = None
    score_detail: Optional[Dict[str, Any]] = None


class RunConfig(BaseModel):
    run_id: str
    description: str
    task_name: str
    task_type: str
    models: List[str]
    created_at: str
    ollama_options: Optional[Dict[str, Any]] = None
    judge: Optional[str] = None


class HWSample(BaseModel):
    ts_ms: int
    vram_mb: int
    gpu_util: int
    gpu_temp: int
    gpu_power_w: float
    ram_mb: int
    cpu_util: float
