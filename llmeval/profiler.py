"""
HW Profiler: 추론 중 GPU/RAM/CPU를 백그라운드 스레드로 샘플링.
- VRAM: nvidia-smi subprocess (Ollama CUDA 할당을 정확히 반영)
- util/temp/power: pynvml (빠른 폴링)
- RAM/CPU: psutil
"""
import subprocess
import threading
import time
from typing import List, Dict

import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    _HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

VRAM_BUDGET_MB = 12282  # RTX 4070 Super 실측값


def _smi_vram_mb() -> int:
    """nvidia-smi로 VRAM 사용량 읽기. Ollama CUDA 할당을 정확히 반영."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=1,
        ).decode().strip()
        return int(out.split()[0])
    except Exception:
        return 0


def _read_gpu() -> dict:
    vram_mb = _smi_vram_mb()
    if not NVML_AVAILABLE:
        return {"vram_mb": vram_mb, "gpu_util": 0, "gpu_temp": 0, "gpu_power_w": 0.0}
    util = pynvml.nvmlDeviceGetUtilizationRates(_HANDLE)
    temp = pynvml.nvmlDeviceGetTemperature(_HANDLE, pynvml.NVML_TEMPERATURE_GPU)
    power_mw = pynvml.nvmlDeviceGetPowerUsage(_HANDLE)
    return {
        "vram_mb": vram_mb,
        "gpu_util": util.gpu,
        "gpu_temp": temp,
        "gpu_power_w": round(power_mw / 1000, 2),
    }


def _read_system() -> dict:
    vm = psutil.virtual_memory()
    return {
        "ram_mb": vm.used // (1024 * 1024),
        "cpu_util": psutil.cpu_percent(),
    }


class HWProfiler:
    def __init__(self, sample_interval_ms: int = 50):
        self._interval = sample_interval_ms / 1000
        self._samples: List[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_ts: float = 0

    def start(self) -> None:
        self._samples = []
        self._stop.clear()
        self._start_ts = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        return self._aggregate()

    def get_raw_samples(self) -> List[dict]:
        return list(self._samples)

    def _loop(self) -> None:
        while not self._stop.is_set():
            ts_ms = int((time.perf_counter() - self._start_ts) * 1000)
            gpu = _read_gpu()
            sys = _read_system()
            self._samples.append({"ts_ms": ts_ms, **gpu, **sys})
            time.sleep(self._interval)

    def _aggregate(self) -> dict:
        if not self._samples:
            return {
                "vram_peak_mb": 0, "vram_avg_mb": 0,
                "vram_budget_mb": VRAM_BUDGET_MB, "vram_headroom_mb": VRAM_BUDGET_MB,
                "gpu_util_avg_pct": 0.0, "gpu_temp_peak_c": 0,
                "gpu_power_avg_w": 0.0, "ram_peak_mb": 0,
                "cpu_util_avg_pct": 0.0, "spillover_detected": False,
                "throttle_detected": False, "sample_count": 0,
                "nvml_available": NVML_AVAILABLE,
            }

        vram = [s["vram_mb"] for s in self._samples]
        util = [s["gpu_util"] for s in self._samples]
        temp = [s["gpu_temp"] for s in self._samples]
        power = [s["gpu_power_w"] for s in self._samples]
        ram = [s["ram_mb"] for s in self._samples]
        cpu = [s["cpu_util"] for s in self._samples]

        vram_peak = max(vram)

        # 쓰로틀링 감지: 전반 20% t/s vs 후반 20% t/s 비교는 runner에서 처리
        # 여기서는 온도 기반으로만 체크
        throttle = max(temp) >= 83

        return {
            "vram_peak_mb": vram_peak,
            "vram_avg_mb": int(sum(vram) / len(vram)),
            "vram_budget_mb": VRAM_BUDGET_MB,
            "vram_headroom_mb": VRAM_BUDGET_MB - vram_peak,
            "gpu_util_avg_pct": round(sum(util) / len(util), 1),
            "gpu_temp_peak_c": max(temp),
            "gpu_power_avg_w": round(sum(power) / len(power), 1),
            "ram_peak_mb": max(ram),
            "cpu_util_avg_pct": round(sum(cpu) / len(cpu), 1),
            "spillover_detected": vram_peak > (VRAM_BUDGET_MB - 200),
            "throttle_detected": throttle,
            "sample_count": len(self._samples),
            "nvml_available": NVML_AVAILABLE,
        }


def snapshot() -> Dict:
    """현재 하드웨어 상태 즉시 스냅샷."""
    gpu = _read_gpu()
    sys = _read_system()
    return {
        "gpu_name": pynvml.nvmlDeviceGetName(_HANDLE) if NVML_AVAILABLE else "N/A",
        "vram_budget_mb": VRAM_BUDGET_MB,
        **gpu,
        **sys,
        "nvml_available": NVML_AVAILABLE,
    }
