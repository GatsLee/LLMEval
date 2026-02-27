# LLMEval — 시스템 아키텍처

---

## 전체 구조

```
┌─────────────────────────────────────────────────────────────────────┐
│                          eval.py  (CLI 진입점)                       │
│                                                                     │
│   run │ report │ compare │ hardware │ analyze                       │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
    ┌───────▼──────┐ ┌──────▼──────┐ ┌─────▼──────────┐
    │    Runner    │ │  HW Profiler│ │  Auto Analyzer │
    │              │ │             │ │                │
    │ YAML 태스크  │ │ pynvml      │ │ ollama         │
    │ 로드 + 실행  │ │ psutil      │ │ (judge 모델)   │
    │ (Ollama API) │ │ 10ms 샘플링 │ │ 분석 텍스트    │
    └───────┬──────┘ └──────┬──────┘ │ 생성           │
            │               │        └─────┬──────────┘
            │               │              │
            └───────┬───────┘              │
                    │                      │
           ┌────────▼──────────────────────▼──────┐
           │            Evaluators                 │
           │                                       │
           │  ┌─────────────┐  ┌───────────────┐  │
           │  │ exact_match │  │     rouge     │  │
           │  └─────────────┘  └───────────────┘  │
           │  ┌──────────────────────────────────┐ │
           │  │          llm_judge               │ │
           │  │  (faithfulness / fluency / 등)   │ │
           │  └──────────────────────────────────┘ │
           └──────────────────┬────────────────────┘
                              │
                   ┌──────────▼──────────┐
                   │    Results Store     │
                   │    (SQLite)          │
                   │                      │
                   │  runs                │
                   │  results             │
                   │  hw_samples (시계열) │
                   └──────────┬──────────┘
                              │
                   ┌──────────▼──────────┐
                   │   Report Engine      │
                   │                      │
                   │  Rich CLI 테이블     │
                   │  CSV / HTML export   │
                   └─────────────────────┘
```

---

## 컴포넌트별 상세 설계

---

### 1. Runner (`llmeval/runner.py`)

```
YAML 파일 로드
      │
      ▼
TaskDefinition 파싱
  ├── name, type, evaluator
  ├── inputs: [{context, question, reference}, ...]
  └── judge_criteria

      │
      ▼
모델 목록 순회
  └── for model in models:
        │
        ├── HW Profiler.start()       ← 백그라운드 샘플링 시작
        │
        ├── for input in task.inputs:
        │     ├── 프롬프트 렌더링 (Jinja2 템플릿)
        │     ├── ollama.chat(model, prompt)
        │     │     ├── TTFT 측정 (첫 토큰 도착 시간)
        │     │     └── 스트리밍 토큰 카운트 → t/s 계산
        │     └── 응답 + 메타데이터 수집
        │
        └── HW Profiler.stop()        ← 샘플 집계 반환
              └── {vram_peak, gpu_util_avg, gpu_temp_peak, ...}
```

**Ollama API 호출 방식:**
```python
import ollama, time

start = time.perf_counter()
first_token_time = None
tokens = []

for chunk in ollama.chat(model=model, messages=messages, stream=True):
    if first_token_time is None:
        first_token_time = time.perf_counter() - start
    tokens.append(chunk['message']['content'])

total_time = time.perf_counter() - start
tps = len(tokens) / total_time
ttft_ms = first_token_time * 1000
```

---

### 2. HW Profiler (`llmeval/profiler.py`)

```
Thread.start()
      │
      ▼ (10ms 간격 루프)
┌─────────────────────────────────┐
│  pynvml.nvmlDeviceGetMemoryInfo │ → vram_used_mb
│  pynvml.nvmlDeviceGetUtilization│ → gpu_util_pct, mem_util_pct
│  pynvml.nvmlDeviceGetTemperature│ → gpu_temp_c
│  pynvml.nvmlDeviceGetPowerUsage │ → gpu_power_mw → W 변환
│  psutil.virtual_memory()        │ → ram_used_mb, ram_available_mb
│  psutil.cpu_percent()           │ → cpu_util_pct
└─────────────────────────────────┘
      │
      ▼
samples[] 에 timestamp + 값 추가
      │
Thread.stop() 호출 시
      │
      ▼
집계 반환:
  {
    vram_peak_mb:    max(samples.vram_used_mb),
    vram_avg_mb:     mean(samples.vram_used_mb),
    vram_budget_mb:  12288,               ← RTX 4070 Super
    vram_headroom_mb: 12288 - vram_peak,
    gpu_util_avg_pct: mean(samples.gpu_util),
    gpu_temp_peak_c: max(samples.gpu_temp),
    gpu_power_avg_w: mean(samples.gpu_power_mw) / 1000,
    ram_peak_mb:     max(samples.ram_used),
    cpu_util_avg_pct: mean(samples.cpu_util),
    spillover_detected: vram_peak > 12288,
    throttle_detected:  (후반 t/s < 초반 t/s × 0.85)
  }
```

**스필오버 감지 로직:**
```
VRAM 초과 시 Ollama는 레이어를 RAM으로 오프로드.
→ RAM 사용량이 baseline (추론 전) 대비 급증.
→ 레이턴시가 비정상적으로 높아짐 (보통 3–5배).

감지 기준:
  vram_peak_mb > (device_vram_mb - 200)   ← 200MB 버퍼
  OR
  ram_peak_mb > ram_baseline_mb + 2048    ← RAM 2GB 이상 급증
```

**쓰로틀링 감지 로직:**
```
추론 중 t/s를 3구간으로 나눔 (초반 / 중반 / 후반).
후반 t/s < 초반 t/s × 0.85 이면 쓰로틀링으로 판정.

원인 분류:
  gpu_temp_peak > 83°C → "온도 쓰로틀링"
  gpu_power_avg > TDP × 0.95 → "전력 쓰로틀링"
  그 외 → "원인 불명"
```

---

### 3. Evaluators (`llmeval/evaluators/`)

#### exact_match.py
```
입력: response (str), reference (str)
처리: reference가 response에 포함되는지 (대소문자 무시)
출력: {score: 0 or 1, matched: bool}
```

#### rouge.py
```
입력: response (str), reference (str)
처리: rouge-score 라이브러리
출력: {rouge1: float, rouge2: float, rougeL: float}
```

#### llm_judge.py
```
입력: context, question, response, criteria (faithfulness|fluency|compliance)
      judge_model (기본: qwen2.5:7b)

프롬프트 템플릿 (Jinja2):
  "다음 답변을 {{ criteria }} 기준으로 평가하세요.
   점수: 1(매우 나쁨) ~ 5(매우 좋음)
   JSON으로만 응답: {\"score\": N, \"reason\": \"...\"}
   [컨텍스트] {{ context }}
   [질문] {{ question }}
   [답변] {{ response }}"

출력: {score: float (1–5), reason: str, raw: str}

일관성 보정:
  동일 입력 3회 실행 → 중앙값 사용 (이상치 제거)
```

**Judge criteria 목록:**

| criteria | 평가 기준 |
|---------|---------|
| `faithfulness` | 답변이 컨텍스트에 근거하는가? 없는 내용을 지어냈는가? |
| `fluency` | 한국어가 자연스러운가? 문법/어색함 없는가? |
| `compliance` | 지시사항을 정확히 따랐는가? (형식, 길이, 언어 등) |
| `correctness` | 정답과 의미적으로 일치하는가? |

---

### 4. Results Store (`llmeval/store.py`)

**SQLite 스키마:**

```sql
-- 실험 단위
CREATE TABLE runs (
    run_id      TEXT PRIMARY KEY,   -- uuid4
    created_at  TEXT,               -- ISO 8601
    description TEXT,               -- 실험 설명
    task_name   TEXT,               -- 태스크명
    models      TEXT,               -- JSON 배열
    hw_snapshot TEXT                -- 실험 시작 시 HW 스냅샷 JSON
);

-- 각 모델 × 입력 결과
CREATE TABLE results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT REFERENCES runs(run_id),
    model           TEXT,
    input_idx       INTEGER,        -- 태스크 inputs[i]
    prompt          TEXT,
    response        TEXT,
    score           REAL,           -- 최종 품질 점수
    score_detail    TEXT,           -- JSON (rouge1, rouge2, judge_reason 등)
    tps             REAL,           -- tokens/sec
    ttft_ms         REAL,           -- time to first token (ms)
    total_ms        REAL,           -- 전체 응답 시간 (ms)
    token_count     INTEGER,
    hw_summary      TEXT            -- JSON (HW Profiler 집계)
);

-- 하드웨어 시계열 (선택적 상세 저장)
CREATE TABLE hw_samples (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT REFERENCES runs(run_id),
    model       TEXT,
    ts_ms       INTEGER,            -- 실험 시작 기준 경과 ms
    vram_mb     INTEGER,
    gpu_util    INTEGER,            -- %
    gpu_temp    INTEGER,            -- °C
    gpu_power_w REAL,
    ram_mb      INTEGER,
    cpu_util    INTEGER             -- %
);
```

---

### 5. Auto Analyzer (`llmeval/analyzer.py`)

```
입력: run_id (SQLite에서 결과 로드)
      │
      ▼
비교 데이터 구성:
  {
    models: [...],
    scores: {model: avg_score},
    tps: {model: avg_tps},
    vram: {model: peak_mb},
    temp: {model: peak_c},
    spillover: {model: bool},
    throttle: {model: bool}
  }
      │
      ▼
분석 프롬프트 생성 (Jinja2):
  "다음은 {{ task_name }} 태스크에 대한 실험 결과입니다.
   타겟 하드웨어: RTX 4070 Super (12GB VRAM), DDR5 64GB, i5-14600K.
   [데이터] {{ comparison_table }}
   다음 항목을 한국어로 분석하세요:
   1. 품질/속도/VRAM 트레이드오프 요약
   2. 이 하드웨어에서 최적 모델 권장 (용도별)
   3. 이상 징후 (스필오버/쓰로틀링) 해석
   4. 다음 실험 제안"
      │
      ▼
ollama.chat(model=analyst_model, prompt=...)
      │
      ▼
분석 텍스트 출력 + 저장
```

---

### 6. Report Engine (`llmeval/report.py`)

**`eval.py report` 출력 레이아웃:**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Run: RAG 충실도 비교 v1    2026-03-15    Task: rag_faithfulness
 Hardware: RTX 4070 Super (12GB) │ DDR5 64GB │ i5-14600K
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 ┌ 품질 지표 ─────────────────────────────────────────┐
 │ 모델              Score   ROUGE-L  Consistency      │
 │ llama3.2:3b       3.2/5   0.61     ±0.3             │
 │ mistral:7b        3.8/5   0.74     ±0.2             │
 │ qwen2.5:7b        4.1/5   0.79     ±0.2    ★ 권장  │
 │ deepseek-r1:7b    4.3/5   0.82     ±0.1             │
 └───────────────────────────────────────────────────┘

 ┌ 하드웨어 지표 ─────────────────────────────────────────────────────┐
 │ 모델            t/s    TTFT    VRAM(MB)  여유(MB)  Temp  Power  ⚠   │
 │ llama3.2:3b     48.1   310ms   2,841     9,447     64°C  89W       │
 │ mistral:7b      31.4   480ms   4,503     7,785     68°C  132W      │
 │ qwen2.5:7b      27.9   520ms   4,823     7,465     71°C  143W  ★  │
 │ deepseek-r1:7b  19.2   680ms   5,112     7,176     74°C  156W      │
 └──────────────────────────────────────────────────────────────────┘

 ┌ 자동 분석 ─────────────────────────────────────────┐
 │ [LLM 생성 텍스트]                                   │
 │ RTX 4070 Super 기준 네 모델 모두 VRAM 안전 범위...  │
 └───────────────────────────────────────────────────┘
```

**`eval.py compare` 출력 레이아웃 (두 run 비교):**

```
 Run A: rag_faithfulness v1 (2026-03-10)
 Run B: rag_faithfulness v2 / 프롬프트 개선 (2026-03-15)

 ┌ 변화 요약 ──────────────────────────────────────────────┐
 │ 모델          Score A → B    t/s A → B    VRAM A → B     │
 │ qwen2.5:7b    3.9 → 4.1 ↑   27.4 → 27.9  4,801 → 4,823  │
 └─────────────────────────────────────────────────────────┘
```

---

## 데이터 흐름 요약

```
[YAML 태스크]
      │
      ▼
[Runner] ──────────────── [HW Profiler (Thread)]
  │  Ollama API 호출              │  pynvml + psutil
  │  스트리밍 응답 수신            │  10ms 샘플링
  │  TTFT / t/s 계산              │  집계 → HW Summary
  └─────────────┬─────────────────┘
                │
                ▼
          [Evaluators]
           exact_match
           rouge
           llm_judge (로컬 모델)
                │
                ▼
        [Results Store (SQLite)]
          runs / results / hw_samples
                │
         ┌──────┴──────┐
         │             │
  [Report Engine]  [Auto Analyzer]
   Rich 테이블       ollama (로컬)
   CSV/HTML          한국어 분석 텍스트
```

---

## 핵심 설계 결정 및 이유

| 결정 | 이유 |
|------|------|
| SQLite (vs JSON 파일) | 실험 간 비교 쿼리가 필요. `run_id`로 조인 가능. 의존성 없음. |
| pynvml (vs nvidia-smi 파싱) | subprocess보다 빠름, 10ms 샘플링 가능. Python 바인딩 공식 지원. |
| 백그라운드 스레드로 HW 샘플링 | Ollama 추론을 블로킹하지 않으면서 연속 샘플링. GIL 무관 (IO bound). |
| judge 모델 = 로컬 모델 | OpenAI API 비용/의존성 제거. 완전 오프라인 실험 가능. |
| judge 3회 실행 후 중앙값 | 로컬 모델의 judge 점수 분산이 크므로 안정화 필요. |
| Jinja2 프롬프트 템플릿 | 프롬프트를 코드에서 분리. 실험 없이 프롬프트만 교체 가능. |
| Rich (vs 일반 print) | 컬러 테이블/진행바로 가독성 확보. 터미널 출력이 포트폴리오 스크린샷이 됨. |

---

## 확장 경로 (3월 이후)

```
현재 MVP
    │
    ├── FastAPI + 간단한 웹 대시보드 (실험 히스토리 브라우저)
    │
    ├── PageNode 직접 연동
    │     실제 RAG 파이프라인 출력 → faithfulness 자동 평가 → 모델 교체 추천
    │
    ├── GUIDEBOOK 연동
    │     요약 품질 → korean_summarization 태스크로 자동 측정
    │
    └── CI 통합 (GitHub Actions)
          모델 업데이트 시 자동 실험 실행 → 품질 회귀 감지
```
