# LLMEval — 하드웨어 인식 로컬 LLM 평가 파이프라인

**목적:** Ollama 기반 로컬 LLM을 태스크 품질 + 실제 하드웨어 성능을 동시에 측정하여,
"내 GPU와 RAM 기준으로 최적의 모델과 설정은 무엇인가"를 수치로 결정한다.

**마감:** 2026년 3월 말
**타겟 하드웨어:** RTX 4070 Super (12GB VRAM) + DDR5 64GB + i5-14600K
**연결 프로젝트:** PageNode (RAG 모델 선택 근거), GUIDEBOOK (요약 품질 평가)

---

## 왜 하드웨어 메트릭이 필요한가

품질 점수만으로는 "어떤 모델을 쓸까"에 답할 수 없다.

| 질문 | 품질 점수만 있을 때 | 하드웨어 메트릭 포함 시 |
|------|-----------------|-------------------|
| Q4 vs Q8 중 뭘 써야 하나? | "Q8이 더 좋다" | "Q8은 품질 +8%, VRAM +3.1GB, 속도 -22% → Q5_K_M이 최적" |
| 12GB VRAM에 7B 모델 올릴 수 있나? | 모름 | "7.2GB 사용, 4.8GB 여유 → 컨텍스트 32K까지 안전" |
| 왜 갑자기 느려졌지? | 모름 | "VRAM 초과 → RAM 스필오버 → 레이턴시 3.2배 증가" |
| 장시간 추론 시 안정적인가? | 모름 | "30분 후 GPU 온도 83°C → 클럭 다운 → t/s -18%" |

---

## 측정 지표

### 품질 지표
| 지표 | 설명 | 평가 방식 |
|------|------|---------|
| Task Score | 태스크별 정확도/품질 | exact_match / rouge / llm_judge |
| Faithfulness | RAG 컨텍스트 이탈 여부 | llm_judge (1–5) |
| Korean Fluency | 한국어 자연성 | llm_judge (1–5) |
| Consistency | 동일 프롬프트 N회 분산 | std_dev(scores) |

### 하드웨어 지표 (추론 중 샘플링)
| 지표 | 출처 | 설명 |
|------|------|------|
| VRAM Used (MB) | pynvml | 피크 / 평균 |
| VRAM Available (MB) | pynvml | 12,288MB 기준 잔여량 |
| GPU Util (%) | pynvml | 추론 중 평균 GPU 연산 점유율 |
| GPU Temp (°C) | pynvml | 피크 온도 |
| GPU Power (W) | pynvml | 평균 소비 전력 |
| RAM Used (MB) | psutil | 피크 시스템 메모리 |
| CPU Util (%) | psutil | 추론 중 CPU 점유율 |
| Tokens/sec | Ollama API | 생성 속도 |
| TTFT (ms) | 측정 | Time To First Token |
| Total Latency (ms) | 측정 | 전체 응답 시간 |

### 교차 분석 (자동 생성)
- **VRAM 효율성:** `Task Score / VRAM Used` → 품질 대비 메모리 비용
- **전력 효율성:** `Tokens/sec / GPU Power` → 와트당 토큰
- **스필오버 감지:** `RAM Used > baseline + 500MB` → VRAM 초과 경고
- **쓰로틀링 감지:** `t/s[후반] < t/s[초반] × 0.85` → 온도 쓰로틀링 경고

---

## 핵심 기능 (MVP)

### 1. Task Registry (YAML)
```yaml
# tasks/rag_faithfulness.yaml
name: RAG 충실도 평가
type: qa_with_context
evaluator: llm_judge
judge_criteria: faithfulness
inputs:
  - context: "조선은 1392년 이성계가 건국하였다."
    question: "조선은 언제 건국되었나?"
    reference: "1392년"
```

### 2. Hardware Profiler (백그라운드 스레드)
추론 실행 중 10ms 간격으로 GPU/RAM/CPU 샘플링 → 시계열 저장 → 집계.

```python
# 샘플링 결과 예시
{
  "vram_peak_mb": 4823,
  "vram_avg_mb": 4651,
  "gpu_util_avg_pct": 94,
  "gpu_temp_peak_c": 71,
  "gpu_power_avg_w": 143,
  "ram_peak_mb": 8241,
  "cpu_util_avg_pct": 12,
  "spillover_detected": False
}
```

### 3. Runner
YAML 태스크 + 모델 목록 → Ollama API 호출 + HW 프로파일링 동시 실행.

### 4. Evaluators
- `exact_match` — 정답 포함 여부
- `rouge` — ROUGE-1/2/L
- `llm_judge` — 로컬 judge 모델이 채점 (외부 API 불필요)

### 5. Auto Analyzer
측정 결과를 받아 한국어 자동 분석 텍스트 생성 (LLM 사용).

```
[자동 분석 예시]
qwen2.5:7b (Q4_K_M) vs mistral:7b (Q4_K_M) 비교:

품질: qwen2.5:7b가 0.4점 높음 (4.1 vs 3.7, faithfulness 기준)
속도: qwen2.5:7b가 9 t/s 느림 (28 vs 37 t/s)
VRAM: qwen2.5:7b가 320MB 더 사용 (4823 vs 4503 MB)
GPU 온도: 두 모델 모두 안전 범위 (71°C vs 68°C)

→ RTX 4070 Super 기준 PageNode RAG 용도로는 qwen2.5:7b 권장.
  품질 차이가 유의미하고, VRAM 여유 7.4GB로 32K 컨텍스트 안전.
```

### 6. CLI + 리포트
```bash
$ python eval.py run --task rag_faithfulness --models llama3.2,mistral,qwen2.5
$ python eval.py report --run-id latest
$ python eval.py compare --run-id A --run-id B
$ python eval.py hardware          # 현재 하드웨어 상태 스냅샷
$ python eval.py analyze --run-id latest  # 자동 분석 텍스트 출력
```

---

## 평가 태스크 목록

### Phase 1 — MVP 태스크
| 태스크 | 타입 | 평가 | 연결 |
|--------|------|------|------|
| `rag_faithfulness` | QA with context | llm_judge | PageNode |
| `korean_summarization` | 요약 | rouge + llm_judge | GUIDEBOOK |
| `structured_output` | JSON 형식 준수 | regex | GUIDEBOOK 런북 |
| `instruction_following` | 지시 이행 | llm_judge | 일반 |

### Phase 2 — 하드웨어 특화 태스크
| 태스크 | 목적 |
|--------|------|
| `quantization_sweep` | Q4/Q5/Q8/FP16 품질 vs VRAM 트레이드오프 |
| `context_length_sweep` | 컨텍스트 길이 증가 시 VRAM/품질 변화 |
| `thermal_endurance` | 30분 연속 추론 시 온도/속도 변화 |
| `concurrent_load` | 다른 서비스와 동시 실행 시 성능 저하 측정 |

---

## 기술 스택

```
Python 3.11+
├── ollama              — 로컬 LLM 추론 (Python 클라이언트)
├── pynvml              — NVIDIA GPU 메트릭 (VRAM, 온도, 전력, 사용률)
├── psutil              — CPU / RAM 메트릭
├── typer               — CLI 인터페이스
├── rich                — 터미널 테이블, 진행바, 색상 출력
├── sqlite3 (built-in)  — 결과 및 시계열 저장
├── rouge-score         — ROUGE 계산
├── pyyaml              — 태스크 YAML 로드
└── jinja2              — 프롬프트 / 리포트 템플릿
```

외부 API 불필요. 완전 로컬 실행.

---

## 디렉토리 구조

```
02_LLMEval/
├── PLAN.md                        ← 이 파일
├── ARCHITECTURE.md                ← 시스템 설계도
├── README.md                      ← 완성 후 작성 (스크린샷 포함)
│
├── eval.py                        ← CLI 진입점 (typer)
│
├── llmeval/
│   ├── __init__.py
│   ├── runner.py                  ← Ollama 호출 + 태스크 실행 관리
│   ├── profiler.py                ← 하드웨어 프로파일러 (백그라운드 스레드)
│   ├── analyzer.py                ← 자동 분석 텍스트 생성 (LLM)
│   ├── evaluators/
│   │   ├── __init__.py
│   │   ├── exact_match.py
│   │   ├── rouge.py
│   │   └── llm_judge.py
│   ├── store.py                   ← SQLite 읽기/쓰기 (runs, results, hw_samples)
│   ├── report.py                  ← Rich 테이블/차트 렌더링
│   └── models.py                  ← Pydantic 데이터 모델
│
├── tasks/
│   ├── rag_faithfulness.yaml
│   ├── korean_summarization.yaml
│   ├── structured_output.yaml
│   ├── instruction_following.yaml
│   └── quantization_sweep.yaml
│
├── results/                       ← SQLite DB (.gitignore)
├── exports/                       ← CSV / HTML 리포트 출력
└── requirements.txt
```

---

## 개발 페이즈

### Phase 0: 스캐폴딩 + 하드웨어 연결 (1일) ✅
- [x] 디렉토리 구조 생성, `requirements.txt` 작성
- [x] `pynvml` 연결 — RTX 4070 Super VRAM/온도/전력 읽기 확인
- [x] `psutil` 연결 — RAM/CPU 읽기 확인
- [x] Ollama Python 클라이언트 — `ollama.chat()` 정상 동작 확인
- [x] SQLite 스키마 생성 (`runs`, `results`, `hw_samples` 테이블)

### Phase 1: 핵심 파이프라인 (3–4일) ✅
- [x] `profiler.py` — 백그라운드 스레드로 50ms 간격 샘플링, 집계 반환
- [x] `runner.py` — YAML 태스크 로드 → Ollama 호출 + 프로파일러 동시 실행
- [x] `exact_match.py`, `rouge.py` 구현
- [x] `store.py` — run/results/hw_samples 저장
- [x] `eval.py run` 커맨드 동작 확인 (태스크 1개, 모델 1개, HW 메트릭 포함)

### Phase 2: LLM Judge + Auto Analyzer (2–3일) ✅
- [x] `llm_judge.py` — faithfulness / fluency / correctness / compliance 기준 프롬프트 설계
- [x] judge 모델 3회 반복 → 중앙값 사용
- [x] `analyzer.py` — 비교 결과 → 한국어 자동 분석 텍스트 생성
- [x] 스필오버/쓰로틀링 자동 감지 로직

### Phase 3: CLI 리포트 (2일) ✅ (CSV/HTML export 제외)
- [x] `report.py` — Rich 테이블: 품질 점수 + HW 메트릭 통합 뷰
- [x] `eval.py compare` — run A vs B 나란히 비교
- [x] `eval.py hardware` — 현재 HW 스냅샷 출력
- [x] `eval.py analyze` — LLM 자동 분석 출력
- [x] `graph.py` — plotext 터미널 그래프 + matplotlib PNG export
- [ ] CSV / HTML export

### Phase 4: 하드웨어 특화 실험 (3–4일) ✅
- [x] `quantization_sweep.yaml` — 양자화 수준별 비교 태스크 생성
- [x] `context_length_sweep.yaml` — 컨텍스트 길이별 VRAM/품질 변화 태스크 생성
- [x] `thermal_endurance.yaml` — 장시간 추론 내구성 테스트 (20회 연속)
- [x] GPU 베이스라인 실험 실행 (llama3.1:8b, mistral:7b, qwen2.5:7b)
- [x] RAG 충실도, 한국어 요약, 구조화 출력, 지시 이행 전 태스크 3모델 비교 완료
- [x] matplotlib PNG 그래프 + 터미널 그래프 출력
- [x] LLM 자동 분석 텍스트 생성 검증
- [ ] PageNode 실제 청크로 `rag_faithfulness` 실험 실행 (PageNode Phase 1 이후)

### Phase 5: README + 공개 준비 (1–2일)
- [ ] README: 문제 정의 → 설계 결정 → 핵심 발견 (수치 포함)
- [ ] 터미널 출력 스크린샷
- [ ] 대표 실험 결과 1개 블로그 포스팅 원고로 정리
- [ ] GitHub 업로드

---

## 완성 기준 (Definition of Done)

다음 두 커맨드가 정상 동작하고:

```bash
python eval.py run \
  --task tasks/rag_faithfulness.yaml \
  --models llama3.2:3b,mistral:7b,qwen2.5:7b \
  --description "RAG 충실도 + HW 비교 v1"

python eval.py report --run-id latest
```

출력 테이블이 이 형태를 포함해야 한다:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Run: RAG 충실도 + HW 비교 v1  │  2026-03-XX  │  Task: rag_faithfulness │
├──────────────────┬───────┬────────┬─────────┬──────────┬───────────────┤
│ 모델             │ Score │ t/s    │ VRAM MB │ GPU Temp │ Spillover     │
├──────────────────┼───────┼────────┼─────────┼──────────┼───────────────┤
│ llama3.2:3b      │ 3.2   │ 48.1   │ 2,841   │ 64°C     │ No            │
│ mistral:7b       │ 3.8   │ 31.4   │ 4,503   │ 68°C     │ No            │
│ qwen2.5:7b       │ 4.1   │ 27.9   │ 4,823   │ 71°C     │ No            │
│ deepseek-r1:7b   │ 4.3   │ 19.2   │ 5,112   │ 74°C     │ No            │
└──────────────────┴───────┴────────┴─────────┴──────────┴───────────────┘

[자동 분석]
RTX 4070 Super (12GB) 기준: 네 모델 모두 VRAM 여유 있음 (최대 사용 5,112MB).
PageNode RAG 용도 권장: qwen2.5:7b — 품질/속도 균형 최적.
deepseek-r1:7b는 품질 최고이나 t/s가 2.5배 느림 → 배치 처리에는 비효율.
```
