# Archived Tasks

Tasks moved here showed no meaningful differentiation between models.

## structured_output.yaml (구조화 출력 준수 평가)
- **Archived**: 2026-03-01
- **Reason**: All 8 LLM models scored exactly 1.00/5.00 — zero spread, zero standard deviation.
  The task is too easy; every model passes perfectly. No value for model comparison.
- **Original evaluator**: structured_output (JSON parse + schema validation)
- **Recommendation**: Replace with a harder structured output challenge if needed (the `structured_output_json.yaml` task already provides meaningful differentiation with spread=1.55).
