"""
구조화 출력 평가기: JSON 파싱 + 스키마 검증 + 필드 정확도.
"""
import json
import re
from typing import Any, Dict, List, Optional


def _extract_json(text: str) -> Optional[Any]:
    """Extract JSON from response text (handles markdown code blocks)."""
    # Try code block first
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if match:
        text = match.group(1).strip()

    # Try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try finding JSON object or array
    for pattern in [r'(\{[\s\S]*\})', r'(\[[\s\S]*\])']:
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
    return None


def _check_schema(data: Any, schema: dict) -> dict:
    """Simple JSON schema validation (type + required fields)."""
    errors = []
    expected_types = {
        "object": dict, "array": list, "string": str,
        "number": (int, float), "integer": int, "boolean": bool,
    }

    if "type" in schema:
        if schema["type"] in expected_types:
            if not isinstance(data, expected_types[schema["type"]]):
                errors.append(f"Expected type {schema['type']}, got {type(data).__name__}")
                return {"valid": False, "errors": errors}

    if isinstance(data, dict):
        for field in schema.get("required", []):
            if field not in data:
                errors.append(f"Missing required field: {field}")

        for field, field_schema in schema.get("properties", {}).items():
            if field in data and "type" in field_schema:
                if field_schema["type"] in expected_types:
                    if not isinstance(data[field], expected_types[field_schema["type"]]):
                        errors.append(f"Field '{field}' expected {field_schema['type']}")

    return {"valid": len(errors) == 0, "errors": errors}


def evaluate(
    response: str,
    reference: str = "",
    schema: Optional[dict] = None,
    required_fields: Optional[List[str]] = None,
    **kwargs,
) -> dict:
    parsed = _extract_json(response)
    parse_success = parsed is not None

    schema_valid = False
    schema_errors = []
    if parse_success and schema:
        result = _check_schema(parsed, schema)
        schema_valid = result["valid"]
        schema_errors = result["errors"]
    elif parse_success:
        schema_valid = True

    # Field accuracy
    field_score = 0.0
    field_details: Dict[str, bool] = {}
    if parse_success and required_fields:
        present = 0
        for field in required_fields:
            found = False
            if isinstance(parsed, dict):
                found = field in parsed and parsed[field] not in (None, "", [])
            field_details[field] = found
            if found:
                present += 1
        field_score = present / len(required_fields)
    elif parse_success:
        field_score = 1.0

    # Reference match
    ref_match = False
    if reference:
        ref_match = reference.lower().strip() in response.lower()

    # Composite score (0-5)
    parse_pts = 1.0 if parse_success else 0.0
    schema_pts = 1.0 if schema_valid else 0.0
    ref_pts = 1.0 if ref_match else 0.0
    composite = (parse_pts * 0.4 + schema_pts * 0.2 + field_score * 0.2 + ref_pts * 0.2) * 5.0

    return {
        "score": round(composite, 2),
        "parse_success": parse_success,
        "schema_valid": schema_valid,
        "schema_errors": schema_errors,
        "field_accuracy": round(field_score, 4),
        "field_details": field_details,
        "reference_match": ref_match,
    }
