#!/usr/bin/env python3
"""Validate NEoWave operationalization staging outputs.

The validator is intentionally small and file-based. It checks JSONL parsing,
primary ID duplication, curated reference integrity, AUTO_SRC normalization,
quote length policy, image-rights policy, and promotion/review split.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "neowave_dataset"
STAGING = DATASET / "02_staging"

PRIMARY_KEYS = {
    "sources.staging.jsonl": "source_id",
    "sources_automation.staging.jsonl": "source_id",
    "documents.staging.jsonl": "document_id",
    "image_candidates.staging.jsonl": "image_candidate_id",
    "pattern_candidates.staging.jsonl": "pattern_candidate_id",
    "rule_candidates.staging.jsonl": "rule_id",
    "monowave_definition_candidates.staging.jsonl": "definition_id",
    "swing_point_criteria_candidates.staging.jsonl": "criteria_id",
    "pattern_subtype_criteria.staging.jsonl": "subtype_id",
    "operational_rule_gaps.staging.jsonl": "gap_id",
    "labeled_examples_inventory.staging.jsonl": "example_id",
    "chart_image_rights_review.staging.jsonl": "image_candidate_id",
    "rejected_automation_sources.staging.jsonl": "rejected_id",
}

ALLOWED_NULL_GAP_TYPES = {
    "missing_input_data",
    "copyright_limited",
    "requires_human_expertise",
    "needs_backtest",
    "unsupported_by_public_sources",
    "needs_labeled_examples",
}

NEW_OPERATIONALIZATION_FILES = {
    "monowave_definition_candidates.staging.jsonl",
    "swing_point_criteria_candidates.staging.jsonl",
    "pattern_subtype_criteria.staging.jsonl",
    "operational_rule_gaps.staging.jsonl",
    "labeled_examples_inventory.staging.jsonl",
    "chart_image_rights_review.staging.jsonl",
}


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.exists():
        return rows, [f"{path}: missing file"]

    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path}:{line_no}: {exc}")
            continue
        if not isinstance(value, dict):
            errors.append(f"{path}:{line_no}: JSONL row is not an object")
            continue
        rows.append(value)
    return rows, errors


def read_json(path: Path) -> tuple[dict[str, Any], list[str]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - validation should report any parse failure
        return {}, [f"{path}: {exc}"]
    if not isinstance(value, dict):
        return {}, [f"{path}: JSON document is not an object"]
    return value, []


def quote_word_count(value: str) -> int:
    return len(re.findall(r"\b[\w'./%-]+\b", value))


def is_allowed_null_gap(row: dict[str, Any]) -> bool:
    return (
        row.get("needs_human_review") is True
        and bool(row.get("source_needed"))
        and bool(row.get("why_it_blocks_automation"))
        and row.get("gap_type") in ALLOWED_NULL_GAP_TYPES
    )


def row_id(row: dict[str, Any]) -> str:
    for key in (
        "definition_id",
        "criteria_id",
        "subtype_id",
        "gap_id",
        "example_id",
        "image_candidate_id",
        "rejected_id",
        "rule_id",
        "document_id",
        "pattern_candidate_id",
        "source_id",
    ):
        if row.get(key):
            return str(row[key])
    return "<unknown>"


def collect_auto_src_refs(paths: list[Path]) -> dict[str, list[str]]:
    refs: dict[str, list[str]] = {}
    excluded = {"sources_automation.staging.jsonl"}
    for path in paths:
        if path.name in excluded:
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for source_id in re.findall(r"AUTO_SRC\d{3}", line):
                refs.setdefault(source_id, []).append(f"{path.relative_to(ROOT)}:{line_no}")
    return refs


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    staging_paths = sorted(STAGING.glob("*.staging.jsonl"))
    rejected_path = DATASET / "99_rejected" / "rejected_automation_sources.staging.jsonl"
    paths = staging_paths + [rejected_path]

    records: dict[Path, list[dict[str, Any]]] = {}
    for path in paths:
        rows, path_errors = read_jsonl(path)
        records[path] = rows
        errors.extend(path_errors)

    manifest, manifest_errors = read_json(DATASET / "08_human_review" / "MANIFEST.json")
    errors.extend(manifest_errors)

    curated_sources, curated_source_errors = read_jsonl(DATASET / "03_postgres" / "sources.curated.jsonl")
    curated_rules, curated_rule_errors = read_jsonl(DATASET / "03_postgres" / "rules.curated.jsonl")
    curated_patterns, curated_pattern_errors = read_jsonl(DATASET / "03_postgres" / "patterns.curated.jsonl")
    errors.extend(curated_source_errors + curated_rule_errors + curated_pattern_errors)

    curated_source_ids = {row["source_id"] for row in curated_sources if row.get("source_id")}
    curated_rule_ids = {row["rule_id"] for row in curated_rules if row.get("rule_id")}
    curated_pattern_ids = {row["pattern_id"] for row in curated_patterns if row.get("pattern_id")}
    curated_urls = {row.get("url"): row.get("source_id") for row in curated_sources if row.get("url")}

    for path, rows in records.items():
        primary_key = PRIMARY_KEYS.get(path.name)
        if not primary_key:
            continue
        values = [row.get(primary_key) for row in rows if row.get(primary_key)]
        duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
        if duplicates:
            errors.append(f"{path}: duplicate {primary_key}: {duplicates}")
        missing = [idx for idx, row in enumerate(rows, 1) if not row.get(primary_key)]
        if missing:
            errors.append(f"{path}: missing {primary_key} on lines {missing}")

    auto_refs = collect_auto_src_refs(paths)
    manifest_sources = manifest.get("new_source_ids", {}) if manifest else {}
    manifest_auto_ids = set(manifest_sources)

    sources_auto_path = STAGING / "sources_automation.staging.jsonl"
    sources_auto = records.get(sources_auto_path, [])
    sources_auto_ids = {row.get("source_id") for row in sources_auto if row.get("source_id")}

    if manifest_auto_ids and sources_auto_ids != manifest_auto_ids:
        errors.append(
            "sources_automation.staging.jsonl AUTO_SRC coverage mismatch: "
            f"expected={sorted(manifest_auto_ids)} actual={sorted(sources_auto_ids)}"
        )

    for row in sources_auto:
        source_id = row.get("source_id")
        expected_used = source_id in auto_refs
        if row.get("used_in_staging_jsonl") is not expected_used:
            errors.append(
                f"{sources_auto_path}: {source_id} used_in_staging_jsonl="
                f"{row.get('used_in_staging_jsonl')} expected={expected_used}"
            )
        expected_status = "ready_for_curated_review" if expected_used else "manifest_only"
        if row.get("status") != expected_status:
            errors.append(f"{sources_auto_path}: {source_id} status={row.get('status')} expected={expected_status}")
        url = row.get("url")
        if url in curated_urls:
            warnings.append(f"{source_id}: URL duplicates curated source {curated_urls[url]}; reuse curated source_id")

    known_source_ids = curated_source_ids | sources_auto_ids
    for path, rows in records.items():
        if path.name == "sources_automation.staging.jsonl":
            continue
        for row in rows:
            source_id = row.get("source_id")
            if source_id and source_id not in known_source_ids:
                errors.append(f"{path}: {row_id(row)} references unknown source_id {source_id}")

            related_rule_id = row.get("related_rule_id")
            if related_rule_id and related_rule_id not in curated_rule_ids:
                errors.append(f"{path}: {row_id(row)} references unknown related_rule_id {related_rule_id}")

            related_pattern_id = row.get("related_pattern_id")
            if isinstance(related_pattern_id, str) and "/" not in related_pattern_id and related_pattern_id:
                if related_pattern_id not in curated_pattern_ids:
                    warnings.append(f"{path}: {row_id(row)} references non-curated related_pattern_id {related_pattern_id}")

            for list_key in ("related_rule_ids", "source_rule_ids", "rule_ids"):
                values = row.get(list_key)
                if isinstance(values, list):
                    for value in values:
                        if isinstance(value, str) and value.startswith("RULE") and value not in curated_rule_ids:
                            errors.append(f"{path}: {row_id(row)} references unknown {list_key} value {value}")

    allowed_null_quote_gaps: list[str] = []
    for path, rows in records.items():
        for line_no, row in enumerate(rows, 1):
            for quote_key in ("source_quote", "short_evidence_quote"):
                if quote_key not in row:
                    continue
                quote = row.get(quote_key)
                if quote is None or quote == "":
                    if path.name == "operational_rule_gaps.staging.jsonl" and quote_key == "source_quote":
                        if is_allowed_null_gap(row):
                            allowed_null_quote_gaps.append(str(row.get("gap_id")))
                            warnings.append(
                                f"{path}:{line_no}: allowed_null_source_quote_gap {row.get('gap_id')}"
                            )
                        else:
                            errors.append(f"{path}:{line_no}: disallowed null source_quote for {row_id(row)}")
                    else:
                        errors.append(f"{path}:{line_no}: null or empty {quote_key} for {row_id(row)}")
                    continue
                if quote_word_count(str(quote)) > 25:
                    errors.append(f"{path}:{line_no}: {quote_key} exceeds 25 words for {row_id(row)}")

    for row in records.get(STAGING / "chart_image_rights_review.staging.jsonl", []):
        if row.get("can_store_original_image") is True:
            errors.append(f"chart image {row_id(row)} allows original image storage")
        if row.get("can_store_metadata_only") is not True:
            errors.append(f"chart image {row_id(row)} does not allow metadata-only storage")
        if row.get("copyright_risk") in {"high", "unknown"} and row.get("needs_human_review") is not True:
            errors.append(f"chart image {row_id(row)} has copyright risk without human review")

    promote: dict[str, list[str]] = {}
    review: dict[str, list[str]] = {}
    for path, rows in records.items():
        if path.name not in NEW_OPERATIONALIZATION_FILES:
            continue
        for row in rows:
            target = review if row.get("needs_human_review") is True else promote
            target.setdefault(path.name, []).append(row_id(row))

    result = {
        "status": "failed" if errors else "passed",
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "sources_automation_rows": len(sources_auto),
            "auto_src_refs": {key: len(value) for key, value in sorted(auto_refs.items())},
            "manifest_auto_sources": sorted(manifest_auto_ids),
            "allowed_null_source_quote_gaps": sorted(set(allowed_null_quote_gaps)),
            "promotion_candidates": promote,
            "human_review_required": review,
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
