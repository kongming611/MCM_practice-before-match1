"""Deterministic, auditable industry mapping for Q3.

The Q2 target table contains masked enterprise names and no industry label.
Q3 therefore uses only explicitly registered high-specificity name keywords.
Unknown and ambiguous names are retained as ``unknown`` instead of being
forced into a category.  The implementation is deliberately rule based: it
does not inspect enterprise identifiers and does not call an LLM or a learned
classifier.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import yaml

try:  # Script entry points expose ``src`` on sys.path; tests import ``src``.
    from _internal.q3_baseline_preflight import PreflightError, run_q3_preflight
except ModuleNotFoundError:  # pragma: no cover - import-path compatibility
    from src._internal.q3_baseline_preflight import PreflightError, run_q3_preflight


INDUSTRY_CODES = (
    "agriculture",
    "manufacturing_industry",
    "construction",
    "wholesale_retail",
    "transport_storage_post",
    "accommodation_catering",
    "finance",
    "real_estate",
    "information_software_it",
    "leasing_business_services",
    "other_services",
    "unknown",
)

OUTPUT_COLUMNS = (
    "enterprise_id",
    "enterprise_name",
    "normalized_name",
    "industry_code",
    "industry_label",
    "matched_keyword",
    "mapping_status",
    "confidence",
    "mapping_source",
    "notes",
)

REVIEW_EVIDENCE_COLUMNS = (
    "review_priority",
    "enterprise_id",
    "enterprise_name",
    "current_industry_code",
    "mapping_status",
    "matched_keyword",
    "rule_notes",
    "required_external_evidence",
    "recommended_review_action",
)


class IndustryMappingError(RuntimeError):
    """Raised when Q3 industry mapping cannot pass its acceptance checks."""


def _repo_root_from_file() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _relative_path(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_name(value: Any) -> str:
    """Normalize a display name without adding information.

    NFKC normalizes full-width punctuation/digits, while whitespace is removed
    only to make keyword matching stable.  Masking characters and all Chinese
    characters are retained for auditability.
    """

    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    return re.sub(r"\s+", "", text)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise IndustryMappingError(f"Q3 config must be a mapping: {path}")
    return value


def _as_keyword_map(value: Any, *, field: str) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, dict):
        raise IndustryMappingError(f"q3_config.industry_mapping.{field} must be a mapping")
    result: dict[str, tuple[str, ...]] = {}
    for code, keywords in value.items():
        if code not in INDUSTRY_CODES:
            raise IndustryMappingError(f"unknown industry code in {field}: {code}")
        if isinstance(keywords, str):
            keywords = [keywords]
        if not isinstance(keywords, list) or not all(isinstance(item, str) for item in keywords):
            raise IndustryMappingError(f"keywords for {code} in {field} must be a string list")
        cleaned = tuple(dict.fromkeys(item.strip() for item in keywords if item.strip()))
        result[code] = cleaned
    return result


def _load_mapping_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    settings = config.get("industry_mapping")
    if not isinstance(settings, dict):
        raise IndustryMappingError("q3_config.industry_mapping is missing")
    allowed = settings.get("allowed_codes")
    if allowed is None:
        allowed = list(INDUSTRY_CODES)
    if tuple(allowed) != INDUSTRY_CODES:
        raise IndustryMappingError("industry_mapping.allowed_codes must match the Q3 canonical order")
    labels = settings.get("labels")
    if not isinstance(labels, dict) or set(labels) != set(INDUSTRY_CODES):
        raise IndustryMappingError("industry_mapping.labels must cover every canonical code")
    high = _as_keyword_map(settings.get("high_specific_keywords"), field="high_specific_keywords")
    low = _as_keyword_map(settings.get("low_specific_keywords", {}), field="low_specific_keywords")
    generic_regex = settings.get("generic_name_regex", r"^个体经营E[0-9]+$")
    try:
        generic_pattern = re.compile(str(generic_regex))
    except re.error as exc:
        raise IndustryMappingError(f"invalid generic_name_regex: {generic_regex!r}") from exc
    return {
        "input": settings.get("input", "data/processed/q2_risk_rating_scores.csv"),
        "output": settings.get("output", "data/processed/q3_enterprise_industry_mapping.csv"),
        "review_evidence_output": settings.get(
            "review_evidence_output",
            "outputs/q3/tables/q3_industry_mapping_review_evidence.csv",
        ),
        "validation_output": settings.get(
            "validation_output", "outputs/q3/reports/q3_industry_mapping_validation.json"
        ),
        "report_output": settings.get(
            "report_output", "outputs/q3/reports/q3_industry_mapping_validation.md"
        ),
        "generic_expected_count": int(settings.get("generic_expected_count", 56)),
        "generic_pattern": generic_pattern,
        "labels": {str(code): str(labels[code]) for code in INDUSTRY_CODES},
        "high": high,
        "low": low,
    }


def load_mapping_settings(config_path: Path | None = None) -> dict[str, Any]:
    """Load the validated stage-two rule settings for unit-level checks."""

    path = config_path or (_repo_root_from_file() / "src" / "_internal" / "q3_config.yaml")
    return _load_mapping_settings(_read_yaml(path))


def _keyword_hits(
    name: str,
    keyword_map: Mapping[str, tuple[str, ...]],
    *,
    include_unknown: bool = False,
) -> dict[str, tuple[str, ...]]:
    hits: dict[str, tuple[str, ...]] = {}
    for code in INDUSTRY_CODES:
        if code == "unknown" and not include_unknown:
            continue
        found = tuple(keyword for keyword in keyword_map.get(code, ()) if keyword in name)
        if found:
            hits[code] = found
    return hits


def classify_name(name: Any, settings: Mapping[str, Any]) -> dict[str, Any]:
    """Map one enterprise name using the registered deterministic rules."""

    normalized = normalize_name(name)
    labels = settings["labels"]
    generic_pattern = settings["generic_pattern"]
    if not normalized:
        return {
            "normalized_name": normalized,
            "industry_code": "unknown",
            "industry_label": labels["unknown"],
            "matched_keyword": "",
            "mapping_status": "empty_name_unknown",
            "confidence": 0.0,
            "mapping_source": "q3_name_keyword_rules_v1",
            "notes": "企业名称为空，需人工复核",
        }

    # Generic names are a declared hard rule.  The identifier suffix is never
    # parsed as an industry signal.
    if generic_pattern.fullmatch(normalized):
        return {
            "normalized_name": normalized,
            "industry_code": "unknown",
            "industry_label": labels["unknown"],
            "matched_keyword": "",
            "mapping_status": "generic_unknown",
            "confidence": 0.0,
            "mapping_source": "q3_name_keyword_rules_v1",
            "notes": "个体经营E编号不含行业证据，按阶段合同强制unknown",
        }

    high_hits = _keyword_hits(normalized, settings["high"])
    low_hits = _keyword_hits(normalized, settings["low"], include_unknown=True)
    matched = [f"{code}:{','.join(values)}" for code, values in sorted(high_hits.items())]
    if len(high_hits) == 1:
        code, values = next(iter(high_hits.items()))
        return {
            "normalized_name": normalized,
            "industry_code": code,
            "industry_label": labels[code],
            "matched_keyword": ";".join(values),
            "mapping_status": "high_specific_rule",
            "confidence": 1.0,
            "mapping_source": "q3_name_keyword_rules_v1",
            "notes": "单一行业命中高特异关键词",
        }
    if len(high_hits) > 1:
        return {
            "normalized_name": normalized,
            "industry_code": "unknown",
            "industry_label": labels["unknown"],
            "matched_keyword": ";".join(matched),
            "mapping_status": "ambiguous_unknown",
            "confidence": 0.0,
            "mapping_source": "q3_name_keyword_rules_v1",
            "notes": "多个行业命中高特异关键词，保守保留unknown",
        }

    low_matched = [f"{code}:{','.join(values)}" for code, values in sorted(low_hits.items())]
    return {
        "normalized_name": normalized,
        "industry_code": "unknown",
        "industry_label": labels["unknown"],
        "matched_keyword": ";".join(low_matched),
        "mapping_status": "low_specific_or_unmatched_unknown",
        "confidence": 0.0,
        "mapping_source": "q3_name_keyword_rules_v1",
        "notes": (
            "仅命中低特异关键词，未直接分类"
            if low_matched
            else "未命中高特异关键词，保留unknown"
        ),
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise IndustryMappingError(f"Q2 risk table does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "enterprise_id" not in rows[0] or "enterprise_name" not in rows[0]:
        raise IndustryMappingError("Q2 risk table must contain enterprise_id and enterprise_name")
    return rows


def _classify_rows(rows: list[dict[str, str]], settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        enterprise_id = str(row.get("enterprise_id", "")).strip()
        name = row.get("enterprise_name", "")
        result = classify_name(name, settings)
        output.append(
            {
                "enterprise_id": enterprise_id,
                "enterprise_name": "" if name is None else str(name),
                **result,
            }
        )
    return sorted(output, key=lambda item: item["enterprise_id"])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in OUTPUT_COLUMNS} for row in rows)


def _build_review_evidence(mapped_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a deterministic queue for the human industry-label review.

    The queue does not guess a replacement industry.  It records exactly why
    each unknown row needs external registration or business-scope evidence.
    """

    priority = {
        "ambiguous_unknown": 1,
        "empty_name_unknown": 1,
        "low_specific_or_unmatched_unknown": 2,
        "generic_unknown": 3,
    }
    action = {
        "ambiguous_unknown": "核对登记行业；多个高特异关键词冲突，禁止仅凭名称择一",
        "empty_name_unknown": "补充企业名称和登记行业后复核",
        "low_specific_or_unmatched_unknown": "核对登记行业或主营业务；低特异词不足以直接分类",
        "generic_unknown": "必须补充登记行业或主营业务；企业编号不含行业信息",
    }
    evidence: list[dict[str, Any]] = []
    for row in mapped_rows:
        if row["industry_code"] != "unknown":
            continue
        status = str(row["mapping_status"])
        evidence.append(
            {
                "review_priority": priority.get(status, 2),
                "enterprise_id": row["enterprise_id"],
                "enterprise_name": row["enterprise_name"],
                "current_industry_code": row["industry_code"],
                "mapping_status": status,
                "matched_keyword": row["matched_keyword"],
                "rule_notes": row["notes"],
                "required_external_evidence": "企业登记行业、经营范围或可核验的主营业务资料",
                "recommended_review_action": action.get(status, "补充外部行业证据后复核"),
            }
        )
    return sorted(
        evidence,
        key=lambda item: (int(item["review_priority"]), str(item["enterprise_id"])),
    )


def _write_review_evidence(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(REVIEW_EVIDENCE_COLUMNS),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            {column: row.get(column, "") for column in REVIEW_EVIDENCE_COLUMNS}
            for row in rows
        )


def _validate(
    q2_rows: list[dict[str, str]],
    mapped_rows: list[dict[str, Any]],
    settings: Mapping[str, Any],
    output_path: Path,
    repo_root: Path,
    repeat_rows: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    review_path: Path,
) -> dict[str, Any]:
    q2_ids = {str(row.get("enterprise_id", "")).strip() for row in q2_rows}
    q2_pairs = {
        (str(row.get("enterprise_id", "")).strip(), str(row.get("enterprise_name", "")))
        for row in q2_rows
    }
    mapped_ids = [str(row.get("enterprise_id", "")) for row in mapped_rows]
    mapped_pairs = {(row["enterprise_id"], row["enterprise_name"]) for row in mapped_rows}
    generic_pattern = settings["generic_pattern"]
    generic_rows = [
        row for row in mapped_rows if generic_pattern.fullmatch(row["normalized_name"])
    ]
    allowed_codes = set(INDUSTRY_CODES)
    checks = {
        "row_count_302": len(q2_rows) == 302 and len(mapped_rows) == 302,
        "q2_ids_unique": len(q2_ids) == len(q2_rows) and all(q2_ids),
        "mapped_ids_unique": len(mapped_ids) == len(set(mapped_ids)) and all(mapped_ids),
        "q2_id_name_set_match": q2_pairs == mapped_pairs,
        "allowed_industry_codes": all(row["industry_code"] in allowed_codes for row in mapped_rows),
        "no_enterprise_omitted": q2_ids == set(mapped_ids),
        "generic_count_56": len(generic_rows) == settings["generic_expected_count"],
        "generic_all_unknown": all(
            row["industry_code"] == "unknown" and row["mapping_status"] == "generic_unknown"
            for row in generic_rows
        ),
        "rule_deterministic": mapped_rows == repeat_rows,
        "output_columns_complete": set(OUTPUT_COLUMNS).issubset(mapped_rows[0] if mapped_rows else {}),
    }
    status_counts = Counter(str(row["mapping_status"]) for row in mapped_rows)
    industry_counts = Counter(str(row["industry_code"]) for row in mapped_rows)
    confidence_counts = Counter(str(row["confidence"]) for row in mapped_rows)
    unknown_count = int(industry_counts.get("unknown", 0))
    unknown_ids = {str(row["enterprise_id"]) for row in mapped_rows if row["industry_code"] == "unknown"}
    review_ids = [str(row["enterprise_id"]) for row in review_rows]
    checks.update(
        {
            "review_evidence_rows_equal_unknown": len(review_rows) == unknown_count,
            "review_evidence_ids_exact": set(review_ids) == unknown_ids and len(review_ids) == len(set(review_ids)),
            "review_evidence_columns_complete": set(REVIEW_EVIDENCE_COLUMNS).issubset(
                review_rows[0] if review_rows else {}
            ),
            "review_evidence_prioritized": all(
                int(row["review_priority"]) in {1, 2, 3} for row in review_rows
            ),
        }
    )
    contract = {
        "contract_version": "q3_industry_mapping_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "input": {
            "path": _relative_path(repo_root, _resolve_path(repo_root, settings["input"])),
            "row_count": len(q2_rows),
            "sha256": _sha256_bytes(
                _resolve_path(repo_root, settings["input"]).read_bytes()
            ),
        },
        "output": {
            "path": _relative_path(repo_root, output_path),
            "sha256": _sha256_bytes(output_path.read_bytes()) if output_path.is_file() else None,
            "row_count": len(mapped_rows),
        },
        "review_evidence": {
            "path": _relative_path(repo_root, review_path),
            "sha256": _sha256_bytes(review_path.read_bytes()) if review_path.is_file() else None,
            "row_count": len(review_rows),
            "priority_counts": dict(
                sorted(Counter(str(row["review_priority"]) for row in review_rows).items())
            ),
        },
        "checks": checks,
        "counts": {
            "industry_code": dict(sorted(industry_counts.items())),
            "mapping_status": dict(sorted(status_counts.items())),
            "confidence": dict(sorted(confidence_counts.items())),
            "unknown_count": unknown_count,
            "generic_count": len(generic_rows),
        },
        "manual_review_risk": {
            "unknown_rows_require_review": unknown_count,
            "unknown_is_intentional": True,
            "reason": "名称掩码、低特异关键词或多行业命中不能支持可靠的行业判断",
        },
        "rule_set": {
            "mapping_source": "q3_name_keyword_rules_v1",
            "enterprise_id_used_as_signal": False,
            "llm_used": False,
            "generic_regex": generic_pattern.pattern,
        },
    }
    return contract


def _write_report(path: Path, contract: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checks = contract["checks"]
    counts = contract["counts"]
    lines = [
        "# Q3 企业行业映射验收报告",
        "",
        f"最终状态：**{contract['status']}**",
        "",
        "## 验收检查",
        "",
        "| 检查 | 结果 |",
        "| --- | --- |",
    ]
    lines.extend(f"| {name} | {'PASS' if passed else 'FAIL'} |" for name, passed in checks.items())
    lines.extend(
        [
            "",
            "## 计数",
            "",
            f"- 行业代码计数：`{json.dumps(counts['industry_code'], ensure_ascii=False, sort_keys=True)}`",
            f"- 映射状态计数：`{json.dumps(counts['mapping_status'], ensure_ascii=False, sort_keys=True)}`",
            f"- 置信度计数：`{json.dumps(counts['confidence'], ensure_ascii=False, sort_keys=True)}`",
            f"- unknown 行数：**{counts['unknown_count']}**（其中 generic 行数：{counts['generic_count']}）",
            "",
            "## 边界与人工复核",
            "",
            "规则只使用企业名称中的已登记高特异关键词；企业编号没有作为特征，未使用 LLM。",
            "unknown 是有意保留的结果，低特异、多重命中、名称掩码和无命中行需人工复核，不能据此宣称真实行业归属。",
            f"逐行复核证据队列：`{contract['review_evidence']['path']}`；共 {contract['review_evidence']['row_count']} 行，已按冲突、低特异/无命中、泛化名称排序。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def run_q3_industry_mapping(
    repo_root: Path | None = None,
    config_path: Path | None = None,
    *,
    preflight_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run stage-two mapping after the Q2 baseline preflight gate."""

    root = (repo_root or _repo_root_from_file()).resolve()
    resolved_config = config_path or (root / "src" / "_internal" / "q3_config.yaml")
    config = _read_yaml(resolved_config)
    settings = _load_mapping_settings(config)
    if preflight_contract is None:
        try:
            preflight_contract = run_q3_preflight(root, resolved_config)
        except PreflightError as exc:
            raise IndustryMappingError(str(exc)) from exc
    if preflight_contract.get("status") != "PASS":
        raise IndustryMappingError("Q3 industry mapping requires a PASS Q2 baseline preflight")

    input_path = _resolve_path(root, settings["input"])
    baseline_artifact = (
        preflight_contract.get("artifact_hashes", {}).get("q2_risk_rating_scores", {})
        if isinstance(preflight_contract.get("artifact_hashes"), dict)
        else {}
    )
    baseline_path = baseline_artifact.get("path")
    if baseline_path:
        if _relative_path(root, input_path) != str(baseline_path).replace("\\", "/"):
            raise IndustryMappingError(
                "industry mapping input must be the Q2 risk table accepted by preflight"
            )
    else:
        raise IndustryMappingError("preflight contract lacks q2_risk_rating_scores artifact evidence")
    output_path = _resolve_path(root, settings["output"])
    review_path = _resolve_path(root, settings["review_evidence_output"])
    validation_path = _resolve_path(root, settings["validation_output"])
    report_path = _resolve_path(root, settings["report_output"])
    q2_rows = _read_rows(input_path)
    mapped_rows = _classify_rows(q2_rows, settings)
    # Serialize the same in-memory result in a stable order.  This check is
    # intentionally independent of the previous output file, so reruns cannot
    # inherit a stale PASS.
    repeat_rows = _classify_rows(q2_rows, settings)
    if mapped_rows != repeat_rows:
        raise IndustryMappingError("industry mapping rules are not deterministic")
    _write_csv(output_path, mapped_rows)
    review_rows = _build_review_evidence(mapped_rows)
    _write_review_evidence(review_path, review_rows)
    contract = _validate(
        q2_rows,
        mapped_rows,
        settings,
        output_path,
        root,
        repeat_rows,
        review_rows,
        review_path,
    )
    contract["preflight_status"] = preflight_contract.get("status")
    contract["config_path"] = _relative_path(root, resolved_config)
    contract["config_sha256"] = _sha256_bytes(resolved_config.read_bytes())
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_report(report_path, contract)
    if contract["status"] != "PASS":
        raise IndustryMappingError(
            f"Q3 industry mapping acceptance failed; see {_relative_path(root, validation_path)}"
        )
    return contract
