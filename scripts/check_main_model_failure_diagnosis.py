from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIR = PROJECT_ROOT / "validation_layer"
FORMAL_DIR = PROJECT_ROOT / ("formal" + "_layer")

DIAGNOSIS_MD_PATH = VALIDATION_DIR / "main_model_failure_diagnosis.md"
DIAGNOSIS_CSV_PATH = VALIDATION_DIR / "main_model_failure_diagnosis.csv"
RECOMMENDATION_PATH = VALIDATION_DIR / "main_model_repair_recommendation.json"
FORMAL_STATUS_PATH = FORMAL_DIR / "formal_status.md"
FORMAL_CANDIDATES_PATH = FORMAL_DIR / "formal_candidates.csv"

ALLOWED_REPAIR_IDS = {
    "repair_return_ranking_features",
    "repair_score_weighting",
    "redefine_return_target",
    "review_target_or_data_sufficiency",
    "ready_for_formal_review",
}

REQUIRED_DIAGNOSTIC_AREAS = {
    "score_band_ordering",
    "holdout_vs_market",
    "dev_holdout_drift",
    "head_diagnostics",
    "head_generalization",
    "feature_learnability",
    "return_ranking_probe",
    "score_weighting",
    "concentration",
    "return_failure_concentration",
}


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def require_path(path: Path) -> None:
    if not path.exists():
        fail(f"missing required diagnosis output: {path.relative_to(PROJECT_ROOT)}")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    for path in [DIAGNOSIS_MD_PATH, DIAGNOSIS_CSV_PATH, RECOMMENDATION_PATH]:
        require_path(path)

    rows = read_csv_rows(DIAGNOSIS_CSV_PATH)
    if not rows:
        fail("diagnosis csv must contain at least one row")
    if "diagnostic_area" not in rows[0]:
        fail("diagnosis csv must include diagnostic_area")
    found_areas = {row["diagnostic_area"] for row in rows}
    missing_areas = REQUIRED_DIAGNOSTIC_AREAS - found_areas
    if missing_areas:
        fail("diagnosis csv missing areas: " + ", ".join(sorted(missing_areas)))
    score_weighting_checks = {row.get("check_name", "") for row in rows if row.get("diagnostic_area") == "score_weighting"}
    for check_name in [
        "selected_weight_development_monthly_stability",
        "selected_weight_min_monthly_success_lift",
        "selected_weight_min_monthly_return_lift",
    ]:
        if check_name not in score_weighting_checks:
            fail(f"diagnosis csv missing score weighting check: {check_name}")

    recommendation = json.loads(RECOMMENDATION_PATH.read_text(encoding="utf-8"))
    repair_id = recommendation.get("recommended_repair_id")
    if repair_id not in ALLOWED_REPAIR_IDS:
        fail(f"recommended_repair_id must be one of {sorted(ALLOWED_REPAIR_IDS)}")
    alternatives = recommendation.get("alternative_repair_ids", [])
    if repair_id in alternatives:
        fail("recommended repair must not also appear as an alternative")
    if alternatives:
        fail("diagnosis must recommend one repair direction without alternatives")
    if recommendation.get("formal_outputs_unchanged") is not True:
        fail("recommendation must explicitly state formal outputs were unchanged")

    formal_hashes = recommendation.get("formal_hashes", {})
    expected_status_hash = formal_hashes.get("formal_status_sha256")
    expected_candidates_hash = formal_hashes.get("formal_candidates_sha256")
    if not expected_status_hash or not expected_candidates_hash:
        fail("recommendation must record formal output hashes")
    if expected_status_hash != sha256(FORMAL_STATUS_PATH):
        fail("formal_status.md hash does not match recommendation record")
    if expected_candidates_hash != sha256(FORMAL_CANDIDATES_PATH):
        fail("formal_candidates.csv hash does not match recommendation record")

    md = DIAGNOSIS_MD_PATH.read_text(encoding="utf-8")
    required_phrases = [
        "主模型",
        "唯一建議",
        "同日報酬排序根因",
        "整合分數權重根因",
        repair_id,
    ]
    for phrase in required_phrases:
        if phrase not in md:
            fail(f"diagnosis md missing phrase: {phrase}")

    print("OK: main model failure diagnosis contract passed")
    print(f"REPORT: {DIAGNOSIS_MD_PATH}")


if __name__ == "__main__":
    main()
