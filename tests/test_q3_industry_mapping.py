from __future__ import annotations

import hashlib
import csv
import subprocess
import sys
import unittest
from pathlib import Path

from src._internal.q3_industry_mapping import (
    classify_name,
    load_mapping_settings,
    run_q3_industry_mapping,
)


ROOT = Path(__file__).resolve().parents[1]


class Q3IndustryMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.settings = load_mapping_settings()

    def test_high_specific_keyword_maps_directly(self) -> None:
        result = classify_name("测试装饰工程有限公司", self.settings)
        self.assertEqual(result["industry_code"], "construction")
        self.assertEqual(result["mapping_status"], "high_specific_rule")
        self.assertEqual(result["confidence"], 1.0)

    def test_property_service_maps_to_real_estate(self) -> None:
        for name in ["测试物业服务有限公司", "测试物业管理有限公司"]:
            with self.subTest(name=name):
                self.assertEqual(classify_name(name, self.settings)["industry_code"], "real_estate")

    def test_professional_technical_activity_maps_to_other_services(self) -> None:
        for name in [
            "测试工程检测有限公司",
            "测试测绘服务有限公司",
            "测试质量检验测试有限公司",
            "测试节能服务有限公司",
            "测试设计服务有限公司",
            "测试职业技能服务有限公司",
            "测试电器维护服务有限公司",
        ]:
            with self.subTest(name=name):
                self.assertEqual(classify_name(name, self.settings)["industry_code"], "other_services")

    def test_low_specific_keyword_is_conservative_unknown(self) -> None:
        result = classify_name("测试科技有限公司", self.settings)
        self.assertEqual(result["industry_code"], "unknown")
        self.assertEqual(result["mapping_status"], "low_specific_or_unmatched_unknown")
        self.assertIn("科技", result["matched_keyword"])

    def test_multiple_industry_hits_are_unknown(self) -> None:
        result = classify_name("测试建筑装饰工程商贸有限公司", self.settings)
        self.assertEqual(result["industry_code"], "unknown")
        self.assertEqual(result["mapping_status"], "ambiguous_unknown")
        self.assertIn("construction:", result["matched_keyword"])
        self.assertIn("wholesale_retail:", result["matched_keyword"])

    def test_generic_individual_name_is_forced_unknown(self) -> None:
        result = classify_name("个体经营E999", self.settings)
        self.assertEqual(result["industry_code"], "unknown")
        self.assertEqual(result["mapping_status"], "generic_unknown")
        self.assertEqual(result["confidence"], 0.0)

    def test_conservative_regressions_remain_unknown(self) -> None:
        for name in [
            "测试劳务有限公司",
            "测试服务部",
            "测试医疗器械有限公司",
            "测试医疗设备有限公司",
            "测试电子科技有限公司",
            "测试智能科技有限公司",
            "测试纺织品有限公司",
            "测试鞋材有限公司",
            "测试包装材料有限公司",
            "测试硬质合金有限公司",
            "测试不锈钢材料有限公司",
            "测试塑胶有限公司",
            "测试石材工艺品有限公司",
            "测试演艺设备有限公司",
            "测试生态魔芋有限公司",
        ]:
            with self.subTest(name=name):
                self.assertEqual(classify_name(name, self.settings)["industry_code"], "unknown")

    def test_building_labor_is_construction_without_generic_labor_ambiguity(self) -> None:
        result = classify_name("测试建筑劳务有限公司", self.settings)
        self.assertEqual(result["industry_code"], "construction")
        self.assertEqual(result["mapping_status"], "high_specific_rule")

    def test_full_stage_passes_and_repeats_deterministically(self) -> None:
        first = run_q3_industry_mapping()
        output = ROOT / "data/processed/q3_enterprise_industry_mapping.csv"
        review_output = ROOT / "outputs/q3/tables/q3_industry_mapping_review_evidence.csv"
        first_bytes = output.read_bytes()
        first_hash = hashlib.sha256(first_bytes).hexdigest()
        second = run_q3_industry_mapping()
        second_bytes = output.read_bytes()
        self.assertEqual(first["status"], "PASS")
        self.assertEqual(second["status"], "PASS")
        self.assertEqual(first["counts"]["generic_count"], 56)
        self.assertEqual(first["counts"]["unknown_count"], 145)
        self.assertEqual(first["review_evidence"]["row_count"], 145)
        with review_output.open("r", encoding="utf-8-sig", newline="") as handle:
            review_rows = list(csv.DictReader(handle))
        with output.open("r", encoding="utf-8-sig", newline="") as handle:
            mapped_rows = list(csv.DictReader(handle))
        self.assertEqual(len(review_rows), 145)
        self.assertEqual({row["enterprise_id"] for row in review_rows}, {
            row["enterprise_id"]
            for row in mapped_rows
            if row["industry_code"] == "unknown"
        })
        self.assertEqual(
            {row["review_priority"] for row in review_rows if row["mapping_status"] == "ambiguous_unknown"},
            {"1"},
        )
        self.assertEqual(first_hash, hashlib.sha256(second_bytes).hexdigest())
        self.assertEqual(first_bytes, second_bytes)

    def test_cli_classify_is_available(self) -> None:
        command = [sys.executable, str(ROOT / "src/q3.py"), "--stage", "classify"]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("stage=classify", completed.stdout)


if __name__ == "__main__":
    unittest.main()
