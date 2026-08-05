"""Contract tests for the enterprise-level feature builder."""

from __future__ import annotations

import sys
import unittest
import importlib
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from q1_common import collapse_invoice_lines  # noqa: E402
feature_module = importlib.import_module("02_build_features")
BOUNDED_FEATURES = feature_module.BOUNDED_FEATURES
FEATURE_NAMES = feature_module.FEATURE_NAMES
_build_monthly_panel = feature_module._build_monthly_panel
_aggregate_enterprise_features = feature_module._aggregate_enterprise_features


def _config() -> dict:
    return {
        "processing": {
            "amount_unit_divisor": 10000,
            "amount_tolerance_yuan": 0.01,
            "status_aliases": {"valid": ["有效", "有效发票"], "void": ["作废", "作废发票"]},
        },
        "input": {
            "required_columns": {
                "invoice": [
                    "企业代号",
                    "发票号码",
                    "开票日期",
                    "金额",
                    "税额",
                    "价税合计",
                    "发票状态",
                ]
            }
        },
    }


class FeatureContractTests(unittest.TestCase):
    def test_dictionary_has_21_features_and_bounded_subset(self) -> None:
        self.assertEqual(len(FEATURE_NAMES), 21)
        self.assertTrue(set(BOUNDED_FEATURES).issubset(FEATURE_NAMES))

    def test_enterprise_builder_keeps_negative_void_and_zero_statistics(self) -> None:
        info = pd.DataFrame(
            {
                "enterprise_id": ["E1", "E2"],
                "enterprise_name": ["one", "two"],
                "credit_rating": ["A", "B"],
                "default_label": [0, 1],
            }
        )
        input_raw = pd.DataFrame(
            [
                ["E1", "I1", "2020-01-01", "V1", 100, 0, 100, "有效发票"],
                ["E1", "I2", "2020-02-01", "V1", -10, 0, -10, "有效发票"],
                ["E1", "I3", "2020-02-01", "V1", 0, 0, 0, "作废发票"],
            ],
            columns=["企业代号", "发票号码", "开票日期", "销方单位代号", "金额", "税额", "价税合计", "发票状态"],
        )
        output_raw = pd.DataFrame(
            [
                ["E1", "O1", "2020-01-01", "C1", 200, 0, 200, "有效发票"],
                ["E1", "O2", "2020-02-01", "C1", -20, 0, -20, "有效发票"],
            ],
            columns=["企业代号", "发票号码", "开票日期", "购方单位代号", "金额", "税额", "价税合计", "发票状态"],
        )
        input_ledger, _ = collapse_invoice_lines(input_raw, "input", _config())
        output_ledger, _ = collapse_invoice_lines(output_raw, "output", _config())
        months = pd.period_range("2020-01", "2020-02", freq="M")
        monthly = _build_monthly_panel(info, input_ledger, output_ledger, months)
        result = _aggregate_enterprise_features(info, input_ledger, output_ledger, monthly, months)
        self.assertEqual(result.shape[0], 2)
        self.assertEqual(result["enterprise_id"].nunique(), 2)
        self.assertEqual(int(result.loc[result["enterprise_id"].eq("E1"), "audit_negative_invoice_count"].iloc[0]), 2)
        self.assertEqual(int(result.loc[result["enterprise_id"].eq("E1"), "audit_void_invoice_count"].iloc[0]), 1)
        self.assertEqual(
            result.loc[result["enterprise_id"].eq("E1"), "zero_amount_invoice_rate"].iloc[0],
            0,
        )
        self.assertTrue(
            pd.isna(result.loc[result["enterprise_id"].eq("E2"), "zero_amount_invoice_rate"].iloc[0])
        )
        self.assertTrue(pd.isna(result.loc[result["enterprise_id"].eq("E2"), "purchase_sales_ratio"]).iloc[0])


if __name__ == "__main__":
    unittest.main()
