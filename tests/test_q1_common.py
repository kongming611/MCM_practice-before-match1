"""Fast unit tests for data-identity and feature helper rules."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from q1_common import collapse_invoice_lines, longest_consecutive_run, normalize_status, safe_ratio  # noqa: E402


def _config() -> dict:
    return {
        "processing": {
            "amount_unit_divisor": 10000,
            "amount_tolerance_yuan": 0.01,
            "status_aliases": {
                "valid": ["有效", "有效发票"],
                "void": ["作废", "作废发票"],
            },
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


class Q1CommonTests(unittest.TestCase):
    def test_status_normalization_strips_and_maps_aliases(self) -> None:
        raw, canonical = normalize_status(
            pd.Series(["有效发票", " 作废发票 ", "未知状态", None]),
            _config(),
        )
        self.assertEqual(raw.iloc[0], "有效发票")
        self.assertEqual(canonical.iloc[0], "有效")
        self.assertEqual(canonical.iloc[1], "作废")
        self.assertEqual(canonical.iloc[2], "未知")
        self.assertEqual(canonical.iloc[3], "未知")

    def test_safe_ratio_marks_zero_denominator_as_na(self) -> None:
        result = safe_ratio(pd.Series([2.0, 1.0]), pd.Series([4.0, 0.0]))
        self.assertAlmostEqual(result.iloc[0], 0.5)
        self.assertTrue(np.isnan(result.iloc[1]))

    def test_longest_run(self) -> None:
        self.assertEqual(longest_consecutive_run([False, True, True, False, True]), 2)
        self.assertEqual(longest_consecutive_run([False, False]), 0)

    def test_exact_duplicates_once_and_invoice_lines_sum(self) -> None:
        frame = pd.DataFrame(
            [
                ["E1", "I1", "2020-01-01", "V1", 10, 1, 11, "有效发票"],
                ["E1", "I1", "2020-01-01", "V1", 10, 1, 11, "有效发票"],
                ["E1", "I2", "2020-01-01", "V1", 2, 0, 2, "有效发票"],
                ["E1", "I2", "2020-01-01", "V1", 3, 0, 3, "有效发票"],
                ["E1", "I3", "2020-01-01", "V1", 0, 0, 0, "作废发票"],
            ],
            columns=[
                "企业代号",
                "发票号码",
                "开票日期",
                "销方单位代号",
                "金额",
                "税额",
                "价税合计",
                "发票状态",
            ],
        )
        ledger, diagnostics = collapse_invoice_lines(frame, "input", _config())
        self.assertEqual(diagnostics["exact_duplicate_extra_rows"], 1)
        self.assertEqual(len(ledger), 3)
        i2 = ledger.loc[ledger["invoice_number"].eq("I2"), "total_10k"].iloc[0]
        self.assertAlmostEqual(i2, 0.0005)
        self.assertEqual(int(ledger["is_void"].sum()), 1)


if __name__ == "__main__":
    unittest.main()
