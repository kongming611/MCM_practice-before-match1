"""Contract tests for the validation/EDA handoff files."""

from pathlib import Path
import sys
import unittest
from importlib.util import module_from_spec, spec_from_file_location

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_validation_module():
    spec = spec_from_file_location("q1_validation", ROOT / "src" / "03_validate_features.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import validation module")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FeatureValidationContractTests(unittest.TestCase):
    def test_dictionary_has_21_main_names(self) -> None:
        module = _load_validation_module()
        names, _ = module._dictionary_names(module.DEFAULT_DICTIONARY)
        self.assertEqual(len(names), 21)
        features = pd.read_csv(ROOT / "results" / "features" / "enterprise_features_123.csv", encoding="utf-8-sig")
        self.assertTrue(set(names).issubset(features.columns))

    def test_feature_table_primary_key_contract(self) -> None:
        path = ROOT / "results" / "features" / "enterprise_features_123.csv"
        frame = pd.read_csv(path, encoding="utf-8-sig")
        self.assertEqual(frame.shape[0], 123)
        self.assertTrue(frame["enterprise_id"].is_unique)
        self.assertTrue(frame["default_label"].notna().all())
        self.assertTrue(frame["credit_rating"].notna().all())

    def test_eda_outputs_have_chart_index(self) -> None:
        index = pd.read_csv(ROOT / "results" / "feature_validation" / "eda_chart_index.csv", encoding="utf-8-sig")
        self.assertEqual(len(index), 7)
        for relative_path in index["file"]:
            self.assertTrue((ROOT / relative_path).exists(), relative_path)


if __name__ == "__main__":
    unittest.main()
