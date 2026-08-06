"""Audit attachment 3 and fit the independent monotone churn curves."""

from __future__ import annotations

import sys

from _internal.data_pipeline import load_config
from _internal.credit_strategy import read_attachment3, write_churn_outputs
import pandas as pd


def main() -> int:
    config = load_config()
    normalized, audit, meta = read_attachment3(config)
    paths = write_churn_outputs(config, normalized, audit, meta)
    crossings = int(pd.read_csv(paths["cross"])["fitted_crossing_flag"].sum())
    print(
        "06_fit_churn_curves.py SUCCEEDED: "
        f"rates={meta['observed_rate_count']} audit_pass={meta['audit_pass']} "
        f"crossings={crossings}"
    )
    for key in ("audit", "normalized", "fitted", "metrics", "cross", "manifest"):
        print(f"  {key}={paths[key].relative_to(paths[key].parents[2]).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
