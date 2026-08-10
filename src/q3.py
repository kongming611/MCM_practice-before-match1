"""Q3 thin entry point.

The implemented stages are the Q2 baseline preflight, deterministic
enterprise-industry mapping, and the industry-pressure risk overlay.

Run from the repository root::

    python src/q3.py --stage preflight
    python src/q3.py --stage classify
    python src/q3.py --stage stress
    python src/q3.py --stage optimize
    python src/q3.py --stage analyze
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _internal.q3_baseline_preflight import PreflightError, run_q3_preflight
from _internal.q3_industry_mapping import IndustryMappingError, run_q3_industry_mapping
from _internal.q3_stress_scenarios import StressScenarioError, run_q3_stress_scenarios
from _internal.q3_robust_optimization import RobustOptimizationError, run_q3_robust_optimization
from _internal.q3_sensitivity_analysis import (
    SensitivityAnalysisError,
    build_q3_full_validation,
    run_q3_sensitivity_analysis,
)
from _internal.q3_figures import FigureGenerationError, run_q3_figures


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the implemented Q3 stages")
    parser.add_argument("--stage", choices=["preflight", "classify", "stress", "optimize", "analyze"], default="preflight")
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        contract = run_q3_preflight(config_path=args.config)
        if args.stage == "preflight":
            print(
                "q3.py PASS: "
                f"Q2 baseline contract status={contract['status']} "
                "(stage=preflight)"
            )
            return 0
        mapping = run_q3_industry_mapping(
            config_path=args.config,
            preflight_contract=contract,
        )
        if args.stage == "classify":
            print(
                "q3.py PASS: "
                f"industry mapping status={mapping['status']} "
                f"unknown_count={mapping['counts']['unknown_count']} "
                "(stage=classify)"
            )
            return 0
        stress = run_q3_stress_scenarios(
            config_path=args.config,
            preflight_contract=contract,
            mapping_contract=mapping,
        )
        if args.stage == "stress":
            print(
                "q3.py PASS: "
                f"stress status={stress['status']} "
                f"unknown_count={stress['counts']['unknown_rows']} "
                "(stage=stress)"
            )
            return 0
        robust = run_q3_robust_optimization(
            config_path=args.config,
            preflight_contract=contract,
            mapping_contract=mapping,
            stress_contract=stress,
        )
        if args.stage == "optimize":
            print(
                "q3.py PASS: "
                f"robust optimization status={robust['status']} "
                f"scenarios={len(robust['scenario_summary'])} "
                "(stage=optimize)"
            )
            return 0
        sensitivity = run_q3_sensitivity_analysis(
            config_path=args.config,
            preflight_contract=contract,
            mapping_contract=mapping,
            stress_contract=stress,
            optimization_contract=robust,
        )
        figures = run_q3_figures(config_path=args.config, sensitivity_contract=sensitivity)
        full = build_q3_full_validation(
            None,
            args.config,
            preflight_contract=contract,
            mapping_contract=mapping,
            stress_contract=stress,
            optimization_contract=robust,
            sensitivity_contract=sensitivity,
            figure_contract=figures,
        )
    except (PreflightError, IndustryMappingError, StressScenarioError, RobustOptimizationError, SensitivityAnalysisError, FigureGenerationError) as exc:
        print(f"q3.py FAILED: {exc}")
        return 1
    print(
        "q3.py PASS: "
        f"full validation status={full['status']} "
        f"sensitivity_scenarios={sensitivity['scenario_count']} "
        f"figures={figures['figure_count']} "
        "(stage=analyze)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
