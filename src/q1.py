"""问题一统一入口。

运行方式：
    python src/q1.py --stage all
    python src/q1.py --stage final
    python src/q1.py --stage validate
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "_internal" / "q1_config.yaml"
STAGES = {
    "all": [
        "src/_internal/stages/01_data_audit.py",
        "src/_internal/stages/02_build_features.py",
        "src/_internal/stages/03_validate_features.py",
        "src/_internal/stages/04_exploratory_analysis.py",
        "src/_internal/stages/05_train_risk_models.py",
        "src/_internal/stages/06_fit_churn_curves.py",
        "src/_internal/stages/07_optimize_credit_strategy.py",
        "src/_internal/stages/08_credit_sensitivity_analysis.py",
        "src/_internal/validation.py",
    ],
    "final": [
        "src/_internal/stages/06_fit_churn_curves.py",
        "src/_internal/stages/07_optimize_credit_strategy.py",
        "src/_internal/stages/08_credit_sensitivity_analysis.py",
        "src/_internal/validation.py",
    ],
    "validate": ["src/_internal/validation.py"],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行正式问题一流程")
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    config_path = args.config.resolve()
    runtime_log_dir = ROOT / "data" / "processed" / "_runtime" / "logs"
    runtime_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = runtime_log_dir / f"q1_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    started = datetime.now().isoformat(timespec="seconds")
    lines = [
        f"pipeline_stage={args.stage}",
        f"config={config_path}",
        f"started={started}",
        f"python={sys.executable}",
        "",
    ]
    environment = os.environ.copy()
    environment["Q1_CONFIG_PATH"] = str(config_path)
    source_path = str(ROOT / "src")
    environment["PYTHONPATH"] = source_path + os.pathsep + environment.get("PYTHONPATH", "")

    status = "PASS"
    overall_start = time.perf_counter()
    for script in STAGES[args.stage]:
        command = [sys.executable, str(ROOT / script)]
        step_start = datetime.now().isoformat(timespec="seconds")
        timer = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            env=environment,
            shell=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        elapsed = time.perf_counter() - timer
        step_status = "PASS" if completed.returncode == 0 else "FAIL"
        lines.extend([
            f"command={' '.join(command)}",
            f"step_started={step_start}",
            f"step_ended={datetime.now().isoformat(timespec='seconds')}",
            f"elapsed_seconds={elapsed:.6f}",
            f"status={step_status}",
            "stdout:",
            completed.stdout.rstrip(),
            "stderr:",
            completed.stderr.rstrip(),
            "",
        ])
        if completed.stdout:
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="" if completed.stderr.endswith("\n") else "\n")
        if completed.returncode != 0:
            status = "FAIL"
            break

    lines.extend([
        f"ended={datetime.now().isoformat(timespec='seconds')}",
        f"elapsed_total_seconds={time.perf_counter() - overall_start:.6f}",
        f"status={status}",
    ])
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"pipeline_log={log_path.relative_to(ROOT).as_posix()}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
