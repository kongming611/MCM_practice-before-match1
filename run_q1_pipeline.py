"""Run the question-one stages with fail-fast, shell-free subprocess calls."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STAGES = {
    "all": [
        "src/01_data_audit.py",
        "src/02_build_features.py",
        "src/03_validate_features.py",
        "src/04_exploratory_analysis.py",
        "src/05_train_risk_models.py",
        "src/06_fit_churn_curves.py",
        "src/07_optimize_credit_strategy.py",
        "src/08_credit_sensitivity_analysis.py",
        "src/09_validate_q1_final.py",
    ],
    "final": [
        "src/06_fit_churn_curves.py",
        "src/07_optimize_credit_strategy.py",
        "src/08_credit_sensitivity_analysis.py",
        "src/09_validate_q1_final.py",
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the formal question-one pipeline")
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    args = parser.parse_args()

    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"q1_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    started = datetime.now().isoformat(timespec="seconds")
    lines = [f"pipeline_stage={args.stage}", f"started={started}", f"python={sys.executable}", ""]
    overall_start = time.perf_counter()
    status = "PASS"
    for script in STAGES[args.stage]:
        command = [sys.executable, script]
        step_start = datetime.now().isoformat(timespec="seconds")
        timer = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
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
