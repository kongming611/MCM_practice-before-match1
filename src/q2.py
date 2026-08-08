"""问题二数据与模型统一入口。

运行方式：
    python src/q2.py --stage all
    python src/q2.py --stage audit
    python src/q2.py --stage features
    python src/q2.py --stage eda
    python src/q2.py --stage validate
    python src/q2.py --stage model

本入口先完成数据审计、问题一同口径特征构建、分布比较和OOD诊断，再运行锁定的风险、
评级和Label Spreading交叉检查模型。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = ROOT / "data" / "processed" / "_runtime" / "q2" / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

from _internal.q2_data import run_q2
from _internal.q2_credit_optimization import run_q2_optimization
from _internal.q2_delivery import run_q2_delivery
from _internal.q2_models import run_q2_models


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行问题二数据与模型流程")
    parser.add_argument(
        "--stage",
        choices=["all", "audit", "features", "eda", "validate", "model", "optimize", "deliver"],
        default="all",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--optimization-config", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        if args.stage == "deliver":
            return run_q2_delivery()
        if args.stage in {"all", "model", "optimize"}:
            run_q2(args.config, "all")
            model_status = run_q2_models(args.config)
            if model_status != 0:
                return model_status
            if args.stage in {"all", "optimize"}:
                optimization_status = run_q2_optimization(args.optimization_config)
                if optimization_status != 0:
                    return optimization_status
                return run_q2_delivery()
            return model_status
        return run_q2(args.config, args.stage)
    except Exception as exc:
        print(f"q2.py FAILED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
