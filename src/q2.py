"""问题二数据层统一入口。

运行方式：
    python src/q2.py --stage all
    python src/q2.py --stage audit
    python src/q2.py --stage features
    python src/q2.py --stage eda
    python src/q2.py --stage validate

本入口只完成数据审计、问题一同口径特征构建、分布比较、OOD诊断和自动验收，
不训练违约模型、评级模型或任何信贷优化模型。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _internal.q2_data import run_q2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行问题二数据层流程")
    parser.add_argument(
        "--stage",
        choices=["all", "audit", "features", "eda", "validate"],
        default="all",
    )
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        return run_q2(args.config, args.stage)
    except Exception as exc:
        print(f"q2.py FAILED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
