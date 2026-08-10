# Q3 复现说明

本文档给出不依赖本机私有目录的重建顺序、命令、输入输出索引和故障门禁。Q3 运行会重建 `data/processed/q3_*` 与 `outputs/q3/*`，但按当前契约不写入 Q1/Q2 产物。

## 1. 环境

当前 manifest 记录的验证环境为：Python 3.10.20、NumPy 2.2.5、pandas 2.3.3、SciPy 1.15.3、PyYAML 6.0.2，Windows 平台。建议使用兼容的 Python 环境并安装 [requirements.txt](../../requirements.txt)；不要在文档中写入本机用户目录或专用解释器路径。

绘图后端为 Python/Matplotlib，图形格式由 Q3 阶段5的既有脚本生成；本阶段只引用现有图，不重画。

## 2. 阶段依赖与命令

从仓库根目录执行：

```text
python src/q3.py --stage preflight
python src/q3.py --stage classify
python src/q3.py --stage stress
python src/q3.py --stage optimize
python src/q3.py --stage analyze
```

阶段依赖为：

```text
Q2 manifest/artifacts
        ↓
preflight → classify → stress → optimize → sensitivity + figures + full validation
```

`analyze` 是完整链，会先执行前四个阶段，再生成敏感性表、稳定性表、三张图、图形 QA、`q3_full_validation.json` 和 `q3_run_manifest.json`。单独运行某一阶段时，入口也会先执行其前置契约；出现非零返回码应停止，不应跳过门禁。

## 3. 输入索引

| 输入 | 作用 |
|---|---|
| [data/processed/q2_risk_rating_scores.csv](../../data/processed/q2_risk_rating_scores.csv) | Q2风险、评级和不确定性字段 |
| [data/processed/q2_credit_strategy.csv](../../data/processed/q2_credit_strategy.csv) | Q2固定策略对照 |
| [outputs/q2/reports/q2_credit_optimization_manifest.json](../../outputs/q2/reports/q2_credit_optimization_manifest.json) | Q2优化结果和哈希来源 |
| [outputs/q2/reports/q2_risk_rating_run_manifest.json](../../outputs/q2/reports/q2_risk_rating_run_manifest.json) | Q2风险运行契约 |
| [src/_internal/q2_config.yaml](../../src/_internal/q2_config.yaml) | Q2风险配置 |
| [src/_internal/q2_optimization_config.yaml](../../src/_internal/q2_optimization_config.yaml) | Q2优化配置、LGD、资金成本和阈值 |
| [src/_internal/q3_config.yaml](../../src/_internal/q3_config.yaml) | Q3阶段路径、压力参数、敏感性网格和图形契约 |
| [国家统计局2020Q1数据](https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900688.html) | 分行业回顾性外部压力标尺 |

## 4. 输出索引

### 4.1 数据与表

- [q3_enterprise_industry_mapping.csv](../../data/processed/q3_enterprise_industry_mapping.csv)：302行行业规则审计。
- [q3_industry_mapping_review_evidence.csv](../../outputs/q3/tables/q3_industry_mapping_review_evidence.csv)：145家 unknown 的逐行人工复核证据队列；只记录规则证据和所需外部材料，不猜测真实行业。
- [q3_scenario_risk_scores.csv](../../data/processed/q3_scenario_risk_scores.csv)：逐企业四档场景、robust 风险和来源元数据。
- [q3_robust_credit_strategy.csv](../../data/processed/q3_robust_credit_strategy.csv)：robust 风险下的Q2优化适配结果。
- [q3_industry_shock_params.csv](../../outputs/q3/tables/q3_industry_shock_params.csv)：11个已知行业加unknown的压力参数。
- [q3_portfolio_scenario_summary.csv](../../outputs/q3/tables/q3_portfolio_scenario_summary.csv)：identity/light/medium/severe/robust组合摘要。
- [q3_fixed_q2_strategy_under_robust_risk.csv](../../outputs/q3/tables/q3_fixed_q2_strategy_under_robust_risk.csv)：固定Q2策略在robust风险下的模型隐含对照。
- [q3_industry_exposure_comparison.csv](../../outputs/q3/tables/q3_industry_exposure_comparison.csv)：行业额度、选中数和HHI。
- [q3_strategy_adjustments.csv](../../outputs/q3/tables/q3_strategy_adjustments.csv)：Q2到robust的企业级变化。
- [q3_sensitivity_summary.csv](../../outputs/q3/tables/q3_sensitivity_summary.csv)、[q3_sensitivity_enterprise_strategies.csv](../../outputs/q3/tables/q3_sensitivity_enterprise_strategies.csv)：13场景敏感性和企业级长表。
- [q3_enterprise_strategy_stability.csv](../../outputs/q3/tables/q3_enterprise_strategy_stability.csv)：排除探索性组合后的12场景选择频率标签。

### 4.2 Contracts、报告和图

- [q3_q2_baseline_contract.json](../../outputs/q3/reports/q3_q2_baseline_contract.json)：Q2 baseline preflight。
- [q3_industry_mapping_validation.json](../../outputs/q3/reports/q3_industry_mapping_validation.json)：行业映射验收。
- [q3_stress_validation.json](../../outputs/q3/reports/q3_stress_validation.json)：压力场景验收。
- [q3_robust_optimization_validation.json](../../outputs/q3/reports/q3_robust_optimization_validation.json)：Q2 MILP 复用验收。
- [q3_sensitivity_validation.json](../../outputs/q3/reports/q3_sensitivity_validation.json)：敏感性验收。
- [q3_figure_qa.json](../../outputs/q3/reports/q3_figure_qa.json)：三张图的自动和人工 QA 记录。
- [q3_full_validation.json](../../outputs/q3/reports/q3_full_validation.json)：全链状态、行数、哈希和门禁汇总。
- [q3_run_manifest.json](../../outputs/q3/reports/q3_run_manifest.json)：代码/配置/输入/输出 SHA-256、环境和确定性摘要。
- [q3_industry_shock_heatmap.pdf](../../outputs/q3/figures/q3_industry_shock_heatmap.pdf)、[q3_industry_allocation_shift.pdf](../../outputs/q3/figures/q3_industry_allocation_shift.pdf)、[q3_sensitivity_results.pdf](../../outputs/q3/figures/q3_sensitivity_results.pdf)：既有图形导出；同名 SVG/PNG/TIFF 也应存在。

## 5. 测试与重建后检查

```text
python -m unittest discover -v
python -m py_compile src/q3.py src/_internal/q3_baseline_preflight.py src/_internal/q3_industry_mapping.py src/_internal/q3_stress_scenarios.py src/_internal/q3_robust_optimization.py src/_internal/q3_sensitivity_analysis.py src/_internal/q3_figures.py
python src/q3.py --stage analyze
```

复现后至少确认：

1. unittest 全部通过（当前全仓阶段测试共31项）；
2. `q3_full_validation.json` 的 `status=PASS`，contracts 的 preflight/mapping/stress/optimization/sensitivity/figures 全为 `PASS`；
3. 302行、145 unknown、13行敏感性摘要、5个主场景和3张图均存在；
4. `q2_artifact_hashes_before` 与 `q2_artifact_hashes_after` 相等，Q2文件未被覆盖；
5. `q3_run_manifest.json` 中输入/输出哈希与重建文件一致，确定性摘要为真；
6. 图形 QA 无 FAIL，PDF/SVG/PNG/TIFF 文件非空。

## 6. 故障门禁

- **preflight 失败**：停止，不运行行业映射或压力覆盖；先修复 Q2 输入/manifest 连接。
- **mapping 失败**：停止，不把低特异性命中强行升级为行业；检查 unknown、冲突和 ID/名称连接。
- **stress 失败**：停止，不运行优化；检查官方增长值、百分点评分、positive-growth 中性和 unknown 最大策略。
- **optimization 失败**：停止，不发布额度结论；必须是每场 `optimal`、无 fallback、预算严格等式通过。
- **sensitivity/full validation 失败**：停止，不引用结果；13个场景必须可重算且 Q2 哈希不变。
- **figure QA 失败**：停止，不引用图；先检查来源表、图形脚本和导出质量。
- **任何自动门禁通过但人工确认项未完成**：不得把材料标为最终参赛稿，须在 [q3_submission_checklist.md](q3_submission_checklist.md) 保留未勾选状态。
- 人工确认时统一使用 [q3_manual_confirmation_packet.md](q3_manual_confirmation_packet.md)，参数、unknown策略、行业上限、外部资料与论文表述应逐项签字。
- Q3 的 AI 辅助范围和核验记录见 [q3_ai_usage_record.md](q3_ai_usage_record.md)，正式竞赛提交时须与其他章节记录合并。
