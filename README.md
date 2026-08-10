# 中小微企业信贷决策项目

本分支按“原始数据—企业特征—OOF风险—流失率情景—信贷优化—验收”整理问题一正式流程，并在同一数据契约下提供问题二和问题三的独立入口。问题一、问题二、问题三的代码、产物和验收报告彼此分开；Q3 的压力参数是回顾性外部压力标尺，不是疫情因果识别或真实 PD。

## 目录

```text
data/
├── raw/                         # 题目 DOCX、附件1/2/3，只读
└── processed/
    ├── q1_enterprise_features.csv
    ├── q1_risk_scores.csv
    ├── q2_enterprise_features.csv
    ├── q2_risk_rating_scores.csv
    ├── q2_credit_strategy.csv
    ├── q3_enterprise_industry_mapping.csv
    ├── q3_scenario_risk_scores.csv
    └── q3_robust_credit_strategy.csv
src/
├── q1.py                        # 问题一统一入口
├── q2.py                        # 问题二统一入口
├── q3.py                        # 问题三统一入口
└── _internal/                   # Q1/Q2/Q3 内部实现与配置
outputs/q1/
├── final/                       # 最终Excel、清单、验收报告
├── tables/                      # 正式策略、模型和敏感性表
├── figures/{eda,model,credit}/  # 29张正式图片
└── reports/                     # 数据、模型、附件3和基准策略报告
outputs/q2/
├── tables/                      # Q2 风险、策略、组合和敏感性表
├── figures/                     # Q2 模型与优化图
└── reports/                     # Q2 manifest、验收和优化报告
outputs/q3/
├── tables/                      # 行业冲击、策略、组合、稳定性和敏感性表
├── figures/                     # 行业冲击、额度迁移和敏感性图（PDF/SVG/PNG/TIFF）
└── reports/                     # Q3 各阶段 contract、full validation 和 run manifest
paper/
├── project_plan.md
├── diagrams/
├── guidelines/
├── q1/
├── q2/
└── q3/
```

`data/processed/_runtime/` 只保存可重建的审计、月度表、逐折预测和临时日志，已加入忽略规则；原始文件不会被流程覆盖。

## 环境

建议使用 Python 3.12 或兼容版本，并安装锁定依赖：

```text
python -m pip install -r requirements.txt
```

绘图统一使用 Matplotlib，内部不再调用 PowerShell 绘图后端。

## 运行

从项目根目录执行：

```text
python src/q1.py --stage all
```

`all` 从 `data/raw/` 开始执行数据审计、特征构造、特征验收、EDA、风险建模、附件3曲线、基准信贷优化、30个正式敏感性情景和最终验收。阶段失败会立即返回非零状态。

已有正式特征和风险结果、只需重建最终信贷材料时执行：

```text
python src/q1.py --stage final
```

只检查当前正式结果时执行：

```text
python src/q1.py --stage validate
```

配置默认使用 `src/_internal/q1_config.yaml`，也可以显式指定：

```text
python src/q1.py --stage all --config src/_internal/q1_config.yaml
```

问题二入口（已有 Q2 产物时，可按需重建或验收）：

```text
python src/q2.py --stage all
python src/q2.py --stage model
python src/q2.py --stage validate
python src/q2.py --stage optimize
```

问题三按依赖顺序运行单阶段，或直接运行完整链：

```text
python src/q3.py --stage preflight
python src/q3.py --stage classify
python src/q3.py --stage stress
python src/q3.py --stage optimize
python src/q3.py --stage analyze
```

`analyze` 会依次执行 `preflight → classify → stress → optimize`，再完成敏感性分析、图形生成和全链验收；`q3.py` 不提供未实现的 `all`/`final`/`validate` 阶段。

## 问题一正式口径

- 输入为附件1的123家有信贷记录企业；企业级特征共21项。
- 主模型为弹性网 Logistic，随机种子为 `20260805`，采用5折×10重复分层交叉验证。
- 正式风险表使用企业级OOF结果，`selected_model_risk_score` 必须在 `[0,1]`。
- 信誉评级不进入发票行为风险模型，只用于附件3曲线匹配、D级业务约束和展示。
- A/B/C流失率按附件3实际观测利率点分别做递增保序拟合；客户流失率不是违约率。
- D级企业额度为0；已放贷额度为10～100万元，利率为4%～15%。
- 基准预算为 `0.50 × B_max = 4950` 万元；30个正式敏感性场景均要求最优。
- 风险、收益、信用损失、客户接受率和资金成本均按参数化情景解释，不宣称为真实一年期PD或真实银行利润。

## 问题三正式口径

- Q3 以 Q2 的302家企业、评级概率、D级阈值、流失曲线和 MILP 约束为基线，只替换 `main_risk_score`；Q2 原始数据和产物不被覆盖。
- 行业映射为确定性的名称关键词规则，11个粗行业之外保留 `unknown`。当前映射中有145家 `unknown`，不能把映射标签当作企业法定行业或因果变量。
- 行业压力来自国家统计局2020年一季度国内生产总值分行业同比数据（含2020年3月，2020-04-18发布、初步核算）：[官方来源](https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900688.html)。它是 Q2 发票窗口之后的回顾性外部压力标尺，不是 Q2 训练输入、事前验证或疫情因果估计。
- 设 `d_h=max(0,-g_h)`（百分点）、`S_h=clip(d_h/40,0,1)`，再按轻/中/重场景放大相对风险；`unknown` 主策略使用已知行业最大 `S_h`，P50 仅在敏感性场景中报告。所有 `c=40`、`rho=0.25`、场景倍数、LGD=0.50 和资金成本=0.03 均为建模假设或运行默认。
- Q3 风险仍是“历史发票行为相对违约倾向”的模型量；Q3 组合的收益、信用损失和额度迁移都是模型隐含的情景量，不是实际利润、实际损失或真实 PD。正式稳健策略为场景风险最大值；在当前单调场景设置下它恰好等于 severe，不构成额外独立证据。
- 名义预算严格等式为 `10000` 个 `10k CNY` 单位（1亿元），每家已放贷额度10～100万元；没有新增行业额度上限。

## 正式产物

核心数据：

- `data/processed/q1_enterprise_features.csv`：123家企业的正式特征表。
- `data/processed/q1_risk_scores.csv`：123家企业的最终OOF风险表。

正式交付：

- `outputs/q1/final/q1_final_delivery.xlsx`：当前12张工作表的最终Excel。
- `outputs/q1/final/q1_final_output_index.csv`：正式文件路径、大小和SHA-256索引。
- `outputs/q1/final/q1_final_manifest.json`：配置、输入、代码和输出哈希。
- `outputs/q1/final/q1_final_validation_report.md`：最终验收报告，成功时状态为 `PASS`。
- `outputs/q1/tables/`：企业策略、组合汇总、模型指标、流失率和敏感性汇总。
- `outputs/q1/figures/`：EDA、模型和信贷优化正式图片，共29张。
- `outputs/q1/reports/`：数据质量、特征验收、模型训练、附件3审计和基准策略报告。
- `paper/q1/`：特征字典、模型规格、模型结果、最终结果、假设局限和提交清单。

Q2 产物入口：

- `data/processed/q2_risk_rating_scores.csv`、`data/processed/q2_credit_strategy.csv`：Q2 风险和策略基线。
- `outputs/q2/reports/q2_credit_optimization_manifest.json`、`outputs/q2/reports/q2_credit_optimization_report.md`：Q2 优化契约和解释。
- `paper/q2/`：Q2 模型规格、结果和限制材料。

Q3 产物入口：

- `data/processed/q3_enterprise_industry_mapping.csv`：302家企业的行业映射及审计字段。
- `data/processed/q3_scenario_risk_scores.csv`：identity/light/medium/severe/robust 风险覆盖表。
- `data/processed/q3_robust_credit_strategy.csv`：稳健风险下的 Q2 MILP 复用结果。
- `outputs/q3/tables/`：行业参数、组合情景、固定 Q2 对照、策略调整、稳定性和敏感性表。
- `outputs/q3/figures/`：`q3_industry_shock_heatmap`、`q3_industry_allocation_shift`、`q3_sensitivity_results` 的四种格式导出。
- `outputs/q3/reports/q3_full_validation.json`、`outputs/q3/reports/q3_run_manifest.json`：全链 PASS 门禁、输入输出哈希和可复现环境记录。
- `paper/q3/`：模型规格、可写入论文的证据要点、假设局限、提交清单和复现说明。

最终Excel工作表为：`README`、`RiskScores`、`ChurnRaw`、`ChurnFitted`、`BaselineStrategy`、`BaselinePortfolio`、`BudgetSensitivity`、`ParameterSensitivity`、`StrategyStability`、`ModelMetrics`、`Assumptions`、`Validation`。

## 验收重点

`src/_internal/validation.py` 会检查：

1. 原始输入与配置能被读取，123家特征和风险记录唯一；
2. 每家企业恰好有10次OOF测试记录，风险值合法；
3. 附件3审计通过，拟合流失率和接受概率单调且位于 `[0,1]`；
4. 基准MILP最优，D级零额度，额度/利率边界和收益分解恒等式成立；
5. 30个敏感性场景均最优，策略稳定性表包含123家企业；
6. 最终Excel可重读、包含12张工作表、29张图片非空，输出索引哈希一致。

Q3 额外门禁：Q2 baseline preflight、行业映射、压力覆盖、Q2 优化复用、13场景敏感性和图形 QA 必须全部为 `PASS`；302行 ID/名称连接、145行 unknown 主策略、风险覆盖范围、严格预算等式、Q2 产物哈希不变和确定性重跑均需在当前 contract 中有证据。门禁通过只表示当前模型和文件契约成立，不等于外部违约验证或业务上线授权。

## 论文材料

总方案见 `paper/project_plan.md`，框架图见 `paper/diagrams/`，编程和建模交付要求见 `paper/guidelines/`；问题一、问题二和问题三的论文材料分别见 `paper/q1/`、`paper/q2/`、`paper/q3/`。论文材料是结构化证据和审稿辅助，不是未经人工确认即可提交的完整参赛论文。
