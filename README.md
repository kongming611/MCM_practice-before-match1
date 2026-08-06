# 中小微企业信贷决策项目

本分支按“原始数据—企业特征—OOF风险—流失率情景—信贷优化—验收”整理问题一正式流程。当前只交付问题一；问题二、问题三保留入口占位，不把历史程序或历史验证结果混入新结构。

## 目录

```text
data/
├── raw/                         # 题目 DOCX、附件1/2/3，只读
└── processed/
    ├── q1_enterprise_features.csv
    └── q1_risk_scores.csv
src/
├── q1.py                        # 问题一统一入口
├── q2.py                        # 尚未实现占位
├── q3.py                        # 尚未实现占位
└── _internal/                   # 问题一内部实现与配置
outputs/q1/
├── final/                       # 最终Excel、清单、验收报告
├── tables/                      # 正式策略、模型和敏感性表
├── figures/{eda,model,credit}/  # 29张正式图片
└── reports/                     # 数据、模型、附件3和基准策略报告
paper/
├── project_plan.md
├── diagrams/
├── guidelines/
└── q1/
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

问题二、问题三目前会明确提示尚未实现并返回非零状态：

```text
python src/q2.py
python src/q3.py
```

## 问题一正式口径

- 输入为附件1的123家有信贷记录企业；企业级特征共21项。
- 主模型为弹性网 Logistic，随机种子为 `20260805`，采用5折×10重复分层交叉验证。
- 正式风险表使用企业级OOF结果，`selected_model_risk_score` 必须在 `[0,1]`。
- 信誉评级不进入发票行为风险模型，只用于附件3曲线匹配、D级业务约束和展示。
- A/B/C流失率按附件3实际观测利率点分别做递增保序拟合；客户流失率不是违约率。
- D级企业额度为0；已放贷额度为10～100万元，利率为4%～15%。
- 基准预算为 `0.50 × B_max = 4950` 万元；30个正式敏感性场景均要求最优。
- 风险、收益、信用损失、客户接受率和资金成本均按参数化情景解释，不宣称为真实一年期PD或真实银行利润。

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

最终Excel工作表为：`README`、`RiskScores`、`ChurnRaw`、`ChurnFitted`、`BaselineStrategy`、`BaselinePortfolio`、`BudgetSensitivity`、`ParameterSensitivity`、`StrategyStability`、`ModelMetrics`、`Assumptions`、`Validation`。

## 验收重点

`src/_internal/validation.py` 会检查：

1. 原始输入与配置能被读取，123家特征和风险记录唯一；
2. 每家企业恰好有10次OOF测试记录，风险值合法；
3. 附件3审计通过，拟合流失率和接受概率单调且位于 `[0,1]`；
4. 基准MILP最优，D级零额度，额度/利率边界和收益分解恒等式成立；
5. 30个敏感性场景均最优，策略稳定性表包含123家企业；
6. 最终Excel可重读、包含12张工作表、29张图片非空，输出索引哈希一致。

## 论文材料

总方案见 `paper/project_plan.md`，框架图见 `paper/diagrams/`，编程和建模交付要求见 `paper/guidelines/`，问题一可直接用于论文的材料见 `paper/q1/`。问题二历史验证结果已删除，未来实现必须在新接口和新输出契约下重新生成。
