# 问题二论文手交付总入口

本目录是问题二交给论文手的唯一入口。材料已按根目录 `README.md` 的 V1、V2、V3、最终版四阶段要求重新组织；模型数字全部来自当前分支锁定环境的正式重跑，不从历史计划或截图抄录。

## 0. 当前交付状态

| 项目 | 当前状态 | 证据 |
|---|---|---|
| 数据层 | `PASS`，原始数据审计为已披露的 `WARN` | `outputs/q2/reports/q2_validation_report.md`、`q2_data_quality_report.md` |
| 风险/评级/半监督交叉检查 | `PASS` | `outputs/q2/reports/q2_model_report_for_paper.md` |
| 1亿元信贷优化 | `optimal / PASS` | `outputs/q2/reports/q2_credit_optimization_report.md` |
| 24个主/敏感性情景 | 24/24 `optimal / PASS` | `outputs/q2/tables/q2_strategy_validation.csv` |
| 论文材料自动验收 | 运行 `python src/q2.py --stage deliver` 后以报告为准 | `outputs/q2/final/q2_delivery_validation_report.md` |
| 另一位建模同学人工复核与签字 | **待团队完成，不得代签** | `q2_submission_checklist.md` 末尾签字区 |

正式运行环境为 Python 3.12.13；`requirements.txt` 中八个直接依赖均按精确版本安装。默认 Python 3.11 不满足 `numpy==2.5.1` 的版本要求，不能作为正式复现环境。

## 1. 论文手阅读顺序

1. `q2_v1_problem_analysis.md`：回答“这一问要做什么、为什么选这条路线”。
2. `q2_v2_data_and_model_report.md`：给出字段、清洗、21项特征、假设、符号、可编辑公式、参数、模型和算法流程。
3. `q2_v3_results_and_validation_report.md`：给出完整结果入口、所有企业结果、图片与源数据、误差/合理性/灵敏度/稳健性证据及摘要数字。
4. `q2_final_paper_materials.md`：最终拼装页，提供可直接改写进论文的段落骨架、公式、核心表、图注、文献和关键数字追溯。
5. `q2_assumptions_and_limitations.md`：写作边界；出现口径冲突时以本文件和正式配置为准。
6. `q2_submission_checklist.md`：提交前逐项技术核对和人工复核签字。

`q2_model_spec.md`、`q2_model_results_for_paper.md` 和 `q2_final_results_for_paper.md` 保留为技术证据层，不再要求论文手从这些文件自行拼装完整章节。

## 2. README要求覆盖矩阵

| 根README要求 | 交付位置 |
|---|---|
| V1：题意、数据、输入输出、约束、与Q1关系 | `q2_v1_problem_analysis.md` 第1～5节 |
| V1：至少两种模型、优缺点、最终选择 | `q2_v1_problem_analysis.md` 第6～7节 |
| V1：完整步骤/流程图、风险 | `q2_v1_problem_analysis.md` 第8～9节 |
| V2：字段、缺失/异常/作废/负数、指标经济含义 | `q2_v2_data_and_model_report.md` 第1～4节 |
| V2：假设、符号、单位、完整公式、目标与约束 | `q2_v2_data_and_model_report.md` 第5～10节 |
| V2：参数、求解、流程图、阶段结果、候选比较 | `q2_v2_data_and_model_report.md` 第11～14节 |
| V3：完整表、302家风险/排序/策略 | `q2_v3_results_and_validation_report.md` 第1、5节及CSV附录 |
| V3：高清图、源数据、标题/坐标/单位/含义 | `q2_v3_results_and_validation_report.md` 第6节 |
| V3：结论、误差、合理性、灵敏度、稳健性、局限、摘要数字 | `q2_v3_results_and_validation_report.md` 第2～4、7～12节 |
| 最终版：可用文字、可编辑公式、符号、求解、表、图注 | `q2_final_paper_materials.md` 第1～8节 |
| 最终版：优缺点/改进、文献、关键数字代码位置 | `q2_final_paper_materials.md` 第9～11节 |
| 复现：代码、顺序、环境、参数、输出、图片 | 本文件第7节及根README“问题二正式口径” |
| 技术核对与人工签字 | `q2_submission_checklist.md` |

## 3. 论文中直接回答第二问的结论

在当前锁定主场景下，模型从302家无信贷记录企业中选择164家授信，名义授信总额严格为10000万元（1亿元）：128家各50万元、36家各100万元；所选年利率只出现9.85%、10.25%、10.65%、12.25%和14.25%。模型情景下预期实际发放额为2122.868113万元，情景预期净收益为103.824198万元。12家企业因 πD≥0.80 被拒贷，246家适用高不确定性额度上限。

上述数字必须同时带上三条限定：

- 302家没有真实违约标签，风险值是历史发票行为对应的相对违约倾向，不是经外部验证的一年期PD；
- 10000万元是严格名义授信额，不是实际发放额；实际发放还要乘客户接受概率；
- 净收益、损失和资金成本来自 LGD=0.50、资金成本率=0.03 等参数情景，不是已实现银行利润。

## 4. 正文、附录与文件分工

正文建议只放：

- 方法流程、关键公式和主场景参数；
- 123家代理验证的主指标；
- 302家分布迁移/不确定性摘要；
- 1亿元策略汇总、5个代表性企业和单因素敏感性；
- 4～6张最能回答题目的图。

附录直接挂接：

- 附录A：`outputs/q2/tables/q2_credit_strategy.csv`，302家完整策略；
- 附录B：`data/processed/q2_risk_rating_scores.csv`，302家风险、评级概率和不确定性；
- 附录C：`outputs/q2/tables/q2_model_metrics.csv`、`q2_rating_metrics.csv` 和 `q2_model_comparison_bootstrap.csv`；
- 附录D：`outputs/q2/tables/q2_optimization_sensitivity.csv`、`q2_budget_identity.csv` 和 `q2_strategy_validation.csv`；
- 附录E：`outputs/q2/final/q2_delivery_output_index.csv`，正式文件大小和SHA-256索引。

## 5. 不得混淆的五个概念

| 概念 | 正确写法 | 禁止写法 |
|---|---|---|
| 历史违约倾向 | 由附件1历史标签与发票行为学习的相对风险分数 | 302家真实违约概率/一年期PD |
| 评级概率 | 有序Logistic对A/B/C/D历史评级类别的概率分配 | 企业真实评级 |
| 客户流失率 | 附件3中评级—利率关系下的客户行为比例 | 违约率或LGD |
| 贷款接受概率 | A/B/C评级概率与非流失概率的混合 | `1-风险值` |
| 情景预期净收益 | 给定风险、LGD、资金成本和接受率假设的模型输出 | 已实现或保证利润 |

## 6. 关键正式文件

| 用途 | 文件 |
|---|---|
| 题目原文 | `data/raw/2020C-中小微企业的信贷决策.docx` |
| 统一入口 | `src/q2.py` |
| 数据/模型配置 | `src/_internal/q2_config.yaml` |
| 优化配置 | `src/_internal/q2_optimization_config.yaml` |
| 风险/评级实现 | `src/_internal/q2_models.py` |
| 概率评级MILP实现 | `src/_internal/q2_credit_optimization.py` |
| 302家风险与评级概率 | `data/processed/q2_risk_rating_scores.csv` |
| 302家完整策略 | `data/processed/q2_credit_strategy.csv` |
| 模型结果 | `outputs/q2/reports/q2_model_report_for_paper.md` |
| 优化结果 | `outputs/q2/reports/q2_credit_optimization_report.md` |
| 最终自动验收 | `outputs/q2/final/q2_delivery_validation_report.md` |

## 7. 复现命令

PowerShell示例：

```powershell
conda create --prefix .\.conda-q2-delivery python=3.12 pip -y
.\.conda-q2-delivery\python.exe -m pip install -r requirements.txt
.\.conda-q2-delivery\python.exe src\q2.py --stage all
.\.conda-q2-delivery\python.exe src\q2.py --stage deliver
```

`all` 正式运行约需数分钟，成功时依次出现 `q2_stage=all ... PASS`、`q2_model_stage=PASS`、`q2 optimization completed: status=optimal, validation=PASS` 和 `q2_delivery_stage=PASS`。只复核当前交付包时运行 `deliver`。

## 8. 人工复核边界

自动验收不能代替另一位建模同学复核。提交负责人之前，另一位建模同学必须核对公式与代码、参数与配置、企业编号与策略行、单位、图片与正文、Q1/Q2衔接，并在 `q2_submission_checklist.md` 填写姓名、时间和结论。当前仓库不会替任何人预先勾选或签字。
