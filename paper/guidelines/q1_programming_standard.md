# 问题一编程与交付规范

本规范只覆盖当前分支保留的问题一正式流程。所有数值必须由程序从原始附件重建，不能从历史报告复制；运行产生的中间文件不进入正式交付目录。

## 1. 固定目录与入口

- `data/raw/`：题目 DOCX 和附件 1、2、3，只读，程序不得覆盖。
- `data/processed/`：只保留正式的 `q1_enterprise_features.csv` 和 `q1_risk_scores.csv`；审计、月度表、逐折预测和大型中间表写入被忽略的 `_runtime/`。
- `src/q1.py`：问题一统一入口；内部实现位于 `src/_internal/`。
- `src/q2.py`、`src/q3.py`：未实现占位，必须明确提示“尚未实现”并返回非零状态。
- `src/_internal/q1_config.yaml`：唯一正式配置文件。
- `outputs/q1/`：正式问一的最终文件、表格、图像和报告。
- `paper/`：方案、规格、论文材料、交付要求和框架图。

统一入口：

```text
python src/q1.py --stage all
python src/q1.py --stage final
python src/q1.py --stage validate
```

入口也接受 `--config`；未指定时使用 `src/_internal/q1_config.yaml`。配置路径通过环境变量传给内部阶段，所有相对路径均相对于项目根目录解析。

## 2. 原始数据保护

原始 DOCX/XLSX 只能以只读方式打开。禁止改写、另存、覆盖或在原始文件内清洗。正式流程开始前后都应记录原始文件 SHA-256；迁移前后哈希必须完全一致。原始文件只允许位于 `data/raw/`。

## 3. 运行文件与正式文件边界

运行日志和中间状态统一写入 `data/processed/_runtime/`，该目录被 `.gitignore` 忽略。不得把月度表、Parquet、逐折预测、完整诊断明细和临时缓存复制到正式目录。

正式文件范围如下：

| 类型 | 正式位置 |
|---|---|
| 企业特征 | `data/processed/q1_enterprise_features.csv` |
| 企业级OOF风险 | `data/processed/q1_risk_scores.csv` |
| EDA图 | `outputs/q1/figures/eda/` |
| 模型图 | `outputs/q1/figures/model/` |
| 信贷图 | `outputs/q1/figures/credit/` |
| 最终表格 | `outputs/q1/tables/` |
| 最终Excel、清单、验收 | `outputs/q1/final/` |
| 数据、模型和策略报告 | `outputs/q1/reports/` |
| 论文材料 | `paper/q1/` |

## 4. 固定参数与可复现性

随机种子固定为 `20260805`。正式风险模型仍为弹性网 Logistic：`saga`、`elasticnet`、`C=0.2`、`l1_ratio=0.8`；树模型仅作为对照。交叉验证固定为 5 折 × 10 重复，bootstrap 与经济情景参数全部从配置读取。

企业级 CSV 必须按 `enterprise_id` 稳定排序，UTF-8-SIG 编码、逗号分隔、无索引、缺失值为空。金额特征和额度使用万元；比例、风险值、流失率和接受概率使用 `[0,1]` 小数。

## 5. 数据与特征契约

1. 企业表主键为 `enterprise_id`，最终特征表恰好 123 行且主键唯一。
2. 发票先按既定规则审计、去重和原子化，再聚合到企业级；严禁发票级随机切分。
3. 21 个派生特征必须与 `paper/q1/q1_feature_dictionary.md` 一致。
4. `zero_amount_invoice_rate` 仅作审计，不得进入主模型或正式特征敏感性矩阵。
5. 标签、评级、企业名称和主键不得进入发票行为风险模型的数值 Pipeline。
6. 缺失值填补、缩尾、变换和标准化必须在训练折内拟合，不能先对全样本计算统计量。

## 6. 风险模型与输出

主模型使用企业级重复分层 OOF 预测，主指标包括 PR-AUC、Brier、LogLoss 和最高风险 20% 召回率，ROC-AUC、Balanced Accuracy、F1 与校准误差作辅助。最终 `q1_risk_scores.csv` 中必须存在 `selected_model=elastic_net_logistic` 和 `[0,1]` 范围内的 `selected_model_risk_score`。

模型正式表包括模型指标汇总、置信区间、配对 bootstrap、最终 Logistic 系数、系数稳定性、特征集敏感性和评级泄漏敏感性；逐折预测只保存在 `_runtime/model_training/`。

## 7. 信贷优化与验收

附件 3 的 A/B/C 流失曲线分别做递增保序拟合，只使用实际观测利率点。D 级企业额度必须为 0；已放贷企业额度为 10～100 万元、利率为 4%～15%。基准预算为 `0.50 × B_max = 4950` 万元，基准和全部 30 个正式敏感性情景都必须达到最优状态。

验收模块由 `src/_internal/validation.py` 实现，通过 `python src/q1.py --stage validate` 执行。验收至少检查：

- 123 家特征和风险记录唯一且风险值位于 `[0,1]`；
- OOF 测试次数为每家 10 次；
- 附件 3 审计通过、曲线单调且拟合值/接受概率合法；
- D 级零额度、额度/利率边界、收益分解恒等式和预算等式；
- 30 个情景最优、最终 Excel 可重读且包含 12 张工作表；
- 29 张正式 PNG 存在且非空；最终验收报告状态为 `PASS`。

## 8. 提交前检查

提交前应检查 `git status`、`git diff --cached --name-status` 和实际文件清单，确保每次提交只包含当前模块。禁止提交日志、缓存、`.vscode`、旧目录、旧比较脚本、问题二历史结果以及大型诊断明细。所有提交说明使用中文；最终推送后需分别核对新分支本地 SHA、远端 SHA、上游关系和旧 `feature/q1-q2` SHA。
