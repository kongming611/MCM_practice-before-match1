# 问题一编程强制规范

本规范只覆盖问题一。未经数据审计、运行日志和输出契约验证，不得把任何数值交给论文手。

## 1. 推荐目录

- config/q1.yaml：路径、窗口、清洗、特征、模型、优化与敏感性参数。
- data/ 或 data/raw/：原始附件，只读；现有文件无需为本轮移动。
- src/：01_data_audit.py、02_build_features.py，后续脚本继续使用两位序号。
- tests/：数据口径、特征公式、Pipeline和输出契约测试。
- docs/：本轮四份规格文档。
- results/audit/：机器可读质量检查；results/intermediate/：可重建中间表。
- results/models/、results/tables/、results/figures/、results/reports/：模型、表、图、报告。
- logs/：逐次运行日志。

## 2. 脚本与代码命名

入口脚本采用“两位序号_动作_对象.py”，如 01_data_audit.py；模块、函数、变量用 snake_case，类用 PascalCase，常量用 UPPER_SNAKE_CASE。禁止把探索性 notebook 当正式流水线。

## 3. 原始数据保护

原始 xlsx 只读打开，禁止原地改名、改工作表、覆盖或另存。清洗结果只能写入 results/。运行清单必须记录输入文件大小、修改时间和 SHA-256。

## 4. 路径与输出位置

必须创建且只使用以下约定路径：

| 阶段 | 输出 |
|---|---|
| 审计明细 | results/audit/table_profile.csv、missing_values.csv、enterprise_coverage.csv、invoice_status_sign_summary.csv、date_range.csv、duplicate_key_issues.csv、enterprise_record_counts.csv、amount_anomalies.csv |
| 审计结论 | results/audit/q1_data_quality.json、results/reports/q1_data_quality_report.md |
| 清洗中间表 | results/intermediate/q1_clean_invoice_ledger.parquet、results/intermediate/q1_enterprise_monthly.parquet |
| 特征 | results/enterprise_features_123.csv、results/audit/q1_feature_quality.json、results/reports/q1_feature_quality_report.md |
| 风险模型（后续） | results/models/q1_elastic_net.joblib、results/tables/q1_cv_metrics.csv、results/q1_enterprise_risk.csv |
| 定价优化（后续） | results/tables/q1_churn_curve.csv、results/q1_credit_strategy.csv、results/tables/q1_budget_summary.csv |
| 图表（后续） | results/figures/fig_q1_01_pr_curve.png、fig_q1_02_roc_curve.png、fig_q1_03_calibration_curve.png、fig_q1_04_churn_curve.png、fig_q1_05_sensitivity.png |
| 复现清单 | results/reports/q1_run_manifest.json、logs/q1_{script}_{timestamp}.log |

禁止在项目根目录散落新结果文件。

## 5. 随机性

唯一全局随机种子为 20260805。Python random、NumPy、scikit-learn、交叉验证、bootstrap和求解器均显式设置；无法设种子的组件不得进入正式流程。

## 6. 路径不得硬编码

入口通过 pathlib 从项目根目录和 config/q1.yaml 解析相对路径；禁止出现盘符、个人用户名、固定工作目录或 E:/、D:/ 等绝对路径。

## 7. 参数集中配置

q1.yaml 至少包含：输入文件与工作表、必需字段、analysis_start/end、边界月规则、金额单位、重复键、金额核验容差、缩尾分位数、特征列表、交叉验证参数、模型参数、利率网格、budget、budget_mode、lgd、funding_cost、日志级别。脚本内只允许放不可变默认常量；运行时把最终配置副本写入复现清单。

## 8. 单位

原始金额按元读取；所有企业级金额特征和贷款额度统一为万元，字段名带 _10k 或 _10k_cny。比例和年利率用 \([0,1]\) 小数，不用百分数文本。输出不得混用元、万元和亿元。

## 9. 唯一主键

企业代号重命名为 enterprise_id，是所有企业级表的唯一主键；去除首尾空格但不得改写编号。企业名称不是主键。连接前后都验证基数和未匹配集合。

## 10. 数据切分

先完成发票原子化和企业聚合，再按 enterprise_id 做重复分层交叉验证。严禁发票级随机切分、同一企业跨训练/测试折或先看全体标签后手工分组。

## 11. Pipeline与折内拟合

缺失填补、缩尾阈值、对数变换、标准化、特征选择、概率校准和分类器必须封装在 Pipeline 或等价复合估计器中。所有含数据统计量的步骤只在训练折 fit，再对验证折 transform；禁止全样本预处理后交叉验证。

## 12. 预处理顺序

企业级原始特征表保持未变换值。模型Pipeline固定为：训练折中位数填补 → 训练折分位数缩尾 → 按字典执行 log1p/有符号log1p → 训练折标准化 → 弹性网Logistic。边界比例只校验不缩尾；标签、主键、名称、评级不进入数值Pipeline。

## 13. 类别不平衡

以分层交叉验证和PR-AUC为主；主模型先使用未重采样的惩罚似然，以保留概率含义。禁止在切分前做SMOTE或全局欠采样。若测试 class_weight 或重采样，只能在训练折内完成，使用同一折作配对比较，并重新检查Brier、LogLoss和校准；不得仅因准确率提高而采用。

## 14. 日志

日志使用UTF-8，同时写控制台和文件，记录：脚本/版本、开始结束时间、配置、随机种子、输入哈希、表行列数、每步保留/隔离数量、断言、警告、模型参数、软件版本和输出路径。不得记录整张敏感明细。

## 15. 异常与错误

禁止空 except 和静默填零。缺工作表/字段、主键重复、标签缺失、未知状态、金额不可解析或企业数不符时失败退出；可恢复异常写入质量报告并给出数量和样例键。任何删行、去重、隔离都必须有 reason_code 和前后行数。

## 16. 数据质量断言

至少断言：

1. 工作表及字段名与规格一致，企业表 enterprise_id 非空且唯一。
2. 发票企业均能匹配企业表；日期、金额、税额、价税合计可解析。
3. 发票状态只含已配置合法值；“金额+税额≈价税合计”的超差记录已报告。
4. 完全重复、复合发票键重复和同号元数据冲突分别统计。
5. 附件1标签只含“是/否”，映射后无缺失；评级只含A/B/C/D。
6. 企业特征无无穷值，边界比例在 \([0,1]\)，HHI不小于最大占比平方。
7. 最终特征表恰好123行、主键唯一、标签完整，特征列与字典一一对应。

## 17. 可复现

同一输入哈希、配置、代码版本和环境下重复运行，CSV排序、数值容差内结果和图表数据必须一致。CSV按 enterprise_id 稳定排序；记录 Python及依赖版本、Git提交或工作区状态、求解器版本。模型与结果文件不得手工编辑。

## 18. CSV字段契约

所有CSV使用 UTF-8-SIG、逗号分隔、点号小数、无行索引；缺失值为空，不写“无”或0替代NA。

| 文件 | 必需字段 |
|---|---|
| enterprise_features_123.csv | enterprise_id、enterprise_name、credit_rating、default_label，以及 q1_feature_dictionary.md 的21个英文变量；评级为元数据，不进入主模型 |
| q1_cv_metrics.csv | repeat_id、fold_id、n_train、n_test、n_default_test、pr_auc、brier、log_loss、roc_auc |
| q1_enterprise_risk.csv | enterprise_id、default_label、oof_risk_mean、oof_risk_sd、risk_rank |
| q1_churn_curve.csv | credit_rating、annual_rate、observed_churn_rate、fitted_churn_rate、acceptance_probability |
| q1_credit_strategy.csv | enterprise_id、credit_rating、risk_value、lend_flag、loan_amount_10k_cny、annual_rate、acceptance_probability、expected_net_return_10k_cny、reason_code |
| q1_budget_summary.csv | budget_10k_cny、budget_mode、allocated_10k_cny、expected_accepted_10k_cny、expected_net_return_10k_cny、expected_loss_10k_cny、unused_10k_cny、solver_status |

## 19. 图表

文件名固定使用 fig_q1_序号_主题.png；白底、300 dpi、统一中文字体和色板，轴名必须含单位，概率轴限定0～1。图由保存的CSV重绘，不手工改图；同时保存绘图数据或在运行清单中指明来源。PR、ROC、校准图必须使用交叉验证外预测。

## 20. 单元测试与验收

tests/ 至少覆盖：正/负/零/作废口径、完全重复与多明细发票、冲突同号、缺月补零、金额换算、21个公式、零分母、HHI/最大占比、最长连续月份、标签映射、企业级切分、折内缩尾/标准化、CSV字段与123行断言。提交前运行全套测试、01和02脚本；任一失败不得交付下游。

## 21. src/01_data_audit.py

输入：config/q1.yaml、附件1三个工作表和附件3 Sheet1；全部只读。

处理：

1. 核对工作表、列名、类型和企业主键。
2. 对每张表输出行列数与逐列缺失。
3. 计算企业表与进/销项表的覆盖率、未匹配企业及每家记录数。
4. 分表统计发票状态，以及价税合计为正、负、零的记录数。
5. 输出开票日期最小值、最大值、不可解析数和月覆盖。
6. 分别检查完全重复行、复合键重复、同一“企业—方向—发票号”的元数据冲突。
7. 以配置阈值检查极端金额、非有限值、金额恒等式超差和异常税额。
8. 生成上述审计CSV、q1_data_quality.json和简短Markdown报告；报告必须给出 PASS/WARN/FAIL。

01不得清洗或覆盖原始文件；FAIL时返回非零退出码并阻止02正式产出。

## 22. src/02_build_features.py

输入：config/q1.yaml、附件1、通过审计的 results/audit/q1_data_quality.json，以及 q1_feature_dictionary.md 规定的口径。

处理：

1. 规范字段类型和单位，按审计结论处理完全重复并构造原子发票。
2. 有效正数、负数、零额与作废记录严格分流；未知状态不得进入。
3. 按企业—月份补齐 \(\mathcal T\)，生成进/销项月表和企业—交易对手汇总。
4. 只计算字典中的21个原始企业级特征；不得加入评级、名称或标签衍生特征。
5. 左连接企业信息，映射 default_label，并执行主键、行数、范围、缺失和公式一致性检查。

输出：

- results/intermediate/q1_clean_invoice_ledger.parquet；
- results/intermediate/q1_enterprise_monthly.parquet；
- results/enterprise_features_123.csv：一行一家企业，必须恰好123行，enterprise_id不重复，default_label完整；
- results/audit/q1_feature_quality.json；
- results/reports/q1_feature_quality_report.md：逐特征缺失率、分位数、常量列、无穷值、异常值、范围检查和与字典的一致性结论。

02不做训练、风险预测或信贷优化；其特征计算口径必须与 q1_feature_dictionary.md 完全一致。
