# 第三批问题一风险模型训练报告

本报告只使用data/processed/q1_enterprise_features.csv中已经通过阶段A验收的企业级特征；没有调用历史模型脚本，也没有从原始Excel重新构造特征。

## 1. 样本、标签和输入契约

- 企业样本数：123；违约数：27；未违约数：96；违约率：21.95%。
- 每家企业外层测试次数：10；重复分层交叉验证为5折×10次，共50个Logistic正式折。
- 企业代号、企业名称和信誉评级仅用于标识/展示；default_label仅作标签；audit_字段不进入模型矩阵。

## 2. 特征角色

- 共构造21个企业级派生特征，其中15个进入主模型，5个用于特征集敏感性分析，zero_amount_invoice_rate仅作审计。
- 本轮主模型实际使用：sales_scale_10k, purchase_scale_10k, operating_net_inflow_proxy_10k, sales_growth_trend, sales_monthly_cv, invoice_activity_per_month, sales_return_rate, purchase_return_rate, void_invoice_rate, customer_count, supplier_count, customer_hhi, supplier_hhi, purchase_sales_ratio, active_month_ratio。
- zero_amount_invoice_rate的精确/容差核验与排除理由见outputs/q1/reports/q1_zero_amount_invoice_rate_decision.md；程序实际核验的有效原子发票分母为349767，精确零值数为0，0.01元容差零值数为100；两种口径的差异使该字段只保留为审计字段。

## 3. 交叉验证与折内预处理

- 两类算法使用完全相同的50个企业级外层切分；所有中位数、1%/99%缩尾边界、log1p变换和RobustScaler均在训练折Pipeline内拟合。
- 主模型为未使用SMOTE、未使用class_weight=balanced的弹性网Logistic；对照为预先锁定参数的HistGradientBoostingClassifier。

## 4. 企业聚合OOF主要指标

| 模型 | PR-AUC | Brier | LogLoss | ROC-AUC | 校准误差 | Balanced Accuracy | F1 | Top20召回 | Top20精确率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 弹性网Logistic | 0.6814 | 0.1091 | 0.3605 | 0.8646 | 0.0849 | 0.6829 | 0.5238 | 0.7037 | 0.7600 |
| HistGradientBoosting | 0.6903 | 0.1045 | 0.3463 | 0.8742 | 0.0412 | 0.7703 | 0.6667 | 0.6296 | 0.6800 |

PR-AUC、Brier、LogLoss和Top20召回用于主要比较；阈值0.5的指标只作辅助结果。

## 5. 配对bootstrap和主模型选择

- Bootstrap按123家企业抽样，要求至少2000次；跳过单一类别样本次数为0。
| 指标 | 树模型相对Logistic差值 | 树模型优势定义 | 95%区间 | 树模型更优概率 |
|---|---:|---|---|---:|
| pr_auc | 0.0088 | tree-logistic | [-0.1439, 0.1564] | 0.5520 |
| brier | -0.0046 | logistic-tree | [-0.0178, 0.0080] | 0.7505 |
| log_loss | -0.0142 | logistic-tree | [-0.0468, 0.0197] | 0.8040 |
| top20_recall | -0.0741 | tree-logistic | [-0.1600, 0.1053] | 0.2180 |

- 按运行前锁定的规则，树模型在4项主要评价中稳定优于Logistic的项数为0；最终主模型为弹性网Logistic，树模型作为稳健性对照。

## 6. Logistic系数稳定性

- 稳定正向风险特征：sales_monthly_cv（中位数=0.5775，正号比例=100.00%）；void_invoice_rate（中位数=0.1071，正号比例=98.00%）；customer_hhi（中位数=0.1372，正号比例=74.00%）。
- 稳定负向风险特征：sales_scale_10k（中位数=-0.8872，负号比例=100.00%）；operating_net_inflow_proxy_10k（中位数=-0.1655，负号比例=94.00%）；supplier_hhi（中位数=-0.0508，负号比例=54.00%）。
- 所有正式折均收敛。 全样本最终展示系数另存于final_logistic_coefficients.csv；全样本拟合只用于解释和后续部署，不用于评价自身性能。

## 7. 特征集敏感性与评级敏感性

| 特征集 | 模型 | 特征数 | PR-AUC | Brier | LogLoss | Top20召回 |
|---|---|---:|---:|---:|---:|---:|
| primary_model_features | 弹性网Logistic | 15 | 0.6814 | 0.1091 | 0.3605 | 0.7037 |
| sensitivity_model_features_full_nonconstant | 弹性网Logistic | 20 | 0.6807 | 0.1078 | 0.3573 | 0.7037 |
| sales_scale_replaced_by_net_sales | 弹性网Logistic | 15 | 0.6831 | 0.1088 | 0.3593 | 0.7037 |
| hhi_replaced_by_max_share | 弹性网Logistic | 15 | 0.6920 | 0.1105 | 0.3623 | 0.7037 |
| active_month_replaced_by_longest_streak | 弹性网Logistic | 15 | 0.7136 | 0.1080 | 0.3578 | 0.7037 |

- 评级实验明确标记为rating_leakage_sensitivity_only，仅用于说明评级与标签绑定，不替代行为主模型：弹性网Logistic: PR-AUC=0.9271, Brier=0.0598, LogLoss=0.2396, Top20召回=0.8519；HistGradientBoosting: PR-AUC=0.9542, Brier=0.0309, LogLoss=0.1507, Top20召回=0.8889。

## 8. 典型企业和概率解释

- 选定模型的最高风险企业示例：E96（***土地整理有限公司，风险值=0.8080）；E115（***装饰工程有限公司，风险值=0.7835）；E123（***创科技有限责任公司，风险值=0.7820）；E113（***美居科技有限公司，风险值=0.7592）；E120（***陈列广告有限公司，风险值=0.7212）。
- 选定模型的最低风险企业示例：E3（***电子(中国)有限公司***分公司，风险值=0.0215）；E9（***生活用品服务有限公司***分公司，风险值=0.0217）；E7（***家电有限公司***分公司，风险值=0.0258）；E8（***科学研究院有限公司，风险值=0.0261）；E2（***技术有限责任公司，风险值=0.0283）。
- 这些风险值是历史发票行为下的相对违约倾向，不是严格一年期违约概率；相关关系不解释为因果关系。

## 9. 局限和可复现文件

- 样本只有123家企业且违约标签不均衡；外层重复折彼此重叠，不能当成50组独立样本。
- 没有违约发生日期和明确观察期，可能存在时间信息限制；评级是人工先验且附件2缺失，因此主模型不使用评级。
- 主要输出运行文件见data/processed/_runtime/model_training；正式表格见outputs/q1/tables，图表见outputs/q1/figures/model；输入特征哈希、配置哈希、代码版本和环境见运行目录中的manifest和environment文件。
