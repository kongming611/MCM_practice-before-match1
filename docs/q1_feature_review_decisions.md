# 问题一特征评审与处理决定

本文件由 results/feature_validation/feature_validation_report.md 的实际输出生成；分类只表示模型角色，不覆盖原始特征表。
共构造21个企业级派生特征，其中15个进入主模型，5个用于特征集敏感性分析，zero_amount_invoice_rate仅作审计。
‘经营净流入代理’与‘经营差额’均不是利润；评级、标签、企业身份和audit_字段不进入正式风险模型矩阵。

## 分类总表

| 特征 | 模型角色 | 依据与原因 |
|---|---|---|
| business_scale_10k | 特征集敏感性分析 | 规模/右偏风险较强，按字典在训练折内log1p或有符号log1p并缩尾；参与|Spearman|>0.85高相关对，按预先锁定的替代变量组处理；与销售、采购规模高度重复，仅放入完整敏感性特征集 |
| sales_scale_10k | 主模型特征 | 规模/右偏风险较强，按字典在训练折内log1p或有符号log1p并缩尾；参与|Spearman|>0.85高相关对，按预先锁定的替代变量组处理 |
| purchase_scale_10k | 主模型特征 | 规模/右偏风险较强，按字典在训练折内log1p或有符号log1p并缩尾；参与|Spearman|>0.85高相关对，按预先锁定的替代变量组处理 |
| net_sales_10k | 特征集敏感性分析 | 规模/右偏风险较强，按字典在训练折内log1p或有符号log1p并缩尾；参与|Spearman|>0.85高相关对，按预先锁定的替代变量组处理；与sales_scale_10k高度相关，仅放入销售规模替代敏感性组 |
| operating_net_inflow_proxy_10k | 主模型特征 | 规模/右偏风险较强，按字典在训练折内log1p或有符号log1p并缩尾；只能解释为销售净额减采购净额的经营差额代理，不代表利润 |
| sales_growth_trend | 主模型特征 | 按特征字典定义在折内完成缺失填补、缩尾、变换和标准化 |
| sales_monthly_cv | 主模型特征 | 规模/右偏风险较强，按字典在训练折内log1p或有符号log1p并缩尾 |
| invoice_activity_per_month | 主模型特征 | 规模/右偏风险较强，按字典在训练折内log1p或有符号log1p并缩尾；参与|Spearman|>0.85高相关对，按预先锁定的替代变量组处理 |
| sales_return_rate | 主模型特征 | 规模/右偏风险较强，按字典在训练折内log1p或有符号log1p并缩尾 |
| purchase_return_rate | 主模型特征 | 规模/右偏风险较强，按字典在训练折内log1p或有符号log1p并缩尾 |
| void_invoice_rate | 主模型特征 | 按特征字典定义在折内完成缺失填补、缩尾、变换和标准化 |
| zero_amount_invoice_rate | 审计型派生字段，不进入任何正式模型 | 当前样本为全常数；零金额字段只作为已知审计事实，不阻断主模型验收；正式定义和精确/容差核验见q1_zero_amount_invoice_rate_decision.md；作废行为由void_invoice_rate刻画 |
| customer_count | 主模型特征 | 规模/右偏风险较强，按字典在训练折内log1p或有符号log1p并缩尾 |
| supplier_count | 主模型特征 | 规模/右偏风险较强，按字典在训练折内log1p或有符号log1p并缩尾；参与|Spearman|>0.85高相关对，按预先锁定的替代变量组处理 |
| customer_hhi | 主模型特征 | 参与|Spearman|>0.85高相关对，按预先锁定的替代变量组处理 |
| supplier_hhi | 主模型特征 | 参与|Spearman|>0.85高相关对，按预先锁定的替代变量组处理 |
| max_customer_share | 特征集敏感性分析 | 参与|Spearman|>0.85高相关对，按预先锁定的替代变量组处理；分别作为对应HHI的替代变量组，仅用于敏感性分析 |
| max_supplier_share | 特征集敏感性分析 | 参与|Spearman|>0.85高相关对，按预先锁定的替代变量组处理；分别作为对应HHI的替代变量组，仅用于敏感性分析 |
| purchase_sales_ratio | 主模型特征 | 规模/右偏风险较强，按字典在训练折内log1p或有符号log1p并缩尾 |
| active_month_ratio | 主模型特征 | 参与|Spearman|>0.85高相关对，按预先锁定的替代变量组处理 |
| longest_active_streak_ratio | 特征集敏感性分析 | 参与|Spearman|>0.85高相关对，按预先锁定的替代变量组处理；与active_month_ratio高度相关，仅用于连续性替代敏感性组 |

## 已锁定的主模型特征选择规则

- business_scale_10k与销售、采购规模高度重复，只作敏感性分析。
- sales_scale_10k与net_sales_10k高度相关，主模型保留sales_scale_10k。
- customer_hhi与max_customer_share高度相关，主模型保留HHI。
- supplier_hhi与max_supplier_share高度相关，主模型保留HHI。
- active_month_ratio与longest_active_streak_ratio高度相关，主模型保留active_month_ratio。
- 被替代变量保留在特征集敏感性分析中，不从特征表删除。

## 高相关变量处理

高相关变量不从原始企业特征表物理删除；正式主模型和敏感性模型使用配置中的明确特征列表。

| feature_a | feature_b | spearman_rho | abs_spearman_rho | n_pair |
|---|---|---|---|---|
| sales_scale_10k | net_sales_10k | 0.9985360694431905 | 0.9985360694431905 | 123 |
| business_scale_10k | sales_scale_10k | 0.9836001083437592 | 0.9836001083437592 | 123 |
| business_scale_10k | net_sales_10k | 0.9835356180108602 | 0.9835356180108602 | 123 |
| supplier_hhi | max_supplier_share | 0.9770213534377681 | 0.9770213534377681 | 123 |
| customer_hhi | max_customer_share | 0.9745456656047259 | 0.9745456656047259 | 123 |
| active_month_ratio | longest_active_streak_ratio | 0.9511727435609926 | 0.9511727435609926 | 123 |
| business_scale_10k | purchase_scale_10k | 0.900072229172847 | 0.900072229172847 | 123 |
| business_scale_10k | invoice_activity_per_month | 0.8800139299355086 | 0.8800139299355086 | 123 |
| invoice_activity_per_month | supplier_count | 0.8698814596566875 | 0.8698814596566875 | 123 |
| net_sales_10k | invoice_activity_per_month | 0.862853024683224 | 0.862853024683224 | 123 |
| sales_scale_10k | invoice_activity_per_month | 0.8620468942222936 | 0.8620468942222936 | 123 |

## 变量排除边界

- zero_amount_invoice_rate保留在features.names和企业级特征表中，但不在primary_model_features或sensitivity_model_features中。
- credit_rating只作单独敏感性实验，default_label只作标签；enterprise_id和enterprise_name只作标识或展示。
- audit_开头字段只作审计辅助，不进入任何模型矩阵。
