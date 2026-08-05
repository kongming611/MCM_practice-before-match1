# 问题一特征评审与处理决定

本文件由 `results/feature_validation/feature_validation_report.md` 的实际输出生成；本轮没有训练风险模型。
分类只表示进入后续建模前的处理建议，不覆盖原始特征表。‘经营净流入代理’与‘经营差额’均不是利润。

## 分类总表

| 特征 | 决定 | 依据与原因 |
|---|---|---|
| `business_scale_10k` | 仅用于描述性分析 | 规模/右偏风险较强，按字典在训练折内log1p或有符号log1p并缩尾；参与|Spearman|>0.85高相关对，不能与相关变量同时入模而不做审查；是销售和采购规模之和，适合报告总体规模；与分项规模重叠 |
| `sales_scale_10k` | 保留但需要转换 | 规模/右偏风险较强，按字典在训练折内log1p或有符号log1p并缩尾；参与|Spearman|>0.85高相关对，不能与相关变量同时入模而不做审查 |
| `purchase_scale_10k` | 保留但需要转换 | 规模/右偏风险较强，按字典在训练折内log1p或有符号log1p并缩尾；参与|Spearman|>0.85高相关对，不能与相关变量同时入模而不做审查 |
| `net_sales_10k` | 保留但需要转换 | 规模/右偏风险较强，按字典在训练折内log1p或有符号log1p并缩尾；参与|Spearman|>0.85高相关对，不能与相关变量同时入模而不做审查 |
| `operating_net_inflow_proxy_10k` | 保留但需要转换 | 规模/右偏风险较强，按字典在训练折内log1p或有符号log1p并缩尾；只能解释为销售净额减采购净额的经营差额代理，不代表利润 |
| `sales_growth_trend` | 保留为主模型候选特征 | 边界在[0,1]或趋势/连续性定义清晰，可保留并在折内预处理 |
| `sales_monthly_cv` | 保留但需要转换 | 规模/右偏风险较强，按字典在训练折内log1p或有符号log1p并缩尾 |
| `invoice_activity_per_month` | 保留但需要转换 | 规模/右偏风险较强，按字典在训练折内log1p或有符号log1p并缩尾；参与|Spearman|>0.85高相关对，不能与相关变量同时入模而不做审查 |
| `sales_return_rate` | 保留但需要转换 | 规模/右偏风险较强，按字典在训练折内log1p或有符号log1p并缩尾 |
| `purchase_return_rate` | 保留但需要转换 | 规模/右偏风险较强，按字典在训练折内log1p或有符号log1p并缩尾 |
| `void_invoice_rate` | 保留为主模型候选特征 | 边界在[0,1]或趋势/连续性定义清晰，可保留并在折内预处理 |
| `zero_amount_invoice_rate` | 暂时删除 | 当前样本为全常数，不能提供主模型区分度；需要确认零金额作废票的业务含义；当前特征全常数 |
| `customer_count` | 保留但需要转换 | 规模/右偏风险较强，按字典在训练折内log1p或有符号log1p并缩尾 |
| `supplier_count` | 保留但需要转换 | 规模/右偏风险较强，按字典在训练折内log1p或有符号log1p并缩尾；参与|Spearman|>0.85高相关对，不能与相关变量同时入模而不做审查 |
| `customer_hhi` | 保留为主模型候选特征 | 参与|Spearman|>0.85高相关对，不能与相关变量同时入模而不做审查 |
| `supplier_hhi` | 保留为主模型候选特征 | 参与|Spearman|>0.85高相关对，不能与相关变量同时入模而不做审查 |
| `max_customer_share` | 需要建模手决定 | 参与|Spearman|>0.85高相关对，不能与相关变量同时入模而不做审查 |
| `max_supplier_share` | 需要建模手决定 | 参与|Spearman|>0.85高相关对，不能与相关变量同时入模而不做审查 |
| `purchase_sales_ratio` | 保留但需要转换 | 规模/右偏风险较强，按字典在训练折内log1p或有符号log1p并缩尾 |
| `active_month_ratio` | 保留为主模型候选特征 | 参与|Spearman|>0.85高相关对，不能与相关变量同时入模而不做审查 |
| `longest_active_streak_ratio` | 保留为主模型候选特征 | 参与|Spearman|>0.85高相关对，不能与相关变量同时入模而不做审查 |

## 高相关变量处理

高相关只作诊断，不在本轮自动删除。后续建模手需要在弹性网、变量组选择或保留一个代表变量之间做决定。

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

## 当前明确的口径缺口

- 需确认审计报告中完全重复行是否确属重复导入；当前构建按 `keep_first`。
- 需确认是否纳入2016-10与2020-02两个可能不完整的边界月份。
- 3条金额+税额与价税合计超差记录目前仅标记并保留，是否修正/剔除由建模手决定。
- 信誉评级和是否违约保留在表中做标签/辅助分析，但不进入主行为特征矩阵；`audit_`列同样只作审计辅助。
