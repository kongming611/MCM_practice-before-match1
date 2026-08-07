# 问题二企业级特征质量报告

- 状态：**PASS**
- 训练参考企业数：123
- 目标企业数：302
- 构造特征数：21
- 正式主模型特征数：15
- 共同月份窗口：2016-10, 2016-11, 2016-12, 2017-01, 2017-02, 2017-03, 2017-04, 2017-05, 2017-06, 2017-07, 2017-08, 2017-09, 2017-10, 2017-11, 2017-12, 2018-01, 2018-02, 2018-03, 2018-04, 2018-05, 2018-06, 2018-07, 2018-08, 2018-09, 2018-10, 2018-11, 2018-12, 2019-01, 2019-02, 2019-03, 2019-04, 2019-05, 2019-06, 2019-07, 2019-08, 2019-09, 2019-10, 2019-11, 2019-12, 2020-01, 2020-02
- 目标表不包含信誉评级和是否违约字段。
- 未执行风险模型训练；本阶段只生成数据层特征和审计输出。

## 特征合法性

| check | value |
| --- | --- |
| status | PASS |
| row_count | 302 |
| enterprise_id_unique | True |
| constructed_feature_count | 21 |
| primary_model_feature_count | 15 |
| constructed_feature_names | ["business_scale_10k", "sales_scale_10k", "purchase_scale_10k", "net_sales_10k", "operating_net_inflow_proxy_10k", "sales_growth_trend", "sales_monthly_cv", "invoice_activity_per_month", "sales_return_rate", "purchase_return_rate", "void_invoice_rate", "zero_amount_invoice_rate", "customer_count", "supplier_count", "customer_hhi", "supplier_hhi", "max_customer_share", "max_supplier_share", "purchase_sales_ratio", "active_month_ratio", "longest_active_streak_ratio"] |
| primary_model_features | ["sales_scale_10k", "purchase_scale_10k", "operating_net_inflow_proxy_10k", "sales_growth_trend", "sales_monthly_cv", "invoice_activity_per_month", "sales_return_rate", "purchase_return_rate", "void_invoice_rate", "customer_count", "supplier_count", "customer_hhi", "supplier_hhi", "purchase_sales_ratio", "active_month_ratio"] |
| bounded_violations | {} |
| target_label_columns_present | [] |
| month_window | ["2016-10", "2016-11", "2016-12", "2017-01", "2017-02", "2017-03", "2017-04", "2017-05", "2017-06", "2017-07", "2017-08", "2017-09", "2017-10", "2017-11", "2017-12", "2018-01", "2018-02", "2018-03", "2018-04", "2018-05", "2018-06", "2018-07", "2018-08", "2018-09", "2018-10", "2018-11", "2018-12", "2019-01", "2019-02", "2019-03", "2019-04", "2019-05", "2019-06", "2019-07", "2019-08", "2019-09", "2019-10", "2019-11", "2019-12", "2020-01", "2020-02"] |
| reference_feature_quality | {"rows": 123, "enterprise_ids_unique": true, "feature_columns": ["business_scale_10k", "sales_scale_10k", "purchase_scale_10k", "net_sales_10k", "operating_net_inflow_proxy_10k", "sales_growth_trend", "sales_monthly_cv", "invoice_activity_per_month", "sales_return_rate", "purchase_return_rate", "void_invoice_rate", "zero_amount_invoice_rate", "customer_count", "supplier_count", "customer_hhi", "supplier_hhi", "max_customer_share", "max_supplier_share", "purchase_sales_ratio", "active_month_ratio", "longest_active_streak_ratio"]} |
| target_labels_absent_by_design | True |

## 错误

- 无

训练参考特征和目标特征均调用问题一的原子发票、共同月份面板和21项特征函数；标准化、填补和缩尾未在数据层执行。
