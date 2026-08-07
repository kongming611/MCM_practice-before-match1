# 问题二数据质量报告

- 审计状态：**WARN**
- 训练附件：data/raw/附件1：123家有信贷记录企业的相关数据.xlsx
- 目标附件：data/raw/附件2：302家无信贷记录企业的相关数据.xlsx
- 训练附件SHA-256：450df5f7184aa43b3b1cddaa4387cceb91d4881a666e667b7c84c480a8b2b600
- 目标附件SHA-256：1399075d2b39a05da6d7722af16cb96ed322cf13b96dd780d5d42a7686bc634b
- 共同月份窗口：2016-10, 2016-11, 2016-12, 2017-01, 2017-02, 2017-03, 2017-04, 2017-05, 2017-06, 2017-07, 2017-08, 2017-09, 2017-10, 2017-11, 2017-12, 2018-01, 2018-02, 2018-03, 2018-04, 2018-05, 2018-06, 2018-07, 2018-08, 2018-09, 2018-10, 2018-11, 2018-12, 2019-01, 2019-02, 2019-03, 2019-04, 2019-05, 2019-06, 2019-07, 2019-08, 2019-09, 2019-10, 2019-11, 2019-12, 2020-01, 2020-02

## 企业覆盖

| dataset | enterprise_count | input_covered_count | output_covered_count | any_covered_count | both_directions_covered_count | uncovered_enterprise_count | errors | warnings |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| training | 123 | 123 | 123 | 123 | 123 | 0 | 无 | input_amount_arithmetic_mismatch_present, input_business_key_duplicates_present, input_exact_duplicates_present, input_invoice_number_conflicts_present, output_amount_arithmetic_mismatch_present, output_business_key_duplicates_present, output_exact_duplicates_present, output_invoice_number_conflicts_present |
| target | 302 | 302 | 302 | 302 | 302 | 0 | 无 | input_amount_arithmetic_mismatch_present, input_business_key_duplicates_present, input_exact_duplicates_present, input_invoice_number_conflicts_present, output_business_key_duplicates_present, output_exact_duplicates_present, output_invoice_number_conflicts_present |

## 表、缺失、状态、日期、重复和金额检查

详细结果见 q2_table_profile.csv、q2_missing_values.csv、q2_invoice_status_summary.csv、q2_date_summary.csv、q2_duplicate_summary.csv 和 q2_amount_anomalies.csv。

### 状态摘要

| dataset | sheet | direction | status_raw | status_canonical | row_count |
| --- | --- | --- | --- | --- | --- |
| training | 进项发票信息 | input | 作废发票 | 作废 | 7608 |
| training | 进项发票信息 | input | 有效发票 | 有效 | 203339 |
| training | 销项发票信息 | output | 作废发票 | 作废 | 11206 |
| training | 销项发票信息 | output | 有效发票 | 有效 | 151278 |
| target | 进项发票信息 | input | 作废发票 | 作废 | 17236 |
| target | 进项发票信息 | input | 有效发票 | 有效 | 377939 |
| target | 销项发票信息 | output | 作废发票 | 作废 | 27555 |
| target | 销项发票信息 | output | 有效发票 | 有效 | 303280 |

### 日期摘要

| dataset | sheet | direction | invalid_date_count | min_date | max_date |
| --- | --- | --- | --- | --- | --- |
| training | 进项发票信息 | input | 0 | 2016-10-04 | 2020-02-21 |
| training | 销项发票信息 | output | 0 | 2016-10-07 | 2020-02-21 |
| target | 进项发票信息 | input | 0 | 2016-10-08 | 2020-02-21 |
| target | 销项发票信息 | output | 0 | 2016-10-08 | 2020-02-21 |

### 重复摘要

| dataset | sheet | direction | raw_exact_duplicate_rows | normalized_exact_duplicate_extra_rows | business_key_duplicate_extra_rows | conflicting_invoice_number_groups |
| --- | --- | --- | --- | --- | --- | --- |
| training | 进项发票信息 | input | 803 | 803 | 803 | 15 |
| training | 销项发票信息 | output | 4436 | 4436 | 4436 | 1 |
| target | 进项发票信息 | input | 4681 | 4681 | 4681 | 37 |
| target | 销项发票信息 | output | 1745 | 1745 | 1745 | 2048 |

### 金额异常摘要

| dataset | sheet | direction | raw_rows | missing_amount_yuan | missing_tax_yuan | missing_total_yuan | arithmetic_error_rows | positive_total_rows | negative_total_rows | zero_total_rows | min_total_yuan | max_total_yuan | q01_total_yuan | q99_total_yuan | configured_lower_quantile | configured_upper_quantile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| training | 进项发票信息 | input | 210947 | 0 | 0 | 0 | 2 | 209066 | 1881 | 0 | -11100000.0 | 11699934.0 | 1.0 | 344930.5400000012 | 0.01 | 0.99 |
| training | 销项发票信息 | output | 162484 | 0 | 0 | 0 | 1 | 152249 | 8556 | 1679 | -1169732.68 | 4615590.0 | -100000.0 | 1157520.0 | 0.01 | 0.99 |
| target | 进项发票信息 | input | 395175 | 0 | 0 | 0 | 2 | 391545 | 3630 | 0 | -1022995.0 | 8868000.0 | 1.0 | 307950.5680000006 | 0.01 | 0.99 |
| target | 销项发票信息 | output | 330835 | 0 | 0 | 0 | 0 | 322798 | 5697 | 2340 | -1101380.0 | 1167257.0 | -10000.0 | 1000000.0 | 0.01 | 0.99 |

## 错误与警告

### 错误

- 无

### 警告

- input_amount_arithmetic_mismatch_present
- input_business_key_duplicates_present
- input_exact_duplicates_present
- input_invoice_number_conflicts_present
- output_amount_arithmetic_mismatch_present
- output_business_key_duplicates_present
- output_exact_duplicates_present
- output_invoice_number_conflicts_present

原始工作簿只读；重复、负数、零额和分位数外金额均不在审计阶段静默删除。
