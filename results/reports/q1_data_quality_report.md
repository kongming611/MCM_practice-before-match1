# 问题一数据质量报告

- 审计状态：**WARN**
- 输入文件：data/附件1：123家有信贷记录企业的相关数据.xlsx
- 输入SHA-256：450df5f7184aa43b3b1cddaa4387cceb91d4881a666e667b7c84c480a8b2b600
- 企业数：123
- 可进入特征构建：**是**

## 工作表概况

| workbook | sheet_name | role | n_rows | n_columns | fields | field_types |
|---|---|---|---|---|---|---|
| data/附件1：123家有信贷记录企业的相关数据.xlsx | 企业信息 | enterprise | 123 | 4 | ["企业代号", "企业名称", "信誉评级", "是否违约"] | {"企业代号": "string", "企业名称": "string", "信誉评级": "string", "是否违约": "string"} |
| data/附件1：123家有信贷记录企业的相关数据.xlsx | 进项发票信息 | input_invoice | 210947 | 8 | ["企业代号", "发票号码", "开票日期", "销方单位代号", "金额", "税额", "价税合计", "发票状态"] | {"价税合计": "float64", "企业代号": "string", "发票号码": "string", "发票状态": "string", "开票日期": "datetime64[us]", "税额": "float64", "金额": "float64", "销方单位代号": "string"} |
| data/附件1：123家有信贷记录企业的相关数据.xlsx | 销项发票信息 | output_invoice | 162484 | 8 | ["企业代号", "发票号码", "开票日期", "购方单位代号", "金额", "税额", "价税合计", "发票状态"] | {"价税合计": "float64", "企业代号": "string", "发票号码": "string", "发票状态": "string", "开票日期": "datetime64[us]", "税额": "float64", "购方单位代号": "string", "金额": "float64"} |

## 企业覆盖

| enterprise_count | input_covered_count | output_covered_count | any_covered_count | unmatched_input_ids | unmatched_output_ids | missing_enterprise_ids_in_info |
|---|---|---|---|---|---|---|
| 123 | 123 | 123 | 123 | [] | [] | 0 |

## 发票状态与金额符号

| workbook | sheet_name | direction | status_raw | status_canonical | record_count | positive_amount_count | negative_amount_count | zero_amount_count | missing_amount_count |
|---|---|---|---|---|---|---|---|---|---|
| data/附件1：123家有信贷记录企业的相关数据.xlsx | 进项发票信息 | input | 作废发票 | 作废 | 7608 | 7587 | 21 | 0 | 0 |
| data/附件1：123家有信贷记录企业的相关数据.xlsx | 进项发票信息 | input | 有效发票 | 有效 | 203339 | 201479 | 1860 | 0 | 0 |
| data/附件1：123家有信贷记录企业的相关数据.xlsx | 销项发票信息 | output | 作废发票 | 作废 | 11206 | 9395 | 132 | 1679 | 0 |
| data/附件1：123家有信贷记录企业的相关数据.xlsx | 销项发票信息 | output | 有效发票 | 有效 | 151278 | 142854 | 8424 | 0 | 0 |

## 日期范围

| workbook | sheet_name | direction | record_count | invalid_date_count | min_date | max_date | month_count |
|---|---|---|---|---|---|---|---|
| data/附件1：123家有信贷记录企业的相关数据.xlsx | 进项发票信息 | input | 210947 | 0 | 2016-10-04 | 2020-02-21 | 41 |
| data/附件1：123家有信贷记录企业的相关数据.xlsx | 销项发票信息 | output | 162484 | 0 | 2016-10-07 | 2020-02-21 | 41 |

## 重复与疑似重复

| workbook | sheet_name | direction | check_type | group_count | extra_row_count | affected_enterprise_count | sample_keys |
|---|---|---|---|---|---|---|---|
| data/附件1：123家有信贷记录企业的相关数据.xlsx | 进项发票信息 | input | exact_duplicate_rows | 803 | 803 | 1 |  |
| data/附件1：123家有信贷记录企业的相关数据.xlsx | 进项发票信息 | input | business_key_repeated | 803 | 803 | 1 | [["E22", "4133754", "2016-10-08", "A00314", "有效"], ["E22", "4133752", "2016-10-08", "A00314", "有效"], ["E22", "4133753", "2016-10-08", "A00314", "有效"], ["E22", "2889788", "2016-10-25", "A05891", "有效"], ["E22", "2890081", "2016-10-26", "A05891", "有效"]] |
| data/附件1：123家有信贷记录企业的相关数据.xlsx | 进项发票信息 | input | invoice_number_metadata_conflict | 15 | 0 | 8 | [["E2", "22890241"], ["E2", "37561620"], ["E2", "37967314"], ["E2", "3552575"], ["E6", "12589191"]] |
| data/附件1：123家有信贷记录企业的相关数据.xlsx | 销项发票信息 | output | exact_duplicate_rows | 4436 | 4436 | 5 |  |
| data/附件1：123家有信贷记录企业的相关数据.xlsx | 销项发票信息 | output | business_key_repeated | 4436 | 4436 | 5 | [["E2", "9177643", "2019-01-21", "B01134", "有效"], ["E2", "9177644", "2019-01-21", "B01134", "有效"], ["E2", "9177645", "2019-01-21", "B01134", "有效"], ["E2", "9177646", "2019-01-21", "B01353", "有效"], ["E2", "9177647", "2019-01-21", "B01353", "有效"]] |
| data/附件1：123家有信贷记录企业的相关数据.xlsx | 销项发票信息 | output | invoice_number_metadata_conflict | 1 | 0 | 1 | [["E22", "1861804"]] |

## 金额分布与异常

| workbook | sheet_name | direction | record_count | non_numeric_total_count | non_finite_total_count | arithmetic_mismatch_count | min_yuan | q01_yuan | q05_yuan | q25_yuan | median_yuan | q75_yuan | q95_yuan | q99_yuan | max_yuan | below_q01_count | above_q99_count |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| data/附件1：123家有信贷记录企业的相关数据.xlsx | 进项发票信息 | input | 210947 | 0 | 0 | 2 | -11100000.0 | 1.0 | 58.0 | 224.0 | 520.0 | 9480.9 | 100000.0 | 344930.5400000012 | 11699934.0 | 2048 | 2110 |
| data/附件1：123家有信贷记录企业的相关数据.xlsx | 销项发票信息 | output | 162484 | 0 | 0 | 1 | -1169732.68 | -100000.0 | -1203.0 | 3018.4 | 19792.08 | 100000.0 | 700000.0 | 1157520.0 | 4615590.0 | 1503 | 1614 |

## 错误与警告

### 错误

- 无

### 警告

- input存在金额+税额与价税合计超差记录
- output存在金额+税额与价税合计超差记录
- 存在完全重复、重复业务键或同号元数据冲突，详见duplicate_check.csv
- 金额存在配置分位数外记录；原始记录未删除

原始工作簿只读；任何重复、负数、零额和分位数外金额均未在审计阶段删除。
