# Q3 企业行业映射验收报告

最终状态：**PASS**

## 验收检查

| 检查 | 结果 |
| --- | --- |
| row_count_302 | PASS |
| q2_ids_unique | PASS |
| mapped_ids_unique | PASS |
| q2_id_name_set_match | PASS |
| allowed_industry_codes | PASS |
| no_enterprise_omitted | PASS |
| generic_count_56 | PASS |
| generic_all_unknown | PASS |
| rule_deterministic | PASS |
| output_columns_complete | PASS |
| review_evidence_rows_equal_unknown | PASS |
| review_evidence_ids_exact | PASS |
| review_evidence_columns_complete | PASS |
| review_evidence_prioritized | PASS |

## 计数

- 行业代码计数：`{"accommodation_catering": 3, "agriculture": 2, "construction": 53, "information_software_it": 8, "leasing_business_services": 16, "manufacturing_industry": 7, "other_services": 20, "real_estate": 3, "transport_storage_post": 11, "unknown": 145, "wholesale_retail": 34}`
- 映射状态计数：`{"ambiguous_unknown": 5, "generic_unknown": 56, "high_specific_rule": 157, "low_specific_or_unmatched_unknown": 84}`
- 置信度计数：`{"0.0": 145, "1.0": 157}`
- unknown 行数：**145**（其中 generic 行数：56）

## 边界与人工复核

规则只使用企业名称中的已登记高特异关键词；企业编号没有作为特征，未使用 LLM。
unknown 是有意保留的结果，低特异、多重命中、名称掩码和无命中行需人工复核，不能据此宣称真实行业归属。
逐行复核证据队列：`outputs/q3/tables/q3_industry_mapping_review_evidence.csv`；共 145 行，已按冲突、低特异/无命中、泛化名称排序。
