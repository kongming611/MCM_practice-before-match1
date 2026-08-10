# Q3 疫情行业压力场景验证

状态：**PASS**

本阶段仅覆盖 Q2 `main_risk_score` 的相对风险倾向；不改变评级概率、流失曲线或其他 Q2 参数。
官方分行业 GDP 数据是包含 2020 年 3 月的回顾性外部压力标尺，不是 Q2 训练输入或事前预测验证。

## 门禁

| 检查 | 结果 |
| --- | --- |
| row_count_302 | PASS |
| q2_ids_unique | PASS |
| output_ids_unique | PASS |
| exact_id_name_join | PASS |
| finite_values | PASS |
| risk_values_legal | PASS |
| multipliers_ge_one | PASS |
| identity_exact_q2 | PASS |
| scenario_order_monotone | PASS |
| robust_is_exact_max | PASS |
| positive_growth_neutral | PASS |
| unknown_count_145_max_known_policy | PASS |
| source_metadata_complete | PASS |
| official_growth_values_match | PASS |
| risk_override_only | PASS |
| deterministic_rows | PASS |
| parameter_columns_complete | PASS |
| output_columns_complete | PASS |

## 场景风险变化摘要

```json
{
  "by_industry": {
    "accommodation_catering": {
      "count": 3,
      "max_robust_change_from_q2": 0.032007927275000014,
      "q2_mean_risk": 0.10557875244666666,
      "robust_mean_risk": 0.13197344055833335,
      "severe_mean_risk": 0.13197344055833335
    },
    "agriculture": {
      "count": 2,
      "max_robust_change_from_q2": 0.010971725656500064,
      "q2_mean_risk": 0.2836601615,
      "robust_mean_risk": 0.2911062407393751,
      "severe_mean_risk": 0.2911062407393751
    },
    "construction": {
      "count": 53,
      "max_robust_change_from_q2": 0.1257352767890625,
      "q2_mean_risk": 0.18902759072849054,
      "robust_mean_risk": 0.22003992983238355,
      "severe_mean_risk": 0.22003992983238355
    },
    "information_software_it": {
      "count": 8,
      "max_robust_change_from_q2": 0.0,
      "q2_mean_risk": 0.30724054417125,
      "robust_mean_risk": 0.30724054417125,
      "severe_mean_risk": 0.30724054417125
    },
    "leasing_business_services": {
      "count": 16,
      "max_robust_change_from_q2": 0.05679011254312505,
      "q2_mean_risk": 0.26129473924187496,
      "robust_mean_risk": 0.28432133813756527,
      "severe_mean_risk": 0.28432133813756527
    },
    "manufacturing_industry": {
      "count": 7,
      "max_robust_change_from_q2": 0.01896336384806252,
      "q2_mean_risk": 0.11232623400714285,
      "robust_mean_risk": 0.12306743013407591,
      "severe_mean_risk": 0.12306743013407591
    },
    "other_services": {
      "count": 20,
      "max_robust_change_from_q2": 0.010821774867187495,
      "q2_mean_risk": 0.16317312269849998,
      "robust_mean_risk": 0.16592666914403717,
      "severe_mean_risk": 0.16592666914403717
    },
    "real_estate": {
      "count": 3,
      "max_robust_change_from_q2": 0.011433445943812487,
      "q2_mean_risk": 0.09918260018666665,
      "robust_mean_risk": 0.10485460513484167,
      "severe_mean_risk": 0.10485460513484167
    },
    "transport_storage_post": {
      "count": 11,
      "max_robust_change_from_q2": 0.08587701710437501,
      "q2_mean_risk": 0.13126619810727272,
      "robust_mean_risk": 0.14849488660885227,
      "severe_mean_risk": 0.14849488660885227
    },
    "unknown": {
      "count": 145,
      "max_robust_change_from_q2": 0.18966613717500003,
      "q2_mean_risk": 0.1833231013026206,
      "robust_mean_risk": 0.22767396561103442,
      "severe_mean_risk": 0.22767396561103442
    },
    "wholesale_retail": {
      "count": 34,
      "max_robust_change_from_q2": 0.1355800546,
      "q2_mean_risk": 0.24330578192470592,
      "robust_mean_risk": 0.2833517146604831,
      "severe_mean_risk": 0.2833517146604831
    }
  },
  "parameter_rows": 12,
  "scenario": {
    "identity": {
      "clipped_at_upper_bound_count": 0,
      "max_change_from_q2": 0.0,
      "max_risk": 0.9142122803,
      "mean_change_from_q2": 0.0,
      "mean_risk": 0.1926709517095034,
      "min_risk": 0.02157733009
    },
    "light": {
      "clipped_at_upper_bound_count": 1,
      "max_change_from_q2": 0.09458809268378132,
      "max_risk": 0.999999,
      "mean_change_from_q2": 0.013908148568253861,
      "mean_risk": 0.20657910027775717,
      "min_risk": 0.023957579315553125
    },
    "medium": {
      "clipped_at_upper_bound_count": 2,
      "max_change_from_q2": 0.1673803660569375,
      "max_risk": 0.999999,
      "mean_change_from_q2": 0.02737782129952904,
      "mean_risk": 0.22004877300903236,
      "min_risk": 0.02633782854110625
    },
    "severe": {
      "clipped_at_upper_bound_count": 4,
      "max_change_from_q2": 0.18966613717500003,
      "max_risk": 0.999999,
      "mean_change_from_q2": 0.033891999024995755,
      "mean_risk": 0.22656295073449914,
      "min_risk": 0.0269716626125
    }
  }
}
```
