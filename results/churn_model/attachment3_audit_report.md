# 附件3审计与利率—客户流失率单调拟合报告

本报告由程序从附件3实际数据生成。客户流失率是客户接受贷款利率后流失的情景参数，不是违约率。

## 1. 数据审计

- 工作簿：`data/附件3：银行贷款年利率与客户流失率关系的统计数据.xlsx`；工作表：`Sheet1`。
- 检测到29个实际利率点；利率和流失率统一为0—1小数。
- 百分数转换仅在原始非空数值最大值超过1且不超过100时执行，并在audit CSV中逐项记录；本次转换结果不能由文档预设。
- 利率集合使用附件3实际观测点，不生成区间外或额外精细利率。

## 2. 拟合方法

对A、B、C分别拟合独立的 `IsotonicRegression(increasing=True, y_min=0, y_max=1, out_of_bounds='clip')`。拟合值只在实际观测利率点使用；接受概率定义为 `A_g(r)=1-L_g(r)`。

## 3. 拟合指标

```text
credit_rating                     fit_method  observed_point_count  raw_monotonic_violation_count  fitted_monotonic_violation_count      mae     rmse  maximum_absolute_adjustment  mean_absolute_adjustment  flat_interval_count  raw_min  raw_max  fitted_min  fitted_max
            A isotonic_regression_increasing                    29                              3                                 0 0.000689 0.001819                     0.005746                  0.000689                    3      0.0 0.922061         0.0    0.922061
            B isotonic_regression_increasing                    29                              2                                 0 0.000623 0.001949                     0.007189                  0.000623                    2      0.0 0.885865         0.0    0.885865
            C isotonic_regression_increasing                    29                              2                                 0 0.000521 0.001406                     0.004001                  0.000521                    2      0.0 0.895165         0.0    0.895165
```

## 4. 评级间顺序诊断

独立保序拟合后，在相同利率点发现28个A级≤B级≤C级不成立的点。该顺序只作业务合理性诊断，不作为本基准模型硬约束；若存在交叉，基准仍使用独立拟合，不静默调整。

## 5. 输出解释边界

后续优化中的收益、损失和组合策略均为参数化情景结果，不是银行真实利润或真实损失预测。基准优化使用独立评级保序拟合曲线。

## 6. 审计状态

`attachment3_audit.csv`中的所有强制检查均为PASS。
