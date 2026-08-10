# Q3 模型规格（结构化写作材料）

本文档规定 Q3 的可复现模型接口、变量、假设和验收边界。它是论文结构和证据索引，不是可直接提交的完整参赛论文；最终文字、队内口径和人工复核仍需由团队确认。

## 1. 问题拆解与范围

Q3 将 Q2 的历史发票行为相对风险映射到行业压力场景，并在不改动 Q2 评级概率、客户流失曲线、D 级规则和 MILP 约束的前提下，比较固定 Q2 策略与重新优化后的额度分配。

1. **基线门禁**：检查 Q2 的风险表、策略表、优化配置和哈希；入口为 `preflight`。
2. **行业映射**：对 [q2_risk_rating_scores.csv](../../data/processed/q2_risk_rating_scores.csv) 的302家企业使用确定性名称关键词规则，输出 [q3_enterprise_industry_mapping.csv](../../data/processed/q3_enterprise_industry_mapping.csv)，并为全部 unknown 行生成 [逐行人工复核证据队列](../../outputs/q3/tables/q3_industry_mapping_review_evidence.csv)。
3. **压力覆盖**：读取国家统计局分行业同比数据，计算 identity/light/medium/severe 风险覆盖，并将未知行业按主策略规则处理，输出 [q3_scenario_risk_scores.csv](../../data/processed/q3_scenario_risk_scores.csv)。
4. **稳健优化**：仅将 Q2 的 `main_risk_score` 替换为场景覆盖风险，复用 Q2 的候选利率、流失曲线、D 规则和 MILP，输出 [q3_robust_credit_strategy.csv](../../data/processed/q3_robust_credit_strategy.csv)。
5. **比较和敏感性**：比较 identity、各压力场景、robust 与 fixed-Q2-under-robust，并对登记的13个确定性场景进行敏感性分析。

Q3 风险是“历史发票行为相对违约倾向”的模型量，不是观测 PD；组合收益和信用损失是模型隐含的情景数量，不应写成真实利润、真实损失或疫情因果效应。

## 2. 数据、来源与行业映射

| 数据/参数 | 当前文件或来源 | 用途 | 事实/假设边界 |
|---|---|---|---|
| Q2 风险、评级概率和不确定性字段 | [data/processed/q2_risk_rating_scores.csv](../../data/processed/q2_risk_rating_scores.csv) | 302家企业的 `q2_main_risk` 与 Q2 评级输入 | Q2 已验收的模型输出；仍不是真实 PD |
| Q2 策略和 MILP | [data/processed/q2_credit_strategy.csv](../../data/processed/q2_credit_strategy.csv)、[src/_internal/q2_credit_optimization.py](../../src/_internal/q2_credit_optimization.py) | 固定策略对照和重新优化 | 复用代码契约，不重新估计 Q2 模型 |
| 企业行业映射 | [data/processed/q3_enterprise_industry_mapping.csv](../../data/processed/q3_enterprise_industry_mapping.csv) | 将企业连接到粗行业 | 名称关键词推断；不是法定行业证明 |
| 分行业增长 | [国家统计局 2020Q1 页面](https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900688.html) | 提供外部压力标尺 `g_h` | 2020-04-18 发布的初步核算，含2020年3月；为回顾性外部压力标尺，不是 Q2 训练输入或事前验证 |

当前11个已知行业的同比增长（百分点）为：农业 -2.8、制造业 -10.2、建筑业 -17.5、批发和零售业 -17.8、交通运输仓储和邮政业 -14.0、住宿和餐饮业 -35.3、金融业 +6.0、房地产业 -6.1、信息传输软件和信息技术服务业 +13.2、租赁和商务服务业 -9.4、其他服务业 -1.8。`manufacturing_industry` 使用制造业 -10.2，不使用工业总体 -8.5。企业映射的当前分布为：农业2、制造业7、建筑53、批零34、交通11、住宿餐饮3、金融0、房地产业3、信息技术8、租赁商务16、其他服务20、未知145。

映射只允许上述11类加 `unknown`。当前规则审计中 `high_specific_rule=157`、`generic_unknown=56`、`ambiguous_unknown=5`、`low_specific_or_unmatched_unknown=84`；这些状态是规则证据，不是行业真实标签质量的外部认证。

## 3. 符号表与单位

| 符号 | 含义 | 单位/范围 |
|---|---|---|
| `i` | 企业索引 | `1..302` |
| `h` | 行业代码 | 11类或 `unknown` |
| `g_h` | 行业同比增长 | 百分点；例如 -35.3 表示 -35.3 个百分点 |
| `d_h` | 负向冲击幅度 `max(0,-g_h)` | 百分点 |
| `c` | 冲击归一化尺度 | 40 个百分点；建模假设 |
| `S_h` | 行业压力强度 | `[0,1]` |
| `s` | 场景 | identity、light、medium、severe |
| `lambda_s` | 场景放大倍数 | identity=0、light=0.5、medium=1、severe=1.5 |
| `D_hs` | 场景压力量 | `[0,1]`，模型量，不是 PD |
| `rho` | 风险放大系数 | 0.25；建模假设 |
| `m_hs` | 风险乘数 | `>=1` |
| `q2_i` | Q2 `main_risk_score` | `[0,1]`，历史相对风险倾向 |
| `r_i_s` | 场景风险覆盖 | `[0,1-eps]` |
| `eps` | 上界裁剪余量 | `1e-6` |
| `x_ik` | 企业 `i` 是否选择候选利率 `k` | 0/1 |
| `s_ik` | 名义放贷额度 | `10^4` 元；即万元 |
| `B` | 名义预算 | 10000 个 `10^4` 元单位，即1亿元 |
| `LGD` | 损失率 | 0.50；建模假设 |
| `c_f` | 资金成本率 | 0.03；运行默认/建模假设 |

金额字段后缀 `_10k` 均以万元计；`_yuan` 为元。预算约束使用名义额度，不将预期实际放款额替代预算。

## 4. 行业压力与风险覆盖

先将增长率换成非正向压力：

```text
d_h = max(0, -g_h)
S_h = clip(d_h / c, 0, 1), 其中 c = 40 个百分点
```

对每个场景：

```text
D_hs = clip(lambda_s * S_h, 0, 1)
m_hs = 1 + rho * D_hs, 其中 rho = 0.25
```

企业风险覆盖为：

```text
r_i_s = clip(q2_i * m_h(i)_s, 0, 1 - eps), 其中 eps = 1e-6
```

identity 的 `lambda=0`，因此必须逐行精确重现 Q2 风险。正增长行业金融业（+6.0）和信息软件 IT（+13.2）取 `d=S=D=0`，不提供“风险奖励”，乘数保持1。住宿餐饮的中心值为 `S=35.3/40=0.8825`，故 light 的 `m=1.1103125`、medium 的 `m=1.220625`，severe 的 `D=1`、`m=1.25`。

`unknown` 不伪造增长率：主策略取11个已知行业的最大 `S_h=0.8825`，并在输出中写明 `unknown_policy=max_known_S` 与 `growth_source=derived_policy_max_known_S`。`p50_known_S` 只在敏感性分析中运行，不作为中心策略。

稳健风险定义为：

```text
r_i_robust = max(r_i_identity, r_i_light, r_i_medium, r_i_severe)
```

由于本设置中 `lambda_s >= 0`、`S_h >= 0`、乘数随场景单调增加，当前 `robust` 与 `severe` 数值完全相同，因而最优策略也相同。这是公式结构结果，不是额外的稳健性证据。

## 5. Q2 MILP 复用、目标与约束

Q3 不重估评级或流失曲线，只将 Q2 的 `main_risk_score` 换成上式 `r_i_s`。Q2 的评级概率 `pi_ig`、D级阈值 `tau_D=0.8`、A/B/C 客户流失曲线、候选利率观测点（4%～15%，29点）、额度边界（10～100万元）和高不确定性上限（50万元）保持不变。

候选利率下的单位预期量由 Q2 代码计算：

```text
I_ik = A_i(r_k) * (1 - r_i_s) * r_k
L_ik = A_i(r_k) * r_i_s * LGD
F_ik = A_i(r_k) * c_f
N_ik = I_ik - L_ik - F_ik
目标：max sum_i,k N_ik * s_ik
```

这里 `A_i(r_k)` 是 Q2 的评级混合客户接受概率，不等于 `1-r_i_s`；D概率达到阈值的企业先拒贷，D质量不外推流失曲线。名义预算采用严格等式：

```text
sum_i,k s_ik = B = 10000
```

并同时满足每家企业至多选一个利率、额度上下界、D/不确定性上限、二元选择约束和 Q2 已登记的策略条件。求解器必须返回 `optimal`，不允许用 fallback 静默替代。

## 6. 正式比较与敏感性设计

正式组合比较包括：

- identity：Q2 风险不变；
- light、medium、severe：压力覆盖下重新求解；
- robust：上述场景风险最大值重新求解；
- fixed-Q2-under-robust：保持 Q2 的决策、利率和额度，仅用 robust 风险重算模型隐含经济量。

敏感性表共13行：formal robust 中心场景、`c in {20,60}`、`rho in {0.10,0.50}`、`lambda_severe in {1.125,1.875}`、unknown 的 `p50_known_S`、`LGD in {0.30,0.70}`、资金成本率 `in {0.02,0.04}`，以及一个探索性组合 `exploratory_combined_adverse`。中心值和单因素场景用于敏感性说明；探索性组合不是预测，也不纳入正式选择频率标签。

## 7. 验收门

以下门禁必须在 [q3_full_validation.json](../../outputs/q3/reports/q3_full_validation.json) 和 [q3_run_manifest.json](../../outputs/q3/reports/q3_run_manifest.json) 中有当前运行证据：

- 302行、ID/名称唯一且与 Q2 精确连接；行业映射和压力覆盖 `PASS`；
- 风险有限、identity 与 Q2 逐行相等、场景单调、robust 精确等于场景最大值；
- 评级概率、D规则、流失曲线和 Q2 输入哈希不变，只覆盖风险列；
- 5个主优化场景和13个敏感性场景均 `optimal`、无 fallback、严格名义预算等式成立；
- 重新运行确定性一致；图形 QA 为 `PASS`；
- 全部假设、回顾性时间关系、unknown 策略和“不是 PD/因果”的解释在论文材料中保留。
