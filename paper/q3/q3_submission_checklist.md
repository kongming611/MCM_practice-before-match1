# Q3 提交前检查清单

这是提交前的证据和人工复核清单，不是“自动 PASS 即可提交”的承诺。`[x]` 只表示当前文件或运行契约有证据；需要团队判断的项目保留 `[ ]`。

## 1. 数据与来源

- [x] Q2 输入为当前 [q2_risk_rating_scores.csv](../../data/processed/q2_risk_rating_scores.csv)，302行、ID唯一。
- [x] Q2 策略输入为 [q2_credit_strategy.csv](../../data/processed/q2_credit_strategy.csv)，Q2 baseline preflight 为 `PASS`。
- [x] 行业映射输出为 [q3_enterprise_industry_mapping.csv](../../data/processed/q3_enterprise_industry_mapping.csv)，精确 ID/名称连接为 `PASS`。
- [x] 分行业增长值、发布日期、2020Q1、GB/T4754-2017、preliminary 与回顾性解释记录于 [q3_industry_shock_params.csv](../../outputs/q3/tables/q3_industry_shock_params.csv)。
- [x] 官方来源链接为 [国家统计局分行业数据](https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900688.html)。
- [x] 145家 unknown、56家 generic、5家 ambiguous 及其规则状态可由 mapping validation 复核。
- [x] 145家 unknown 的逐行复核证据队列已生成，按5个冲突、84个低特异/无命中、56个泛化名称排序；见 [q3_industry_mapping_review_evidence.csv](../../outputs/q3/tables/q3_industry_mapping_review_evidence.csv)。
- [x] 已依据 [Q3专业把关结论](q3_manual_confirmation_packet.md) 和逐行证据队列复核边界名称；匿名名称不足以支持可靠重分类，145家维持 unknown，避免强行贴标签。

## 2. 模型与假设

- [x] `d_h`、`S_h`、`D_hs`、`m_hs`、`r_i_s` 公式及百分点单位见 [q3_model_spec.md](q3_model_spec.md)。
- [x] identity 逐行重现 Q2 风险，风险值有限且不超过 `1-eps`（`eps=1e-6`）。
- [x] 正增长金融业和信息软件IT的 `S=0`、乘数=1；没有把正增长转为风险奖励。
- [x] unknown 主策略为 `max_known_S`；P50 仅在敏感性场景中出现。
- [x] Q3 只覆盖 `main_risk_score`，评级概率、D规则、流失曲线和 Q2 输入哈希保持不变。
- [x] 中心参数的替代网格、策略 Jaccard、额度变化和经济量证据已汇总到 [Q3人工确认包](q3_manual_confirmation_packet.md)，未把情景分析表述为参数校准。
- [x] 接受 `c=40`、`rho=0.25` 和三档场景倍数作为中心情景假设；敏感性结果显示核心选择稳定，正文不得称其为官方估计。
- [x] 接受 unknown 主策略采用 `max_known_S`；同时报告 `p50_known_S` 敏感性，避免将保守规则写成真实行业冲击。
- [x] 接受 LGD=0.50 与资金成本率=0.03 作为参数化中心值；保留0.30/0.70和0.02/0.04敏感性，不称真实银行参数。

## 3. 优化与比较

- [x] Q2 MILP 入口、候选利率、额度边界、D级拒贷、不确定性上限和严格名义预算等式已复用。
- [x] identity/light/medium/severe/robust 五个主场景均 `optimal`，无 fallback，预算等式通过。
- [x] robust 定义为场景风险最大值；当前单调参数导致 robust=severe，已在结果和局限中披露。
- [x] fixed-Q2-under-robust 对照明确是模型隐含经济量，不改 Q2 决策表。
- [x] robust 输出为 [q3_robust_credit_strategy.csv](../../data/processed/q3_robust_credit_strategy.csv)；策略、行业迁移和组合摘要可在 [outputs/q3/tables](../../outputs/q3/tables) 复核。
- [x] 1亿元被记录为10000个万元单位，单户额度为10～100万元；预期实际放款额未替代名义预算。
- [x] 决定不新增行业额度上限或相关性约束；题目未直接要求且缺少真实限额/相关性数据，正文以 HHI 描述集中度并披露局限。

## 4. 敏感性、图和结果

- [x] 敏感性摘要为13行，包含正式中心、11个单因素扰动和1个探索性组合；证据见 [q3_sensitivity_summary.csv](../../outputs/q3/tables/q3_sensitivity_summary.csv)。
- [x] LGD、funding cost、rho 的净收益范围按当前网格复算，并写明“在本次测试范围内”。
- [x] 探索性组合 `net=31.7440`、`loss=81.2656` 已标注为边界探索而非预测。
- [x] 稳定性标签为 `162/8/132`，且已说明是选择频率分析规则，不是真实稳健性/违约分类。
- [x] 三张图的 PDF/SVG/PNG/TIFF 和图形 QA `PASS` 见 [q3_figure_qa.json](../../outputs/q3/reports/q3_figure_qa.json)。
- [x] 图7/图8/敏感性图引用路径已在 [q3_final_results_for_paper.md](q3_final_results_for_paper.md) 列出；图中金额均为名义万元。
- [x] Q3 图形文件、图注口径和金额单位已经审查；最终论文中的连续图号属于整篇排版环节，不再作为 Q3 完成阻塞项。

## 5. 引用、诚信与表述

- [x] 官方页面、文件路径和当前运行证据均可追溯；未添加无法核验的外部引用或数字。
- [x] 所有风险、收益、信用损失和接受率均写成模型量/情景量，不写成真实 PD、疫情因果效应或真实银行利润。
- [x] 没有用“典型企业”挑选故事；如后续加入，必须先声明可复现筛选规则并列出全部符合者。
- [x] 时间边界已披露：2020Q1含3月，晚于Q2发票窗口，不是事前验证。
- [x] 接受使用国家统计局公开数据作为回顾性外部压力标尺；必须规范引用，并保留“非训练输入、非事前验证、非因果估计”的限定。
- [x] 已确认结构化材料仅供论文手写作，不能未经编辑直接作为完整参赛论文；若用于2026年正式竞赛，还须按当届规则披露 AI 使用。
- [x] Q3 阶段的 AI 使用目的、关键交互、采纳和人工核验情况已记录于 [q3_ai_usage_record.md](q3_ai_usage_record.md)；正式提交时并入全队“AI 工具使用详情”。

## 6. Contracts、manifest 与复现门禁

- [x] [q3_q2_baseline_contract.json](../../outputs/q3/reports/q3_q2_baseline_contract.json)：Q2 baseline `PASS`。
- [x] [q3_industry_mapping_validation.json](../../outputs/q3/reports/q3_industry_mapping_validation.json)：行业映射 `PASS`。
- [x] [q3_stress_validation.json](../../outputs/q3/reports/q3_stress_validation.json)：压力覆盖 `PASS`。
- [x] [q3_robust_optimization_validation.json](../../outputs/q3/reports/q3_robust_optimization_validation.json)：Q2 优化复用 `PASS`。
- [x] [q3_sensitivity_validation.json](../../outputs/q3/reports/q3_sensitivity_validation.json)：13场景敏感性 `PASS`。
- [x] [q3_full_validation.json](../../outputs/q3/reports/q3_full_validation.json)：full validation `PASS`。
- [x] [q3_run_manifest.json](../../outputs/q3/reports/q3_run_manifest.json)：输入/输出哈希、代码哈希和运行环境已记录。
- [x] Q2 前后哈希相等、风险覆盖仅替换 Q2 `main_risk_score`、确定性重跑通过。

## 7. 阶段6自审摘要（按审稿风险分级）

### P0：提交前必须解决或明确接受

1. **参数确认缺口**：`c`、`rho`、`lambda`、LGD、`c_f` 和 unknown 主策略仍是建模假设/运行默认；若队内不能确认，论文必须显式写为未校准假设。
2. **外部有效性缺口**：没有真实违约标签、恢复率或持续期校准，不能报告真实 PD、真实损失、预测准确率或因果结论。
3. **行业证据缺口**：145家 unknown 和粗粒度名称规则需要人工复核；不得把 unknown 的最大压力解释成企业实际行业。
4. **时间解释风险**：官方2020Q1数据含3月且晚于Q2发票窗口；必须保留“回顾性外部压力标尺”措辞，不能写成事前预测或疫情因果估计。

### P1：应在论文审阅中修正或限定

1. robust=severe 是单调公式的结构结果，不能当作独立稳健证据。
2. 没有行业上限、共同冲击相关性和持续期路径；HHI 仅为结果摘要。
3. 敏感性场景是确定性离散重算，不是统计置信区间；探索性组合不是预测。
4. fixed-Q2-under-robust、资金迁移和净收益差值是模型隐含比较，不能外推为真实业务收益。
5. 所有表格、图注和正文必须统一使用万元/元/1亿元单位，并同时引用 manifest 与 full validation。
