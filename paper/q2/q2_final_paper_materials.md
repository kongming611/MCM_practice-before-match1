# 问题二最终论文材料包

本文件是论文手的最终拼装页，提供可直接改写的段落骨架、可编辑LaTeX公式、符号、结果表、图注、检验、优缺点、资料来源和关键数字追溯。它不是一篇完整参赛论文；论文手仍需按全篇结构衔接问题一、问题三并完成引用编号。

## 1. 模型选择说明（可直接改写）

问题二的核心困难是附件2只有302家企业的发票信息，没有真实违约结果和信誉评级。为保持与问题一一致并控制小样本过拟合，主方案沿用仅依赖两份附件共有发票特征的弹性网Logistic模型，将附件1中123家企业的历史违约标签所包含的关系迁移到附件2；同时建立有序Logistic模型输出A/B/C/D四级评级概率，使风险预测能够与附件3的评级—利率—客户流失关系衔接。另以K近邻Label Spreading作为半监督交叉检查，利用302家无标签企业的局部图结构识别主模型分歧，但不替换或平均主风险。123家企业聚合OOF结果显示，主方案在PR-AUC、Brier、LogLoss和Top-20%召回四项主要指标的点估计均优于Label Spreading，因此最终采用“监督迁移为主、半监督传播作不确定性诊断”的组合路线。

## 2. 数据处理说明（可直接改写）

模型首先将附件1和附件2的进项、销项发票统一到2016年10月至2020年2月的41个月窗口，并以企业为单位聚合，防止同一企业的发票同时进入训练与验证。正式金额取“价税合计”并除以10000换算为万元；有效正数发票用于购销规模，有效负数发票保留为退货信息，作废发票不计入有效金额但进入作废率，零金额有效发票只用于审计。完全重复行按登记规则保留首行，发票号码冲突和金额算术不一致保留审计警告，不改写原始工作簿。共构造21项企业特征，其中15项进入正式模型；训练折内依次完成中位数填补、1%～99%缩尾、对数变换和稳健尺度化，并把同一参数应用于验证折和附件2，避免目标分布泄漏。

## 3. 可编辑核心公式

### 3.1 历史违约倾向

\[
\hat p_i=\frac{1}{1+\exp[-(\beta_0+z_i^\top\beta)]},
\]

\[
\min_{\beta_0,\beta}
-\frac1n\sum_i[y_i\log\hat p_i+(1-y_i)\log(1-\hat p_i)]
+\lambda\left[\alpha\lVert\beta\rVert_1+
\frac{1-\alpha}{2}\lVert\beta\rVert_2^2\right].
\]

### 3.2 有序评级概率

\[
P(R_i\le h\mid z_i)=\sigma(\theta_h-z_i^\top\gamma),
\quad h=0,1,2,\quad \theta_0<\theta_1<\theta_2.
\]

\[
\begin{aligned}
\pi_{iA}&=F_{i0},\\
\pi_{iB}&=F_{i1}-F_{i0},\\
\pi_{iC}&=F_{i2}-F_{i1},\\
\pi_{iD}&=1-F_{i2}.
\end{aligned}
\]

### 3.3 概率混合客户接受率

\[
\boxed{A_i(r_k)=\sum_{g\in\{A,B,C\}}\pi_{ig}[1-L_g(r_k)]}.
\]

D级概率不重新归一化，故 $A_i(r_k)\le1-\pi_{iD}$。客户接受率不是 $1-p_i$。

### 3.4 高不确定性和D级资格

\[
H_i=I\{\mathrm{OOD}_i\lor W_i\ge0.20\lor
\Delta_i^{LS}\ge0.20\lor E_i^\pi\ge0.75\},
\]

\[
e_i^D=I\{\pi_{iD}<0.80\},\qquad
U_i=\begin{cases}50,&H_i=1,\\100,&H_i=0.\end{cases}
\]

主场景 $p_i=\texttt{main\_risk\_score}$；高不确定性只改变额度上限，不自动使用p90。p90是单独敏感性情景。

### 3.5 1亿元MILP

\[
\kappa_{ik}=A_i(r_k)[(1-p_i)r_k-c_f-p_i\ell],
\]

\[
\boxed{
\begin{aligned}
\max_{x,s}\quad&\sum_{i,k}\kappa_{ik}s_{ik}\\
\text{s.t.}\quad
&\sum_kx_{ik}\le e_i^D,&&\forall i,\\
&10x_{ik}\le s_{ik}\le U_ix_{ik},&&\forall i,k,\\
&\sum_{i,k}s_{ik}=10000,\\
&x_{ik}\in\{0,1\},\quad s_{ik}\ge0.
\end{aligned}}
\]

\[
\begin{aligned}
\mathrm{ED}&=\sum_{i,k}A_i(r_k)s_{ik},\\
\mathrm{Interest}&=\sum_{i,k}A_i(r_k)(1-p_i)r_ks_{ik},\\
\mathrm{EL}&=\sum_{i,k}A_i(r_k)p_i\ell s_{ik},\\
\mathrm{FundingCost}&=\sum_{i,k}A_i(r_k)c_fs_{ik},\\
\mathrm{NM}&=\mathrm{Interest}-\mathrm{EL}-\mathrm{FundingCost}.
\end{aligned}
\]

## 4. 符号说明表

| 符号 | 含义 | 单位/取值 |
|---|---|---|
| $z_i$ | 企业$i$经训练折预处理的15维特征 | 无量纲 |
| $y_i$ | 附件1历史违约标签 | 0/1 |
| $\hat p_i$ / $p_i$ | 主风险输出/优化使用的历史违约倾向代理 | $[0,1]$ |
| $\pi_{ig}$ | 企业$i$属于评级$g$的模型概率 | $[0,1]$，和为1 |
| $L_g(r_k)$ | 评级$g$在利率$r_k$下的保序流失率 | $[0,1]$ |
| $A_i(r_k)$ | 企业$i$在报价$r_k$下的模型接受概率 | $[0,1]$ |
| $H_i$ | 高不确定性标记 | 0/1 |
| $e_i^D$ | 通过D概率政策的资格 | 0/1 |
| $x_{ik}$ | 是否给企业$i$选择利率$r_k$ | 0/1 |
| $s_{ik}$ | 企业$i$在利率$r_k$下的名义授信额 | 万元 |
| $U_i$ | 单家额度上限 | 50或100万元 |
| $\ell$ | 违约损失率LGD情景参数 | 0.50 |
| $c_f$ | 资金成本率情景参数 | 0.03 |
| $\kappa_{ik}$ | 每1万元名义额度的情景净边际收益 | 比例 |
| $\mathrm{ED}/\mathrm{EL}/\mathrm{NM}$ | 预期实际发放/预期信用损失/预期净收益 | 万元 |

## 5. 模型求解过程

1. 只读审计附件1/2/3，统一字段、月份和金额单位；
2. 按企业聚合发票，构造21项特征并锁定15项主特征；
3. 在123家企业上做5折×10次重复分层OOF，评价弹性网风险和有序评级；
4. 在相同遮蔽小块上运行Label Spreading，按123家企业聚合后做1000次配对bootstrap；
5. 用全部123家拟合最终风险和评级模型，输出302家风险、四级概率和重复折不稳定性；
6. 用附件1参考分布计算OOD，并合并风险区间、LS分歧和评级熵形成保守额度标记；
7. 将四级概率与附件3 A/B/C保序流失率混合，计算302×29个候选利率的接受率和单位经济学；
8. 用HiGHS求解二元利率选择+连续额度MILP；
9. 重算预算、收益、损失、D规则、额度和利率约束；
10. 运行24个主/单因素敏感性情景和最终论文材料验收。

## 6. 最终结果表

### 6.1 代理验证与迁移诊断

| 指标 | 结果 |
|---|---:|
| 主风险PR-AUC / Brier / LogLoss | 0.681437 / 0.109099 / 0.360483 |
| 主风险Top-20%召回 | 0.703704 |
| 有序评级宏F1 / 加权Kappa | 0.440997 / 0.637358 |
| 评级相邻一档内比例 | 0.951220 |
| OOD企业 | 50/302 |
| 主模型与LS风险差≥0.20 | 76家 |
| 模型阶段高不确定性 | 257家 |

### 6.2 1亿元主场景

| 指标 | 结果 |
|---|---:|
| 求解与验收 | `optimal / PASS` |
| 获贷/不贷 | 164 / 138家 |
| D概率拒贷 | 12家 |
| 优化阶段不确定性上限 | 246家 |
| 名义授信总额 | 10000万元=1亿元 |
| 额度构成 | 128家×50万元；36家×100万元 |
| 利率构成 | 9.85%×31；10.25%×15；10.65%×5；12.25%×39；14.25%×74 |
| 预期实际发放 | 2122.868113万元 |
| 情景预期利息 | 230.469558万元 |
| 情景预期信用损失 | 62.959317万元 |
| 情景预期资金成本 | 63.686043万元 |
| 情景预期净收益 | 103.824198万元 |
| 预算差额/目标重算差额 | 0 / 0万元 |

302家完整结果见 `outputs/q2/tables/q2_credit_strategy.csv`，不得只提交上表。

## 7. 图片及建议图注

建议正文优先使用以下6张，剩余图放附录：

1. `standardized_mean_difference.png`：**附件1参考企业与附件2目标企业的标准化均值差。** 横轴为15项共同发票特征，纵轴为仅按附件1均值和标准差计算的SMD；进销比差异最大，说明迁移时必须报告分布支持。
2. `q2_risk_probability_calibration_oof.png`：**弹性网Logistic与Label Spreading在123家代理样本上的企业聚合OOF校准。** 横轴为平均预测风险，纵轴为实际违约比例；该图不表示302家外部校准。
3. `q2_risk_ranking_disagreement.png`：**302家企业主模型与半监督交叉检查的风险分歧。** 红色表示OOD，大点表示高不确定性，偏离对角线的企业需要降额或人工复核。
4. `q2_rating_probability_heatmap_302.png`：**按主风险排序的302家A/B/C/D评级概率。** 颜色表示概率，避免把单一硬评级当成确定真值。
5. `q2_credit_allocation_risk.png`：**主场景风险、D级概率与名义授信额度。** 横轴为历史违约倾向，纵轴为额度（万元），颜色为D级概率。
6. `q2_credit_sensitivity_net_return.png`：**问题二单因素敏感性下的情景预期净收益。** 横轴为参数水平，纵轴为净收益（万元），用于区分稳定结论和参数敏感结论。

11张图的完整标题、坐标、单位、含义和源数据见 `q2_v3_results_and_validation_report.md` 第6节。

## 8. 结果分析与模型检验文字（可直接改写）

在123家有标签企业的重复分层代理验证中，弹性网Logistic的PR-AUC、Brier、LogLoss和Top-20%召回分别为0.681437、0.109099、0.360483和0.703704，四项主要指标的点估计均优于Label Spreading。1000次企业级配对bootstrap显示，LogLoss的主模型优势区间不跨0，但PR-AUC、Brier和Top-20%召回区间仍跨0，因此结果支持“主模型优先、半监督作交叉检查”，而不支持全面统计显著优势的表述。

在302家企业上，50家至少有一项主特征超出附件1的1%～99%参考区间，主模型与Label Spreading有76家风险差超过0.20。模型将这些信号与风险区间宽度和评级熵组合成高不确定性标记，并通过50万元额度上限控制外推风险。主场景最终选择164家，名义额度严格等于1亿元，D概率达到阈值的12家均未获贷；全部利率来自附件3观测点，预算、收益分解和目标函数重算差额均为0。

单因素敏感性表明，D阈值和OOD阈值的局部变化没有改变当前最优组合，但风险改用p90、LGD、资金成本和不确定性额度上限会明显改变情景净收益或获贷家数。因此，策略在可行性和部分排序上具有稳定性，但经济结果仍依赖未由题目观测的情景参数。

## 9. 优点、缺点和改进方向

### 优点

- 两问使用完全相同的企业级特征函数，避免迁移口径漂移；
- 以企业为切分单位并做5折×10次重复验证，避免发票级泄漏和单次切分偶然性；
- 主模型、评级概率、客户流失率和违约风险概念分离；
- 半监督模型只作交叉检查，且用企业级配对bootstrap量化不确定性；
- MILP同时决定企业、额度和利率，并逐项验证预算、D规则和收益恒等式；
- 每张图都有可编辑源数据，关键数字可追溯到代码和CSV。

### 缺点

- 302家没有真实违约、评级和接受结果，无法进行真正外部验证；
- 123家、27个违约标签对复杂关系的支持有限；
- 缺少违约时间和统一观察期，风险不是严格一年期PD；
- 评级模型宏F1不高，且历史人工评级可能包含主观先验；
- LGD、资金成本和政策阈值属于情景假设；
- 统一50万元上限没有区分不确定性原因和严重程度。

### 改进方向

- 收集附件2后续真实表现并做外部校准、漂移监控和滚动回测；
- 引入行业、企业年龄、现金流、交易网络和宏观变量；
- 用时间切分或生存模型建立明确预测窗口；
- 将不确定性上限改为按OOD程度、模型分歧和评级熵连续变化的鲁棒额度；
- 使用银行真实LGD、资金成本和审批政策重新估计并验证策略。

## 10. 参考文献与资料来源

1. 2020年高教社杯全国大学生数学建模竞赛C题，《中小微企业的信贷决策》，`data/raw/2020C-中小微企业的信贷决策.docx`。
2. 题目附件1、附件2、附件3，均保存在 `data/raw/`，输入SHA-256记录于Q2运行manifest。
3. Zou, H., & Hastie, T. (2005). Regularization and Variable Selection via the Elastic Net. *Journal of the Royal Statistical Society: Series B*, 67(2), 301-320. DOI: https://doi.org/10.1111/j.1467-9868.2005.00503.x
4. McCullagh, P. (1980). Regression Models for Ordinal Data. *Journal of the Royal Statistical Society: Series B*, 42(2), 109-127. https://doi.org/10.1111/j.2517-6161.1980.tb01109.x
5. Zhu, X., & Ghahramani, Z. (2002). Learning from Labeled and Unlabeled Data with Label Propagation. CMU-CALD-02-107. https://reports-archive.adm.cs.cmu.edu/anon/cald/abstracts/02-107.html
6. SciPy Developers. `scipy.optimize.milp` official documentation; the interface calls HiGHS. https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html

论文最终参考文献编号需按整篇论文统一调整；不得虚构未实际使用的文献或银行参数来源。

## 11. 关键数字追溯表

| 论文数字/结论 | 结果文件 | 代码/配置位置 |
|---|---|---|
| 123家、27违约、5折×10次 | `q2_model_report_for_paper.md`、`q2_model_metrics.csv` | `q1_config.yaml:model_training.cv`、`q2_models.py:_run_risk_and_label_spreading` |
| 主风险PR-AUC/Brier/LogLoss/Top20 | `q2_model_metrics.csv` | `q2_models.py:_summarize_binary_models` |
| 1000次配对bootstrap区间 | `q2_model_comparison_bootstrap.csv` | `q2_models.py:_paired_bootstrap_comparison` |
| 有序评级指标与阈值 | `q2_rating_metrics.csv`、`q2_final_ordered_logistic_parameters.csv` | `q2_models.py:OrderedLogistic` |
| 50家OOD、SMD | `q2_feature_distribution_comparison.csv`、`q2_ood_scores.csv` | `q2_data.py:run_eda`、`q2_config.yaml:eda` |
| 257家模型阶段高不确定性 | `q2_risk_rating_scores.csv` | `q2_models.py:_add_uncertainty_flags` |
| 12家D拒贷、246家额度上限 | `q2_credit_strategy.csv`、`q2_portfolio_summary.csv` | `q2_credit_optimization.py:_build_policy` |
| 164家、10000万元、额度/利率分布 | `q2_credit_strategy.csv` | `q2_credit_optimization.py:_solve_scenario/_build_strategy` |
| 预期发放、利息、损失、成本、净收益 | `q2_portfolio_summary.csv` | `q2_credit_optimization.py:_portfolio_summary` |
| 24个敏感性情景 | `q2_optimization_sensitivity.csv` | `q2_optimization_config.yaml:sensitivity` |
| 预算/目标/约束均通过 | `q2_budget_identity.csv`、`q2_strategy_validation.csv` | `q2_credit_optimization.py:_validate_strategy` |
| 最终材料文件哈希与总状态 | `outputs/q2/final/q2_delivery_manifest.json`、`q2_delivery_validation_report.md` | `q2_delivery.py:run_q2_delivery` |

## 12. 不可省略的写作边界

302家没有真实违约标签，不能把附件1代理验证指标写成附件2外部准确率；风险值不是经验证的一年期PD。10000万元是名义授信额，不是预期实际发放额。客户流失率不是违约率，贷款接受概率不是 `1-风险值`。情景预期净收益不是已实现利润。所有参数和结果均应遵守赛事规则、课程要求与学术诚信，并由团队成员核验后使用。
