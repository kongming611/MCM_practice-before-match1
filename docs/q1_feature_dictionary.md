# 问题一企业级特征字典

本字典定义21个仅依赖附件1、2共有发票字段的企业级派生特征。根据配置，15个进入主模型，5个作为替代变量用于特征集敏感性分析，zero_amount_invoice_rate仅作审计型派生字段。信誉评级、企业名称和是否违约均不属于行为特征：评级只作单独敏感性实验、辅助验证或展示；是否违约仅作标签。

## 1. 统一符号与处理码

方向 \(S/P\) 分别为销项/进项；\(q\) 为有效原子发票“价税合计”除以 \(10^4\) 后的万元值，\(q^+=\max(q,0)\)，\(q^-=\max(-q,0)\)；\(G_{it}^d=\sum q^+\)，\(N_{it}^d=\sum q\)，\(G_i^d=\sum_tG_{it}^d\)。月份集合 \(\mathcal T\) 对所有企业一致、缺月补零，\(m\) 为月份数；上标 \(v,nz,void,all\) 分别表示有效、非零、作废和全部已知状态原子发票。客户/供应商份额只按有效正向金额计算。zero_amount_invoice_rate的正式分子和分母均在去重、原子发票聚合后计算；零值同时按total_yuan==0和abs(total_yuan)<=配置的zero_amount_tolerance_yuan核对。

缺失码：Z=无合格记录时记0；N=零分母或无法定义时保留NA，在训练折用中位数填补。异常码：U=原始值保留并审计，企业级特征在训练折按配置分位数缩尾；B=校验应在 \([0,1]\)，越界即报错、不缩尾。所有特征以原始值写入CSV；“取对数”和“标准化”均在Pipeline的训练折内执行。

## 2. 定义、来源与聚合

| 中文名称 | 英文变量名 | 经济含义 | 数学公式 | 来源数据表 | 原始字段 | 聚合层级 |
|---|---|---|---|---|---|---|
| 经营规模 | business_scale_10k | 总体购销规模 | \(G_i^S+G_i^P\) | 进项+销项 | 企业代号、价税合计、状态 | 发票→月→企业 |
| 销售规模 | sales_scale_10k | 正向销售额 | \(G_i^S\) | 销项 | 企业代号、价税合计、状态 | 发票→月→企业 |
| 采购规模 | purchase_scale_10k | 正向采购额 | \(G_i^P\) | 进项 | 企业代号、价税合计、状态 | 发票→月→企业 |
| 净销售额 | net_sales_10k | 扣除销售退货后的销售净额 | \(\sum_tN_{it}^S\) | 销项 | 企业代号、价税合计、状态 | 发票→月→企业 |
| 经营净流入代理 | operating_net_inflow_proxy_10k | 销售净额减采购净额；不是利润 | \(\sum_t(N_{it}^S-N_{it}^P)\) | 进项+销项 | 企业代号、价税合计、状态 | 发票→月→企业 |
| 销售增长趋势 | sales_growth_trend | 正向销售规模的时间趋势 | \(\log(1+G_{it}^S)=a_i+b_it+e_{it}\) 中 \(b_i\) | 销项 | 企业代号、开票日期、价税合计、状态 | 发票→月→企业回归 |
| 销售月度波动 | sales_monthly_cv | 销售不稳定程度 | \(sd_t(G_{it}^S)/mean_t(G_{it}^S)\) | 销项 | 企业代号、开票日期、价税合计、状态 | 发票→月→企业 |
| 月均交易活跃度 | invoice_activity_per_month | 月均有效非零发票数 | \((n_i^{S,v,nz}+n_i^{P,v,nz})/m\) | 进项+销项 | 企业代号、发票号码、日期、价税合计、状态 | 原子发票→企业 |
| 销售退货率 | sales_return_rate | 销售退款相对正向销售 | \(\sum q_{ij}^{S,-}/G_i^S\) | 销项 | 企业代号、价税合计、状态 | 原子发票→企业 |
| 采购退货率 | purchase_return_rate | 采购退款相对正向采购 | \(\sum q_{ij}^{P,-}/G_i^P\) | 进项 | 企业代号、价税合计、状态 | 原子发票→企业 |
| 作废率 | void_invoice_rate | 取消交易占全部已知状态发票比例 | \(n_i^{void}/n_i^{all}\) | 进项+销项 | 企业代号、发票号码、状态 | 原子发票→企业 |
| 零金额发票比例 | zero_amount_invoice_rate | 有效原子发票零额审计比例 | \(n_i^{v,q=0}/n_i^v\)；同时报告精确零值与配置容差零值 | 进项+销项 | 企业代号、发票号码、日期、交易对手、价税合计、状态 | 去重明细→原子发票→企业 |
| 客户数量 | customer_count | 正向销售覆盖的客户数 | \(\operatorname{card}\{c:\sum_jq_{ijc}^{S,+}>0\}\) | 销项 | 企业代号、购方单位代号、价税合计、状态 | 发票→客户→企业 |
| 供应商数量 | supplier_count | 正向采购覆盖的供应商数 | \(\operatorname{card}\{v:\sum_jq_{ijv}^{P,+}>0\}\) | 进项 | 企业代号、销方单位代号、价税合计、状态 | 发票→供应商→企业 |
| 客户集中度HHI | customer_hhi | 销售依赖少数客户的程度 | \(\sum_c w_{ic}^2,\ w_{ic}=\sum_jq_{ijc}^{S,+}/G_i^S\) | 销项 | 企业代号、购方单位代号、价税合计、状态 | 发票→客户→企业 |
| 供应商集中度HHI | supplier_hhi | 采购依赖少数供应商的程度 | \(\sum_v w_{iv}^2,\ w_{iv}=\sum_jq_{ijv}^{P,+}/G_i^P\) | 进项 | 企业代号、销方单位代号、价税合计、状态 | 发票→供应商→企业 |
| 最大客户占比 | max_customer_share | 第一大客户销售占比 | \(\max_c w_{ic}\) | 销项 | 企业代号、购方单位代号、价税合计、状态 | 发票→客户→企业 |
| 最大供应商占比 | max_supplier_share | 第一大供应商采购占比 | \(\max_v w_{iv}\) | 进项 | 企业代号、销方单位代号、价税合计、状态 | 发票→供应商→企业 |
| 进销比 | purchase_sales_ratio | 正向采购与正向销售的匹配关系 | \(G_i^P/G_i^S\) | 进项+销项 | 企业代号、价税合计、状态 | 发票→企业 |
| 活跃月份比例 | active_month_ratio | 观察期内有真实交易的月份覆盖 | \(m^{-1}\sum_tI_{it}\)，\(I_{it}=1\{\sum_d\sum_j\operatorname{abs}(q_{ijt}^d)>0\}\) | 进项+销项 | 企业代号、日期、价税合计、状态 | 发票→月→企业 |
| 最长连续交易比例 | longest_active_streak_ratio | 经营连续性 | \(\max\{\text{连续 }I_{it}=1\text{ 的月数}\}/m\) | 进项+销项 | 企业代号、日期、价税合计、状态 | 发票→月序列→企业 |

## 3. 预处理与解释风险

| 英文变量名 | 缺失值处理 | 异常值处理 | 是否取对数 | 是否标准化 | 预期与违约风险关系 | 可能的解释风险 |
|---|---|---|---|---|---|---|
| business_scale_10k | Z | U | log1p | 是 | 通常负向 | 与销售、采购规模共线；大额不等于优质 |
| sales_scale_10k | Z | U | log1p | 是 | 通常负向 | 行业价格和企业年龄造成差异 |
| purchase_scale_10k | Z | U | log1p | 是 | 不确定/通常负向 | 高采购也可能是积压或成本压力 |
| net_sales_10k | Z | U | 有符号log1p | 是 | 越高通常风险越低 | 含税票据净额不是收入审计值 |
| operating_net_inflow_proxy_10k | Z | U | 有符号log1p | 是 | 越高通常风险越低 | 不是利润，未含工资、费用和融资现金流 |
| sales_growth_trend | 无销售时N | U | 否，月额已log | 是 | 正增长通常负向；极端增长不确定 | 受季节性、窗口和低基数影响 |
| sales_monthly_cv | 均值为0时N | U | log1p | 是 | 正向 | 缺月补零会放大新设/停业企业波动 |
| invoice_activity_per_month | Z | U | log1p | 是 | 通常负向 | 多行拆票会抬高活跃度，须先原子化 |
| sales_return_rate | 销售正额为0时N | U | log1p | 是 | 通常正向 | 行业退货惯例不同，相关不等于因果 |
| purchase_return_rate | 采购正额为0时N | U | log1p | 是 | 通常正向 | 可能反映议价能力而非经营恶化 |
| void_invoice_rate | 总票数为0时N | B | 否 | 是 | 通常正向 | 系统或开票习惯也会造成作废 |
| zero_amount_invoice_rate | 有效原子发票数为0时N | B | 否 | 否（审计型） | 不进入主模型 | 精确零值与容差零值必须实际核对；作废零额票不改写为有效交易 |
| customer_count | Z | U | log1p | 是 | 通常负向 | 单位代号不一定等同独立客户 |
| supplier_count | Z | U | log1p | 是 | 通常负向 | 供应链模式和行业差异显著 |
| customer_hhi | 销售正额为0时N | B | 否 | 是 | 正向 | 大客户长期合同也可能提高稳定性 |
| supplier_hhi | 采购正额为0时N | B | 否 | 是 | 正向 | 核心供应商关系不必然是风险 |
| max_customer_share | 销售正额为0时N | B | 否 | 是 | 正向 | 与客户HHI高度相关，系数需谨慎解读 |
| max_supplier_share | 采购正额为0时N | B | 否 | 是 | 正向 | 与供应商HHI高度相关 |
| purchase_sales_ratio | 销售正额为0时N | U | log1p | 是 | 偏离合理区间时可能正向 | 合理区间随行业变化，线性系数可能掩盖U形 |
| active_month_ratio | Z | B | 否 | 是 | 通常负向 | 观察窗和季节性决定可比性 |
| longest_active_streak_ratio | Z | B | 否 | 是 | 通常负向 | 短观察窗会虚高连续性 |

## 4. 模型角色与替代变量组

| 角色 | 特征 |
|---|---|
| 主模型（15项） | sales_scale_10k、purchase_scale_10k、operating_net_inflow_proxy_10k、sales_growth_trend、sales_monthly_cv、invoice_activity_per_month、sales_return_rate、purchase_return_rate、void_invoice_rate、customer_count、supplier_count、customer_hhi、supplier_hhi、purchase_sales_ratio、active_month_ratio |
| 敏感性分析额外5项 | business_scale_10k、net_sales_10k、max_customer_share、max_supplier_share、longest_active_streak_ratio |
| 审计型排除 | zero_amount_invoice_rate |

主模型选择规则已锁定：business_scale_10k不与销售、采购规模同时作为主模型变量；sales_scale_10k替代net_sales_10k；customer_hhi替代max_customer_share；supplier_hhi替代max_supplier_share；active_month_ratio替代longest_active_streak_ratio。被替代变量保留在sensitivity_model_features中，不从企业特征表删除。

## 5. 禁用与辅助变量

- credit_rating：不得进入主违约模型特征矩阵；可单独用于评级—违约列联验证和标记为rating_leakage_sensitivity_only的实验，不得替代行为主模型。
- enterprise_name：仅用于结果展示，不做文本或行业特征。
- default_label：由“是否违约”映射，是监督标签而非输入特征。
- zero_amount_invoice_rate：保留在features.names和企业级特征表中，仅作为审计型派生字段；不进入primary_model_features或sensitivity_model_features。
- audit_开头字段：仅作审计辅助，不进入任何模型矩阵。
- 任何只存在于附件1、无法在附件2同口径计算的变量，均不得加入主特征集。
