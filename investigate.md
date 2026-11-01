# 偏离点干预：从开环推动到救援方法的调查

本文汇总 C61–C64 这一轮的结论，并据此给出"检测 + 选择 + 干预"方法的下一步。实验脚本与结果在 [interpret/](../interpret)，编号沿用 [interpret/document/proposal.md](../interpret/document/proposal.md) 的「五、实验编号」；方案在 [interpret/document/proposal_candidates.md](../interpret/document/proposal_candidates.md)。按约定，C6x 的结果不写进 results.md。

## 背景

模型：冻结的 OmniNav fast system（Qwen2.5-VL 3B），R2R-CE val_unseen。偏离点 = 失败 episode 里人工标注的第一次走错方向的那一步（k = t − d）。步、头、阈值同 C33：主头 L28 h1；选对 = |th4 − θ\*| ≤ 20°；θ\* 为参考路径上前进 1 m 处的方位；只用离路径 < 1 m 的非判停步。

C60（[c60_deviation.py](../interpret/c60_deviation.py)）已确认偏离点上的两条线索：
- **不是指令把方向拉偏**：把指令对模板的注意力全部切断（cut_all）也不更好。
- **而是无指令时区分度低**：注意力剖面在方位上不集中。cut_all 的 tvr 比成功对照低 0.09 [0.007, 0.17]，判据 H2 通过；H1（指令拉离）的逐层判据没通过。

C61（[c61_intervene.py](../interpret/c61_intervene.py)）的开环推动：在 L22–28 全部头上，给模板 query 对某方向 ±10° 视觉列的注意力 logit 加 log β。本轮把 β 从 8 加到 16/32/64，并把这套"检测—选择—干预"展开成四步。

## 一、开环推动：β 加到 64（C61）

H2 目标集（区分度低，n ≈ 180），范围 L22–28 全部头，落进目标 ±20° 的比例：

| β | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| 推向 θ\* | 0.32 | 0.34 | 0.35 | 0.38 | 0.43 | 0.47 | 0.52 |
| 推向镜像 | 0.33 | 0.34 | 0.36 | 0.37 | 0.38 | 0.39 | 0.40 |

- 推向 θ\* 随 β 单调上升，没有饱和；β = 64 比原指令高 0.20 [0.15, 0.26]。
- 推向镜像（以当前朝向为轴、与 θ\* 对称、同角距离）几乎推不动，尽管 L28 上目标列的注意力质量被推到和 θ\* 一样高（0.72 vs 0.69）。
- 判停翻转 ≤ 0.9%。
- 结果：`c61_intervene_dev_b64.json`，补跑数据在 `data/intervene_b64`。

## 二、推动只是在模型已有的候选里重新排序（C63）

[c63_push_candidates.py](../interpret/c63_push_candidates.py)。用 C61 的 609 步，只取推之前 th4 离目标至少 20° 的 315 步。按构造，镜像与 θ\* 离 th4 一样远，是控制了"推动距离"的对照。

| | θ\* | 镜像 |
|---|---|---|
| 目标处有注意力候选峰（4 条件任一） | 57% | 22% |
| 目标处有 navmesh 开口 | 87% | 46% |
| β = 64 落进 ±20°：目标**是**候选 | 36% | 34% |
| β = 64 落进 ±20°：目标**不是**候选 | 5% | 2% |

- **候选决定能否推动**：目标处没有候选时，β = 64 也推不动；有候选时，θ\* 和镜像推得一样好。
- **GT 方向没有额外优势**：线性概率模型里，控制住候选有无和候选强度后，"是不是 θ\*"的系数是 −0.02 [−0.09, +0.05]。θ\* 好推只是因为那里更常有候选、候选也更强。
- **起作用的是注意力候选，不是开口**：加入注意力候选后，navmesh 开口的系数接近 0。
- **这条路的上限**：部署时只有 true 和 cut_all 两次前向可用。θ\* 落在这两者候选里的步只有 39%，加上改写指令（部署时拿不到）也只到 57%。所以"在模型自己的候选里重排"最多能修约四成选错的步，其余要从外部补候选。

结论：推动 = 候选重排。它不能无中生有地造出一个方向，选择器必须解决"选哪个已有候选"。

## 三、检测器：何时干预（C64）

[c64_detect.py](../interpret/c64_detect.py)。只用部署时拿得到的量（true 与 cut_all 两次前向），不训练、不调阈值，比较各读数的 AUC（按路径 bootstrap）。方法来源见 [interpret/document/investigate_look.md](../interpret/document/investigate_look.md)，表中括号给出对应文献。

| 读数（出处） | 偏离点 | 选错 | 可救的选错 |
|---|---|---|---|
| 以 cut_all 为基线的比值 max log₂(r_true/r_cut)（[Zhang et al. 2025](https://arxiv.org/abs/2502.17422)） | 0.60 | 0.64 | 0.71 |
| −tvr_true（现有做法） | 0.59 | 0.59 | 0.71 |
| 有效方向数 exp(熵)（空间熵，[Kang et al. 2025](https://arxiv.org/abs/2503.06287)） | 0.59 | 0.59 | 0.70 |
| true 与 cut_all 的 th4 之差 | 0.56 | 0.58 | 0.71 |
| EMD(true, cut_all)（[Bylinskii et al. 2016](https://arxiv.org/abs/1604.03605)） | 0.56 | 0.53 | 0.71 |
| 按阈值数峰（候选个数、第一/第二峰比） | 0.54 | 0.56 | 0.68 |
| cut_all 自身的 tvr / 熵（H2 标记） | 0.54 | 0.54 | 0.66 |
| 比值、−tvr、th4 差三者的名次平均 | 0.61 | 0.64 | **0.77** |
| 去尾空格前后的 th4 差（无语义扰动当测量噪声；swap2 子集） | — | 0.65 | 0.70 |
| 参照：改写指令下的 th4 差（部署时拿不到） | 0.72 | 0.68 | 0.75 |

三个任务的正例：偏离点（k = −3..0 vs 成功 episode）、选错（swap_all 全部步，基率 30%）、可救的选错（选错且 θ\* 在候选里，基率 10.5%）。

- **整体偏弱**：可部署的读数在"选错"上 AUC 都 ≤ 0.64。名次平均在 10% 预算下命中可救选错 34%（基率 10.5%）。意味着多数干预会落在本来走对的步上，所以选择器必须保守：没有别的候选就不动。
- **比值类 > 按阈值数峰**：支持换掉现在的突出度阈值，改用不依赖阈值的汇总量或带置信带的持续性（[Fasy et al. 2013](https://arxiv.org/abs/1303.7117)）。
- **16 头相加比 L28 h1 低 0.02–0.08**：h1 本来按它与输出的关系挑出，这部分优势可能来自挑选本身；改用不看输出的准则挑头可以排除（[Kang et al. 2025](https://arxiv.org/abs/2503.06287)）。
- **最强信号是改写分歧**（AUC 0.68–0.75），但它部署时拿不到；测试时自造扰动或用 LLM 现生成改写可以替代，见第八节 B（SPUQ）与 C（CLARA）。

investigate_look.md 里适合 / 不适合检测器的方法：
- 适合（按优先级）：
  1. 以 cut_all 为基线的比值，加 value 范数加权（[Kobayashi et al. 2020](https://arxiv.org/abs/2004.10102)），并检查注意力 sink / 高范数 token（[Kang et al. 2025](https://arxiv.org/abs/2503.03321)、[Darcet et al. 2023](https://arxiv.org/abs/2309.16588)）。需重放记 value。
  2. 无语义扰动或自动改写下的行为不稳。
  3. 13 个模板 query 做 bootstrap，给峰的持续性加置信带（[Fasy et al. 2013](https://arxiv.org/abs/1303.7117)）。需重放。
  4. tuned lens 读逐层 th4（[Belrose et al. 2023](https://arxiv.org/abs/2303.08112)），自监督、不用 GT。
- 不适合：
  - A/B 混合拟合（[Caruso et al. 2018](https://www.nature.com/articles/s41467-018-05121-8)）、跨条件持续图（[Kim et al. 2023](https://arxiv.org/abs/2305.04826)）：要多条指令的剖面。
  - 圆形多峰检验（[Bolón et al. 2024](https://arxiv.org/abs/2407.05957)）、临界带宽检验（[Zhang & Wang 2026](https://arxiv.org/abs/2605.18686)）：假设剖面来自 i.i.d. 样本。
  - DAS（[Wu et al. 2023](https://arxiv.org/abs/2305.08809)）、SAE（[Pach et al. 2025](https://arxiv.org/abs/2504.02821)）：要训练。
  - 调温度（[Veličković et al. 2024](https://arxiv.org/abs/2410.01104)）：是干预，不是检测。

### 逐头注意力比 + 线性层（C65，Lookback Lens）

[c65_lookback.py](../interpret/c65_lookback.py)，做法照 [Chuang et al. 2024](https://arxiv.org/abs/2407.07071)：每个头算模板 query 的"三视图 /（三视图 + 指令）"注意力比，接 L2 逻辑回归。只用 true 与 cut_all 两次前向，L13–36 × 16 头；按路径分组 5 折交叉验证，正则强度在训练折内再分 3 折选。

| 特征 | 维数 | 选错 | 可救的选错 | 偏离点 | 迁移：选错 → 偏离点 |
|---|---|---|---|---|---|
| lookback（逐头比，true） | 384 | 0.743 | 0.766 | 0.635 | 0.592 |
| lookback+（再加三视图份额 / 熵 / 峰份额，true 与 cut_all） | 2688 | **0.783** | **0.785** | **0.669** | 0.589 |
| 同样特征按 16 头平均 | 168 | 0.745 | 0.751 | 0.643 | 0.585 |
| 只 L28 的 16 头 | 112 | 0.727 | 0.760 | 0.641 | 0.591 |
| 只 L28 h1 | 7 | 0.660 | 0.748 | 0.603 | 0.585 |
| 混杂对照：步号、历史帧数、指令长度 | 5 | 0.495 | 0.526 | 0.492 | 0.533 |
| lookback+ 加混杂 | 2693 | 0.783 | 0.785 | 0.669 | 0.589 |
| C64 最好的无训练读数 | — | 0.639 | 0.770 | 0.610 | — |

95% CI：选错约 ±0.025，偏离点约 ±0.045（折外分数按路径 bootstrap，未计入折间方差）。

- **加线性层有用**：选错从 0.64 升到 0.78，偏离点从 0.61 升到 0.67。C64 低的一部分原因是只用了一个头、一个读数。
- **不是混杂**：步号、历史帧数、指令长度单独只有 0.49–0.53，加进 lookback+ 也不改变结果。
- **逐头展开贡献不大**：按头平均后仍有 0.745。增益主要来自覆盖全部层和组合几种读数；lookback+ 的额外读数再加 0.04。
- **迁移弱**：用"选错"标签训练、在偏离点上评估，所有特征集都在 0.585–0.592，和 C64 无训练读数相当。能预测"这一步选错"的模式，不能迁移到"导致偏离的那一步"。部署时拿不到人工偏离点标签，所以对检测器来说，真正可用的是迁移这一列。
- **权重大的头分散在各层**（L25 h0、L21 h0、L29 h8、L15 h6、L24 h11 …），L28 h1 不在其中。

## 四、单步闭环救援：已放弃（C62）

闭环只在偏离点 d 这一步干预，之后按原策略走完。变体：**push_gt**（推向 θ\*）、**push_mr**（同 β 推向镜像，强度相同、方向无关的对照）、**exec_gt**（直接把转角换成 θ\*，上界）。

**只有一版结果（61 条，β ≤ 8，`c62_rescue_dev.json`）**：

| | push_gt | push_mr | exec_gt |
|---|---|---|---|
| ΔSR vs 原样 | +0.098 [0.032, 0.175] | +0.049 [0, 0.111] | +0.492 [0.361, 0.618] |

- push_gt 减 push_mr 的 ΔSR = +0.049 [−0.018, +0.117]，含 0，不显著。
- push_gt 救回的 6 条里只有 1 条在 d 处推中了 θ\*。

**结论：单步闭环得不出有效结论，这条线停掉。** 原因是闭环混沌（C50：去尾空格一处改动翻转 20% 的 episode），只在 d 这一步干预一次、之后放任原策略走完，下游的成败被混沌主导，而不是被这一步的干预决定。于是：
- push_gt 和 push_mr 无法区分（上面 CI 含 0），救回的少数 episode 更像扰动发散而非真救援；
- 每臂只跑 1 次，单步干预的效应淹没在混沌里，再加样本也改不了这个设计缺陷。

扩大到 220 条的闭环（`dev_rx`）已**停止**，不再跑；β ≤ 64 下原 61 条（`dev_b64`）跑了一部分即停，未汇总。220 步的开环推动（β = 2..64，`data/intervene_rx`）已跑完并保留，C63 用的是 C61 的 609 步，尚未用到它；d 处能被某个 β 推进 θ\* ±20° 的：61 条里 21 条、220 条里 45 条。

要让闭环能得结论，必须换评测设计，不是加样本——见第七节第 1 条：在 d 处分叉、每臂配 K 个空扰动副本、做配对的随时有效检验，把干预效应从混沌里分出来。而且干预也不应只在 d 一步，应配合第三节的检测器在每个触发步都介入。

## 五、两个被现成数据否证的思路

文献建议里有两个用现有数据就能先判的，都不成立：

- **回归头把两个候选平均掉了**（[CompCPZ 2026](https://arxiv.org/abs/2608.17717) 的现象）：**不成立**。偏离点附近 823 步里 83% 的 th4 落在某个候选峰 ±10° 内，只有 9.7% 夹在两峰之间，且这两类的选错率一样（0.67）。"吸附到最近峰"反而把选对率从 0.33 降到 0.24。模型是选中了一个峰、只是选错了。
- **CFG 外推** θ = θ_cut + γ(θ_true − θ_cut)：**不成立**。γ = 0/1/1.5/2/3 时偏离点选对率 0.37/0.33/0.30/0.29/0.25，成功对照 0.72/0.78/0.75/0.71/0.64。γ > 1 两组都更差——放大一个本就没帮上忙的指令只会更糟。

这两条和 C63 一致：方向的候选来自视觉，指令在偏离点上没提供有用的区分，所以选择器需要一个**和 OmniNav 独立**的信号。

## 六、选择器的思路（文献，待做）

两条警示正好对上数据：[Paraphrase Consensus Is Not Self-Consistency 2026](https://arxiv.org/abs/2610.19438)（多改写取共识只降方差、去不掉共有偏差——所以改写分歧能当检测器，不能当选择器）；[BOKBO 2026](https://arxiv.org/abs/2605.30660)（策略内部置信度/采样分歧主要反映噪声超参，与真实结果无关）。

最值得先试：
1. **独立地标验证器**（[CA-Nav 2024](https://arxiv.org/abs/2412.10137)、[InstructNav 2024](https://arxiv.org/abs/2406.04882)）：指令离线拆成子指令，闭环中用 SigLIP 量级小模型（< 1 GB）给每个候选峰所在视图裁块与当前子指令地标打分。信号与 OmniNav 独立，单卡放得下。[Fast-SmartWay 2025](https://arxiv.org/abs/2511.00933) 设定相同（三前视图、零样本、VLN-CE）带消歧模块，值得读正文。
2. **conformal 候选集**（[KnowNo 2023](https://arxiv.org/abs/2307.01928)、[Introspective Planning 2024](https://arxiv.org/abs/2402.06529)）：在独立校准集（val_seen/train）标定分数使 θ\* ∈ 集合概率 ≥ 1−α；集合为 1 不干预，>1 交给验证器。把检测与选择统一成一个接口。

备选：逐候选遮挡对比（[PCD 2025](https://arxiv.org/abs/2505.13255)，每候选多 2 次前向）；残差 steering 切换候选（[CAA 2023](https://arxiv.org/abs/2312.06681)）；先走一步再看观测、必要时回退（[CGPI 2026](https://arxiv.org/abs/2609.37591)，降 SPL、含轻量适配）。

## 七、端到端评测的思路（文献，待做）

1. **分叉 + 空扰动**（[PEGASUS](https://arxiv.org/abs/1301.3878) 的 common random numbers）：在检测步保存仿真状态分叉，各臂（干预 / 空扰动 ×K / 随机候选 / 镜像）从同一状态出发，估每条 episode 的成功概率，报配对差；检验用对配对差随时有效的置信序列（[betting CS 2020](https://arxiv.org/abs/2010.09686)），[STEP 2025](https://arxiv.org/abs/2503.10966) 按独立试次建模不适合配对。这是唯一能把干预效应和约 20% 混沌翻转分开的设计——现 C62 每臂只跑 1 次，push_gt 与 push_mr 的差别淹没在这个量级里。
2. **按预算的 Qini 曲线**（[RATE 2021](https://arxiv.org/abs/2111.07966)）：按检测分数只干预前 k% 的步，画净救回（救回 − 弄坏）随 k，随机候选 / 镜像在同预算下对照。
3. **监控器协议**（[OGAM 2026](https://arxiv.org/abs/2610.05878) 同基于注意力、[FIPER 2025](https://arxiv.org/abs/2510.09459)、[SAFE 2025](https://arxiv.org/abs/2506.09937)）：只用成功 episode 校准阈值，报成功 episode 上误报率、对人工偏离点召回、报警时刻与偏离点步差；可加对伤害率的 conformal 上界（[Learning What to Say 2026](https://arxiv.org/abs/2606.12299)）。
4. 同类 VLN 插件（[AVERT-VLN 2026](https://arxiv.org/abs/2609.39579)）只报总 SR 增益、无配对检验与空干预对照——配对 + 空扰动 + 净收益的报告方式本身可作一项贡献。

## 八、更多方法（文献，待做）

第六、七节之外再调研的一轮，按"干预 / 检测 / 选择 / 评测"四个环节分。方法描述默认来自摘要，都还没有在本模型上试过。

### A. 干预手段：除注意力 logit 偏置之外

- **注意力偏置的直接对照**：[Prasad et al. 2026](https://arxiv.org/abs/2608.17095) 在 Qwen3-VL 骨干的驾驶 VLA 上，对检测框内视觉 token 加有界的 pre-softmax 偏置（forward pre-hook，不改权重）。下游同样是回归式轨迹头，剂量响应单调，作用集中在后层。和 C61 几乎同款，可作外部对照，它的"零偏置配对"写法也可照搬。
- **表征 steering**：ITI（[Li et al. 2023](https://arxiv.org/abs/2306.03341)）、ActAdd（[Turner et al. 2023](https://arxiv.org/abs/2308.10248)）、RepE（[Zou et al. 2023](https://arxiv.org/abs/2310.01405)）。用"th4 落在左峰 vs 右峰"两组步的激活差构造方向，加到 L22–28 模板 token 的残差上，绕开注意力。用途：检验"推不动非候选方向"卡在注意力还是卡在表征。免训练，只需离线收集激活。
- **动作 token 层面的引导**：[Chan & Kao 2026](https://arxiv.org/abs/2606.15021) 把低维输入注入 VLA 的动作 token 表征，"引导而不覆盖"。介于注意力推动（约 10%）和直接换转角（49% 上界）之间；但它针对自回归动作 token，我们的动作头是回归，要改成在动作头 query 或输入 token 上加偏移。
- **像素空间的视觉提示**：Set-of-Mark（[Yang et al. 2023](https://arxiv.org/abs/2310.11441)）、红圈（[Shtedritski et al. 2023](https://arxiv.org/abs/2304.06712)）、PIVOT（[Nasiriany et al. 2024](https://arxiv.org/abs/2402.07872)）。在候选方位画圈或描边，相当于像素版的 β 偏置，能测试视觉提示能不能把非候选方向变成候选，也就是突破 C63 的限制。风险是图像分布偏移，可能整体伤害 3B 模型。
- **测试时优化视觉 token**：ControlMLLM（[Wu et al. 2024](https://arxiv.org/abs/2407.21534)）用能量函数优化视觉 token，让注意力落到指定区域，可检验 logit 偏置推不动是不是因为偏置形式太粗。每步要反传几次，12 GB 下要开梯度检查点。Prompt Highlighter（[Zhang et al. 2023](https://arxiv.org/abs/2312.04302)）对高亮片段做类 CFG 加权，可只高亮指令里的地标词，作 cut_all 的反面，代价是两次前向。
- **稳定视觉表征**：VTI（[Liu et al. 2024](https://arxiv.org/abs/2410.15778)）离线估计视觉与文本隐空间的偏移方向，测试时注入。对口"无指令时区分度低"：看注入后方位剖面的熵是否下降。

最值得先试：Prasad et al. 2026 作对照；ActAdd 式方向向量，最便宜，又能区分注意力瓶颈和表征瓶颈。

### B. 检测：何时干预

- **逐头注意力比 + 线性层**：Lookback Lens（[Chuang et al. 2024](https://arxiv.org/abs/2407.07071)）每个头算"对上下文 / 对新 token"的注意力比，再接线性分类器。和 tvr 同类，但按头展开。可检验 C64 的 AUC 0.56–0.64 是不是跨头平均把信号抹掉了。要训一个很小的探针，不动 VLN 模型。
- **自造扰动的不确定性**：SPUQ（[Gao et al. 2024](https://arxiv.org/abs/2403.02509)）对输入做扰动，看多次输出的一致性；语义熵（[Kuhn et al. 2023](https://arxiv.org/abs/2302.09664)）先把等义输出聚类再算熵。改写分歧是 C64 里最强的信号，SPUQ 给了部署时自己造扰动的办法（同义替换、去空格）。把 th4 按方位分箱，即得"方向语义熵"。代价是 K 次前向。
- **蒸馏成单次前向**：语义熵探针（[Kossen et al. 2024](https://arxiv.org/abs/2406.15927)）用线性探针从隐状态预测语义熵。离线用改写分歧作标签，从最后层模板 token 预测，把 K 次前向压回一次。
- **VLN 里的"该问"检测**：[Abraham et al. 2024](https://arxiv.org/abs/2411.05831) 用基于注意力的指令模糊度模块，判断何时缺信息。任务和我们同构，但模块是训练出来的，只适合作参考基线。
- **conformal 门控计算模式**：CovNav（[Wu et al. 2026](https://arxiv.org/abs/2610.27677)）免训练，用 conformal prediction 把快速模式的不确定性转成"何时切到慢思考"（ObjectNav）。可把"何时多跑一次 cut_all、何时调用外部选择器"形式化成带覆盖率保证的门控，需要标定集。

最值得先试：逐头注意力比加线性层，成本最低；自造扰动加蒸馏成单次前向的探针。

### C. 选择：不用 GT 在候选里选

- **CLIP 对各方向打 grounding 分**：CLIP-Nav（[Dorbala et al. 2022](https://arxiv.org/abs/2211.16649)）零样本 VLN，把指令拆成关键短语，用 CLIP 给各方向视图打分来选方向。直接用法：用 CLIP / SigLIP（< 1 GB）给每个候选峰方位的视图裁片和当前子指令打分，选出后施加 β 或直接替换转角。免训练，单卡可行。
- **开放集检测定位地标**：Grounding DINO（[Liu et al. 2023](https://arxiv.org/abs/2303.05499)）给文本短语就能定位物体框。先检测地标在哪个方位，再与候选峰求交；顺带统计"地标匹配多个位置"的频次，量化歧义。Swin-T 版约 0.7 GB。
- **语用重排（speaker）**：[Fried et al. 2017](https://arxiv.org/abs/1711.04987)、Speaker-Follower（[Fried et al. 2018](https://arxiv.org/abs/1806.02724)）用 P(指令 | 路径) 给 follower 的候选重排。原文 speaker 是训练出来的；免训练版可让某个 VLM 判断"指令描述的是不是这个方向"，选 P(指令 | 候选) 最大的。
- **LLM 判别候选**：NavGPT（[Zhou et al. 2023](https://arxiv.org/abs/2305.16986)）、MapGPT（[Chen et al. 2024](https://arxiv.org/abs/2401.07314)）、DiscussNav（[Long et al. 2023](https://arxiv.org/abs/2309.11382)）、Open-Nav（[Qiao et al. 2024](https://arxiv.org/abs/2409.18794)，连续环境、开源 LLM）。只在检测为高风险的步把 2–3 个候选方向的描述交给外部 LLM 判别。单卡放不下，要走 API；只在门控选中的少数步调用，成本可控。
- **回溯与跨步价值累积**：SmartWay（[Shi et al. 2025](https://arxiv.org/abs/2503.10069)）在 VLN-CE 里加历史感知推理和回溯，给"走错后回到岔路"第二次机会，也绕开单步干预被混沌淹没的问题；其路点预测器是训练过的。VLFM（[Yokoyama et al. 2023](https://arxiv.org/abs/2312.03275)）用 VLM 打分构建前沿价值图，可借来跨步累积候选方位的 CLIP 分、降低单步噪声；原文用 BLIP-2，显存紧时换 SigLIP。
- **指令侧消歧**：CLARA（[Park et al. 2023](https://arxiv.org/abs/2306.10376)）用 LLM 判断指令是否模糊并生成澄清。判为模糊时生成补全改写，同时作为 B 节的扰动源。需要外部 LLM。

最值得先试：CLIP / SigLIP 在候选峰内打分，再接 β 或换转角，最便宜；语用重排作框架。

### D. 评测与统计

- **个体反事实的形式化**：[Oberst & Sontag 2019](https://arxiv.org/abs/1905.05824) 用 Gumbel-max 结构因果模型对已观测轨迹做反事实推断。说明"同一 episode 换一个干预"要固定外生噪声（同 seed、同指令字节）才算个体反事实，且这类量一般不可识别、需额外假设。
- **伤害的定义**：[Richens et al. 2022](https://arxiv.org/abs/2204.12993) 把伤害定义为"干预让结果比不干预更差"。用来把"救回 10%"拆成救回 a%、弄坏 b%。
- **救回与弄坏的不对称效用**：[Ben-Michael et al. 2022](https://arxiv.org/abs/2206.10479) 给救回和弄坏赋不对称效用，在部分可识别下给界。配对结果的四格表（成→成、成→败、败→成、败→败）可直接套用。
- **异质效应**：因果森林（[Wager & Athey 2015](https://arxiv.org/abs/1510.04342)）以"目标是否候选""剖面熵""改写分歧"为协变量，估计哪类步上干预效果为正，把检测问题转成 CATE 估计。CPU 即可。
- **OPE 的适用边界**：双稳健 OPE（[Jiang & Li 2015](https://arxiv.org/abs/1511.03722)）。动作头是确定性回归、没有行为概率，重要性采样不能直接用，只能对"干预或不干预"的离散门控做 OPE。论文里要讲明。
- **少 run 下的报告方式**：[Agarwal et al. 2021](https://arxiv.org/abs/2108.13264) 主张用分层 bootstrap CI、IQM 和性能剖面代替点估计。可把多种无语义扰动当作多个 run 来报告。

最值得先试：四格表的 harm/benefit 分解，加上固定外生噪声的配对（与第七节第 1 条的分叉设计合用）。

## 下一步

1. **先把闭环评测设计立起来**（否则闭环结论都不可信）：30 条小子集、在 d 处分叉、每臂 K = 5 空扰动副本、做配对的随时有效检验，确认能把干预效应从混沌里分出来，再铺开。单步 C62 已停，不再用它的协议。
2. 检测器：以 cut_all 为基线的比值加 value 范数加权 + sink 检查（需重放记 value）。
3. 选择器：先搭一个 SigLIP 量级的独立地标验证器，在 d 步的候选集上先做开环，看它选对候选的比例能否超过 C63 给的四成上限。
4. 干预不应只在 d 一步：配合检测器在每个触发步介入，再进上面的闭环评测。

脚本：[c61_intervene.py](../interpret/c61_intervene.py)（+ [replay.py](../interpret/replay.py) `intervene` 的 `--betas/--scopes/--push-only/--sets`）、[c62_rescue.py](../interpret/c62_rescue.py)、[c63_push_candidates.py](../interpret/c63_push_candidates.py)、[c64_detect.py](../interpret/c64_detect.py)、[c60_deviation.py](../interpret/c60_deviation.py) `--rescue all`。

## 参考文献

按首次出现的节分组。arXiv 条目的作者、年份、标题均用 arXiv API 核对（2026-10-22）；方法描述默认来自摘要，2026 年的多为 v1 预印本，结论待读正文确认。

**三、检测器：何时干预（C64）**

- Zhang et al. 2025. MLLMs Know Where to Look: Training-free Perception of Small Visual Details with Multimodal LLMs. [arXiv:2502.17422](https://arxiv.org/abs/2502.17422)
- Kang et al. 2025. Your Large Vision-Language Model Only Needs A Few Attention Heads For Visual Grounding. [arXiv:2503.06287](https://arxiv.org/abs/2503.06287)
- Bylinskii et al. 2016. What do different evaluation metrics tell us about saliency models?. [arXiv:1604.03605](https://arxiv.org/abs/1604.03605)
- Fasy et al. 2013. Confidence sets for persistence diagrams. [arXiv:1303.7117](https://arxiv.org/abs/1303.7117)
- Kobayashi et al. 2020. Attention is Not Only a Weight: Analyzing Transformers with Vector Norms. [arXiv:2004.10102](https://arxiv.org/abs/2004.10102)
- Kang et al. 2025. See What You Are Told: Visual Attention Sink in Large Multimodal Models. [arXiv:2503.03321](https://arxiv.org/abs/2503.03321)
- Darcet et al. 2023. Vision Transformers Need Registers. [arXiv:2309.16588](https://arxiv.org/abs/2309.16588)
- Belrose et al. 2023. Eliciting Latent Predictions from Transformers with the Tuned Lens. [arXiv:2303.08112](https://arxiv.org/abs/2303.08112)
- Kim et al. 2023. Peak-Persistence Diagrams for Estimating Shapes and Functions from Noisy Data. [arXiv:2305.04826](https://arxiv.org/abs/2305.04826)
- Caruso et al. 2018. Single neurons may encode simultaneous stimuli by switching between activity patterns. Nature Communications 9, 2715. [doi](https://www.nature.com/articles/s41467-018-05121-8)
- Bolón et al. 2024. A likelihood ratio test for circular multimodality. [arXiv:2407.05957](https://arxiv.org/abs/2407.05957)
- Zhang & Wang 2026. critband: A Python Package for Critical Bandwidth Analysis of Multimodal Distributions. [arXiv:2605.18686](https://arxiv.org/abs/2605.18686)
- Wu et al. 2023. Interpretability at Scale: Identifying Causal Mechanisms in Alpaca. [arXiv:2305.08809](https://arxiv.org/abs/2305.08809)
- Pach et al. 2025. Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models. [arXiv:2504.02821](https://arxiv.org/abs/2504.02821)
- Veličković et al. 2024. Softmax is not Enough (for Sharp Size Generalisation). [arXiv:2410.01104](https://arxiv.org/abs/2410.01104)

**五、两个被现成数据否证的思路**

- Zhang et al. 2026. CompCPZ: Preserving Multi-Modal Intent in Language-Guided Robot Manipulation. [arXiv:2608.17717](https://arxiv.org/abs/2608.17717)

**六、选择器的思路（文献，待做）**

- Li et al. 2026. Paraphrase Consensus Is Not Self-Consistency: Instruction Reliance and Robustness in VLA Policies. [arXiv:2610.19438](https://arxiv.org/abs/2610.19438)
- Singh et al. 2026. BOKBO (Best of K Bad Options): Calibrated Abstention for VLA Policies. [arXiv:2605.30660](https://arxiv.org/abs/2605.30660)
- Chen et al. 2024. Constraint-Aware Zero-Shot Vision-Language Navigation in Continuous Environments. [arXiv:2412.10137](https://arxiv.org/abs/2412.10137)
- Long et al. 2024. InstructNav: Zero-shot System for Generic Instruction Navigation in Unexplored Environment. [arXiv:2406.04882](https://arxiv.org/abs/2406.04882)
- Shi et al. 2025. Fast-SmartWay: Panoramic-Free End-to-End Zero-Shot Vision-and-Language Navigation. [arXiv:2511.00933](https://arxiv.org/abs/2511.00933)
- Ren et al. 2023. Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners. [arXiv:2307.01928](https://arxiv.org/abs/2307.01928)
- Liang et al. 2024. Introspective Planning: Aligning Robots' Uncertainty with Inherent Task Ambiguity. [arXiv:2402.06529](https://arxiv.org/abs/2402.06529)
- Wu et al. 2025. Policy Contrastive Decoding for Robotic Foundation Models. [arXiv:2505.13255](https://arxiv.org/abs/2505.13255)
- Panickssery et al. 2023. Steering Llama 2 via Contrastive Activation Addition. [arXiv:2312.06681](https://arxiv.org/abs/2312.06681)
- Li et al. 2026. Credit-Guided Policy Improvement for Test-time Adaptive Vision-Language Navigation. [arXiv:2609.37591](https://arxiv.org/abs/2609.37591)

**七、端到端评测的思路（文献，待做）**

- Ng & Jordan 2013. PEGASUS: A Policy Search Method for Large MDPs and POMDPs. [arXiv:1301.3878](https://arxiv.org/abs/1301.3878)
- Waudby-Smith & Ramdas 2020. Estimating means of bounded random variables by betting. [arXiv:2010.09686](https://arxiv.org/abs/2010.09686)
- Snyder et al. 2025. Is Your Imitation Learning Policy Better than Mine? Policy Comparison with Near-Optimal Stopping. [arXiv:2503.10966](https://arxiv.org/abs/2503.10966)
- Yadlowsky et al. 2021. Evaluating Treatment Prioritization Rules via Rank-Weighted Average Treatment Effects. [arXiv:2111.07966](https://arxiv.org/abs/2111.07966)
- Darwish et al. 2026. OGAM: Connecting Systematic Testing to Runtime Assurance through Object-Grounded Attention Monitoring for VLA Policies. [arXiv:2610.05878](https://arxiv.org/abs/2610.05878)
- Römer et al. 2025. Failure Prediction at Runtime for Generative Robot Policies. [arXiv:2510.09459](https://arxiv.org/abs/2510.09459)
- Gu et al. 2025. SAFE: Multitask Failure Detection for Vision-Language-Action Models. [arXiv:2506.09937](https://arxiv.org/abs/2506.09937)
- Jeong et al. 2026. Learning What to Say to Your VLA: Mostly Harmless Vision Language Action Model Steering. [arXiv:2606.12299](https://arxiv.org/abs/2606.12299)
- Liu et al. 2026. AVERT-VLN: Abstention-aware Visual Error Recovery and Training for Vision-and-Language Navigation. [arXiv:2609.39579](https://arxiv.org/abs/2609.39579)

**八、更多方法（文献，待做）**

- Prasad et al. 2026. Inference-Time Attention Steering for Vision-Language-Action Driving Models. [arXiv:2608.17095](https://arxiv.org/abs/2608.17095)
- Li et al. 2023. Inference-Time Intervention: Eliciting Truthful Answers from a Language Model. [arXiv:2306.03341](https://arxiv.org/abs/2306.03341)
- Turner et al. 2023. Steering Language Models With Activation Engineering. [arXiv:2308.10248](https://arxiv.org/abs/2308.10248)
- Zou et al. 2023. Representation Engineering: A Top-Down Approach to AI Transparency. [arXiv:2310.01405](https://arxiv.org/abs/2310.01405)
- Chan & Kao 2026. Steering Autoregressive Vision-Language-Action Policies via Action Token Intervention. [arXiv:2606.15021](https://arxiv.org/abs/2606.15021)
- Yang et al. 2023. Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V. [arXiv:2310.11441](https://arxiv.org/abs/2310.11441)
- Shtedritski et al. 2023. What does CLIP know about a red circle? Visual prompt engineering for VLMs. [arXiv:2304.06712](https://arxiv.org/abs/2304.06712)
- Nasiriany et al. 2024. PIVOT: Iterative Visual Prompting Elicits Actionable Knowledge for VLMs. [arXiv:2402.07872](https://arxiv.org/abs/2402.07872)
- Wu et al. 2024. ControlMLLM: Training-Free Visual Prompt Learning for Multimodal Large Language Models. [arXiv:2407.21534](https://arxiv.org/abs/2407.21534)
- Zhang et al. 2023. Prompt Highlighter: Interactive Control for Multi-Modal LLMs. [arXiv:2312.04302](https://arxiv.org/abs/2312.04302)
- Liu et al. 2024. Reducing Hallucinations in Vision-Language Models via Latent Space Steering. [arXiv:2410.15778](https://arxiv.org/abs/2410.15778)
- Chuang et al. 2024. Lookback Lens: Detecting and Mitigating Contextual Hallucinations in Large Language Models Using Only Attention Maps. [arXiv:2407.07071](https://arxiv.org/abs/2407.07071)
- Gao et al. 2024. SPUQ: Perturbation-Based Uncertainty Quantification for Large Language Models. [arXiv:2403.02509](https://arxiv.org/abs/2403.02509)
- Kuhn et al. 2023. Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation. [arXiv:2302.09664](https://arxiv.org/abs/2302.09664)
- Kossen et al. 2024. Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs. [arXiv:2406.15927](https://arxiv.org/abs/2406.15927)
- Abraham et al. 2024. To Ask or Not to Ask? Detecting Absence of Information in Vision and Language Navigation. [arXiv:2411.05831](https://arxiv.org/abs/2411.05831)
- Wu et al. 2026. CovNav: Conformal Prediction-Guided Adaptive Inference for Object Goal Navigation. [arXiv:2610.27677](https://arxiv.org/abs/2610.27677)
- Dorbala et al. 2022. CLIP-Nav: Using CLIP for Zero-Shot Vision-and-Language Navigation. [arXiv:2211.16649](https://arxiv.org/abs/2211.16649)
- Liu et al. 2023. Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection. [arXiv:2303.05499](https://arxiv.org/abs/2303.05499)
- Fried et al. 2017. Unified Pragmatic Models for Generating and Following Instructions. [arXiv:1711.04987](https://arxiv.org/abs/1711.04987)
- Fried et al. 2018. Speaker-Follower Models for Vision-and-Language Navigation. [arXiv:1806.02724](https://arxiv.org/abs/1806.02724)
- Zhou et al. 2023. NavGPT: Explicit Reasoning in Vision-and-Language Navigation with Large Language Models. [arXiv:2305.16986](https://arxiv.org/abs/2305.16986)
- Chen et al. 2024. MapGPT: Map-Guided Prompting with Adaptive Path Planning for Vision-and-Language Navigation. [arXiv:2401.07314](https://arxiv.org/abs/2401.07314)
- Long et al. 2023. Discuss Before Moving: Visual Language Navigation via Multi-expert Discussions. [arXiv:2309.11382](https://arxiv.org/abs/2309.11382)
- Qiao et al. 2024. Open-Nav: Exploring Zero-Shot Vision-and-Language Navigation in Continuous Environment with Open-Source LLMs. [arXiv:2409.18794](https://arxiv.org/abs/2409.18794)
- Shi et al. 2025. SmartWay: Enhanced Waypoint Prediction and Backtracking for Zero-Shot Vision-and-Language Navigation. [arXiv:2503.10069](https://arxiv.org/abs/2503.10069)
- Yokoyama et al. 2023. VLFM: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation. [arXiv:2312.03275](https://arxiv.org/abs/2312.03275)
- Park et al. 2023. CLARA: Classifying and Disambiguating User Commands for Reliable Interactive Robotic Agents. [arXiv:2306.10376](https://arxiv.org/abs/2306.10376)
- Oberst & Sontag 2019. Counterfactual Off-Policy Evaluation with Gumbel-Max Structural Causal Models. [arXiv:1905.05824](https://arxiv.org/abs/1905.05824)
- Richens et al. 2022. Counterfactual harm. [arXiv:2204.12993](https://arxiv.org/abs/2204.12993)
- Ben-Michael et al. 2022. Policy Learning with Asymmetric Counterfactual Utilities. [arXiv:2206.10479](https://arxiv.org/abs/2206.10479)
- Wager & Athey 2015. Estimation and Inference of Heterogeneous Treatment Effects using Random Forests. [arXiv:1510.04342](https://arxiv.org/abs/1510.04342)
- Jiang & Li 2015. Doubly Robust Off-policy Value Evaluation for Reinforcement Learning. [arXiv:1511.03722](https://arxiv.org/abs/1511.03722)
- Agarwal et al. 2021. Deep Reinforcement Learning at the Edge of the Statistical Precipice. [arXiv:2108.13264](https://arxiv.org/abs/2108.13264)
