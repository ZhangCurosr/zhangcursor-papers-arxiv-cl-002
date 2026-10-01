---
title: "Who-Gets-a-Token-and-What-Does-It-Carry-Unequal-Name-Support"
source: https://arxiv.org/pdf/2609.34065v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 04:27:00"
field: "大语言模型公平性与可解释性"
keywords: ["name-based fairness", "tokenization bias", "LLM interpretability", "concept accessibility", "hidden-state intervention", "lexical comparability", "demographic bias"]
innovations: ["提出NAMETRACE框架，从词法输入到任务相关内部表征的系统性测量", "发现cross-name transfer可解释72.9%-96.6%的支持差异gap", "通过隐藏状态干预验证任务方向的下游决策杠杆"]
benchmarks: ["Florida voter registration first-name dataset (497,583 names)", "Fellowship/hiring/clinical assessment/lending task axes"]
---

# 论文速读：Who-Gets-a-Token-and-What-Does-It-Carry-Unequal-Name-Support

## 一句话总结
本文系统研究了 LLM 对不同姓名在词法层面的不平等支持（直接单 token vs. 多子词碎片化），并提出了 NAMETRACE 框架，证明这种输入端的不平等会延续至任务相关内部表征，且对未见过姓名具有可迁移性，干预测得的任务方向能实际影响后续决策。

## 研究问题与动机
- **现有评估的隐含假设不成立**：姓名公平性评测通常假设对照实验中"被替换的姓名"是可比输入，但实际上不同姓名在 tokenizer 中可能被表示为单个 token（atomic）或多个子词（fragmented），输入层面并不等价。
- **词法不平等是否仅停留在 token 层，还是渗入任务相关表征**：此前研究分别关注了行为级偏差或 tokenization 效应，但缺乏从输入词法到内部概念可及性的系统性追踪链路。
- **模型族间差异未被量化**：不同 LLM 的 tokenizer 对哪些姓名给予直接访问的选择性极高（12 个 tokenizer 中仅 4,052 个姓名在所有模型中均为 atomic），且该差异呈现人口统计结构化特征。
- **可干预性与因果性验证需求**：即便发现词法支持与内部表征差异相关，仍需验证该信号是否具备下游杠杆（downstream leverage），而非仅为静态相关。

## 核心贡献（创新点）
- **提出 NAMETRACE 框架**：一种模型原生、细粒度、行为前（pre-behavioral）的概念可及性测量方法，通过任务对齐的形容词轴和连续权重从中间层读出任务相关概念的可及性；与已有工作本质区别在于不依赖最终生成结果或外部评判器，直接在模型内部概率分布上测量。
- **大规模词法可达性映射**：在近 50 万姓名和 12 个 LLM tokenizer 上系统性量化了直接 token 访问的不平等分布；区别于 An & Rudinger (2023) 等早期小规模分析，本文覆盖规模大且包含跨模型族结构比较。
- **跨姓名迁移验证**：在开发集估计的支持先验可预测未见姓名上的支持差异（相关系数 r=0.992），证明该模式不是个别姓名特异性的噪声；这一迁移检验视角为已有工作所未涉及。
- **隐藏状态干预验证下游杠杆**：沿测得的任务方向对匹配姓名的隐藏状态进行编辑，可显著偏移后续受限决策；区别于仅做观测性测量的工作，本文提供了因果性干预证据。

## 方法详解
- **匹配姓名对构建**：在八个种族/ ethnicity–性别关联层级内，匹配 atomic（在 Qwen、Llama、Ministral 中均为单 token）与 short-fragmented（三模型中均非 atomic，需 2–3 个 token）的姓名对，共 200 对（400 个姓名），按频率、字符长度、人口统计关联强度、元数据置信度和弱正字法线索进行匹配；前 100 对为开发集（定义任务轴和读出层），后 100 对为保留评估集。
- **任务轴自动构建**：基于 SentiWordNet 3.0 提供每个形容词的连续效价 $v_a \in [-5, 5]$，结合任务取向 $p_r \in \{+1, -1\}$，得到任务对齐权重 $w_{r,a} = p_r v_a$；例如临床评估取 $p_r=-1$，使得 "worried" 虽为负面情感但在任务中仍为正对齐权重。
- **可及性评分公式**：在选定层 $\ell$，对名字 $s$ 在任务 $r$、证据条件 $e$ 下的任务相关概念可及性为：
$$S_{m,\ell,r,e}(s) = \sum_{a \in \mathcal{A}_r} P_{m,\ell}(a \mid s,r,e) \cdot w_{r,a}$$
其中 $P_{m,\ell}(a \mid s,r,e)$ 为模型在层 $\ell$ 对形容词 $a$ 的预测概率。匹配对差异 $\Delta = S(h_g) - S(\ell_g)$ 为正表示 atomic 名字的任务对齐概念可及性更高。
- **跨姓名迁移检验**：开发集上的平均 gap $B_{m,r,e}$ 作为支持先验，直接应用于未见评价集，计算残差 $\widetilde{\Delta} = \Delta - B$，若残差趋近于零则说明迁移良好。
- **隐藏状态干预**：从对齐/对立形容词的表示构造任务方向 $d_{m,r}$，在名字 span 处对 atomic 表示加上 $\alpha d_{m,r}$、对 fragmented 表示减去 $\alpha d_{m,r}$，对比正向与反向编辑后模型在受限二选一决策中的概率变化，衡量任务的下游杠杆。

## 实验与结果
- **数据集**：2022年6月佛罗里达州选民登记提取数据（497,583 个单字名）；分析覆盖 12 个 LLM-associated tokenizer；主实验模型为 Qwen3-4B、Llama-3.1-8B、Ministral-3-3B（扩展至 8 个模型）。
- **RQ1 结果**：任何 tokenizer 中 atomic 的姓名仅占 23,095/497,583（4.6%），全 12 个 tokenizer 均为 atomic 的仅 4,052 个。经频率和长度调整后，male-associated 名字原子访问率是 female-associated 的 3.36 倍（49.8% vs 25.7%）；NH Black-associated 名字仅 17.6%，NH White-associated 为 47.2%；交叉分层后从 NH Black 女性（12.1%）到 NH White 男性（64.8%）。
- **RQ2 结果（核心表格）**：
  | 任务轴 | 加权 gap | 95% CI | 对齐概率 gap |
  |---|---|---|---|
  | Fellowship / promise | 0.131 | [0.096, 0.168] | 0.027 |
  | Hiring / competence | 0.072 | [0.055, 0.091] | 0.023 |
  | Clinical assessment / concern | 0.059 | [0.048, 0.072] | 0.022 |
  | Lending / trustworthiness | 0.051 | [0.041, 0.061] | 0.023 |
  所有任务上 atomic 名字的概念可及性均系统性高于 short-fragmented 名字。Qwen3-4B 效果最大（fellowship gap=0.366），Llama 次之但方向一致，Ministral 部分任务接近零或负值。八个种族/性别层级内全部正向。
- **RQ3 跨姓名迁移**：开发集先验可解释未见面 gaps 的 96.6%（fellowship）、88.3%（hiring）、88.2%（clinical）、72.9%（lending）；Pearson r=0.992，Spearman ρ=0.959，符号一致率 94.4%。
- **RQ3 干预结果**：Qwen 对比值 0.153（200/200 对按预期方向移动），Llama 0.155（200/200），Ministral 0.094（195/200），目标方向显著优于无关轴、极性洗牌和随机子空间三种对照。
- **最强结果**：fellowship 任务的加权 gap 0.131 及迁移解释率 96.6% 为最强；跨 8 模型扩展中 fellowship 在 7/8 模型中均显著为正。

## 相关工作脉络
- **Bertrand & Mullainathan (2004)**：经典 correspondence study，发现 White-associated 名字获约 50% 更多电话回呼；本文继承"固定背景换姓名"的因果识别逻辑，但将焦点从行为输出前移至词法输入层。
- **An & Rudinger (2023)**：在 pretrained LM 中证明人口属性、频率和 tokenization 长度均可导致 first-name 偏差；本文将其扩展到现代 LLM tokenizer 面板，并进一步追踪到任务相关内部表征及下游杠杆。
- **Shwartz et al. (2020); Wolfe & Caliskan (2021)**：低频率姓名在预训练模型中表征更不可靠；本文在大规模匹配控制下证明即使同等频率和人口统计属性的姓名，仅因词法支持不同仍产生系统性可及性差异。
- **Eloundou et al. (2025)**（First-Person Fairness）：使用 LLM-as-judge 评估姓名条件对话行为；本文定位为互补——在生成行为发生之前测量内部可及性，无需外部裁判。
- **Li et al. (2023); Rimsky et al. (2024)**：推理时表示编辑方法；本文借鉴并应用于任务方向干预，验证测得概念的下游因果影响力。

## 局限性与未来方向
- 姓名元数据来源于佛罗里达州选民登记记录，仅为 aggregate 统计关联，不能代表个体人口统计属性；且仅限 ASCII 单字 first name，跨语言/脚本/文化场景有待扩展。
- 名字层面的预训练曝光（name-specific pretraining exposure）未直接观测，因此将词法支持视为可预测变量而非隔离的因果处理。
- 碎片化上限为三 token，未考察更极端的 tokenization 失败情况；effect 在 architecture 间差异大（如 Ministral lending gap 为负），泛化性需更多模型验证。
- 行为前测量（pre-behavioral）与自然对话场景之间存在 gap；后续需研究内部可及性差异如何在开放生成中转化为最终行为。

## 研究启发与可借鉴点
- **NAMETRACE 的方法论可直接迁移**：任务对齐形容词轴 + 连续 SentiWordNet 权重 + 冻结后 held-out 评估的设计，可复用于其他敏感属性（如地域词、宗教词、方言变体）的词法公平性评测。
- **跨姓名迁移检验（support prior transfer）** 是验证测量信号非噪声的强证据，其 Pearson r=0.992 的做法值得在其他偏差评测中采用作为稳健性检验。
- **hidden-state intervention 测量 downstream leverage**：将概念方向嵌入隐藏状态并对比正反编辑的受限决策变化，为"内部表征是否真实影响决策"提供了比相关分析更强的因果证据，可推广至其他可解释性研究。
- **Base vs. post-training 比较发现训练阶段的系统性重排**：同一 tokenizer 下 Qwen 从 base 的临床主导转向 post-training 的 fellowship 主导，提示模型开发过程中词法不平等的影响路径可被后续训练改变或抵消，为干预策略提供切入点。
- **匹配协议（MatchScore 加权设计）**：融合频率对数差、长度、人口统计 share 和首字母差异的匹配函数，为姓名公平性研究提供了可复用的配对控制框架。

## 关键术语表
**Atomic name**：在 tokenizer 中被编码为单个 token 且可无损解码回原 surface 的姓名。
**Short-fragmented name**：在目标 tokenizer 中非 atomic，需 2–3 个 subword token 组合表示的姓名。
**NAMETRACE**：本文提出的模型原生、细粒度、行为前框架，通过任务对齐形容词轴测量内部表征中的概念可及性差异。
**Task-aligned weight ($w_{r,a}$)**：由任务取向 $p_r$ 与形容词效价 $v_a$ 相乘得到的连续权重，使负面情感词在特定任务中仍可正对齐。
**Support prior**：从开发集姓名上估计的平均 atomic–short-fragmented gap，直接应用于未见姓名以检验跨姓名迁移性。
**Cross-name transfer**：开发集估计的支持先验对未见姓名 gap 的预测能力，以 Pearson r=0.992 衡量。
**Downstream leverage**：沿任务方向编辑隐藏状态后对后续受限决策产生的偏移量，由 forward-minus-reverse 对比值度量。
**CKA (Centered Kernel Alignment)**：用于比较跨模型共享 atomic 姓名在输入 embedding 层的成对相似性结构，不受 embedding 维度影响。

## 可复现要素
- **数据集**：2022 年 6 月佛罗里达州选民登记提取数据（public record）；论文声明将发布去标识化的 first-name 级别统计和元数据，不含 row-level PII。
- **代码/权重**：未明确声明代码开源；使用模型均为开源权重（Qwen3-4B Apache-2.0、Llama-3.1-8B Community License、Ministral-3-3B Apache-2.0 等，详见 Appendix Table 25）。
- **关键超参**：匹配对的 MatchScore 权重系数（log count: 1.0, log length: 0.25, race share: 1.5, gender share: 1.5, first-char diff: 0.25）；任务轴选择 $K_{\text{pred}}=20$、$K_{\text{axis}}=10$；干预尺度 α Qwen/Llama=20、Ministral=10。
