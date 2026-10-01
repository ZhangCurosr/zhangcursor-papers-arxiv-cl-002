---
title: "TRIADIC-LINEAR-ATTENTION-THREE-DIMENSIONAL-RECURRENT-STATES"
source: https://arxiv.org/pdf/2609.36529v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 16:53:25"
field: "高效长上下文序列建模"
keywords: ["线性注意力", "长上下文", "状态空间模型", "张量积", "RNN", "快速权重编程"]
innovations: ["提出三元线性注意力，将矩阵状态扩展为三阶张量状态以实现参数高效的状态扩容", "适配data-dependent forgetting与delta规则至三维状态并实现chunkwise并行GPU kernel", "验证后训练upcycling策略：预训练模型可扩展状态继续长上下文训练以接近从头训练性能"]
benchmarks: ["PG19", "Wikitext-2", "RULER NIAH", "Arora et al. 2024 Recall Suite", "MQAR", "Zero-shot (10 benchmarks)"]
---

# 论文速读：TRIADIC LINEAR ATTENTION

## 一句话总结
本文提出**三元线性注意力（Triadic Linear Attention）**，通过将线性注意力的矩阵状态扩展为三阶张量状态（引入第二key和第二query），以极小的参数开销实现状态容量的E倍增长，显著提升长上下文语言建模与回忆任务性能。

## 研究问题与动机
1. **核心问题**：线性注意力RNN的记忆状态大小直接决定其召回能力，但朴素增大状态尺寸会显著增加参数量。
2. **现有方法不足**：
   - 增大head维度、value维度、head数量或多value-per-key等替代方案会大幅增加投影矩阵参数量，需削减MLP宽度补偿，导致性能下降甚至退化。
   - 稀疏路由或大状态记忆（如DeltaProduct、TTT-MLP等）带来高昂的计算与内存开销。
3. **理论视角**：线性注意力本质是张量积表征（Smolensky, 1990）与快速权重编程（Schmidhuber, 1992），状态容量由$d^2$限制，无法精确存储超过$d$个正交键值对。
4. **动机**：探索一种**参数高效**的方式，在不显著增加参数的情况下扩展状态容量至三阶张量。

## 核心贡献（创新点）
1. **提出三元线性注意力架构**：将普通线性注意力的二维矩阵状态提升为三阶张量状态$S_t \in \mathbb{R}^{d \times E \times d}$，通过三个向量的三阶外积写入、双query收缩读取，新增参数仅为两个投影矩阵，状态容量提升E倍而参数增长可忽略。
2. **将现代线性注意力关键技术适配到三维状态**：
   - **数据依赖遗忘**：沿第二key轴逐slice分配独立的标量forget gate。
   - **Delta规则**：在写入前联合擦除两个key共同存储的旧关联。
   - **Chunkwise并行训练**：利用Kronecker结构因式化解耦联合key，并将状态沿value轴tiling至不同thread block的寄存器中，避免单SM持有完整三维状态。
3. **系统性实验验证**：在400M和1.3B参数规模下，将Triadic线性注意力应用于GDN和sGLA，在PG19（最长64k）、Wikitext、RULER NIAH及回忆benchmark上全面超越同等状态大小的替代扩容方案（更大head、更宽value、更多head等）。
4. **揭示后训练upcycling可行性**：预训练的普通线性注意力模型可通过复制到每个slice并初始化第二key/query投影，在长上下文扩展阶段转化为Triadic变体，恢复大部分从头训练的收益，启发分阶段状态扩容策略。
5. **在GDN/Transformer混合架构中验证性价比**：在3:1 GDN/GQA-8混合模型中，将GDN层变为Triadic（E=4）比扩大KV cache（GQA-4）在更长上下文获得更低perplexity，且64k时显存占用仅为后者的一半。

## 方法详解
### 基础公式
- **写入**：$\mathbf{S}_t = \mathbf{S}_{t-1} + \mathbf{k}_t \otimes \mathbf{k}'_t \otimes \mathbf{v}_t$，其中$\mathbf{k}_t, \mathbf{v}_t \in \mathbb{R}^d$，$\mathbf{k}'_t \in \mathbb{R}^E$为新引入的第二key。
- **读取**：$\mathbf{o}_t = \mathbf{S}_t \times_1 \mathbf{q}_t \times_2 \mathbf{q}'_t = \sum_{s \leq t} (\mathbf{q}_t^\top \mathbf{k}_s)(\mathbf{q}'_t^\top \mathbf{k}'_s)\mathbf{v}_s$，通过两个query分别与两个key轴做收缩。
- 当$E=1$且$\mathbf{k}'=\mathbf{q}'=1$时退化为普通线性注意力，故普通线性注意力是其特例。

### 遗忘机制
- 每个slice沿第二key轴拥有独立的forget gate：$\mathbf{S}_t = \mathbf{S}_{t-1} \times_2 \text{diag}(\boldsymbol{\alpha}_t) + \mathbf{k}_t \otimes \mathbf{k}'_t \otimes \mathbf{v}_t$，slice间为channelwise decay，slice内为标量decay，参数开销极小。

### Delta规则
- 在写入新键值对前，先联合擦除当前两个key对应的旧值：$\mathbf{S}_t = \mathbf{S}_{t-1} + \beta_t \mathbf{k}_t \otimes \mathbf{k}'_t \otimes (\mathbf{v}_t - \mathbf{S}_{t-1} \times_1 \mathbf{k}_t \times_2 \mathbf{k}'_t)$，其中$\beta_t \in (0,1)$为数据依赖的写入强度。
- 等价形式：将联合key $\boldsymbol{\kappa}_t = \mathbf{k}_t \otimes \mathbf{k}'_t$展平为$d \cdot E$维向量，视作Gated DeltaNet的key维度扩充版，配合每slice标量decay。

### Chunkwise并行实现
- **分离联合key**：利用Kronecker积内积可因式分解的性质$(\mathbf{q}^r \otimes \mathbf{q}'^r)^\top (\mathbf{k}^s \otimes \mathbf{k}'^s) = (\mathbf{q}^{r\top}\mathbf{k}^s)(\mathbf{q}'^{r\top}\mathbf{k}'^s)$，将$C \times C$掩码注意力的计算复杂度从$C^2(d \cdot E)$降至$C^2(d+E)$。
- **状态tiling**：将$ d \times E \times d $状态沿value轴切分为32列的block，每个thread block仅保留自身slice，全程不将整个三维状态驻留于单一SM。例如$E=8$时一个head的完整状态为512 KiB（FP32），超过Hopper SM寄存器文件（256 KiB），但单个tiling block仅需128 KiB。
- 完整chunkwise公式含forget gate与delta规则时，见附录A公式(9)-(11)。

### 网络结构
- 第二key/query通过线性投影+短卷积+softplus激活生成；主key/query/value经SiLU激活与L2归一化。
- 每个slice独立学习forget gate（经$-\exp(-\cdot)$参数化保证正值）。
- Triadic GDN = base GDN + 第二key/query投影 + 每slice scalar forget gate。

## 实验与结果
### 数据集与评估
- **语言建模**：PG19（小说，窗口64k）、Wikitext-2；上下文长度覆盖≤4k、4k-16k、16k-64k。
- **零样本任务**：10个benchmark平均（LAMBADA、HellaSwag、PIQA、ARC-Easy/Challenge、WinoGrande、OpenBookQA、SciQ、BoolQ、COPA）。
- **回忆任务**：Arora et al. (2024)六任务套件（DROP、NQ、TriviaQA、FDA、SQuAD、SWDE）；RULER八项NIAH任务。

### 主要结果
- **400M参数**（Table 2）：Triadic GDN (E=8) 从scratch训练 vs GDN base：
  - WikiText：10.87 vs 11.25（↓0.38）
  - PG19 16k-64k：13.74 vs 14.15（↓0.41）
  - Recall：33.1 vs 26.2（↑26.3%）
  - Upcycled版本：WikiText 11.04、PG19 16k-64k 13.86、Recall 29.7（恢复from-scratch约70%增益）
- **1.3B参数**：Triadic GDN (E=8) Recall达44.4（vs GDN base 36.1，↑23%）；PG19 16k-64k为9.94（vs 10.20）。
- **状态匹配对比**（Table 1，2×和4×状态，参数量持平）：
  - Triadic (E=2/4)在所有PG19分段和WikiText均最低perplexity，Recall最高；替代方案在4×时部分退化。
  - 最大提升：Triadic GDN E=4 Recall 31.1 vs GDN base 26.2（↑18.7%）。
- **混合模型**（Table 3）：3:1 Triadic GDN/GQA-8 (E=4) 在PG19各段perplexity均最优，NIAH 56.3 vs GQA-4基线53.0，64k显存占用220 MB vs GQA-4的407 MB（省46%）。

### 效率
- E=8相比GDN训练开销增加28%-30%；在64k上下文下比Transformer快5.1倍（单H100，batch=4）。

## 相关工作脉络
1. **线性注意力基础**（Katharopoulos et al., 2020）：将softmax注意力替换为核特征映射，状态由向量升为矩阵。本文在此基础上将矩阵进一步升为三阶张量。
2. **Gated DeltaNet**（Yang et al., 2025b）：结合data-dependent forgetting与delta规则的线性RNN。本文将其状态从$ d \times d $扩展为$ d \times E \times d $。
3. **Scalar-gated linear attention (sGLA)**：GDN去除erase项的简化变体。本文同步验证triadic改造在sGLA上的有效性。
4. **状态扩容替代方案**（Gu & Dao, 2024; Dao & Gu, 2024）：更大head/value维度、更多head、多value-per-key。本文证明同等参数预算下triadic扩张显著优于这些方案。
5. **测试时训练（TTT系列）**（Sun et al., 2025; von Oswald et al., 2025; Behrouz et al., 2025）：通过内层学习动态更新状态，状态规模通常较大。本文暗示TTT类方法的增益部分源于更大状态容量，triadic提供参数更高效的替代路径。
6. **高阶关联记忆**（Krotov & Hopfield, 2016; Schlag & Schmidhuber, 2018）：三阶张量积表征存储图结构。本文借鉴同样构造但目标不同——用于现代长上下文序列建模而非图推理。
7. **稀疏记忆路由**（Peng et al., 2022; Zhang et al., 2024; Afzal et al., 2026）：通过路由访问大状态空间。本文采用单一稠密三维状态+高效fused kernel，避免稀疏路由的通信开销。

## 局限性与未来方向
1. **训练开销**：E=8仍比原始GDN慢约30%，虽优于Transformer但仍未达到原生线性效率，需进一步kernel优化。
2. **召回能力尚未完全追平Transformer**：在部分回忆密集任务上仍落后于Transformer，仅以远小于KV cache的状态达成。
3. **仅验证于GDN与sGLA**：未扩展到其他线性注意力变体（如Mamba2、RWKV、xLSTM等）及更广泛的SSM框架。
4. **第二key激活函数选择受限**：ablation显示非负激活（softplus/sigmoid）优于有符号激活（SiLU/线性），但机理尚待系统解释。
5. **未来方向**：将三阶张量状态思想推广至其他线性注意力变体；探索更高阶（n>3）状态；与MoE架构结合（小$d_k$+大$E$可减少投影维度）；设计更激进的tiling策略进一步压缩显存。

## 研究启发与可借鉴点
1. **张量积升阶扩容量**：将线性注意力的outer product从2-向量扩展至n-向量是一种通用范式，可启发探索四阶/五阶状态或异构维度组合（如$d_1 \times d_2 \times d_3$），在参数与容量间寻找新 Pareto前沿。
2. **后训练Upcycling策略**：预训练小状态模型 → 第二阶段扩展状态并继续长上下文训练，可实现"小状态预训练、大状态精调"的分阶段训练范式的可复现流程。
3. **Kronecker因式分解加速**：利用张量积的内积因式分解性质将$C^2 \cdot (d \cdot E)$降至$C^2 \cdot (d+E)$，结合状态tiling将长程状态分布至寄存器，为后续高阶张量注意力kernel设计提供模板。
4. **混合架构中状态分配权衡**：在GDN/Transformer混合模型中，扩大线性状态比扩大KV cache更具显存效率且长上下文性能更优，启发未来研究应关注"线性状态vs softmax缓存"的动态容量分配。
5. **Per-slice独立遗忘门设计**：为每个slice分配独立decay gate的同时保持整体scalar形式，兼顾表达能力与计算效率，可推广至其他多维状态架构。

## 关键术语表
- **Triadic Linear Attention（三元线性注意力）**：将线性注意力的状态从二维矩阵扩展为三阶张量，通过第三向量（第二key）的outer product将状态容量提升E倍的新型序列混合器。
- **Gated DeltaNet (GDN)**：结合data-dependent forgetting与delta erase-write规则的线性RNN变体，本文的核心应用基线之一。
- **Scalar-gated Linear Attention (sGLA)**：GDN去除delta规则中erase项的简化版本，仅保留数据依赖gate与outer-product写入。
- **Chunkwise-parallel Training（分块并行训练）**：将序列按固定长度分块，块内并行计算attention、块间以递归状态传递，实现高效并行化训练线性注意力。
- **Data-dependent Forgetting（数据依赖遗忘）**：通过软门控机制让模型根据当前输入动态控制状态的遗忘速率，而非固定指数衰减。
- **Delta Rule（Delta规则）**：在写入新键值对之前先从状态中擦除与该key对应的旧值，避免过时信息的累积干扰。
- **Multi-query Associative Recall (MQAR)**：基准测试，模型需从状态中回忆N个随机顺序存储的键值对，用于量化状态容量上限。
- **Needle-in-a-Haystack (NIAH)**：长上下文回忆任务，在长文档中检索特定短信息，用于评估模型的实际定位与召回能力。

## 可复现要素
- **数据集**：Fineweb-Edu（预训练）、PG19（长上下文扩展与评估）、Wikitext-2、RULER、Arora et al. (2024)回忆benchmark套件；多数公开可用。
- **代码/权重**：论文未明确声明开源，但提及使用CuTe DSL实现kernel（基于CUTLASS）；附录给出Algorithm 1/2伪代码与详细chunkwise公式(9)-(11)。
- **关键超参**：
  - 400M：24层，$d_{model}=1024$，8 heads，head dim $d=128$；1.3B：24层，$d_{model}=2048$，16 heads，$d=128$。
  - 预训练：50 tokens/parameter（Chinchilla 2.5×），peak LR $3 \times 10^{-4}$，weight decay 0.1，batch ~0.5M tokens。
  - 长上下文扩展：5 tokens/parameter，peak LR $10^{-4}$，上下文4k→64k。
  - Second key维度$E \in \{1, 2, 4, 8\}$，chunk size $C=64$，value轴tiling block 32列。
  - Tokenizer：Llama-2 32k vocab。
