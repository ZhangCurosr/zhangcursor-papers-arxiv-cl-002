---
title: "You-Shouldn-t-Have-Asked-A-Pragmatics-Inspired-Taxonomy-for"
source: https://arxiv.org/pdf/2608.30856v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:36:21"
---

# 论文速读：You-Shouldn-t-Have-Asked-A-Pragmatics-Inspired-Taxonomy-for

## 一句话总结
本文从语用学视角提出首个面向 LLM 拒绝行为（non-compliance）的三层分类体系，结合人工标注与 LLM-as-Judge 流水线，对 16 款主流大模型在 14 类有害场景下的拒绝策略进行大规模实证分析，揭示当前模型拒绝普遍呈现“显式化、道德评判强、人际面子修复薄弱”的特征，并发现近年前沿模型在安全对齐过程中显著减少了道歉性话术。

## 研究问题与动机
- 现有安全对齐工作主要将非合规视为二元安全决策（是否拒绝/为何拒绝），缺乏对拒绝“如何实现”（语言形式、互动策略、面子管理）的系统评估框架。
- 语用学与会话分析将拒绝视为典型的“面子威胁行为”（face-threatening act），人类在拒绝时会通过道歉、缓冲、解释或提供替代方案来缓解互动成本；但 LLM 是否遵循类似策略、是否存在社交冷漠风险尚不明确。
- 拒绝风格直接影响用户情绪反应、信任建立与后续交互意愿，尤其在自残等敏感语境下，生硬或道德化拒绝可能加剧用户羞耻感甚至引发二次伤害。
- 安全对齐的演进可能存在隐性交互代价：随着模型越来越“果断”地拒绝，其礼貌策略与人际修复能力是否在退化，亟需语用维度的量化证据。

## 核心贡献（创新点）
1. **提出三层语用学驱动的 LLM 非合规分类体系**，将拒绝行为解构为 Layer 0（行动合规度）、Layer 1（拒绝理由归因）、Layer 2（实现策略与附带特征），填补了从“是否拒绝”到“如何拒绝”的分析空白。
2. **构建高一致性的 LLM-as-Judge 自动化标注流水线**，通过人工双盲黄金集（κ=1.0）校验 GPT-5.5 非推理模式作为 Judge 的可靠性，实现从百级样本到数千级响应的可扩展评估。
3. **完成 16 款主流模型 × 200 条有害查询的大规模实证**，首次系统刻画跨模型族、参数量、推理模式及危害类别的拒绝风格差异，并揭示 OpenAI 与 Anthropic 近两年显著降低道歉率的历时趋势。
4. **警示安全对齐的潜在社交代价**，指出当前 frontier 模型倾向于以道德评判替代人际缓冲，呼吁对齐评估不仅关注安全性，还需考虑拒绝的语境适应性与互动后果的社会责任。

## 方法详解
- **三层分类体系设计**：
  - **Layer 0 (Action)**：判定输出相对于用户请求的完成度，分为 Full compliance、Partial compliance、Non-compliance。仅当判定为 Non-compliance 时进入后续层级。编码严格以“任务目标是否达成”为准，不以模型自称的安全声明为依据。
  - **Layer 1 (Rationale)**：识别拒绝的显式理由，分为 Bare（无实质理由）、Capacity-based（声称能力/权限/知识缺失）、Policy-based（归因于系统规则/安全策略等外部约束）、Ethics-based（归因于请求行为本身的危害/不道德性）。核心区分在于责任归因位置（外推 vs. 内化道德立场）。
  - **Layer 2 (Realization & Adjuncts)**：实现策略互斥（Explicit non-compliance 含 action-negating marker；Implicit non-compliance 依赖推断或非否定性表达）。附带特征可共现，包括：Apology/regret、Hedge/epistemic softener、Explanatory preface、Positive alignment、Solidarity/empathy、Negative stance、Alternative offer/switch of topic、Executed alternative、Normative suggestion、Statement of principle、Role-based self-positioning。
- **数据构建**：从 SORRY-Bench（132条）与 LMSYS-Chat-1M（68条）抽样，经 Llama Guard 3 重分类后保留 200 条跨 14 类危害的提示词；每类均衡采样，去除模板重复与 benign rewrite 变体。
- **标注与 Judge 校准**：两位作者独立标注 100 对样本构成 Gold Label；筛选 GPT-5.5（非推理模式）作为 Judge，提示词约 6,000 tokens，含 25 个跨 Layer 的校准示例。Judge 与人工仲裁的一致性：Layer 0 κ=0.857，Layer 1 κ=0.860，多数 Layer 2 特征 κ>0.8。
- **分析流程**：按模型属性（家族、规模、推理模式、发布时间）与危害类别分层统计；使用多维尺度分析（MDS）可视化危害类别的拒绝策略聚类；跨时间序列对比 GPT-4o→GPT-5.3 与 Claude Opus 3→Opus 4.6 的风格演变。

## 实验与结果
- **数据集与模型规模**：200 条有害提示（14 类 Llama Guard 3 范畴），16 款模型（OpenAI 4、Anthropic 4、Google/xAI 2、Meta 2、Qwen 4），共生成 3,200 对查询-响应。
- **合规率分布**：整体合规 21%（完全 17%、部分 4%），API 级硬性拦截 2%，不合规 77%。合规率跨度大：Qwen3-32B (reasoning) 最高（38%），Llama-3.1-8B 最低（8%）；Violent Crimes（95%）与 Sex-Related Crimes（92%）几乎全拒，Specialized Advice 仅 47%。
- **Layer 1 理由倾向**：Ethics-based 占绝对主导（70%），Bare 占 23%，Policy-based 6%，Capacity-based 1%。GPT-4o（92%）、Llama-3.1-8B（77%）、Llama-3.1-70B（68%）以 Bare 模板为主，其余模型普遍采用道德正当性辩护。
- **Layer
