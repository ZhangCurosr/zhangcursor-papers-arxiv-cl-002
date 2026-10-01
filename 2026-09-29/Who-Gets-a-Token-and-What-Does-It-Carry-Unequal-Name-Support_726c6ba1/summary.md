---
title: "Who-Gets-a-Token-and-What-Does-It-Carry-Unequal-Name-Support"
source: https://arxiv.org/pdf/2609.34065v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 04:27:14"
---

# 论文速读：Who-Gets-a-Token-and-What-Does-It-Carry-Unequal-Name-Support

## 一句话总结
本文提出 **NAMETRACE** 框架，揭示并量化了 LLM 中“名称表面支持不均”现象：即使社会属性匹配的名字，因分词器表征差异（单 token vs 多子词）也会在任务相关的内部表征中产生系统性可达性差距，且该信号可跨名迁移并对下游受约束决策产生可干预的杠杆效应。

## 研究问题与动机
- **核心假设检验**：现有名字条件公平性评估隐含“匹配名字即等价输入”的前提，但 tokenizer 实际会将部分名字编码为单 token（atomic），另一部分拆分为 2-3 个子词（short-fragmented），造成输入侧词汇支持不平等。
- **传导机制未知**：这种词汇接口差异仅是 tokenizer 属性，还是会渗透进模型任务相关的内部表征，并在中间层产生可测量的概念可达性差异？
- **行为评估的滞后性**：传统评测依赖最终生成文本或 LLM judge，受回复顺序、风格、 verbosity 等混杂因素干扰；需要一种无需参考答案与外部裁判的 pre-behavioral、模型原生测量。
- **缺乏大规模词汇-表征联动证据**：前期工作多关注低频词表征噪声或嵌入偏差，但未在现代 LLM tokenizer 面板上系统刻画名字表面支持的人口统计学结构，及其向任务概念的延伸。

## 核心贡献（创新点）
1. **首次在大尺度上刻画 LLM tokenizer 对第一名的词汇访问不平等**：跨 12 个 tokenizer、近 50 万名字验证 atomic access 高度选择性（仅 4,052 个全共享），且与族裔/性别元数据强相关。本质区别：将“名字分词差异”从噪声提升为可测量的实验控制变量。
2. **提出 NAMETRACE 框架，实现任务对齐的概念可达性测量**：利用 SentiWordNet 连续效价结合任务方向构建加权 $w_{r,a}=p_r v_a$，在中间层读取模型自身概率分布。本质区别：不依赖离散标签或外部 judge，可捕获同向概念的不同强度（如 excellent 比 promising 贡献更大），且允许任务语义反转通用情感极性。
3. **证明词汇支持差异在控制人口属性后仍独立预测四类任务的内部可达性**：在 200 对同族裔-性别层内匹配名上，atomic 始终获得更高的 task-aligned 概念分数（Fellowship +0.131、Hiring +0.072、Clinical +0.059、Lending +0.051）。本质区别：剥离频率、长度、元数据置信度后，仍分离出词汇支持本身的预测力。
4. **建立“跨名迁移+隐状态干预”的完整因果可验证链条**：开发集支持先验可解释未见名 73%-97% 的差距（Pearson $r=0.992$），且沿任务方向对隐状态的微小偏移可系统改变后续二选一决策（forward-minus-reverse 对比 0.094-0.155）。本质区别：从相关性观测推进到可干预性验证，确立词汇支持是下游行为差异的前置可测量源。

## 方法详解
- **词汇访问映射（RQ1）**：基于佛罗里达州选民登记库（497,583 个单字英文名），判定名字是否为 atomic（单 token 无损解码）。构建 logistic 模型，标准化 log frequency 与字符长度后估计 odds ratio，并交叉族裔/性别元数据考察 intersectional allocation。
- **匹配名对构建（RQ2）**：在 8 个 race/ethnicity × gender 层内，筛选 count≥50、 demographic share≥0.65 的高置信名，按匹配评分函数配对原子名与短碎片名（2-3 token）：
  $\text{MatchScore} = |\Delta\log(\text{count})| + 0.25|\Delta\log\text{th}| + 1.5|\Delta\text{race share}| + 1.5|\Delta\text{gender share}| + 0.25\mathbb{1}[\text{首字符不同}]$
  取每层最优 25 对，共 200 对，平分开发/评估集。
- **NAMETRACE 概念可达性评分**：任务轴由开发集自动构建（保留 recurrence≥30、$|v_a|≥0.5$、跨源极性一致、通过 anchor check 的词）。对模型 $m$、层 $\ell$、任务 $r$、证据 $e$：
  $S_{m,\ell,r,e}(s)=\sum_{a\in\mathcal{A}_r} P_{m,\ell}(a\mid s,r,e)\, w_{r,a}$，其中 $w_{r,a}=p_r v_a$（$p_r=+1$ 用于 fellowship/hiring/lending，$p_r=-1$ 用于 clinical assessment，使 adverse-health 词
