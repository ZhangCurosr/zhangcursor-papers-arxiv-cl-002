---
title: "WORLDBENCH-Culturally-Grounded-Benchmark-for-Multilingual-Ag"
source: https://arxiv.org/pdf/2609.01056v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 07:32:55"
field: "多语言智能体评测"
keywords: ["multilingual agent benchmark", "constrained task success", "state preservation", "culturally grounded evaluation", "LLM agent", "human-in-the-loop benchmark construction"]
innovations: ["提出CTS指标同时评估任务完成与状态保持", "基于native-language的文化情境化任务构建流程（非翻译）", "引入distractor机制暴露collateral edit故障"]
benchmarks: ["WORLDBENCH"]
---

# 论文速读：WORLDBENCH-Culturally-Grounded-Benchmark-for-Multilingual-Ag

## 一句话总结
论文提出了 **WORLDBENCH**，一个文化情境化的多语言智能体基准测试，包含1,600个任务、7种语言、8种文化背景，通过引入 **CTS（Constrained Task Success）** 指标同时评估任务完成率和环境状态保持能力，揭示当前前沿LLM智能体在长周期、多语言工作流中仍存在严重脆弱性。

---

## 研究问题与动机
1. **上下文缺失（Contextual Grounding）**：现有基准的任务通常为通用指令，而真实工作流由特定用户的角色、位置、语言和工作习惯塑造，忽略这些上下文会导致智能体基于错误假设行动。
2. **状态保持不足（State Preservation）**：当前评估只关注任务正确性，智能体可能完成任务但意外修改无关文件、覆盖记录或删除文件，这在真实工作流中是关键缺陷。
3. **多语言覆盖薄弱（Multilingual Coverage）**：多数基准以英语为主并简单翻译，未考虑本地化文档惯例和文化差异，无法可靠评估跨语言性能。
4. **现有基准类型局限**：既有benchmark聚焦网页导航、工具使用、编程或操作系统控制，缺乏针对文件based长周期工作流的评估。

---

## 核心贡献（创新点）
1. **提出WORLDBENCH基准**：构建包含1,600个任务、7种语言、8种文化背景的 culturally-grounded 多语言智能体测试集，任务从种子出发经自动化生成+人类审核构建，而非简单翻译。
   - 与已有工作的本质区别：每个任务的 persona、文件、评估标准均在目标语言和文化语境中直接构建，保留 locale-specific 约定，同时保持评估逻辑一致以支持跨语言比较。

2. **设计CTS（Constrained Task Success）指标**：将任务完成正确性与环境状态保持（preserve non-target files）结合为二元指标，首次在同一框架下同时衡量两者。
   - 与已有工作的本质区别：现有基准仅报告 pass rate，WORLDBENCH 通过 preserve(t) 约束捕获"正确答案+破坏无关文件"类失败，更贴合真实工作流安全需求。

3. **引入包含干扰项的测试集设计**：每个 testbed 包含 20 个 distractor 文件（相似文件名、重叠实体、过时版本），迫使智能体精确选择目标 artefact。
   - 与已有工作的本质区别：通过 distractor 机制主动制造保真度陷阱，暴露 collateral edit 类故障。

4. **建立多维度失败分析框架**：将失败归因为 wrong output、collateral edit、iteration-cap hit、malformed loop、execution error 五类，并提供 6 种 recurring error patterns（如 distractor capture、locale mismatch）。
   - 与已有工作的本质区别：不仅报告分数，还提供细粒度诊断，揭示多语言性能差异的主要机制是 locale mismatch（占非英语设置21.4%）。

5. **开源 benchmark 构建流程与实验协议**：提供完整的人审pipeline、确定性+LLM-as-judge混合评估函数、固定解码配置，支持复现与扩展。

---

## 方法详解
### 任务形式化
将 agentic 任务建模为人与沙盒环境的交互式决策过程：
$$
\mathcal{W} = (S, A, O, \delta)
$$
其中 $S$ 为环境状态空间，$A$ 为结构化动作空间，$O$ 为观测空间，$\delta: S \times A \to S \times O$ 为转移函数。

### 工具环境（Table 1）
| Tool family | Operations |
|---|---|
| Spreadsheet | 读取表格记录、编辑单元格值 |
| Document PDF | 读取、创建、更新文本文档；读取PDF内容并支持转换 |
| Messaging | 检查消息记录、撰写消息 |
| Calendar | 列出事件、创建不重叠条目 |
| Shell | 在沙盒内执行文件系统命令 |
| System | 管理全局控制动作和终止执行 |

### 基准构建流程（Figure 1 / Appendix F）
1. **种子阶段**：每语言设置编写20个 culturally-situated persona + 80个人工编写的 seed tasks。
2. **自动扩展**：生成额外30个 persona，每个关联5-6个候选任务，产生150-180个候选。
3. **人类审核（Human Audit）**：40位具备语言/文化专长的 annotators 审查每个任务的真实性、文化适配性、文件充分性，回收文本反馈用于修订或移除无效实例。
4. **最终平衡**：每设置保留200个任务、50个 persona，共计1,600任务/400 persona。

### 评估协议（§3.6）
- **确定性评估函数**（Table 2）：contain、not_contain、excel_cell_value、file_exist、calendar_no_overlap。
- **LLM-as-judge函数**：evaluate_email、evaluate_note（使用 held-out 模型，temperature=0，输出上限10 tokens，YES判定pass）。
- **CTS指标**：
$$
\text{CTS}(t) = \text{pass}(t) \wedge \text{preserve}(t)
$$
其中 preserve(t) 要求所有非目标文件在初始与最终状态间保持 byte-identical。

### 执行超参（Table 6）
- 迭代上限：30 steps/task
- 温度：[0,1]，最大token数1024
- 停滞检测：连续5次相同动作触发 got_stuck
- 干扰项数量：15-20 per testbed
- 每次任务使用沙盒新鲜副本

---

## 实验与结果
### 评估模型（9个）
Gemini-3.1-Pro、Gemini-3.5-Flash、GPT-5、GPT-4o、Qwen-3-32B、Qwen-3-4B、Llama-3.3-70B、Llama-3.1-8B、EuroLLM-9B。

### 主要结果（Table 3）
- **最强模型**：Gemini-3.1-Pro 达 **49.2% CTS**，GPT-5 48.8%，Qwen-3-32B 48.0%。
- **全部模型** CTS 均未超过50%，表明当前智能体在此类任务上仍显著不足。
- ** Preservation Gap**：所有模型 PASS rate > CTS，差距从 Gemini-3.1-Pro 的 10.1 到 Llama-3.3-70B 的 16.8，说明"完成任务但破坏无关文件"是普遍现象。
- **弱模型**：EuroLLM-9B 仅 10.8% CTS，iteration-cap hit 率高达58%。

### 多语言稳健性（Table 4）
- **英语设置表现最优**（EN-US/C 56-58%），**中文最差**（ZH 38.6%，Qwen例外：Qwen-3-32B 在中文达43.5%）。
- Qwen 系列在中文上下降幅度最小，反映其 multilingual 训练优势。
- 语言梯度主要由 **locale mismatch** 驱动（非英语设置中占比21.4%，中文高达26.9%）。

### 工具与任务类型差异
- Calendar/Document 任务得分较高；Messaging/Shell 任务 CTS 低且 preservation gap 大。
- 确定性评估函数通过率高于 open-ended 邮件/笔记 judge 评估。
- 任务复杂度（reference-solution length）越高，CTS 越低。

### 故障模式（Figure 8 / Table 10）
- **Distractor capture**（24.6%）与 **Shell over-reach**（17.8%）合计占42.4%，均为 preservation 失败。
- **Locale mismatch** 是唯一显著依赖语言的错误模式。

---

## 相关工作脉络
1. **网页/环境导航基准**：Mind2Web（Deng et al., 2023）、WebArena（Zhou et al., 2024）、VisualWebArena（Koh et al., 2024）——聚焦网页交互，不涉及文件-based 工作流与状态保持。
2. **操作系统控制基准**：OSWorld（Xie et al., 2024）——多模态桌面控制，但未覆盖多语言与文化情境化任务。
3. **多应用办公自动化基准**：OfficeBench（Wang et al., 2024）、OdysseyBench（Wang et al., 2025b）——针对英语办公场景，缺少跨语言/文化设计。
4. **多语言智能体基准**：X-WebAgentBench（Wang et al., 2025a）、MAPS（Hofman et al., 2026）——从英语翻译得到，未从 native language 角度构建。
5. **文档理解基准**：DocVQA（Mathew et al., 2021）、ParseBench（Zhang et al., 2026）——聚焦信息抽取，不涉及多步骤工作流执行与状态保持。
6. **合成数据生成**：Self-Instruct（Wang et al., 2023）、WizardLM（Xu et al., 2025）——生成指令数据，但未考虑 agent 任务的完整 state 和 evaluator。
   - **本文定位**：WORLDBENCH 填补了"多语言+文化情境化+状态保持约束"三者的交叉空白，强调 native-language 构建而非翻译，并通过 CTS 指标首次显式量化 preservation gap。

---

## 局限性与未来方向
1. **作用域局限**：当前版本仅支持结构化文件操作（电子表格、文档、邮件、日历、shell），**不含视觉桌面控制**和**实时网页交互**（作者明确计划在后续版本补充）。
2. **评估次数限制**：每个任务仅执行一次（single run），未报告方差或多轮稳定性。
3. **LLM Judge 变异性**：evaluate_email 和 evaluate_note 依赖 judge 模型，虽经人工验证（κ=0.76-0.81），但仍存在轻微偏宽松问题。
4. **跨语言可比性边界**：各语言设置独立构建，未做 item-level 对齐，性能差异可能混合了语言能力、本地化难度、任务分布残余差异等因素。
5. **未来方向**：扩展至视觉/网页 agent；增加多次执行与置信区间；补充更多低资源语言与文化背景；探索 preservation-aware 的训练/对齐方法。

---

## 研究启发与可借鉴点
1. **CTS 指标可直接迁移**：任何需要"任务完成+环境干净"的 agent 评估场景（如 RAG pipeline、代码生成、数据处理流水线）均可借鉴 CTS 设计，同时报告 pass rate 与 preservation rate。
2. **Distractor 机制值得采用**：在 testbed 中主动注入相似文件名、过时版本、重叠实体等干扰项，能有效暴露智能体的 collateral edit 倾向，提升基准的区分度。
3. **Native-language 构建优于翻译**：从种子出发在目标语言文化中直接构建任务（而非翻译英语池），可保留 locale-specific 惯例（如日期格式、货币符号、小数分隔符），更真实地检验多语言 grounding 能力。
4. **细粒度失败分类框架**：将故障归因到 distractor capture、locale mismatch、shell over-reach 等具体模式，比单纯报告分数更能指导模型改进方向。
5. **混合评估策略**：确定性函数 + LLM-as-judge 的组合既保证核心指标的客观性，又覆盖开放ended输出评估，两结合可平衡可靠性和覆盖范围。

---

## 关键术语表
**WORLDBENCH**：一个 culturally-grounded 的多语言智能体基准测试，包含1,600个任务、7种语言、8种文化背景，用于评估 agent 在文件工作流中的任务完成与状态保持能力。

**CTS（Constrained Task Success）**：论文提出的核心评估指标，定义为 CTS(t) = pass(t) ∧ preserve(t)，同时要求任务评估函数全通过且非目标文件未被修改。

**State Preservation（状态保持）**：智能体在完成目标任务的同时，不修改、覆盖或删除 testbed 中非目标文件的约束条件。

**Distractor（干扰项）**：testbed 中故意设置的相似文件名、重叠实体或过时版本文件，用于检测智能体是否能精准定位目标 artefact。

**Persona-grounded**：任务围绕特定用户角色（含姓名、职位、地点、工作背景）构建，使指令与上下文深度耦合而非通用抽象。

**Locale Mismatch（本地化失配）**：智能体正确理解指令但输出时使用错误 locale 的格式约定（如美式日期、英文小数点而非逗号），是多语言性能差异的主要归因。

**Native-language Construction**：直接从目标语言和文化语境中构建任务，而非将英语任务翻译得到的设计原则。

**LLM-as-Judge**：使用独立 LLM 对开放性输出（如邮件、笔记）按 rubric 进行 pass/fail 判决的评估机制。

---

## 可复现要素
- **数据集**：论文未明确声明公开仓库链接，但附录提供了完整的构建流程、annotation 问题、超参配置，支持复现。
- **代码**：论文未提及开源代码库（无 GitHub 链接）；工具环境实现细节见 Appendices C/D。
- **权重**：未提供自行训练的模型权重，使用官方 API/预训练模型：Gemini-3.1-Pro、GPT-5、Qwen-3-32B、Llama-3.3-70B 等。
- **关键超参**：迭代上限 30 steps；temperature [0,1]；judge temperature=0，max tokens=10；malformed action 重试1次；stagnation 连续5次相同动作触发。

---
