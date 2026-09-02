---
title: "WORLDBENCH-Culturally-Grounded-Benchmark-for-Multilingual-Ag"
source: https://arxiv.org/pdf/2609.01056v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 19:39:57"
field: "多语言 Agent 评测基准"
keywords: ["agent benchmark", "multilingual evaluation", "state preservation", "culturally-grounded tasks", "constrained task success", "LLM agents"]
innovations: ["提出 WORLDBENCH：1,600 任务、7 语言、8 文化背景的文化锚定多语言 Agent 基准", "引入 CTS 指标：联合任务正确性与非目标文件保持性的二元合取评估", "构建种子驱动流水线：人工种子→自动扩展→母语者审核的任务生成与过滤机制"]
benchmarks: ["WORLDBENCH"]
---

# 论文速读：WORLDBENCH-Culturally-Grounded-Benchmark-for-Multilingual-Ag

## 一句话总结
本文提出 WORLDBENCH，一个涵盖 7 种语言、8 种文化背景的 1,600 任务多智能体基准，要求模型在沙盒环境中执行带有角色锚定的日常办公工作流；同时提出 CTS（Constrained Task Success）指标，将任务正确性与工作区状态保持性联合评估。实验表明，最强模型仅达 49.2% CTS，所有模型均存在显著的"正确性-环境保持性"差距，且非英语场景性能明显下降。

## 研究问题与动机
- **上下文锚定不足**：现有基准多用通用指令，忽略角色、地域、语言和工作习惯等现实语境，导致模型在真实工作流中基于错误假设行动。
- **状态保持性缺失**：模型可能完成目标输出，但同时意外修改无关文件（如覆盖电子表格、删除文档），而仅以任务完成度为指标的评测无法捕捉此类失败。
- **多语言覆盖薄弱**：多数基准以英语为中心构建后简单翻译，未考虑本土化情境、本地文档规范与文化约定，跨语言可靠性存疑。
- **缺乏可扩展的评测框架**：现有工作流类基准（如 OfficeBench、OdysseyBench）以英语为主，缺少从种子生成到人工审核的完整流水线，也难以支持污染检测与持续扩展。

## 核心贡献（创新点）
1. **提出 WORLDBENCH 多语言文化锚定基准**：包含 1,600 任务、7 种语言、8 种文化背景，每个任务由角色锚定指令、沙盒测试环境和可执行评估函数组成，与仅翻译英文任务的基准本质不同。
2. **设计种子驱动的任务构建流水线**：从人工编写的 persona/seeds 出发，经自动化扩展生成候选任务，再由具备语言与文化专业知识的标注员进行过滤与修订，确保任务真实可用，区别于纯合成生成的基准。
3. **引入 CTS（Constrained Task Success）指标**：将任务级评估函数通过性与非目标文件字节级保持性（preserve）取合取，首次在同一级别同时量化"做对"和"不破坏"，与仅报告 pass rate 的基线工作形成本质区别。
4. **提供 9 个前沿 LLM Agent 的实证评估**：揭示当前智能体在长程执行、多语言本地化和状态保持约束下仍高度脆弱，最强模型 CTS 不足 50%。

## 方法详解
- **任务形式化**：将 agentic 任务建模为人与沙盒环境的有向交互，包含 persona & 任务、环境初始状态、工具集、终止条件与最终状态评估五个组件，要求模型输出结构化 JSON 动作并逐步观测结果。
- **转移系统**：定义 $\mathcal{W} = (S, A, O, \delta)$，其中 $S$ 为状态空间，$A$ 为结构化动作空间，$O$ 为观测空间，$\delta: S \times A \to S \times O$ 为转移函数；轨迹 $H_t = [(a_1,o_1),\ldots,(a_{t-1},o_{t-1})]$ 作为上下文输入。
- **工具环境**：包含 5 类工具——Spreadsheet（读写单元格）、Document PDF（读写 PDF/文档）、Messaging（查收发）、Calendar（日程管理）、Shell（文件系统命令）和 System（控制/终止），每类操作受限。
- **构建流水线**：
  1. 人工种子：每语种写 20 个 persona × 4 个任务 = 80 种子；
  2. 自动生成：每语种再扩 30 个 persona × 5-6 任务；
  3. 人工审核：每语种 40 名母语标注员，审核真实性、文化适配性、信息充分性，回收或修订后平衡至每语种 200 任务/50 persona。
- **干扰项设计**：每个测试环境含 15-20 个干扰文件（相似文件名、过期版本、相关但无关记录），用于检测状态保持性。
- **CTS 指标定义**：
  $$\text{CTS}(t) = \text{pass}(t) \wedge \text{preserve}(t)$$
  其中 $\text{pass}(t)$ 为所有任务评估函数全通过，$\text{preserve}(t)$ 要求所有非目标文件在初态与终态间字节完全一致。
- **评估函数**：7 种——contain（含关键词）、not_contain（不含禁止词）、excel_cell_value（精确单元格值）、file_exist（文件存在）、calendar_no_overlap（无日程冲突）、evaluate_email/note（LLM-as-a-Judge，含默认评判准则）。

## 实验与结果
- **数据集**：WORLDBENCH，1,600 任务，7 语言（EN-US、EN-UK、IT、PT、ES、FR、DE、ZH），8 种文化背景，每设置 200 任务。
- **评估模型**：9 个——Gemini-3.1-Pro、Gemini-3.5-Flash、GPT-5、GPT-4o、Qwen-3-32B、Qwen-3-4B、Llama-3.3-70B、Llama-3.1-8B、EuroLLM-9B，统一 prompt 与配置。
- **主要结果**：
  - 最强模型 Gemini-3.1-Pro 达 **49.2% CTS**（PASS=59.3%，PRESV.=82.5%）；GPT-5 为 48.8%，Qwen-3-32B 为 48.0%。
  - 所有模型 PASS rate 均显著高于 CTS，gap 最大达 16.8（Llama-3.3-70B），表明"误改非目标文件"是普遍失败原因。
  - 语言梯度稳定：英语最高，中文最低；Qwen 系列对中文衰退最小，Qwen-3-32B 中文得分最高（43.5%）。
  - Calendar/Document 类任务表现较好，Messaging/Shell 类任务 gap 更大。
  - 确定性评估函数（excel_cell_value、file_exist）通过率高于开放式 Judge 评估（email/note）。
- **迭代上限分析**：30→50 步提升极小（0.3-1.9 CTS），表明增加步数无法根本解决弱模型的失败。
- **失败模式分类**：wrong output（最强模型主要失败）、collateral edit（所有模型均显著）、iteration-cap hit（弱模型主因，EuroLLM-9B 达 43%）、malformed loop、execution error。
- **人工错误分析**：distractor capture（24.6%）+ shell over-reach（17.8%）合计占 42.4% 的失败，且均属于保持性违规；locale mismatch 在非英语设置中升至 21.4%（中文高达 26.9%），是语言梯度的主要机制。

## 相关工作脉络
- **Web/Mind2Web（Deng et al., 2023）**：聚焦网页导航，WORLDBENCH 转向文件型工作流并强调文化锚定与状态保持。
- **OSWorld（Xie et al., 2024）/ AgentBench（Liu et al., 2025）**：OS 级操作评测，WORLDBENCH 采用固定动作接口保证可复现性，聚焦多应用协同工作流。
- **OfficeBench（Wang et al., 2024）/ OdysseyBench（Wang et al., 2025b）**：英语办公工作流基准，WORLDBENCH 扩展至多语言并在每个语言设置中独立构建而非翻译。
- **MAPS（Hofman et al., 2026）**：多语言 agent 安全基准，WORLDBENCH 侧重任务执行正确性与环境保持性的联合度量。
- **WebArena（Zhou et al., 2024）/ VisualWebArena（Koh et al., 2024）**：视觉网页环境，WORLDBENCH 当前不含视觉桌面控制和实时网页交互。
- **合成数据生成（Wang et al., 2023; Chim et al., 2025）**：通用指令合成，WORLDBENCH 要求每个合成任务包含一致状态、可解指令和可执行评估器，并通过人工审核过滤。

## 局限性与未来方向
- **局限性**：
  - 仅支持结构化文件工作流，未包含视觉桌面控制和实时网页交互（论文声明为设计取舍以保障可复现性）。
  - 每任务仅执行一次（单次采样），无法估计方差。
  - LLM-as-a-judge 引入一定方差（不同 judge 模型 pass rate 差异可达 2.4 个百分点），尽管排序结论稳健。
  - 各语言设置独立构建，任务间无逐一对齐，跨语种比较受任务难度残余差异影响。
- **未来方向**：
  - 加入视觉桌面控制和实时网页交互的扩展版本。
  - 设计更强的状态追踪机制与显式保持性优化目标。
  - 探索更好的本地化感知动作锚定方法。
  - 支持通过新种子持续扩展语言/文化/场景。

## 研究启发与可借鉴点
1. **CTS 指标设计可直接迁移**：将"任务完成"与"环境保持"取合取的方式，适用于任何涉及多文件/多应用协同的操作型 agent 评测，避免仅凭 pass rate 掩盖破坏性行为的评估偏差。
2. **干扰项（distractors）策略值得复用**：在测试环境中嵌入相似文件名、过期版本、相关干扰记录，可有效区分模型是否真正理解了"操作边界"，建议在本团队工作流评测中引入类似设计。
3. **种子驱动的构建流水线具有可迁移性**：人工种子→自动扩展→母语者审核的三层流程，既保留了人类真实性判断，又实现了规模扩展，适合构建其他领域（如医疗、法律）的垂直 agent 基准。
4. **locale mismatch 分析框架可用于多语言评测诊断**：将失败按"distractor capture / shell over-reach / locale mismatch / premature termination / schema violation"分类统计，能快速定位模型在多语言场景下的具体弱点。
5. **LLM-as-a-judge 的鲁棒性验证方法**：用 3 个独立 judge 模型 + 多数投票对比人工标注，报告 Cohen's κ 和一致性，可作为后续工作中 judge 可信度的标准验证流程。

## 关键术语表
**CTS（Constrained Task Success）**：论文提出的 headline 指标，等于任务所有评估函数通过且所有非目标文件未被修改的合取条件，衡量 agent 在"做对"和"不破坏"两方面的综合表现。
**Preservation Gap（保持性差距）**：PASS rate 与 CTS 之间的差值，反映模型完成任务但同时意外修改非目标文件的比例，是本文发现的核心失败模式之一。
**Persona（角色）**：锚定任务的虚拟用户画像，包含姓名、角色、所在地、语言和工作上下文，决定任务的本土化内容（货币、日期格式、组织名称等）。
**Distractor（干扰项）**：测试环境中故意植入的相似文件名、过期版本或相关记录，用于检测 agent 是否能精确识别目标文件而不误操作。
**Testbed（测试环境）**：每个任务对应的沙盒工作区，包含目标文件、干扰文件、邮件、日历、电子表格等多类型 artefact。
**LLM-as-a-Judge**：用独立的 LLM 充当裁判，对开放式输出（邮件、笔记）按任务特定评分准则进行 pass/fail 判定。
**Iteration Cap（迭代上限）**：每个任务允许的最大动作步数（本文设为 30），超限则终止并记录为 hit。
**Locale Mismatch（本地化失配）**：agent 正确解析指令但使用错误本地惯例（如美式日期格式代替欧式）写入内容的失败模式，是英语与非英语性能差距的主因。

## 可复现要素
- **数据集**：WORLDBENCH，论文提供 arXiv 源码及附录中的构建细节，数据集与代码见论文仓库（arxiv 链接可查），数据集已公开。
- **代码/权重**：论文未明确声明开源仓库地址，仅说明 via arXiv 提交；模型评估使用官方 API（Gemini、GPT）和本地运行开源模型（Llama、Qwen、EuroLLM）。
- **关键超参**：迭代上限 30 步；agent temperature [0,1]，max tokens 1024；judge temperature [0,1]，max tokens 10；停滞检测阈值 5 次连续相同动作；malformed action 允许 1 次重试；每个任务使用独立沙盒副本；干扰项 15-20 个/测试环境。
