---
title: "Unknown-is-not-normal-separating-language-model-extraction-f"
source: https://arxiv.org/pdf/2609.34112v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 10:55:52"
field: "临床自然语言处理 / 医疗AI"
keywords: ["clinical risk scores", "LLM extraction", "three-valued extraction", "active information gathering", "score bounds", "under-triage", "MedCalc-Bench"]
innovations: ["将LLM三值提取与确定性代码决策分离，通过分数边界仅询问能改变分类的问题", "揭示并量化'Missing=Normal'惯例导致的系统性低治风险（约8.5%）", "证明9B本地模型作为提取器可达准最优准确率（99.8%），显著降低部署成本"]
benchmarks: ["MedCalc-Bench", "MedCalc-Bench Verified"]
---

# 论文速读：Unknown-is-not-normal-separating-language-model-extraction-from-rule-based-decision-logic-for-clinical-risk-scores

## 一句话总结
论文提出将 LLM 的三值信息提取（present/absent/unknown）与确定性代码决策逻辑分离，通过计算"分数边界"仅询问能改变决策的问题，在保持准确率的同时将提问量减半，且彻底消除提前决策和无关提问；同时揭示"缺失=正常"惯例会系统性低治约 8.5% 的患者。

## 研究问题与动机
- **未记录≠正常**：临床笔记常不完整，MedCalc-Bench 将未提及输入视为 absent/normal 的约定会导致分数被低估，使患者被错误归入低风险类别。
- **LLM 在此类任务上表现不佳**：LLM 同时承担提取与决策，会在类别未定时提前承诺，或在已确定时仍犹豫；交互式基准测试显示模型要么问得太多、要么问得太少。
- **缺乏可审计的决策路径**：端到端 agent 难以追踪"为何问了这个问题/为何此时作答"，而确定性代码可显式推理未知的边界范围。
- **现实风险**：在急诊场景中，过度低治（under-triage）意味着延误治疗，后果严重，因此必须区分"未知"与"正常/不存在"。

## 核心贡献（创新点）
1. **提出提取-决策分离架构**：LLM 仅负责三值提取（present/absent/unknown）并附带证据引用与置信度，确定性代码负责计算分数边界和决策——与端到端 agent 的本质区别在于决策逻辑可审计、可复现。
2. **定义"决策相关性"（decision relevance）并实现 bounds 策略**：仅询问那些在当前证据下能改变最终分类的问题；与 ask-all 的本质区别是不再问所有缺失项，而是先判断是否真的需要。
3. **揭示"缺失=正常"的系统性危害**：在六个临床评分上均显示该约定使准确率降至约 91%，并导致 8.5% 的患者被低治，这是 MedCalc-Bench 基线未能捕获的安全漏洞。
4. **验证小模型可达准最优**：9B 本地模型（Qwen3.5-9B）作为提取器即可达到 99.8% 的准最优准确率，大幅降低部署成本——与依赖前沿模型的 agent 方案形成对比。

## 方法详解
- **三值提取**：对每个输入，LLM 返回 status（present/absent/unknown）、value、unit 及最短支持原文引用（exact substring），以及置信度。代码拒绝无法找到原文引用的声明（归为 unknown），并拒绝不合理的数值。
- **单位转换由代码完成**：避免模型做数值换算出错。
- **分数边界计算**：将数值输入按评分阈值划分为区间，枚举所有可能的未知完成（completion），计算分数的上下界；若只有一个分类可能落在边界内，则该分类已"确定"。
- **决策相关性判定**：一个未知输入是 decision-relevant，当且仅当它的不同取值能使最终分类发生变化（例：HEART 已得 2 分、troponin 未知时，可能得分 2–4 跨越低/中风险边界，因此需询问；若已得 0 分则无需问）。
- **Bounds 策略**：每次只问一个 decision-relevant 的未知输入，回答后更新边界，直到分类确定或无更多可问项。
- **Bounds + checks**：对置信度低于阈值的决策关键值，主动要求临床医生确认。
- **Missing = normal**：将未提及输入直接视为 absent，作为对照。
- **端到端 Agent**：Claude Opus 5.5 自主决定何时提问、何时作答。
- **置信度校准**：对 Haiku 和 Qwen 的原始置信度分别用 temperature scaling 和 isotonic regression 校准。
- **模拟器**：理想临床医生从隐藏真值回答；噪声临床医生以 10% 无法回答、5% 答错、数值回答±10% 范围的方式模拟。

## 实验与结果
- **数据集**：6 个计算器（HEART, CURB-65, qSOFA, PERC, Wells PE, Cockcroft-Gault），各 200 例，共 1,200 例合成急诊病例；另用 MedCalc-Bench Verified 的 584 条真实病历作为验证。
- **评估基线**：Ask-all、Bounds、Bounds+checks、Missing=normal、End-to-end Agent。
- **主要结果（Haiku 4.5 提取器，干净笔记，理想临床医生）**：
  - Accuracy：Ask-all 99.4%，Agent 99.6%，Bounds 99.4%，Bounds+checks 99.7%，Missing=normal 91.2%
  - Questions/case：Ask-all 1.78，Agent 0.99，Bounds 0.92，Bounds+checks 0.90，Missing=normal 0.21
  - Irrelevant questions：Ask-all 48.1%，Agent 9.5%，Bounds 0%
  - Premature commitment：Agent 0.2%，Bounds 0%
  - Under-triage：Missing=normal 8.5%（95% CI 7.1–10.2），其余均 ≤0.1%
- **噪声临床医生**：Bounds 准确率 87.0%，Agent 降至 83.5%（p<0.001），Agent 提前承诺 2.7%
- **Qwen3.5-9B 提取器**：Bounds 准确率达 99.8%（oracle 级），questions/case 1.23
- **真实病例（MedCalc-Bench 训练集）**：仅 52% 的病例可从笔记确定分类（HEART 仅 13%）；Missing=normal 下正确率为 69%（HEART 仅 22%）；真实值落在 bounds 内的比例为 93–95%

## 相关工作脉络
1. **MedCalc-Bench**（Khandekar et al., 2024）：当前主流基准，采用"missing=normal"标注惯例；本文指出该惯例掩盖了约 1/12 患者的低治风险，真实病历中仅 52% 能确定分类。
2. **ClinDet-Bench**（Watanabe et al., 2026）：评估 LLM 在临床判断中的可判定性（determinability），但未涉及主动提问；本文在此基础上加入了交互问答环节。
3. **MediQ**（Li et al., 2024）和 **AgentClinic**（Schmidgall et al., 2024）：评估诊断对话中的信息收集能力，关注的是开放式问诊；本文聚焦于固定评分量表的特征获取效率。
4. **CRAFT-MD / MedMCP-Calc**（Zhu et al., 2026；Wang et al., 2025）：评估 LLM 在医疗计算场景下的表现；本文与这些工作共享"计算风险评分"任务设定，但通过分离提取与决策来改进安全性。
5. **Active Feature Acquisition**（Saar-Tsechansky et al., 2009）和 **Selective Prediction**（Geifman & El-Yaniv, 2017）：本文的 bounds 策略可视为前者在临床评分领域的具体应用，后者则是 abstention 行为的理论基础。
6. **HL7 CQL / FHIR CPG**（HL7 International, 2026）：可执行临床指南标准；本文的确定性边界代码与这些标准天然兼容，可作为落地载体。

## 局限性与未来方向
- **合成笔记偏理想**：笔记由同家族 LLM（Claude Sonnet 5）生成，可能高估提取准确率（与 Qwen 笔记对比有约 3.2 点的同家族优势）；真实场景噪声更大。
- **噪声模型简单**：临床医生的噪声以固定概率模拟，未覆盖真实对话中的上下文依赖和情绪因素。
- **端到端 agent 在脏笔记上更强**：Claude Opus 5.5 在 messy notes 下直接读笔记达 99.8%，高于 Haiku 提取的 98.6%，说明小提取器在复杂文本上仍有差距。
- **真实笔记无法测试提问策略**：MedCalc-Bench 不含交互环节，仅能评估提取和计算准确性。
- **置信度来源局限**：Haiku 置信度为 self-reported，Qwen 来自 token log-probabilities，均未经外部金标准严格校准。
- **未来方向**：可在真实电子病历上部署 bounds 系统；将 abstention 转为高风险兜底策略；探索与 CQL/FHIR CPG 标准的集成；研究在连续临床流程中的动态边界计算。

## 研究启发与可借鉴点
1. **"提取-决策分离"设计模式可复用**：任何涉及规则化评分/分类的医疗 AI 系统都可借鉴此模式，将 LLM 限定为提取器，用代码做确定性推理，提升可审计性和安全性。
2. **"决策相关性"定义可推广**：判断"问什么"的标准是"该信息是否能改变最终决策"，这一原则不仅适用于临床评分，也可迁移到法律合规检查、金融风控等阈值决策场景。
3. **置信度校准（temperature scaling + isotonic regression）值得借鉴**：本文将其用于三值提取，在 bounds+checks 策略中有效提升了可靠性，是轻量且可复用的工程实践。
4. **小模型（9B）可达准最优**：提醒团队不必盲目依赖前沿大模型，合理设计提取 schema 和代码逻辑后，本地部署的小模型已能满足精度要求，大幅降低成本。
5. **评测应报告"可判定比例"**：本文揭示 MedCalc-Bench 仅在 52% 的真实病历中可确定分类，建议今后评测类似系统时，除准确率外增加"determined rate"和"under-triage rate"作为核心安全指标。

## 关键术语表
- **Three-valued extraction（三值提取）**：将每个输入标注为 present（存在且有值）、absent（明确不存在）或 unknown（未提及），区别于传统的二值（present/absent）标注。
- **Score bounds（分数边界）**：在未知输入的所有可能取值下，计算评分的最小值和最大值，从而确定分类是否已唯一确定。
- **Decision relevance（决策相关性）**：一个未知输入若其取值能改变最终分类，则被视为决策相关，是 bounds 策略决定是否提问的判据。
- **Premature commitment（提前承诺）**：分类尚未确定时就给出答案的错误行为，是端到端 agent 的典型缺陷。
- **Missing-equals-normal（缺失=正常）**：将笔记未提及的输入视为 absent/normal 的约定，被本文证明会导致系统性低治风险。
- **Under-triage（低治）**：将实际高风险患者错误归类为低风险，是临床评分中最危险的错误类型。
- **Active feature acquisition（主动特征获取）**：根据当前信息选择最有助于决策的特征进行询问，本文的 bounds 策略即为此范式的应用。
- **Selective prediction（选择性预测）**：在不确定性高时拒绝预测（abstain），而非强行给出答案，本文将其与高优先级兜底策略结合。

## 可复现要素
- **数据集**：合成病例可从配置和 seed 重新生成（论文未公开生成后的笔记和模型响应缓存）；真实数据使用 MedCalc-Bench Verified（https://huggingface.co/datasets/nsk7153/MedCalc-Bench-Verified），CC-BY-SA 4.0。
- **代码**：已开源，MIT 许可，GitHub https://github.com/nicoveraz/calc-bounds，Zenodo 存档 https://doi.org/10.5281/zenodo.23004726。
- **关键超参**：缺失率 30%（Synthetic cohort）、confidence 校准阈值 sweep（0.5/0.8/0.9/0.95/0.99/0.999）、噪声设置（10% 无法回答、5% 答错、±10% 数值误差）。
- **硬件/环境**：Qwen3.5-9B 在 16 GB 内存笔记本上通过 Ollama 运行；Claude 通过 headless Claude Code 调用，无 API 费用。
- **随机种子**：所有运行使用固定 seed，确保可复现。
