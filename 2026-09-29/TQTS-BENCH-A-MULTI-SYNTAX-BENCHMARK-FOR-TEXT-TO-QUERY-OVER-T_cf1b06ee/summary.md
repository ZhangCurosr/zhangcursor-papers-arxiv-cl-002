---
title: "TQTS-BENCH-A-MULTI-SYNTAX-BENCHMARK-FOR-TEXT-TO-QUERY-OVER-T"
source: https://arxiv.org/pdf/2609.34783v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 04:21:35"
field: "时序数据库文本到查询"
keywords: ["Text-to-Query", "Time-Series Database", "Benchmark", "Large Language Model", "Multi-Syntax", "Temporal Query Intent"]
innovations: ["首个覆盖多语法、多领域、多时间意图的TSDB文本到查询基准TQTS-BENCH", "提出人类中心AI辅助的QA构建与交叉验证流程，确保6125条数据高质量", "系统揭示TQTS任务的三类核心瓶颈：语法异构、时间意图误解、模式链接困难"]
benchmarks: ["TQTS-BENCH", "Spider", "BIRD", "Spider 2.0", "TEFD", "PromCopilot"]
---

# 论文速读：TQTS-BENCH-A-MULTI-SYNTAX-BENCHMARK-FOR-TEXT-TO-QUERY-OVER-TIME-SERIES-DATABASES

## 一句话总结
本文提出了 **TQTS-BENCH**，首个面向时间序列数据库（TSDB）的多语法文本到查询基准，包含 6,125 条高质量 QA 对，覆盖 97 个 TSDB、23 种查询语法和 22 个应用领域；评估显示现有 LLM 和 Text-to-Query 方法仍面临巨大挑战，最佳模型 Claude-Opus-5 执行准确率仅 48.98%，而人类达 87.34%。

## 研究问题与动机
1. **核心问题**：评估并推动 LLM 在时间序列数据库（TSDB）上的文本到查询（Text-to-Query over Time-Series，TQTS）能力，当前缺乏类似 Spider/BIRD 级别的全面基准。
2. **TSDB 与 RDB 三大本质差异**：
   - **语法异构**：TSDB 查询语法差异大（如 Flux 流水线风格 vs PromQL 嵌套函数），而 RDB 均基于 SQL 方言。
   - **领域覆盖广**：TSDB 常见于 IoT、AIOps、云服务等高频率数据采集场景，现有 RDB 基准极少覆盖这些领域。
   - **时间特定意图**：TSDB 查询包含窗口聚合、时序变化分析、时间定位、关系分析等时间语义，远超 RDB 的时间无关操作。
3. **现有基准不足**：既有 TSDB 基准（如 TEFD、PromCopilot）通常局限于单一 TSDB/语法/领域，无法支撑跨场景的系统评估。
4. **实际 gap**：即便最强 LLM 与现有 Text-to-Query 方法在 TQTS-BENCH 上表现不佳，存在显著优化空间。

## 核心贡献（创新点）
1. **构建首个多语法 TQTS 基准 TQTS-BENCH**：涵盖 97 个 TSDB、23 种查询语法、22 个应用领域、4 类时间特定意图，填补了 RDB 基准之外时序数据查询评测的空白。与 TEFD/PromCopilot 等单一语法/领域基准的本质区别在于**覆盖广度与跨语法通用性**。
2. **提出人类中心的 AI 辅助构建流程**：开发可视化分析工具，由领域专家审核并交叉验证 QA 对（600+ 小时人工投入、30,774 次查询执行），保障数据质量与答案正确性。区别于纯自动生成基准，本工作强调**人在回路的质量控制**。
3. **系统评估与错误分析揭示关键瓶颈**：对比 7 款先进 LLM（开源/闭源）、5 个 RDB Text-to-Query 方法、1 个 TSDB 专用方法；通过误差分析归纳三类主要错误（语法错误 75.29%、意图误解 61.43%、模式链接错误 17.09%）。与已有工作不同，本文**首次量化多语法异构、时间语义理解、模式结构差异三因素对 TQTS 的影响**。
4. **设计适配 TSDB 的评估准则**：引入 Spider 2.0 风格的 Match Policy 缓解因 pivot 等格式差异导致的误判；按意图数量划分易/中/难三级难度，而非单纯依赖 token 长度。

## 方法详解
### 3.1 任务定义（TQTS Task）
给定自然语言问题、数据库 Schema 和目标 TSDB，生成可在该 TSDB 上正确执行的查询语句以检索所需数据。Schema 描述时间序列数据的逻辑组织（时间戳、数值、元数据的结构化方式），不同 TSDB 体现为表/测量、列/字段、标签/标签（tags/labels）等形式差异。

### 3.2 数据集构建流程
**步骤一：TSDB 选取与数据导入**
- 基于 DB-Engines 排名选取 23 个代表性 TSDB 管理系统（考虑可访问性，排除需许可的如 kdb+）。
- 从官方文档、论坛、公开平台收集 427 个候选数据集与 1,405 条种子 QA，经人工筛选后保留 97 个高质量真实 TSDB，导入对应系统。

**步骤二：查询意图识别**
- 协同领域专家分析种子 QA，识别 9 类查询意图：
  - **时间特定（Time-specific）**：I1 窗口聚合与重采样、I2 时序变化分析、I3 时间定位、I4 关系分析
  - **时间无关（Time-agnostic）**：I5 条件过滤、I6 统计计算、I7 排序与排名、I8 元数据查询、I9 领域知识依赖

**步骤三：人机协作 QA 生成**
- 使用视觉分析工具，向 LLM（如 GPT-5.6-Sol）提供意图、Schema 和 TSDB 上下文，生成自然语言问题；
- 标注员审核问题是否符合意图、是否反映真实 TSDB 场景，必要时修正；
- LLM 根据最终问题生成可执行查询，形成未验证 QA 对。

**步骤四：交叉验证**
- 两名标注员独立手写查询并执行，与参考答案比对；
- 结果一致且问题可答则通过；否则提交仲裁员复核并修订，进入下一轮验证；
- 最终 6,125 条 QA 对通过验证，平均每条约 6 分钟审核、5 次查询执行。

### 3.3 评估设置
- **指标**：Execution Accuracy (EX)，判定预测查询结果与预期结果在列层面匹配（含 Match Policy 处理 pivot 格式差异）。
- **难度分级**：按意图组合数划分：1–3 意图（Easy, 31.72%）、4–5 意图（Medium, 44.33%）、6–9 意图（Hard, 23.95%）。
- **基线模型**：
  - LLM：Qwen3.8-Flash、DeepSeek-V4-Pro、GLM-5.3-Flash、Kimi-K3、GPT-6-Sol、Claude-Opus-5、Gemini-3.7-Flash
  - RDB Text-to-Query 方法：DeepEye-SQL、OpenSearch-SQL、RSL-SQL、DAIL-SQL、DIN-SQL（统一以 GPT-4o-mini 为 backbone）
  - TSDB 方法：PromCopilot

## 实验与结果
### 主要结果（执行准确率 EX）
| 方法 | Easy | Medium | Hard | Overall |
|------|------|--------|------|---------|
| Human | *95.71% | *85.52% | *81.94% | *87.34% |
| **Claude-Opus-5** | **62.79%** | **42.91%** | **41.92%** | **48.98%** |
| GPT-6-Sol | 62.74% | 40.63% | 35.38% | 46.38% |
| Kimi-K3 | 58.26% | 37.24% | 34.08% | 43.15% |
| GLM-5.3-Flash | 54.30% | 28.88% | 19.15% | 34.61% |
| Qwen3.8-Flash | 43.75% | 20.07% | 17.79% | 27.04% |
| DeepSeek-V4-Pro | 39.89% | 15.87% | 11.79% | 22.51% |
| Gemini-3.7-Flash | 63.25% | 37.68% | 32.92% | 44.65% |
| DeepEye-SQL (RDB) | 14.82% | 1.92% | 1.16% | 5.83% |
| OpenSearch-SQL (RDB) | 16.37% | 1.84% | 1.09% | 6.27% |
| DIN-SQL (RDB) | 12.25% | 2.17% | 1.23% | 5.14% |
| RSL-SQL (RDB) | 11.99% | 1.33% | 0.95% | 4.62% |
| DAIL-SQL (RDB) | 1.34% | 0.26% | 0.00% | 0.54% |
| PromCopilot (TSDB) | 1.80% | 0.22% | 0.07% | **0.69%** |

### 关键结论
- **最佳 LLM（Claude-Opus-5）EX 仅 48.98%**，与人类 87.34% 差距悬殊；开源模型普遍更低（DeepSeek-V4-Pro 仅 22.51%）。
- **RDB 方法泛化能力极弱**：DeepEye-SQL 在 BIRD 上表现优异，但在 TQTS-BENCH 仅 5.83%，主要因其 prompt 和生成阶段强依赖 SQL 语法，无法适配 TSDB 多样语法。
- **TSDB 专用方法（PromCopilot）通用性差**：仅 0.69% EX；在目标领域（云监控/AIOps）EX 为 3.03%，切换到其他领域降至 0.00%，暴露其领域耦合严重。
- **错误分析**（基于 DeepSeek-V4-Pro 随机 600 样本）：
  - **查询语法错误 75.29%**：结构错误 42.50%（算子/函数顺序不当）、函数/关键字误用 32.79%（参数错误、调用不存在函数）
  - **意图理解错误 61.43%**：时间特定意图错误占 65.41%（I1 窗口聚合 26.31%、I2 时序变化 25.19%、I3 时间定位 7.52%、I4 关系分析 6.39%）
  - **模式链接错误 17.09%**：TSDB 使用 metric/label/tag 而非 table/column，架构异构且大规模 Schema（错误样本平均 381k tokens vs 正常 115k tokens）加剧"lost-in-the-middle"问题。
- **消融实验**：将 TSDB 转换为 RDB（SQLite）后，语法错误案例 EX 从 0.00% 提升至 32.21%，介于 BIRD（56.91%）与 Spider 2.0-lite（22.94%）之间，说明**模型具备理解问题的能力，主要瓶颈在于 TSDB 异构语法而非问题本身难度**。

## 相关工作脉络
1. **Spider / BIRD / Spider 2.0**：RDB Text-to-SQL 基准的代表性工作者，覆盖跨域泛化、大规模真实数据库、企业级复杂工作流等场景；本文与之定位差异在于**TSDB 的语法异构性与时间特定语义是 RDB 基准未覆盖的关键维度**。
2. **TEFD (Wu et al., 2026)**：聚焦 InfluxDB Flux 语法，评测范围仅限单一 TSDB/语法；本文 **TQTS-BENCH 覆盖 23 种语法、97 个 TSDB**，支持更全面的跨语法泛化评估。
3. **PromCopilot (Zhang et al., 2026)**：面向 Prometheus PromQL 的时序查询方法，与云原生 Kubernetes 领域深度耦合；本文通过跨领域实验揭示其**领域特异性导致跨场景泛化失效**（EX 从 3.03% 跌至 0.00%）。
4. **Dranca et al. (2026)**：在 agro-food 场景下评测 Flux 与 InfluxQL；本文**覆盖 22 个应用域**（含 IoT/AIOps/云服务/能源等），领域覆盖面更广。
5. **ScienceBenchmark / SParC / CoSQL / Dr.Spider / LogicCat**：针对科学数据库、对话式查询、鲁棒性/推理能力的 RDB 基准；本文指出这些工作**均基于关系型假设，无法反映 TSDB 的时间聚合、重采样等特有需求**。
6. **Vo et al. (2022) 时序查询工作**：早期探索 NL 接口中的时序问题；本文**首次在多语法、多领域、多意图维度上系统构建 TQTS 评测体系**。

## 局限性与未来方向
1. **基准自身局限**：
   - 仅覆盖 DB-Engines 排名中可公开访问的 23 个 TSDB，排除了如 kdb+ 等需许可的系统，**可能存在覆盖率偏差**。
   - QA 对数量（6,125）相对于 BIRD（数万级）仍较有限，尤其 Hard 难度样本（23.95%）可进一步扩充。
2. **方法局限**：
   - 当前评估聚焦执行准确率，**未深入考察查询效率、资源消耗、长上下文鲁棒性等工程维度**。
   - 误差分析基于 DeepSeek-V4-Pro 单模型，不同模型 error pattern 可能存在差异。
3. **未来方向**（论文讨论与可推断）：
   - 开发**语法自适应的 Text-to-Query 框架**，降低对特定 TSDB 语法的硬编码依赖。
   - 改进**模式链接机制**以应对 TSDB 异构 schema（metric/label/tag）与大规模上下文。
   - 增强模型对**时间特定意图**（尤其是窗口聚合、时序变化分析）的理解与推理能力。
   - 探索**多语法联合训练或语法翻译**策略，提升跨 TSDB 泛化性能。
   - 构建更长上下文、更复杂多意图组合的进阶基准版本。

## 研究启发与可借鉴点
1. **人类中心 AI 辅助标注流程可迁移**：论文提出的"LLM 生成 → 专家审核 → 交叉验证 → 仲裁修订"工作流，对构建其他垂直领域基准（如图数据库、时序分析、生物医学数据查询）具有直接参考价值，尤其是可视化审核工具的交互设计。
2. **意图驱动的数据设计思路**：以 9 类查询意图（含 4 类时间特定）作为 QA 构造的基础框架，实现了跨语法的统一语义锚点；此方法可用于**跨数据库类型的统一评测体系设计**，避免语法差异掩盖语义评估。
3. **Match Policy 评估策略值得借鉴**：针对 TSDB pivot 等操作导致的格式差异引入结果规范化匹配，缓解了误判；类似思路可应用于**任何输出格式多样化的 Text-to-Query 评测场景**（如 JSON/表格/图表）。
4. **消融实验设计严谨**：通过"TSDB→RDB 转换"隔离语法因素，证明模型能力瓶颈主要在语法而非问题理解；这种**控制变量消融**对定位 Text-to-Query 模型失效原因具有方法论示范价值。
5. **可与本团队方向结合的创新机会**：
   - 针对时间特定意图（I1–I4）构建**专用推理模块或 prompt 增强策略**，如引入时间语义解析器。
   - 设计**语法中立表示层**（如将 Flux/PromQL/SQL 统一为中间 IR），再编译为各 TSDB 方言，提升泛化性。
   - 探索**大 Schema 下的模式检索优化**（如语义索引、分层加载）以缓解 lost-in-the-middle 问题。

## 关键术语表
**TQTS（Text-to-Query over Time-Series）**：面向时间序列数据库的自然语言到查询生成任务，是 Text-to-SQL 在时序数据场景的扩展。
**TSDB（Time-Series Database）**：专为高效写入、存储和查询时间序列数据而设计的数据库系统，如 InfluxDB、Prometheus、TimescaleDB。
**Execution Accuracy (EX)**：评估指标，衡量生成查询的实际执行结果是否与预期结果匹配（列级一致）。
**Query Intent（查询意图）**：用户查询背后的语义目标；本文识别 9 类意图，其中 I1–I4 为时间特定意图（窗口聚合、时序变化、时间定位、关系分析），I5–I9 为时间无关意图。
**Match Policy**：为缓解 TSDB 查询因 pivot 等操作导致的输出格式差异而产生的误判，引入的结果规范化匹配策略。
**Flux**：InfluxDB 的查询语言，采用流水线（pipeline）风格的转换算子链语法。
**PromQL**：Prometheus 的查询语言，基于 metric selector 和 range-vector 函数的嵌套表达式语法。
**Schema Linking**：将自然语言提及的实体/属性映射到数据库 Schema 中对应对象的过程，在 TSDB 中因 metric/label/tag 结构而更具挑战性。

## 可复现要素
- **数据集**：TQTS-BENCH 已公开，访问地址：https://anonymous.4open.science/r/TQTS-Bench-00CD
- **代码**：论文未提及代码开源；构建过程中开发了可视化分析工具（详见 App. B），但未提供公开源码链接。
- **权重**：评估使用的 LLM 均为 API 调用（GPT-6-Sol、Claude-Opus-5、Gemini-3.7-Flash、Qwen3.8-Flash、DeepSeek-V4-Pro、GLM-5.3-Flash、Kimi-K3），未提供本地微调权重。
- **关键超参**：论文未明确报告模型温度、top-p、max tokens 等生成超参；文本截断策略为"超过模型最大 token 限制时从开头截断"（遵循 Lei et al. 2025 设置）。
- **Human Eval 子集**：随机抽样 10% 进行人工评估，具体抽样种子未公开。
