---
title: "When-Harness-Beats-Scale-and-When-Reading-Beats-Both"
source: https://arxiv.org/pdf/2609.34366v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 04:26:13"
field: "文档理解与定量推理"
keywords: ["Document Grounded Reasoning", "Program-of-Thoughts", "Retrieval-Augmented Generation", "Structured Output Training", "Graph RAG", "Benchmark Input Shift", "DocSem"]
innovations: ["PoT harness 在紧凑模型（7B）上的联合准确率提升（+0.282）远超等大模型缩放，27B harness 模型匹敌 72B 直接模型，参数比约 2.7:1、碳排放比约 4:1", "通过 7B 谱系对比揭示 PoT 效果依赖结构化输出训练赋予的格式纪律，JSONSchemaBench 有效性可预测 PoT 效果符号方向", "受控退化实验量化测试集光栅扫描输入退化的影响，证明读取质量而非推理架构主导 leaderboard 双峰分布"]
benchmarks: ["DocSem (DocInsights 2026)", "GSM-SEM", "JSONSchemaBench", "IFEval", "GSM8K", "MMLU", "NEREL-Bench", "Librusec History (Russian QA)"]
---

# 论文速读：When-Harness-Beats-Scale-and-When-Reading-Beats-Both

## 一句话总结
本文介绍了作者在 DocInsights 2026 DocSem 评测中的系统，核心发现是：**应用架构（PoT 执行、self-consistency、蒸馏图上下文）比模型规模更能提升指标**——27B 模型配合完整 harness 即可匹敌 72B 直接回答模型，但测试集因 PDF 为低分辨率扫描件+水印导致读取管线崩溃，联合准确率从 0.853 骤降至 13.58%。

---

## 研究问题与动机
- DocSem 任务要求同时完成两项严格匹配：从 PDF 定量段落中计算数值答案，并精确标注支撑证据的布局块标识符（block identifiers），现有方法在大规模模型上的提升边际效益有限。
- 训练/验证数据为干净生成型 PDF，而测试 PDF 为低分辨率光栅扫描件并带有"TESTING COPY"对角水印——输入分布的隐性偏移未被提前检测到，是导致评测崩盘的根因。
- LLM 可提供的两种能力（世界知识 vs. 语言知识）随参数规模呈现不同增长斜率：世界知识陡峭上升，语言知识平缓，因此将世界知识路由到外部工具后，紧凑模型的"语言残留"竞争力被低估。
- 结构化输出训练（structured-output training）是否真正赋予模型格式纪律，还是仅在小模型上有效，缺乏系统性对比证据。

## 核心贡献（创新点）
1. **PoT harness 在紧凑模型上的杠杆效应远超模型缩放**：7B 模型从直接回答切换到 Program-of-Thoughts，联合准确率提升 0.282；而 72B 模型同样切换仅提升 ≤0.005——27B 完整 harness 模型（0.884）即匹敌 72B 直接回答（0.873），参数量仅为后者的约 1/2.7，碳排放约为 1/4。
2. **首次通过 7B 谱系对比（Meno-Lite-0.1 vs. Qwen2.5-7B-Instruct）揭示 PoT 的模型依赖性**：结构化输出训练使紧凑模型成为"harness-ready"而非仅仅"small"，该纪律可迁移至 JSON 有效性（98.1% vs. 30.2%）、GSM8K 严格格式遵循（61.9% vs. 18.7%）等跨任务，而基础模型在自由格式指令上仍占优。
3. **提出"世界知识 vs. 语言知识"二分框架解释 harness 收益不对称性**：DocSem 合成文档中世界知识无价值，LLM 仅需处理语言残留（理解改写查询、提取事实、忠实编写程序），其余路由给解释器/检索器/本体，从而重新定义紧凑模型的竞争边界。
4. **系统性地审计并量化测试集输入退化**：通过受控渲染实验（将验证集 PDF 重渲染为 72-dpi 灰度 JPEG + 水印）复现了 OCR 侧的全部崩溃，同时界定模拟与真实测试之间的差距，证明光栅读取质量而非推理架构主导了 leaderboard 双峰分布。
5. **检索与上下文消融实验给出边界条件**：chunk-level 图的 top-3 蒸馏实体有益（+0.017 on 27B），但完整图视图反而有害（-0.011）；稠密检索承担几乎全部段落选择工作（recall@3=0.9945），BM42 稀疏检索在此任务上几乎无效（recall@3=0.0663）。

## 方法详解
系统基于 RAGU 图检索增强生成工具链，整体管线分为四模块：

**文档摄入（Document Ingestion）**
- 块分割规则："标识符 token 至第一个冒号"，如 `b06:`；表格按行线性化为 `column: value` 形式内嵌于块中。
- 训练/验证 PDF 直接解析；测试光栅页由 Qwen2.5-VL-7B 以 2× 缩放逐页转录，页面分发至多个服务实例后按页索引拼接。
- 针对 38 页损坏页面（26 个文档，含 3 个空白页和 35 个近空白水印页的重复循环），用 Qwen3.8-27B 进行针对性修复。

**混合检索（Hybrid Retrieval）**
- 每份去重文档构建独立索引，块即 chunk。
- 密集向量：`gte-multilingual-base`；稀疏：BM42；两路经 Reciprocal Rank Fusion（RRF）在 Qdrant 中融合。
- `gte-multilingual-reranker` 交叉编码器将 top-8 裁至 top-3 展示给生成器。
- 检索消融：`bge-m3` recall@1=0.348，`gte` recall@1=0.901；rerank 后 `gte` 饱和至 1.000，`bge` 达 0.807。BM42 单独 recall@3=0.0663（因改写查询刻意避开目标段落措辞），稠密向量承担几乎全部选择工作。

**Program-of-Thoughts（PoT）生成**
- 生成器撰写仅读取已展示块的 Python 程序，沙箱解释器执行，最终变量即答案，引用的标识符即证据。
- 服务端通过 vLLM 强制 JSON schema 结构化输出（reasoning + program + evidence）。
- Self-consistency：T=0.8 采样 k=5 个程序，对执行结果投票；标识符经 snapping 校正以对抗转录噪声。
- PoT demonstrations（34 个验证程序）将证据 EM 提升至 1.000，代价是答案准确率下降 0.02–0.03。

**图上下文与最终组装**
- 用 Meno-Lite-0.2（基于 Qwen2-family 的紧凑域适应模型）从检索段落提取 chunk-level mini-graphs，数值本体扩展 NEREL 实体类型，新增 `QUANTITY` 和 `RATE`。
- 生成器 prompt 嵌入与查询在嵌入空间最近的实体（key entities, top-3）。
- 最终提交采用三票多数合并：base PoT 运行 + key-entity 运行 + 多模态升级（将非共识任务页面图像展示给 Qwen3.8-27B）。

## 实验与结果
**内部 Check 集（181 任务，Table 1）**

| 配置 | 答案 EM | 证据 EM | 联合准确率 |
|---|---|---|---|
| Qwen2.5-7B-Instruct, direct | 0.735 | 0.956 | 0.691 |
| Qwen2.5-7B-Instruct, PoT | 0.757 | 0.646 | 0.497 |
| Meno-Lite-0.1 7B, direct | 0.420 | 0.895 | 0.359 |
| Meno-Lite-0.1 7B, PoT | 0.674 | 0.934 | **0.641** (+0.282) |
| Qwen2.5-72B, direct, k=3 | 0.873 | 1.000 | 0.873 |
| Qwen3.8-27B, PoT, full doc, k=5 | 0.884 | 1.000 | **0.884** |
| 27B + key entities (top-3) | 0.901 | 1.000 | **0.901** |

- PoT 对 7B 提升 +0.282，对 72B 最多 +0.005；27B 完整 harness 匹敌 72B（0.884 vs. 0.873），参数比约 2.7:1，CO₂ 排放比约 4:1。
- 全文档 baseline（72B，0.884）与精挑 top-3（0.873）差异在噪声范围内，块选择在紧凑模型线上更有价值。

**验证门户（217 任务，Table 2）**
- Meno-Lite-0.1 + PoT 从 check 的 0.641 降至 portal 的 0.438（-0.203），损失集中在答案端（0.674→0.456）；证据纪律仅轻微下降（0.934→0.899）。
- Qwen3.8-27B + PoT 达 0.848，加上 permutation ensemble + key entities 至 0.853。
- 27B PoT 线超过 72B direct 线（0.848 vs. 0.825）。

**跨任务结构化输出对比（Table 3）**
- JSONSchemaBench（easy）有效 JSON 率：Meno-Lite-0.1 = 0.981 vs. 基础 7B = 0.302；GSM8K 严格格式：0.619 vs. 0.187；IFEval 严格指令遵循：0.679 vs. 0.791。
- 结构化输出训练赋予跨语言的格式纪律，可预测 PoT 效果的符号方向。

**测试集表现（Table 4）**
| 尝试 | 联合准确率 | 答案准确率 | 证据 F1 |
|---|---|---|---|
| 1: Tesseract 读取 | 0.29% | 2.54% | 1.50% |
| 2: VLM 逐页转录 | **13.58%** | 17.57% | 20.34% |
| 3: +合并+多模态升级+修复 | 13.41% | 17.34% | 20.11% |
- 排名 149/163；答案在两次尝试间翻转 89.6%，证明崩溃来自读取而非推理。
- 证据膨胀：单块证据比例从 93.9%（尝试1）降至 64.5%（尝试2），三块以上证据占比从 3.8% 升至 32.5%。

**受控退化研究（Table 6）**
- 将验证集重渲染为 72-dpi 灰度 JPEG + 水印后，Tesseract 侧联合准确率归零（0.000），VLM 侧仅轻微下降（0.756 vs. 0.825），说明 OCR 是测试崩溃的主因，但 VLM 在真实测试中的损害程度远超合成退化。
- Leaderboard 呈双峰分布：22 支队伍在 67.46% 处形成尖峰，其余为长尾读取失败组。

## 相关工作脉络
1. **Program-of-Thoughts（Chen et al., 2023）与 PAL（Gao et al., 2023）**：本文采用 PoT 框架将算术计算卸载至沙箱解释器，但与 PAL 不同，本文强调 PoT 效果高度依赖模型的结构化输出纪律（7B 谱系对比揭示此点），而此前工作多在大模型上评估。
2. **GraphRAG 及其轻量变体（Edge et al., 2024; Guo et al., 2025; Gutiérrez et al., 2024）**：本文与之对话的边界条件是——在短文档单段落算术任务上，仅 top-k 蒸馏的图上下文有益，完整图视图反而有害（-0.011），回应了 Xiang et al. (2026) 关于"何时使用图"的分析。
3. **Self-consistency（Wang et al., 2023）**：本文在 PoT 管线中引入 self-consistency 投票，但发现其对 72B 模型增益极微（k=3→k=5 仅 +0.005），而在紧凑模型上增益显著（k=3→k=5 +0.022），细化了 self-consistency 的收益边界。
4. **RAG 系统（Lewis et al., 2020; Komarov et al., 2026 RAGU）**：本文在 RAGU 基础上加入 chunk-level 图提取与实体 enrichment，消融表明稠密检索（recall@3=0.9945）承担主要选择工作，稀疏 BM42 几乎无效（recall@3=0.0663）。
5. **HybridQA（Chen et al., 2020）与 TaPas（Herzig et al., 2020）**：本文任务在表格+文本混合证据 grounding 上与其一脉相承，但 DocSem 的精确块标识符匹配（exact-match block identifier）对分割漂移零容忍，构成更强约束。
6. **Structured-output 训练相关（JSONSchemaBench, Geng et al., 2025; IFEval, Zhou et al., 2023）**：本文通过 7B 谱系跨任务对比，首次将结构化输出纪律量化为 PoT harness 生效的先决条件（JSON 有效性 0.981 vs. 0.302 可预测 PoT 效果符号方向）。

## 局限性与未来方向
- 单一共享任务研究，读取失败分析基于门户分数和提交诊断，未对测试文档进行人工重新标注验证。
- 视觉读取器仅为 7B VLM 逐页独立转录，未评估更强的文档解析 VLM，因此 pipeline 在光栅输入上的上限未知。
- 7B 谱系对比将英文基准过置于俄语主导模型之上，属于其声明领域的弱侧，结论限定于 harness 而非模型本身。
- 外部 GLM-5.3-Flash judge 仅探测 60 个测试任务，覆盖有限。
- 验证门户分数基于 217 任务，95% 置信区间约 ±4-5 个百分点，小于 1 点的行间差异需谨慎解读。
- 未来方向：在提交前增加输入物理属性审计（分辨率、水印、文字层存在性）作为标准流程；探索更强文档理解 VLM 在光栅 PDF 上的潜力；系统研究结构化输出训练与其他 harness 范式（如 JSON Schema、answer markers）的通用性。

## 研究启发与可借鉴点
1. **"输入审计先于架构设计"的工程纪律**：本文因未检查像素级输入属性（光栅 vs. 生成型 PDF）导致从 0.853 跌至 0.136 的惨痛教训，提示团队在任何 benchmark 提交前必须验证输入分布与训练数据的一致性，建议将分辨率、文字层、水印检测纳入 CI 流水线。
2. **结构化输出训练是 harness-based 架构的先决条件**：PoT/JSON 等刚性输出格式的收益高度依赖模型是否经过结构化输出微调，团队可借鉴用 JSONSchemaBench 有效性作为模型是否"harness-ready"的预筛选指标，避免在不合适的模型上浪费 harness 调优资源。
3. **图上下文在短文档单段落任务中存在收益递减阈值**：top-3 蒸馏实体有益，但完整图反而有害，提示在类似任务中应优先投入检索质量（密集向量+reranker）而非图复杂度。
4. **受控退化实验（controlled degradation study）是可复现的诊断范式**：本文通过将验证集重渲染为测试集物理属性来隔离读取 vs. 推理故障，此方法可直接迁移至团队任何涉及文档理解的 benchmark 失败分析。
5. **世界知识 vs. 语言知识的二分框架指导模型选型**：对于合成/领域特定数据（世界知识无效），应将预算投向架构 harness 而非盲目增大模型，27B harness 模型匹敌 72B 直接模型的案例为此提供定量支撑。

## 关键术语表
**DocSem**：DocInsights 2026 的 document-grounded quantitative reasoning 共享任务，要求从 PDF 定量段落中计算数值答案并精确标注证据块标识符。
**Program-of-Thoughts (PoT)**：让 LLM 生成可执行的 Python 程序而非直接输出答案，将算术计算卸载至沙箱解释器，以分离推理与计算（Chen et al., 2023）。
**Self-consistency**：在相同 prompt 下多次采样（T>0），对执行结果投票取众数，以提升链式推理的稳定性（Wang et al., 2023）。
**Joint Exact Accuracy**：DocSem 主排名指标，要求答案数值和证据块标识符集合均与 gold 完全匹配。
**Meno-Lite-0.1/0.2**：作者团队开发的 7B 域适应 Qwen2-family 模型，面向俄语 RAG 和结构化输出优化，Meno-Lite-0.2 为未发布的后继版本。
**RAGU**：作者团队开发的多步图 RAG 引擎，从块索引到搜索一体化，是本文系统的底层框架。
**结构化输出训练（Structured-output Training）**：针对 JSON schema、固定格式答案等刚性输出范式的微调，本文论证其赋予跨语言的格式纪律，是 PoT harness 生效的前提。
**Reciprocal Rank Fusion (RRF)**：将密集向量检索与稀疏检索（BM42）的排序结果融合的算法（Cormack et al., 2009），本文用于合并双路检索。

## 可复现要素
- **数据集**：DocSem 训练/验证/测试集（908/217/1730 个唯一 PDF），论文未明确说明公开状态；GSM-SEM 框架（Singh et al., 2026）可参考。
- **代码/权重**：RAGU 引擎有 arXiv preprint（Komarov et al., 2026, arXiv:2607.11683）；Meno-Lite-0.1 有 HuggingFace 数据集关联（NEREL-bench）；Meno-Lite-0.2 为未发布版本。论文未提供完整系统代码仓库链接。
- **关键超参**：PoT 采样 T=0.8, k=5；检索 top-k=3（reranker 后）；图实体 top-3；VLM 转录 2× zoom；沙箱解释器 2s 超时；JSON schema 服务端强制。
- **基线模型**：Qwen2.5-7B-Instruct、Qwen2.5-72B、Qwen3.8-27B、Qwen2.5-VL-7B、Meno-Lite-0.1/0.2。
- **检索模型**：gte-multilingual-base（密集）、BM42（稀疏）、gte-multilingual-reranker（交叉编码器重排）、bge-m3（对比用）。
- **推理框架**：vLLM（结构化输出强制）。
