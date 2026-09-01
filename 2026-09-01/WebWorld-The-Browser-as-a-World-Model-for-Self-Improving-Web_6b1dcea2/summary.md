---
title: "WebWorld-The-Browser-as-a-World-Model-for-Self-Improving-Web"
source: https://arxiv.org/pdf/2608.30530v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:34:42"
field: "Web 代码生成与自改进"
keywords: ["web code generation", "browser as world model", "self-improving agents", "interactive HTML", "SFT data construction", "hypothesis-proof separation", "VLM self-repair"]
innovations: ["将浏览器视为 Web 代码的现成世界模型，实现命题-证明分离的自改进闭环", "交互契约+接受证书+质量棘轮的三元认证机制，确保仅证书化转换进入 SFT", "五级证明体系（同轨迹重放→能力增益→目标进展→局部视觉→静态结构）平衡数据质量与吞吐量"]
benchmarks: ["HTMLBench-400", "MiniAppBench-Val"]
---

# 论文速读：WebWorld: The Browser as a World Model for Self-Improving Web Code

## 一句话总结
论文提出 WEBWORLD，将浏览器视为 Web 代码的"世界模型"，通过**交互契约（contract）+ 接受证书（certificate）+ 质量棘轮（quality ratchet）**的三元机制，让 VLM 提出的修复只能在浏览器重执行下通过"目标进展 + 先验能力保留"双重验证后才能进入 SFT 训练池，从而解决 VLM 自改进中"提议者即评判者"导致的视觉合理但行为失效问题。

## 研究问题与动机
- **VLM 自改进的结构缺陷**：VLM 同时负责提议修复和评判修复，且评判基于截图空间，导致"看起来更好"不等于"真正能用"；按钮失活、键盘输入被忽略、修复一处却静默破坏已有功能等问题普遍存在。
- **纯视觉评判无法捕获行为失效**：论文实验表明，最接近朴素 VLM critique-and-rewrite 回路的版本在 HTMLBench 上仅提升 0.4 分（对比完整 WEBWORLD 的 5.3 分），说明缺乏外部可执行证据的闭环几乎无效。
- **现有修复工作依赖测试套件作为预言机，而 Web 代码缺陷多为视觉/交互性**：SWE-bench 等工作依赖现成测试套件，但前端页面缺陷往往无法用静态测试充分覆盖，需要一种可直接可执行的验证机制。
- **Supervision 质量问题**：在自我改进闭环中，如何将运行时探索与训练监督分离，避免噪声数据污染 SFT 出口，是一个未被系统回答的设计问题。

## 核心贡献（创新点）
1. **将浏览器重构为 Web 代码的现成世界模型**：不训练任何世界模型，直接利用浏览器作为确定性可执行模拟器，使监督单元从"最终截图"变为"经认证的状态转换"。与已有工作（仅将浏览器作为导航环境或渲染打分器）的本质区别在于：浏览器在此充当**命题与证明分离**中的"证明方"（counterparty）。
2. **交互契约（Interaction Contract）机制**：将 VLM 自由形式的批评编译为结构化的类型化承诺三元组（目标谓词 + 重执行动作序列 + 保留集合），去除"看起来更精致"等无法被验证的主观评判。与已有 MLLM-as-judge 框架的本质区别在于：所有评判必须锚定到可观测的浏览器状态，而非截图偏好。
3. **接受证书（Acceptance Certificate）的五级证明体系**：同轨迹重放 → 能力增益 → 目标进展 → 局部视觉证据 → 静态结构修复，按优先级尝试，只有浏览器重执行同时满足目标进展和先验保留才颁发证书。与已有工作的本质区别在于：拒绝 VLM 的视觉-only 评判，由可执行证据决定数据准入。
4. **质量棘轮（Quality Ratchet）训练数据出口机制**：严格区分运行时步（ITERSTEP）与质量步（QUALITYSTEP），只有经证书认证的转换才能进入 SFT 导出集和 Agent 能力记忆，后续修复若破坏任何已验证能力则被拒绝。与已有 critique-and-rewrite 循环的本质区别在于：实现了单调递增的能力记忆，防止静默回退。

## 方法详解
**整体框架**（Figure 2）：每轮运行时在真实浏览器中渲染当前 HTML 产物，通过固定的交互探针目录驱动页面，记录截图、DOM、控制台状态、探针结果和执行动作轨迹（observation $o_t$）。VLM 读取 observation 并发出结构化批评；规划器将其编译为交互契约 $r_t$；路由选择匹配的修复技能；技能生成候选补丁 $\hat{x}_{t+1}$；运行时在契约约束下重执行候选；证书门控根据重执行结果颁发接受证书 $e_t$ 或出具类型化拒绝，仅证书认证后的转换进入 SFT 导出。

**世界模型验证轨迹（公式 1-3）**：
$$\mathcal{V}(q, x_t, r_t, \hat{x}_{t+1}, \mathcal{P}_t) \rightarrow \{\text{REJECT}, \text{ACCEPT}(e_t)\}$$
接受条件要求：TARGET($r_t, x_t$) = 0（当前未达标）、TARGET($r_t, \hat{x}_{t+1}$) = 1（目标达成）、$p(\hat{x}_{t+1}) = 1$ $\forall p \in \mathcal{P}_t$（所有先验能力保留）。训练使用完整转换 $(x_t, c_t, r_t, x_{t+1}, e_t)$ 而非仅最终 HTML 文件。

**交互契约（§2.3）**：批判必须命名问题族、依据证据、影响区域、成功条件、保留行为和推荐修复技能。"Make it more polished"不是有效批判。规划器将批判编译为三元承诺：目标谓词（typed predicate over post-repair state）、检查序列（固定动作链 + 稳定选择器）、保留集合（继承自之前认证的谓词）。契约被拒条件包括：锚定于宽泛容器、纯感知描述、无截图可观测后果的 DOM 谓词。候选编辑由类型化修复技能（非自由重写）生成，且受契约影响范围约束。

**接受证书五级证明（§2.4）**：① Same-trace replay：重放契约的精确动作序列；② Capability gain：先前失败的探针宏现在通过；③ Target-issue progress：等价重放下目标谓词为真；④ Localized visual evidence：补丁 diff 和视觉变化均在影响范围内；⑤ Static structural repair：低保留风险的无交互内容添加。每个候选仅携带一个获胜证明级别，确保离线可复现审计。

**质量棘轮与 SFT 导出（§2.5）**：
$$z_t = (q, x_t, o_t, c_t, r_t, \hat{x}_{t+1}, e_t, \rho_t)$$
QUALITYSTEP 判定：TARGET($r_t, \hat{x}_{t+1}$) = 1 且 $p(\hat{x}_{t+1}) = 1$ $\forall p \in \mathcal{P}_t$。当满足时更新能力记忆：$\mathcal{P}_{t+1} = \mathcal{P}_t \cup \text{VERIFIED}(e_t)$。SFT 导出为：
$$\mathcal{D}_{\text{SFT}} = \{(q, o_t, c_t, r_t, x_{t+1}, e_t) \mid \text{QUALITYSTEP}(z_t) = 1\}$$
拒绝的候选留在 ITERSTEP 流中用于路由和调试，但不进入训练池。

**四个构造性保证**：命题-证明分离、类型化声明、单调能力记忆、确定性准入。

## 实验与结果
**数据集与设置**：WEBWORLD 语料库含 32,800 个证书认证的转换；在 4B/9B/27B 规模上匹配训练（相同 backbone、prompt template、optimizer、评估器），仅监督源不同。主评测集：**HTMLBench-400**（6,000 个确定性浏览器测试用例，涵盖 Apps & Tools/Content & Mkt/Data Viz/Games & Sim/3D WebGL/Visual Art 六大类，400 个 item）；迁移评测：**MiniAppBench-Val**。Backbone 为 Qwen3.5 家族，AdamW（LR $2 \times 10^{-5}$，weight decay 0.01，3% warmup），seq len 16,384，训练 3 epochs。

**主要结果**（Table 1）：
- **WEBWORLD-27B**：HTMLBench-400 = **52.7**（Raw-27B = 47.4，+**5.3**分），TC Pass = **43.2%**（Raw: 33.5%，+9.7%），MiniAppBench-Val = **85.5**（Raw-27B = 70.6，+**14.9**分）。
- 达到 **Kimi-K2.6**（49.8 / 85.5）和 **GPT-5.4**（49.2 / 87.7）的交互 HTML 生成水平（HTMLBench 超越前者，MiniApp 接近后者）。
- 提升幅度随容量扩大：4B（+3.4）、9B（+4.8）、27B（+5.3）；提升集中在 TC Pass 和 Functionality 维度，Rendering/Visual/Code 维度差异 <1 分。

**证书机制消融**（9B，Table 2-3）：
| 规则 | Δ HTMLBench vs Raw-9B |
|---|---|
| No certificate | **+0.4** |
| No preserve | +1.5 |
| VLM-only | +3.5 |
| Score gate | +3.6 |
| WEBWORLD (full) | **+4.8** |

NoCertificate 设置下 TC Pass 反而**低于** Raw（31.0 vs 34.1），说明删除证书不仅无效，还注入噪声监督损害行为维度。

**深度诊断**（Table 4）：固定 5,000 样本，按最小认证深度筛选，≥5 步深度数据可达 48.8 分，仅比完整 32,800 样本（49.3 分）差 0.5 分，证明深层轨迹的每条样本携带更高密度的验证能力。

**数据构成**：60K 候选进入重执行阶段，24,350 被拒绝；认证池 42,860 个转换，按证明级别分布：same-trace replay 42.3%、capability gain 22.5%、target-issue progress 16.4%、localized visual evidence 11.5%、static structural repair 7.3%。

## 相关工作脉络
1. **Web 代理与前端评估**（ReAct、WebShop、Reflexion、Mind2Web、WebArena、Web-Voyager、VisualWebBench）：将浏览器用作导航环境或渲染打分器；本文定位差异：浏览器作为**可执行世界模型**，监督单元是证书化的状态转换而非 rollout 或截图分数。
2. **前端生成与评估基准**（Design2Code、WebSight、Image2Struct、DesignBench、FullFront、WebGen-Bench、ArtifactsBench）：聚焦截图到代码或静态渲染评估；本文定位差异：关注**交互行为**的正确性，使用确定性浏览器测试用例（6,000 个）而非视觉打分。
3. **自修复与自我改进**（Self-Refine、Self-Debugging、CodeRL、SWE-bench）：依赖执行反馈或测试套件；本文定位差异：Web 代码缺陷多为视觉/交互性且缺乏等价测试套件，本文用浏览器重执行替代测试套件作为预言机。
4. **MLLMAss-judge 框架**（ArtifactsBench 等）：用视觉评判器扮演预言机角色；本文定位差异：评判器继承提议者的模态偏差（screenshot-only），本文通过浏览器重执行实现命题-证明分离。
5. **合成数据与探索驱动 Web 流水线**（UCoder、Explorer）：探索驱动轨迹合成；本文定位差异：强调准入机制的质量控制，而非轨迹数量。
6. **并发工作 HTMLCure**（Wu et al., 2026b）：提供 HTMLBench-400 测试床和状态感知修复；本文定位差异：使用 HTMLCure 仅用于 held-out 评估，本文贡献是浏览器重执行作为世界模型反方、批判转为交互契约、仅证书认证转换导出 SFT。

## 局限性与未来方向
- **单文件 HTML  artifact 限制**：当前方法隔离于单文件交互 HTML，扩展到多文件或框架密集设置需处理依赖重放、项目组织、Agent scaffolding 等部署变量。
- **感知性缺陷无法被证书覆盖**：审美偏好、品牌身份、无障碍性等真正感知层面的缺陷没有可执行证书，只能回退到 VLM 代理，继承其噪声。
- **错误批判仍浪费修复预算**：验证器降低假阳性但不使 VLM 批判不可错，错误批判仍消耗修复资源；动态应用仍需确定性重放支持。
- **未来方向**：可扩展至多文件/框架项目、结合更多结构化测试语言、将本框架的思想迁移到其他代码域（如移动端 UI 代码、游戏脚本）的自改进。

## 研究启发与可借鉴点
1. **"提议者-证明者分离"范式可迁移至其他代码生成场景**：在 GUI 代码、移动端界面、可视化组件等缺乏现成测试套件的领域，可将运行时环境（模拟器/浏览器/渲染引擎）作为可执行世界模型，替代 VLM 自评，解决"自我评判噪声"问题。
2. **五级证明体系的粒度设计值得借鉴**：从严格的可执行重放到宽松的静态结构修复，形成证据强度梯度；这种分层设计可同时保证数据质量和数据吞吐量，避免过于严格的准入导致数据匮乏。
3. **质量棘轮的单调能力记忆机制适用于迭代式自改进**：累积已验证能力并拒绝破坏先验行为的候选，可防止多轮自改进中的静默回退，在 Agent 工具调用、代码生成流水线中均有应用潜力。
4. **契约编译作为第一道过滤层的设计**：将自由形式批评转为类型化声明，既约束了修复范围，又过滤掉无法被验证的主观判断，可推广至其他需要结构化反馈的语言模型交互场景。
5. **深度筛选作为数据质量代理**：论文证明深度 ≥5 的 5,000 样本（48.8 分）接近全量 32,800 样本（49.3 分），提示"通过更多轮验证的样本质量密度更高"这一信号可用于低成本数据筛选策略。

## 关键术语表
**World Model（世界模型）**：在此指浏览器作为 Web 代码行为的确定性可执行模拟器，无需训练即可提供"如果用户执行某操作，页面将如何响应"的预言能力。
**Interaction Contract（交互契约）**：将 VLM 批评编译为三元组结构化承诺——目标谓词、重执行动作序列、保留谓词集合，使不可验证的主观判断无法进入修复阶段。
**Acceptance Certificate（接受证书）**：浏览器重执行通过后颁发的自包含记录，包含目标重述、补丁 diff、证明级别、验证结果和执行证据（截图、动作轨迹、探针结果），每种证明级别唯一。
**Quality Ratchet（质量棘轮）**：严格区分 ITERSTEP（运行时探索步）和 QUALITYSTEP（证书认证的质量步），只有后者进入 SFT 导出和 Agent 记忆，实现单调递增的能力累积。
**Same-trace Replay（同轨迹重放）**：五级证明中最严格的级别，直接重放契约中的精确动作序列以验证目标谓词。
**Preservation Set（保留集合）$\mathcal{P}_t$**：累积所有已通过证书认证的能力谓词，后续修复必须保留集合中所有元素，否则被拒绝。
**HTMLBench-400**：400 个自然语言提示的静态交互 HTML 基准，配有 6,000 个确定性浏览器测试用例（click/type/hover/JS 断言/screenshot-change），功能分由测试用例通过率决定。
**HTMLCure Runner**：HTMLBench-400 的独立评估运行器，仅用于 held-out 评估，不参与数据构建或训练过程。

## 可复现要素
- **数据集**：WEBWORLD 认证转换语料 32,800 条（从 42,860 条认证池中均匀随机采样 seed=42）；HTMLBench-400（Wu et al., 2026b，公开）；MiniAppBench-Val（Zhang et al., 2026，公开验证集）。论文声明将发布兼容底层 artifact 条款的代码、日志和数据。
- **代码/权重**：论文声明计划发布代码、日志和数据（"We plan to release code, logs, and data"）；模型权重为 Qwen3.5 家族 4B/9B/27B 的 SFT checkpoint，论文未明确声明权重开源状态，需以正式发表为准。
- **关键超参**：Optimizer AdamW，LR $2 \times 10^{-5}$，weight decay 0.01，3% linear warmup；effective batch 128（gradient accumulation）；seq len 16,384；3 epochs；VLM critique 贪心解码（temperature=0, max 4096 tokens）；patch generation temperature=0.7，每契约 1 个候选；headless Chromium/Playwright；seed=42。
- **硬件**：4B/9B 用 64×H100，27B 用 128×H100。
- **探针目录**：14 个确定性浏览器宏（见 Appendix G）。
- **修复技能**：15 个 VLM 技能 + 5 个确定性 patch lane（共 20 个，见 Appendix G）。
