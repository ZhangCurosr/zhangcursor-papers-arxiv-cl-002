---
title: "WHO-Is-LEFT-OF-WHOM-TRACING-SPATIAL-EVIDENCE-AND-ROLE-BINDIN"
source: https://arxiv.org/pdf/2609.35486v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 04:24:49"
field: "视觉语言模型可解释性"
keywords: ["spatial reasoning", "mechanistic interpretability", "vision-language models", "activation patching", "role binding", "location ID", "relative position"]
innovations: ["揭示源端位置信息到查询端角色绑定的分阶段因果通路", "提取目标/参考角色方向并证明其因果作用与零样本迁移能力", "配对一致性测试框架诊断空间推理不一致性"]
benchmarks: ["Synthetic 2-object/3-object", "What'sUp-A/B", "COCO-Spatial", "GQA-Spatial"]
---

# 论文速读：WHO-Is-LEFT-OF-WHOM-TRACING-SPATIAL-EVIDENCE-AND-ROLE-BINDIN

## 一句话总结
论文通过激活修补（activation patching）和目标干预（steering）技术，揭示了 VLM 在相对位置推理中的分阶段表征机制——从源端对象位置信息传递到查询端位置 ID，再经目标/参考角色方向绑定输出关系判断；并证明基于合成数据估计的角色方向可零样本迁移至自然图像基准（What'sUp、COCO-Spatial），在不重训的前提下同时提升准确率与配对一致性。

## 研究问题与动机
1. **准确率悖论**：现有空间推理评测以实例级准确率为核心指标，但模型可能在单条查询上答对，却在对象位置交换（object-location swap）或目标/参考角色反转（target/reference role reversal）时给出矛盾答案，高准确率掩盖了推理不一致性。
2. **表征机制不清**：VLM 内部如何编码对象位置、如何将其与查询角色绑定，尚未有系统理解；已有 mechanistic interpretability 工作仅覆盖图像输入条件，未探索文本输入及源端→查询端的因果传递路径。
3. **定位与角色未分离**：相对位置推理本质需两个独立计算——（i）恢复被查询对象的物理位置；（ii）将对象绑定到 target/reference 角色。现有工作未同时刻画这两类表征及其交互。
4. **缺乏因果验证**：此前研究多停留在相关性分析（如 probing、logit lens），未通过定向干预建立位置信息与角色绑定对最终答案选择的因果链条。

## 核心贡献（创新点）
1. **配对一致性测试框架**：引入对象位置交换与目标/参考角色反转两种扰动测试，揭示"高准确率≠推理一致"现象；相较于仅报告单一准确率的评测，本框架能诊断模型空间推理的脆弱环节。
2. **分阶段表征定位**：通过层-wise 激活修补，首次刻画空间推理的三阶段进展——早期层源端视觉/文本 token 承载位置证据，中间层查询端对象 mention 编码位置 ID，晚期层最终 token 做关系判断；与仅有局部定位的研究相比，本文建立了完整的因果传递链。
3. **Location ID 提取与源→查询因果链路**：在源端和查询端分别提取去属性偏差的对象中心化位置原型（location ID），并通过源端修补证实源端位置信息因果影响查询端位置比较得分与答案偏好；区别于 Kang et al. (2026) 仅报告图像条件的发现，本文在 VLM+text 和 LLM+text 下同样验证该链路。
4. **目标/参考角色方向（Role Direction）**：通过对比同一对象分别作为 target 和 reference 时的查询端隐藏状态差异，提取稳定方向向量；破坏该方向显著降低角色敏感型预测的正确率，而正向引导可零样本迁移至自然图像基准提升配对一致性；这是对角色绑定的首个因果表征分析。

## 方法详解
**1. 配对扰动设计**
- 对象-位置交换 τ_loc：交换两个被查询对象的物理位置，保持查询句式不变，正确答案翻转。
- 目标/参考角色反转 τ_role：保持场景不变，交换 query 中 target 和 reference 的角色，正确答案翻转。

**2. 激活修补（Activation Patching）**
- 对 clean 和 corrupt 输入缓存残差流状态 H^(ℓ)。
- 在层 ℓ 的 token 组 g 上执行 full-vector patching：H_g^(ℓ)(x_patch) ← H_g^(ℓ)(x_clean)，测量 clean-answer recovery rate (CARR)：CARR_(ℓ,g) = E[1{M(x_patch) > 0}]，其中 M(x) = logit_x(a_clean) − logit_x(a_corrupt)。
- 修复组覆盖三类语义 token：source（所有视觉 token / 对象 bbox/strip / 文本描述对象名与位置语句）、query（target+reference 提及 span / 关系选项词）、final（生成答案前的最后一个 token）。

**3. Location ID 提取**
- 对对象 o 在站点 g∈{src, qry} 的残差状态 h_(i,o,ℓ)^g 按属性去均值：ḣ = h − μ_ρ,ℓ^g。
- 位置标签 π 的原型：ID_(π,ℓ)^g = E[ḣ | π_(i,o) = π]。
- 空间轴由对立面 ID 差定义：v_(x,ℓ)^g = ID_(right,ℓ)^g − ID_(left,ℓ)^g，v_(y,ℓ)^g = ID_(above,ℓ)^g − ID_(below,ℓ)^g。

**4. 源→查询因果传递**
- 将 corrupt-run 的源端状态 patch 进 clean-run：H_g^(ℓ)(x_patch^(corrupt,(ℓ,g))) ← H_g^(ℓ)(x_corrupt)。
- 测量下游层 ℓ' 处查询端位置比较得分偏移 Δm→corrupt 与 corrupt-answer flip rate (CAFR)。

**5. 查询端 Location-ID Steering**
- 构造目标/参考位置标签交换干预：π'_t = π_r, π'_r = π_t。
- 在层 ℓ 向查询端对象 token 注入方向差：h_k^(ℓ) ← h_k^(ℓ) + α(ID_(π'_o,ℓ)^qry − ID_(π_o,ℓ)^qry)，k ∈ span(o)。

**6. 角色方向估计与干预**
- 使用 target-first (TF) 与 reference-first (RF) 两种句式模板，对每对角色反转样本计算联合对象级角色对比：d_(p,o)^(ℓ) = ½ Σ_{t∈{TF,RF}}(h_(p,o,t,target)^(ℓ) − h_(p,o,t,reference)^(ℓ))。
- 聚合得 joint role direction：d_role^(ℓ) = (1/|C|) Σ d_(p,o)^(ℓ)。
- 破坏性干预：h_target ← h_target − α·d_role，h_reference ← h_reference + α·d_role；测量候选答案 margin 变化 ΔM 并与正交控制比较（D = ΔM_role − ΔM_orth）。
- 正向 steering：h_k^(ℓ) ← h_k^(ℓ) + α·s_o·d_role^(ℓ)，s_o = +1(target)/−1(reference)。

## 实验与结果
**数据集**：Synthetic（2-object/3-object，336×336 白底，3×3 网格固定位置，颜色-形状 conjunction 对象）、What'sUp-A/B（受控摄影，保留对象身份同时变化空间关系）、COCO-Spatial（2-object subset，440 annotated pairs）。

**模型**：10 个 VLM 及其 LLM backbone，重点报告 LLaVA-1.6-Mistral-7B、InternVL3.5-8B、Pixtral-12B。

**评估指标**：Original Accuracy、Target-Reference Reversal Consistency、Location-Swap Consistency。

**关键结果**：
- Synthetic 上部分模型达满分（InternVL3.5、Qwen3-VL、Pixtral 的 VLM+image 均 100%），但 LLaVA-1.5/1.6-Vicuna 的 LLM+text 仅 50% 准确率（固定答 left/above bias），且配对一致性为 0%。
- What'sUp 上 VLM+image 最优；LLaVA-1.6-Mistral 在 VLM+image 下准确率 88.85%，但 reversal consistency 仅 81.03%、swap consistency 仅 78.31%，暴露不一致性。
- **角色方向 steering 在自然图像上的零样本迁移效果**（表 2）：
  - **Pixtral VLM+text**：Accuracy +4.9%（77.0→81.9），Reversal +7.8%（57.8→65.7），Swap +9.4%（54.7→64.1），为最大提升。
  - **InternVL3.5 LLM+text**：Accuracy +2.4%，Reversal +3.3%，Swap +4.4%。
  - **LLaVA-1.6 VLM+image**：Accuracy +1.8%（82.2→84.0），Reversal +3.3%（70.6→73.9）。
- COCO-Spatial 上 steering 同样提升 reversal consistency 于全部 9 个 setting（表 10）；GQA-Spatial 左/右子集同理（表 11）。
- 最强结果：InternVL3.5 VLM+image 在 Synthetic 上 100% 全指标满分；What'sUp 上 Pixtral VLM+text steering 后 Swap consistency 达 64.1%。

## 相关工作脉络
1. **Kang et al. (2026)**：在 image-conditioned VLM 中发现 object-word 激活携带 spatial ID，操控该 ID 可转移空间信念；本文扩展至文本输入并在 VLM 与 LLM backbone 双重验证，且建立源端→查询端的因果传递链路而非仅孤立检测 ID。
2. **Cui et al. (2026)**：识别 VLM 中主导全局视觉信号的次要通路机制；本文通过源端/查询端分组的 patching 实验精细定位两条通路的具体作用层与 token 组。
3. **Salazar et al. (2026)**：提出 query-token-mediated pathway；本文细化该通路内涵——区分其中编码的 location information 与 role binding 两个正交分量。
4. **Geva et al. (2022)**：建立 activation patching 与 logit lens 在 LLM 中的标准化分析框架；本文将其推广至 VLM 跨模态场景并设计 source-query-final 三阶段分析范式。
5. **Chen et al. (2025a, 2025b)**：诊断 VLM 空间推理失败源于视觉注意力错配与深度特征空间细节损失；本文从表征几何角度给出更底层的解释——位置 ID 与角色方向的解耦与交互机制。
6. **Kamath et al. (2023)**：构建 What'sUp 基准检验 VLM 空间推理；本文在此基础上引入配对一致性测试（reversal/swap consistency）作为诊断工具，并提出无重训的 steering 提升方法。

## 局限性与未来方向
1. **场景简化**：仅限二维网格固定位置、颜色-形状 conjunction 对象的受控场景，未涉及深度、距离、多对象复杂布局与自由空间。
2. **查询句式模板化**：使用规则生成的目标/参考角色反转句式（TF/RF），未覆盖开放式自然语言查询。
3. **错误归因受限**：激活修补仅分析正确回答 pair（clean 与 corrupt 均选各自正确答案），错误案例的表征分解留待未来。
4. **模型规模与架构**：仅覆盖三个 7B–12B 开源模型，MoE 架构与大尺寸/专有模型尚待验证。
5. **干预工程化不足**：角色方向 steering 作为一次性分析工具已验证有效性，但尚未形成可自动化的 post-training 方法，泛化到更广泛的空间任务（如 3D、多跳关系）仍需探索。

## 研究启发与可借鉴点
1. **配对一致性测试可作为通用诊断框架**：位置交换与角色反转的扰动设计逻辑可直接迁移至其他关系推理任务（如因果、时序、拓扑关系），用于暴露"准确率幻觉"。
2. **Location ID 提取流程可复用于其他空间基准**：属性去均值 + 对立面 ID 对比 + 投影验证的 pipeline 简洁且可解释，适合快速在新数据集上检验模型是否编码有意义的位置几何。
3. **角色方向（Role Direction）概念的可迁移性**：target/reference 角色绑定的隐空间方向提取方法可推广至其他需角色/论元区分的任务（如 agent-patient 语义角色、时间视角），为跨任务的表征操控提供模板。
4. **零样本 steering 提升方案**：在合成数据上估计方向、直接应用于自然图像且不更新权重——这一"分析→干预"范式避免了微调成本，适合对已有大模型做低成本一致性增强。
5. **源→查询因果链路验证方法**：通过 corrupted-source patching 测量下游位置比较得分偏移与答案翻转率，为追踪跨模态/跨阶段信息流提供了标准化的因果验证协议。

## 关键术语表
**Location ID**：从对象中心化残差状态中提取的位置原型向量，表征对象在场景中的空间位置（如 left/right/above/below），经对立面 ID 差可定义空间轴。
**Role Direction（目标/参考角色方向）**：由同一对象分别作为 target 与 reference 时的查询端隐藏状态差异聚合而成的稳定方向向量，编码角色绑定信息。
**Activation Patching**：将 clean/corrupt 输入的残差流状态在指定层与 token 组上互换，测量其对答案偏好的因果影响的解释性技术。
**Paired Consistency（配对一致性）**：要求模型在原始查询与扰动后查询（位置交换或角色反转）下均给出正确答案的联合准确率，用于诊断推理稳定性。
**Object-Location Swap（对象-位置交换）**：保持查询句式不变、交换两个被查询对象的物理位置，若模型推理一致则答案应相应翻转。
**Target/Reference Role Reversal（目标/参考角色反转）**：保持场景不变、交换 query 中 target 与 reference 的角色，正确答案随之翻转。
**VLM+image / VLM+text / LLM+text**：三种匹配输入条件，分别指多模态模型接收图像+查询、多模态模型接收文本场景描述+查询、纯语言模型接收文本描述+查询。
**CARR / CAFR**：Clean-Answer Recovery Rate（激活修补恢复正确答案的比例）与 Corrupt-Answer Flip Rate（导致答案翻转的比例），用于量化干预强度。

## 可复现要素
- **数据集**：Synthetic（作者自构建，2-object 共 4,968 questions；3-object 共 4,968 questions）、What'sUp-A/B（Kamath et al., 2023）、COCO-Spatial（Lin et al., 2014 的子集）、GQA-Spatial left/right subset。合成数据集声明 publication 后开源。
- **代码/权重**：复现性声明称"Code, experiment configurations and synthetic datasets will be released upon publication"；预训练模型权重可通过官方渠道获取（LLaVA-1.6-Mistral-7B、InternVL3.5-8B-Instruct、Pixtral-12B 等）。
- **关键超参**：Steering 强度 α ∈ {0, 0.25, 0.5, 1, 1.5}；CARR 筛选阈值 clean-over-corrupt margin ≥ 0.25；解码使用 greedy decoding，max new tokens = 8；随机方向控制用 5 个 seed {42–46}；Bootstrap 95% CI 用 2,000 replicates paired cluster bootstrap。
- **硬件/环境**：单卡 NVIDIA A100-SXM4 40GB；Python 3.13.1、PyTorch 2.9.1、CUDA 12.8、Transformers 4.57.6；LLaVA/Vicuna/Mistral-7B 用 FP16，Pixtral/InternVL/Qwen3 用 BF16。
