---
title: "Where-the-Verifier-Fails-A-Category-Level-Audit-of-Reward-Si"
source: https://arxiv.org/pdf/2609.01354v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 07:33:59"
---

# 论文速读：Where-the-Verifier-Fails-A-Category-Level-Audit-of-Reward-Si

## 一句话总结
本文针对 RLVR 与数学基准评测中普遍依赖的自动答案校验器（verifier），提出一套基于 metamorphic testing 的分类审计协议，在 307,420 次判定上系统分解四类主流 verifier 的错误分布，揭示校验失败并非源于复杂数学语法解析，而是高度集中于空白字符与标点处理（占默认 LaTeX 配置的 93.0%），并首次定量刻画了按数值量级呈阶跃接受错误答案的系统性假阳性机制。

## 研究问题与动机
1. RLVR 与标准 benchmark 评测均将自由文本答案通过自动 verifier 转为二元奖励/准确率，但 verifier 只是工程实现而非数学神谕，其设计决策存在固有失效模式。
2. 现有可靠性文献仅报告聚合指标（如 Hugging Face 称某 harness 自校验约 94%），无法说明错误具体分布在哪些答案形式类别中，导致 practitioner 难以定位修复方向。
3. 聚合错误率掩盖了截然不同的 failure mode：拒绝正确回答（false negative）与执行崩溃/超时报错（no verdict）被混为一谈，相似 aggregate 的 verifier 实际缺陷可能完全相反。
4. 系统性 verifier 错误已被证明会导致 RLVR 策略出现 plateau 或 collapse（Egashira et al., 2026），但缺乏细粒度的 error budget 刻画工具，校正方法所需的错误率参数同样无从精确获取。

## 核心贡献（创新点）
1. **Verifier 端的认证变形测试协议**：将 metamorphic testing 目标从模型反转至评分器，通过构造语义等价变换自动生成 ground truth，任何误拒即为 certified false negative，无需人工仲裁或更强模型比对。
2. **首个分类别的 verifier 错误预算分解**：在 43 种变换、14 个 stratum、307,420 次判定上测量，每单元格附 Wilson 置信区间，并引入 coverage 指标将拒绝与执行失败严格分离。
3. **发现 scale-dependent 确定性假阳性机制**：sympy-cascade 对 off-by-one 错误答案的接受率随金答案量级呈精确阶跃（<10⁴ 拒收，≥10⁴ 全接受），由相对容差 10⁻⁴ 的 scale-invariant 特性导致，aggregate 16.0% 错误率完全掩盖该形状。
4. **Contract matrix 与跨实现可比性论证**：明确各 verifier 声明处理的输入类别边界，区分实现缺陷与规范歧义；证明同库不同配置的接受率在相同输入上分歧高达 49.9%，指出 benchmark 准确率本质是 verifier 配置的函数。

## 方法详解
- **Certified equivalence 判定**：给定金答案 g 与语义保持变换 T，若 $V(g, T(g)) = \text{FALSE}$ 则认证为 false negative；对语义改变变换 $T'$ 若返回 TRUE 则认证为 false positive。Ground truth 由变换构造而非标注产生，样本规模随 golds × transforms × verifiers 组合放大且全程 CPU 运行。
- **Transform classes & Contract matrix**：43 种变换划分为三类：CERTIFIED-EQUIV（数学等价且在 verifier 声明合同内，拒收计 FN）、CONTRACT-DEP（依赖声明合同，反映规范歧义，仅报告不计缺陷）、ADVERSARIAL（改变语义，接受计 FP）。verifier 仅在声明覆盖的 stratum 内评分，避免对从未宣称支持 boxed/unit 的 normalizer 误判。
- **Coverage 分离设计**：定义 $n_{\mathrm{eval}} = n_{\mathrm{TRUE}} + n_{\mathrm{FALSE}}$，coverage = $n_{\mathrm{eval}} / n$。parse exception、timeout、crash 均记为未返回 verdict，Cells 零 coverage 报告 N/A 而非 0%，防止虚假的“正确拒绝”印象。
- **待测对象**：math-verify (LaTeX 默认配置, `mv-latex`)、math-verify (plain-expression, `mv-expr`)、DeepSeek-Math 字符串归一化器 (`strip-string`)、三级引用级联 (`sympy-cascade`：exact string → numeric rel tol 1e-4 → SymPy)。ANTLR4 runtime pinned to 4.13.2。
- **数据与执行**：4,990 个唯一金答案（GSM8K、MATH 七学科、Big-Math 及 2,000 合成标准格式答案）；Azure ML 并行 CPU 管道，每项 5s 超时预算，崩溃与超时单独记录。

## 实验与结果
- **Self-validation 跨度极大**：四类 verifier 在相同输入上的等价变体接受率从 53.8% (`mv-expr`) 到 95.2% (`strip-string`)，跨度 41.3 点；同库 `mv-latex` 与 `mv-expr` 在 49.9% 配对上直接分歧。
- **错误预算高度集中**：空白与标点（S6 whitespace）主导失败：`mv-latex` 占 93.0%，`mv-expr` 占 74.1%，剩余质量分散于集合、分隔符、分数处理；典型失效为表达式后多一个句点或换行即被拒。
- **Coverage 揭示相反缺陷**：`sympy-cascade` coverage 87.3%，但 judged accuracy 100%（全部残差为 parse exception）；`strip-string` coverage 100%，残差为过度拒收；两者 aggregate 误差相近但病因相反。
- **量级阶跃假阳性**：`sympy-cascade` 对 off-by-one 答案接受率在 |gold| < 10⁴ 时为 0%，≥10⁴ 时为 100%（10001 与 10000 相对差仅 0.01%），整体 FP 率 16.0%（95% CI [14.1, 18.1], n=1287）。
- **Corpus-only 稳健性**：剔除 2,000 合成答案后，四者自校验率分别为 57.5%、84.1%、87.8%、88.3%，排序与量级保持不变。
- **Contract-dependent 输入差异**：boxed 答案接受率 0%–75.1%，科学计数法 0%–100%，证明跨论文 benchmark 准确率不可直接比较。

## 相关工作脉络
1. **Cai et al. (2025)** 将 verifier 建模为不对称噪声奖励通道并推导梯度修正，但假设已知错误率；本文测量并分解该率，为其修正项提供实证参数。
2. **Egashira et al. (2026)** 指出 verifier 错误是系统性而非随机性，决定 RLVR 轨迹的是错误形状；本文的 per-stratum 分解刻画了该形状，量级阶跃为其“最大系统性”实例。
3. **Huang et al. (2026)** 对比规则与模型 verifier 发现反向失败方向；本文在 false-negative 侧扩展出分类级分解，并加入认证对抗探针。
4. **Ammanamanchi et al. (2026)** 对 Lean 定理证明 benchmark 做大规模静态审计；本文采用相同审计姿态，但目标转移至自然语言答案的自动化评分器。
5. **Lan et al. (2026)** 在安全扫描器评测中主张 coverage 应与 accuracy 分开报告；本文将此结构论证引入 verifier 评测，证明二者合并会产生误导性 0% 错误率。
6. **Su et al. (2025) / Hua et al. (2025)** 揭示 prompt/格式微小扰动对 MMLU 准确率的大幅影响；本文为 verifier-side 的对应发现，说明下游 accuracy 波动部分源于评分器配置而非模型能力。

## 局限性与未来方向
- 仅审计 verifier 本身，未估计其对任意模型 benchmark 分数的实际影响（真实模型输出不产生等比例变换变体）。
- 未进行任何 RLVR 训练运行，下游 reward shaping 效应未知（仅引用 Zhang 2026 发现 leaky vs hardened reward 影响有界）。
- 仅覆盖开源实现，不对闭源 frontier 系统做任何声明。
- 2,000/4,990 金答案为合成数据（虽已单列 corpus-only 结果验证稳健性）。
- Contract matrix 分配属主观判断，已公开供社区质疑与修订。
- 未来方向：将审计协议延伸至模型输出端错误传播建模、开发 coverage-aware reward clipping 策略、推动 verifier 配置标准报告规范（identity/version/extraction/runtime）。

## 研究启发与可借鉴点
1. **审计范式可迁移**：metamorphic testing 目标反转至 grader/verifier 的设计可复用至代码执行器、形式化证明检查器、表格/图表解析器等自动化评分组件的可靠性评估。
2. **Coverage 作为一等评测指标**：将 rejection 与 execution failure 分离是评估流水线设计的通用最佳实践，建议团队在后续 reward/verifier 论文中强制报告 coverage 曲线与 N/A 单元格。
3. **错误预算揭示高性价比优化点**：93% 失败源于空白标点而非复杂语法，提示 RLVR pipeline 的输入归一化层（trailing punctuation/newline stripping）存在低成本高回报的改进空间。
4. **系统性假阳性可显式校正**：scale-dependent 阶跃接受机制表明固定相对容差会引入确定性 magnitude bias，后续 reward 建模可改用绝对容差、分段容差或对大数答案施加惩罚正则。

## 关键术语表
- **RLVR**：Reinforcement Learning with Verifiable Rewards，利用可自动验证答案正确性生成二元奖励信号的后训练范式。
- **Metamorphic Testing**：变形测试，通过构造输入-输出间的语义等价变换来验证系统一致性的黑盒测试方法。
- **Certified False Negative/Positive**：经变换构造 guarantees 的答案，verifier 误判即为“认证”错误，无需人工仲裁或更强模型比对。
- **Contract Matrix**：记录各 verifier 声明处理的答案形式类别的矩阵，用于界定评分边界并区分实现缺陷与规范歧义。
- **Coverage**：verifier 成功返回裁决的输入比例，与 judged accuracy 分离以避免将 crash/timeout 误计为正确拒绝。
- **Off-by-one Adversarial Answer**：与金答案相差 ±1 的对抗性错误答案，用于探测数值校验器的容差边界与尺度敏感性。
- **Scale-invariant Relative Tolerance**：相对误差容限（如 1e-4
