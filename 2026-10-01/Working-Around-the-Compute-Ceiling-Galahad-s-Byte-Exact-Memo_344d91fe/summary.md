---
title: "Working-Around-the-Compute-Ceiling-Galahad-s-Byte-Exact-Memo"
source: https://arxiv.org/pdf/2609.39358v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 22:02:18"
---

```markdown
# 论文速读：Working-Around-the-Compute-Ceiling-Galahad-s-Byte-Exact-Memo

## 一句话总结
提出 Galahad——一个面向 vLLM / SGLang / llama.cpp 的记忆层，把 LLM 对已读文本的重复 prefill 计算转化为一次性成本：Taliesin 精确持久化 KV 状态（load 后 logits bit-identical），Blaise 按原文分节只传问题所需段落；在 7 个真实数据集上 98.7% 的 prompt tokens 来自记忆而非重算，把推理从 stateless 推向 stateful。

## 研究问题与动机
- **核心问题**：Serving 对同一文档的重复问题会反复 prefill 整篇文档，而实测 98.7% 的 prompt tokens 是模型已经读过的文本，这部分计算本质上是重复劳动。
- **计算天花板不可提升**：接受 Sikka & Sikka [18] 的 $O(N^2 \cdot d)$ 每 token 上限，不试图突破它，转而问“天花板之下的预算有多少花在重复工作上”。
- **现有 KV 复用范围过窄**：vLLM / SGLang 的 prefix cache 只在 GPU 内存内、进程存活期内有效；LMCache / Mooncake 等 offload 到 CPU/磁盘但仍接受近似；CacheBlend / RAGCache 做压缩或融合。
- **正确性门槛**：近似复用会静默改变生成 token（如 vLLM issue #33123 在 AMD MI355X 上 Reported），因此需要 bit-identical 的 accept 测试。
- **cost 语义应翻转**：请求成本应随 new text 增长，而非 total text；对支持/代码/合同/agent 等高返回同一文档的 workload 尤其关键。

## 核心贡献（创新点）
1. **Byte-exact KV 持久化（Taliesin）**：以（输入字节指纹 × 模型指纹 × tenant ID）为 key 存/读 KV state，load 失败即 fallback 到正常 prefill；与 CacheBlend / RAGCache 等近似复用的本质区别在于“logits bit-identical 是唯一接受标准”。
2. **原文分节文本记忆（Blaise）**：CPU 端按文档原始 section 选取，只把该 section 的精确字节送入 model；与 BM25 / 相似度检索的 chunk 不同，目标是 exact section 而非 similarity-ranked chunk。
3. **单库三运行时接入**：`libgalahad.so` 分别挂到 vLLM(KV connector)、SGLang(HiCache backend)、llama.cpp(slot save/restore)，不改 runtime 源码、不改 model weights。
4. **从 RoPE 推导复用粒度**：因 rotary position encoding 使每行 KV 依赖 position，实验测得 0/344,064 行可跨 prompt 共享，故必须整 block 复用——这是设计边界的理论依据。
5. **Fail-closed 安全语义**：confirmation hash / licence signature / tenant separation / byte comparison 四项防御任一失败即重算；以 1/3 载荷刻意失败实验证明 throughput 仍高于 no-Galahad 基线（2.716 vs 2.643 turns/s）。

## 方法详解
- **Taliesin（KV 记忆）**
  - 路径：runtime prefill → KV state → 写入存储（指纹索引）→ 下次同 block 出现时 load → 替换 prefill。
  - 存储格式：论文声明由 pending patent 覆盖，此处不展开；已知约束是 block 内 token position 必须精确匹配。
  - 验证链：confirmation hash + licence signature + tenant separation + load 时 byte comparison；任一失败 → recomputed。
  - 加密：支持 at-rest 加密（NIST T4/A40 CAVP 4,167 vectors, 0 failures）。
- **Blaise（文本记忆）**
  - 文档按原文分 section 保存为 byte-exact 文本；question 到达时 CPU 端选取一 section，把该 section 文本送入 model。
  - 报告的是第一 mode（CPU 选节）；第二 mode（model 读 corpus index、Taliesin 保留那次阅读）留待后续 benchmark。
  - 索引构建：recall test 上耗时 0.1 s、未占用 GPU token 计算。
- **部署形态**
  - 单一共享库 `libgalahad.so`（Linux x86-64），依赖 OpenSSL libcrypto + zstd。
  - 强制 tenant identity + model fingerprint，禁止跨模型/租户状态混用。
  - 运行时接口：vLLM KV connector / SGLang HiCache backend / llama.cpp slot save-restore。
- **设计边界（Section 5）**
  - Block-level reuse 是正确粒度（RoPE 决定）。
  - Fail-closed 是必然选择（approximate reuse 会静默出错）。
  - Memory 只能利用 upstream 已提供的文本；parser 丢弃的内容无法恢复。

## 实验与结果
- **Recall test（核心可控实验）**
  - 设置：100 条 invented-fact 插入 13 篇 Wikipedia（434 KB / 96,726 tokens / 11 blocks）；Gemma 4 31B (4-bit) / RTX A6000；每题 paraphrase 提问，深度 1%-99%。
  - 基线（no Galahad，12k window）：10/100，9.25 s，2,754 J（llama.cpp）。
  - Taliesin only：98/100，3.01 s，572 J（3.1× 更快，79% 更少能耗）；每题平均 53,219 prompt tokens 中 99.5% 来自 Taliesin。
  - Taliesin + Blaise：100/100，0.64 s，213 J（llama.cpp）；0.59 s / 200 J（vLLM）；14.5× 相对基线。
  - 一次性成本回收：存 corpus 耗时 96-108 s、27.8-28.4 kJ；llama.cpp 上第 13 题回收能量、第 16 题回收时间。
  - 外部 RAGFlow v0.27.2（L40S，lightly tuned）：77/100。
- **7 个 real-world datasets（349 questions，未 tuning 数据）**
  - 98.69% prompt tokens 来自 Taliesin（llama.cpp 99.49% / vLLM 99.37% / SGLang 97.21%）。
  - Taliesin only：329-333/349（94-95%）；Taliesin + Blaise：319-321/349（91-92%），3-10× 更快。
  - Messy PDFs 子集：50 题中 9 题 answer 在 PDF 解析阶段已丢失；可答 41 题里 Taliesin+Blaise 得 30。
- **Correctness（Table 1）**
  - Save/restore：262,144/262,144 logits bit-identical（Gemma 4 12B Q8_0，llama.cpp，2026-08-22）。
  - Deterministic prefill：20/20 byte-equal（SHA-256 of logits），跨 Llama 3.1 70B / Gemma 4 E4B / Qwen 3.6 35B-A3B / Ministral 3 8B。
  - Cross-GPU type：64/64 greedy tokens identical（A6000 ↔ RTX 4090）。
  - Stored bytes across 10 GPU types：3/3  each。
  - Hybrid architecture：5/5 token-exact vs Qwen3.5 9B (Gated DeltaNet, TP=2)。
  - Snapshots：276/276 tokens bit-identical after restart（RTX PRO 4500）。
- **Serving performance**
  - TTFT：Gemma 4 12B / vLLM / ~5,200 tokens prompt；RTX PRO 4500 1,138.8→391.2 ms（2.91×），H100 SXM 251.2→191.0 ms（1.32×）。
  - Throughput：H100 / Qwen3-30B-A3B / 16 sessions / concurrency 8 → 2.643→3.424 turns/s（+29.6%）；1/3 载荷刻意失败时 2.716 turns/s 仍高于基线。
  - Restore latency：24,018-token block 从本地磁盘 load 81 ms（±0.5 ms, 10 repeats; A40）；含 8 generated tokens 的请求 314 ms。
  - Storage footprint：96,726-token corpus 占 17.1-17.2 GB（llama.cpp/vLLM, Gemma 4 31B）或 39.0 GB（SGLang）；DeepSeek-V4-Flash 284B 93,157 tokens 占 27.6 GB。
  - Long windows at flat GPU memory：A40 / Gemma 4 / llama.cpp / 200 blocks × 30,000 tokens = 5.97M tokens → 5/5 probes 命中，GPU mem 24.4-24.7 GB；1M→50M tokens ladder 24/25 probes（1 miss 为 harness 解析错误），peak GPU mem 33.8-34.1 GB。
- **Coverage**
  - 30/30 models 在 vLLM 0.30.0 / 4×A10G 上完成 answer/save/load/hit。
  - DeepSeek-V4-Flash 284B（2×H200）：98/100 recall，0.133 s / 143 J per question。
  - Kubernetes（GKE / L4）：pod restart 后 63.9% lookups hit saved state；SIGKILL 后 0 damaged records。

## 相关工作脉络
1. **PagedAttention (vLLM [7]) / RadixAttention (SGLang [20])**：GPU 内存内 prefix 共享；Galahad 通过相同接口把复用扩展到进程寿命与 GPU 内存之外。
2. **LMCache [11] / Mooncake [12]**：KV offload 到 CPU/磁盘/远端；Galahad 区别在于要求 exact reuse（bit-identical logits 为 accept 测试），不Accept approximation。
3. **CacheGen [10] / CacheBlend [19] / RAGCache [6]**：压缩 KV 或融合 retrieved chunks；Galahad 不压缩、不融合，任何 mismatch 视为 miss 并重算。
4. **Prompt Cache [3]**：预定义 prompt module 复用；Galahad 按输入字节指纹动态索引，不按模块硬编码。
5. **Cartridges [2]**：训练 compact KV 表示；Galahad 存 unmodified KV，零训练。
6. **BM25 / lexical ranking [13]**：retrieval baseline；Blaise 目标是 exact section 而非 similarity-ranked chunk，定位不同。
7. **作者 prior work**：Merlin byte-exact deduplication [16,17]、byte-exact KV grafting [14,15]；本文是三者集成到三 runtime 的第一次系统化评估。

## 局限性与未来方向
- **自述局限**
  - Blaise second mode（model 读 corpus index、Taliesin 保留那次阅读）尚未 benchmark。
  - Messy PDFs 的 section selection 仍需改进（9/50 题答案在 PDF 解析阶段已丢失）。
  - 更多 model 与 independent operators 的结果待补充。
- **可推断局限**
  - 依赖上游 parser 质量：parser 丢弃的文本记忆层无法恢复。
  - Beta 阶段仅支持单 GPU、非商业；多 GPU / 多租户 / 生产级 SLA 未验证。
  - Storage 大小受 disk budget 配置约束，long-context 场景的存储成本需进一步评估。
- **未来方向**（论文 Section 7）
  - Benchmark Blaise second mode。
  - 扩展 section selection 到 poorly extracted PDFs。
  - 更多 model 与独立操作者的复现结果。

## 研究启发与可借鉴点
1. **Fail-closed 作为记忆层默认语义**：任何 load 失败都 fallback 到正确路径，性能以 throughput 不低于 no-memory 基线为底线；该原则可迁移到 KV cache、embedding cache、tool response cache 等场景。
2. **Byte-exact 指纹索引三元组**（input bytes × model fingerprint × tenant ID）：同时解决跨模型混用与多租户污染问题，比单纯 content-hash 更严密。
3. **从 RoPE 性质推导复用粒度**：不是经验调 block size，而是从 position encoding 的数学性质证明“行级别不可共享、必须整 block”；该方法论可用于其他 position-aware 结构的缓存设计。
4. **One-time cost 回收的量化框架**：存 corpus 的 27.8 kJ 在第 13 题回收能量、第 16 题回收时间；为“何时启用记忆层”提供 break-even 计算方法。
5. **Long windows at flat GPU memory**：5.97M tokens 仅占 24.4-24.7 GB GPU mem；证明“存储换上下文”在 GPU 显存受限场景可行，可启发超长上下文 serving 的架构选择。

## 关键术语表
- **Taliesin**：Galahad 的 KV 记忆组件，按 block 持久化并精确恢复 attention KV state，load 后 logits bit-identical。
- **Blaise**：Galahad 的文本记忆组件，按原文 section 组织文档，CPU 端选取与问题相关的一节送入 model。
- **Byte-exact memory**：记忆 load 产生与 fresh prefill 完全一致的 logits（bit-identical），与 approximate/compressed reuse 相对。
- **Fail-closed**：任何记忆 load 检查（hash/signature/tenant/byte）失败时回退到正常 prefill，保证正确性不妥协。
- **Compute ceiling**：Sikka & Sikka [18] 提出的 transformer 每 token $O(N^2 \cdot d)$ 计算上限；本文不突破它，而是把重复工作移出门槛。
- **Stateful inference**：推理具有 persistent memory，请求成本随 new text 增长而非 total text。
- **Block-level reuse**：因 RoPE 使每行 KV 依赖 position，只能以整 block（position 精确匹配）为单位复用。
- **RAGFlow**：InfiniFlow 开源 RAG 引擎 v0.27.2，本文用作 external retrieval baseline（77/100）。

## 可复现要素
- **数据集**
  - Recall test：13 篇 Wikipedia articles（434 KB / 96,726 tokens / 11 blocks）+ 100 条 invented facts；深度 1%-99%。
  - 7 个 real-world datasets：help-desk tickets、customer-support logs、messy PDFs、SWEbench Lite、The Stack、AgentBench、WebArena（共 349 questions）。
  - 公开性：recall corpus 来自 Wikipedia（公开）；7 个 dataset 标注为 public/real-world，原文未给统一下载链接。
- **代码/权重**
  - Galahad：https://github.com/corbenicai/galahad（public beta，非商业单 GPU 12 个月，commercial pilots  onRequest）。
  - Raw logs：论文声明将随 accompanying dataset 发布。
  - RAGFlow v0.27.2：https://github.com/infiniflow/ragflow。
- **关键超参**
  - Decoding：greedy（temperature 0）。
  - Block 粒度：按原文 11 blocks（recall test）；具体 token 数论文未明说，由文档结构决定。
  - Disk budget：可配置，有 minimum free-space floor（未给默认值）。
  - 随机种子：20260927。
- **硬件/模型**
  - GPU：RTX A6000、H200、H100 SXM、A40、RTX 4090、RTX PRO 4500、L40S、L4、4×A10G。
  - 模型：Gemma 4 31B (4-bit)、Gemma 4 12B Q8_0、Llama 3.1 70B、Qwen 3.6 35B-A3B、Ministrals 3 8B、Qwen3.5 9B (Gated DeltaNet, TP=2)、DeepSeek-V4-Flash 284B 等共 30 个。
- **运行时版本**
  - vLLM 0.29 / 0.30.0、SGLang 0.5.20、llama.cpp b9189。
- **未提及项**
  - Taliesin 存储格式细节、Blaise 内部索引结构、加密算法具体参数——论文声明由 pending patent 覆盖。

<!--META
{"keywords": ["KV cache persistence", "byte-exact reuse", "stateful inference", "LLM serving", "Taliesin", "Blaise", "Galahad"], "field": "LLM inference optimization", "innovations": ["Byte-exact KV persistence with bit-identical logits acceptance test", "Block-level reuse derived from RoPE position-dependence", "Fail-closed memory layer integrated into vLLM/SGLang/llama.cpp without runtime
