---
title: "When-Harness-Beats-Scale-and-When-Reading-Beats-Both"
source: https://arxiv.org/pdf/2609.34366v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 04:26:24"
---

# 论文速读：When-Harness-Beats-Scale-and-When-Reading-Beats-Both

## 一句话总结
本文针对 DocInsights 2026 的 DocSem 文档定量推理任务，验证了“应用架构（harness）优于参数规模”的设计原则，提出结合混合检索、沙箱化 Program-of-Thoughts 与自一致性的轻量 pipeline；同时在测试集崩溃后揭示了 PDF 物理性质审计的必要性，指出合成文档任务中语言纪律比世界知识更易通过架构补偿。

## 研究问题与动机
- DocSem 要求模型从 PDF 中精确定位定量段落并计算数值答案，同时输出完全匹配的块标识符，现有端到端生成方法常在无关数字干扰下丢失证据或产生算术污染。
- 基准任务使用合成实体与机构，世界知识贡献为零，但现有研究未明确区分“世界知识”与“语言知识”的模型缩放差异，导致盲目堆叠参数。
- 共享任务训练/验证集为原生数字 PDF，测试集实际转为低分辨率光栅扫描件+对角水印，pipeline 设计阶段跳过像素级输入审计
