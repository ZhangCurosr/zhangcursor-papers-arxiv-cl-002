---
title: "WebPageBench-Event-Level-Verification-and-Controlled-UI-Vari"
source: https://arxiv.org/pdf/2609.35026v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 04:25:19"
---

# 论文速读：WebPageBench-Event-Level-Verification-and-Controlled-UI-Vari

## 一句话总结
本文提出 WebPageBench，一个基于接口内生事件日志进行事件级验证的 Web Agent 开源评测框架，首次在同一任务规范下通过切换控件实现生成受控 UI 变体，实现了 Agent 界面形式敏感度的可测量评估，并揭示了当前主流 Agent 完成声明与实际事件匹配之间最高达 41 个百分点的严重鸿沟。

## 研究问题与动机
- 现有 Web Agent 基准在“可验证性”与“规模”之间权衡失当：离线轨迹匹配仅评分路径而非结果，实网基准依赖 judge model 且受内容漂移与反爬机制干扰，自托管环境只能检查episode结束时的最终状态。
- 既有基准无法隔离 Agent 对单一 UI 控件实现的敏感性，聚合分数会将“任务理解失败”与“控件操作失败”混淆（例如弹窗日历与原生 `<input type="date">` 在相同提示词下的表现差异被掩盖）。
- Agent 的完成信号（completion signal）与真实任务达成情况严重脱节，缺乏无需语言模型裁判、直接读取接口行为证据的细粒度验证机制。
- 现有评测管道对浏览器/DOM 类 harness 与纯截图 GUI 类 harness 缺乏统一评估标准，难以公平对比不同动作空间、成本与速度的 Trade-off。

## 核心贡献（创新点）
- **事件级验证契约（Event-Level Verification Contract）**：定义 19 种带类型参数的语义事件及匹配谓词，成功由后端事件日志直接判定（EMS 指标），无需 judge model 且完全语言无关；与现有工作仅比对终态或依赖 LLM 判官的本质区别在于验证依据是“过程性、可溯源的接口行为证据”。
- **受控 UI 变体生成机制**：通过 9 个 UI 配置键（8 个控件 + 配色主题）挂载 33 种实现，将 25 个人工编写任务扩展为 87 个变体，仅替换控件渲染而提示词与成功条件严格保持一致；区别于 WorkArena++ 等仅更换品牌/颜色的做法，真正实现控制变量法下的界面鲁棒性测量。
- **统一 Runner 与双动作空间评测协议**：一套 pytest 评估管道兼容 6 种浏览器/DOM harness 与 5 种纯截图 GUI agent，独立记录 Agent 自报完成信号与确定性验证结果，首次公开量化“完成-验证鸿沟”并支持按领域、交互类别、harness 多维度检索。

## 方法详解
- **Mock Sites 与匿名化构建**：构建 6 个覆盖日常网络行为的模拟站点（marketplace, books & audiobooks, grocery delivery, rail travel, hotel search, document cabinet），使用中性路由与状态命名（如 `bench_catalog_main`）切断品牌线索；书店商品（226 件、75 个作者、229 个书名等）通过确定性脚本（seed 42）替换为合成俄语数据，其余站点图片替换为本地开源许可文件（共 590 条有效路径）。
- **Track 与事件系统**：每次任务运行生成独立 track（独立 URL、应用状态与事件日志，并行不干扰）；前端在用户/Agent 操作时向事件总线写入带类型的结构化事件（如 `basket_add` 携带 item identifier 与 quantity，`select_tariff` 携带
