# TQTS-BENCH: A MULTI-SYNTAX BENCHMARK FOR TEXT-TO-QUERY OVER TIME-SERIES DATABASES

Fei Lyu<sup>∗</sup>, Zhiyi Peng<sup>∗</sup>, Jiaming Liu, Yixuan Yang, Changjian Chen<sup>†</sup>, Zhuo Tang<sup>†</sup>, Jiapeng Zhang, Kenli Li

College of Computer Science and Electronic Engineering, Hunan University

## ABSTRACT

Large language models (LLMs) have significantly advanced natural language querying over relational databases, yet their ability to query time-series databases (TSDBs) remains largely unassessed. Existing benchmarks fail to adequately capture the non-unified query syntaxes, diverse application domains, and unique time-specific query intents inherent to TSDBs. To address this gap, we introduce TQTS-BENCH, a multi-syntax benchmark for evaluating text-to-query capabilities over TSDBs. TQTS-BENCH contains 6,125 high-quality question– answering (QA) pairs spanning 97 TSDBs, 23 distinct query syntaxes, 22 application domains, and 4 types of time-specific query intents. It is constructed through a human-centric AI-assisted workflow, where all QA pairs are carefully reviewed and revised by domain experts to ensure quality and correctness. Extensive evaluations of advanced LLMs and state-of-the-art text-to-query methods reveal challenges in querying TSDBs. Even the best-performing model evaluated, Claude-Opus-5, achieves only 48.98% execution accuracy, while humans reach 87.34%. Error analysis reveals that this performance gap mainly stems from the heterogeneous query syntaxes across different TSDBs, misinterpretation of timespecific intents, and incorrect schema linking. These findings highlight new opportunities to narrow the gap between current LLM capabilities and the requirements of TSDB queries in real-world applications. The benchmark is available at: https://anonymous.4open.science/r/TQTS-Bench-00CD.

## 1 INTRODUCTION

Time-series data are widely used in real-world applications, such as the Internet of Things (IoT), AIOps, and cloud services. This widespread adoption has driven the rapid development of specialized time-series databases (TSDBs) (Jensen et al., 2017). However, analyzing data stored in TSDBs remains challenging because retrieving such data requires learning specialized query syntax (Dranca et al., 2026), particularly for users without expertise in databases. Recently, due to the rapid advancement of large language models (LLMs), text-to-query techniques have progressed significantly, approaching human-level performance on relational database (RDB) benchmarks (e.g., Spider (Yu et al., 2018) and BIRD (Li et al., 2023)). This progress has substantially lowered the barrier to analyzing data stored in databases. More recently, a few studies, such as Zhang et al. (2026) and Dranca et al. (2026), have begun to develop Text-to-Query techniques specifically for TSDBs (TQTS). However, it remains unclear how far these methods have advanced TQTS capabilities due to the lack of a comprehensive TQTS benchmark comparable to Spider and BIRD for RDBs.

Despite its importance, constructing a TQTS benchmark is not a straightforward extension of text to-query benchmarks for RDBs, owing to three key differences (Fig. 1).

• Syntaxes: For RDBs, their query languages share the common SQL syntax with differences mostly limited to small dialects. However, for TSDBs, their query syntaxes can be substantially different (Fig. 1(a)). For example, Flux in InfluxDB adopts a pipeline-style syntax, whereas PromQL in Prometheus relies on nested, function-oriented expressions.

![](images/f101ca27f88bd3a07f02d45d97203647344f92741dcdbf056a94c98f1be11408.jpg)  
Figure 1: The three key differences between RDBs and TSDBs: (a) Syntaxes: unified SQL vs. diverse, non-uniform query syntaxes. (b) Domains: narrow domain coverage vs. broad domain coverage. (c) Intents: time-agnostic operations vs. time-specific analysis.

Moreover, even TSDBs that stay close to SQL, such as QuestDB, extend it with a wide range of time-specific operations and functions, further diverging from basic SQL syntax.

• Domains: In domains that require the continuous collection of high-frequency data, such as IoT and cloud services, TSDBs are predominant. Because RDBs are not specifically designed for such workloads, existing text-to-query benchmarks for RDBs rarely cover these domains, resulting in limited domain coverage (Fig. 1(b)).

• Intents: Queries over RDBs typically focus on time-independent intents such as filtering, sorting, and aggregation. In contrast, TSDB queries involve many time-specific intents, such as change analysis and relationship analysis (Fig. 1(c)). This makes text-to-query for RDBs and TSDBs differ in both question and intent understanding.

Due to the three differences above, rather than simply extending existing text-to-query benchmarks for RDBs, we construct TQTS-BENCH, a new benchmark specifically designed for text-to-query over TSDBs. TQTS-BENCH contains 6,125 text-to-query question-answering (QA) pairs, spanning 97 TSDBs with 23 distinct query syntaxes. These TSDBs span 22 application domains, while the QA pairs cover 4 time-specific query intents. To construct this benchmark, we first identified 23 popular and representative TSDB management systems based on the well-known DB-Engines Ranking of TSDB Management System<sup>1</sup> and collected 427 datasets and 1,405 seed QAs related to these TSDB management systems from public sources such as TSDB official documentation, forums, and technical posts. These 427 datasets were manually filtered based on data authenticity and schema complexity, resulting in 97 datasets. In collaboration with domain experts, we analyzed the 1,405 seed QAs and identified 9 representative query intents. Based on these query intents and the seed QAs, we developed a human-centric AI-assisted workflow supported by a visual analytics too to construct new QA pairs. Finally, these QA pairs underwent a rigorous cross-validation process to ensure their correctness and quality, and 6,125 QA pairs were verified.

We conduct a comprehensive evaluation of advanced LLMs and representative text-to-query methods on TQTS-BENCH. The results show that current LLMs still struggle with TQTS tasks. For example, the best-performing model, Claude-Opus-5, achieves an execution accuracy of only 48.98%, which remains far from human performance (87.34%), highlighting the challenges posed by this benchmark. Meanwhile, both existing RDB methods and TSDB methods also achieve limited performance, suggesting that existing methods still face difficulties in TQTS tasks and exhibit limited transferability and generalizability across different settings. We further conduct an error analysis and identify three main factors behind these failures, including heterogeneous query syntaxes across different TSDBs, misinterpretation of time-specific intents, and incorrect schema linking. Based on these findings, we discuss opportunities and promising directions for improving TQTS performance to enable more effective time-series data analysis in real-world applications.

## 2 RELATED WORK

Text-to-query benchmarks over RDBs. RDB benchmarks have established well-defined evaluation paradigms for the Text-to-SQL task. Spider (Yu et al., 2018) evaluates whether models can generalize to unseen databases in different domains. BIRD (Li et al., 2023) extends this setting to larger and real-world databases, requiring models to use database contents and external knowledge. Spider 2.0 (Lei et al., 2025) further targets complex enterprise environments with large schemas and realistic query workflows. Beyond these general-purpose benchmarks, several studies focus on specific applications. For example, ScienceBenchmark (Zhang et al., 2023) targets scientific databases, while SParC (Yu et al., 2019b) and CoSQL (Yu et al., 2019a) focus on contextual and conversational queries. Additionally, many benchmarks (Vo et al., 2022; Chang et al., 2023; Liu et al., 2026) evaluate model capabilities on reasoning and query robustness. Despite covering broad evaluation settings, these benchmarks are grounded in RDBs and ignore time-series query requirements. However, TSDBs inherently contain multiple query syntaxes and are designed for time-series analysis. Therefore, existing RDB benchmarks are insufficient for evaluating text-to-query over TSDBs.

Text-to-query benchmarks over TSDBs. Recent studies have developed several text-to-query benchmarks for specific TSDBs. For example, TEFD (Wu et al., 2026) evaluates text-to-Flux generation for InfluxDB, while another study (Dranca et al., 2026) evaluates the generation of both Flux and InfluxQL queries in an agro-food production scenario. PromCopilot (Zhang et al., 2026) focuses on generating PromQL queries for Prometheus-based system monitoring. However, these studies are often limited to specific TSDBs, query syntaxes, or application domains, preventing a comprehensive evaluation of LLMs’ text-to-query capabilities across diverse TSDB environments and scenarios. To address this gap, we develop TQTS-BENCH, a multi-syntax benchmark that spans a broad range of TSDBs, query syntaxes, and application domains, enabling a systematic and comprehensive evaluation of LLMs for text-to-query tasks over TSDBs.

## 3 DATASET CONSTRUCTION

## 3.1 TASK DEFINITION

Given a natural-language question, a database schema, and a target TSDB, the TQTS task aims to generate an executable query that correctly retrieves the requested data from the TSDB. Here, the database schema describes the logical organization of time-series data, including how timestamps, data values, and associated metadata are structured and typed (Bader et al., 2017). Depending on the TSDB management system, these elements may be represented as tables or measurements, columns or fields, and tags or labels. The target TSDB refers to a dataset stored in a specified TSDB man agement system (e.g., InfluxDB).

## 3.2 CONSTRUCTION PIPELINE

As shown in Fig. 2, the construction of TQTS-BENCH consists of two main steps: TSDB construction and QA construction and verification. The TSDB construction step collects and curates 97 high-quality TSDBs, and the QA construction and verification step constructs 6,125 QA pairs with a human-centric AI-assisted workflow.

## 3.2.1 TSDB CONSTRUCTION

Based on the well-known DB-Engines Ranking of TSDB Management System, we first select 23 popular and representative TSDB management systems depending on the accessibility. With these systems, we then collect real-world TSDB datasets from diverse application scenarios. Although existing studies (Li et al., 2023; Yu et al., 2018; Zhang et al., 2026; Wu et al., 2026) provide datasets for text-to-query tasks, these datasets are typically distributed as standalone files (e.g., CSV files) rather than as operational TSDBs. Therefore, we gather publicly available datasets from multiple real-world sources, including official websites, hosting platforms that export data from specific TS DBs, and datasets documented as being collected or stored using TSDBs. In parallel, we compiled a set of seed QAs to facilitate subsequent identification of the query intent. These QA pairs were sourced from official TSDB documentation, community forum discussions, and technical posts.

![](images/4622bbca843e24b98caca36f2a654e996b569352fd1761357e4aee9452e6312c.jpg)  
Figure 2: The construction pipeline of TQTS-BENCH.

The collection process results in 427 candidate TSDB datasets and 1,405 seed QAs. We then manually inspect these candidate datasets based on data authenticity and schema complexity, and finally select 97 representative source datasets. Subsequently, we import these datasets into their corresponding TSDB management systems to instantiate the database environments used in our benchmark. For TSDB details, refer to App. A.

## 3.2.2 QA CONSTRUCTION AND VERIFICATION

In TQTS-BENCH, each QA pair consists of a natural-language question and its corresponding answer for a specific TSDB. To construct the QA pairs, we first identify representative query intents from the seed QAs. We also developed a visual analytics tool that supports a human-centric AIassisted workflow to help annotators directly explore and interact with the underlying TSDB.

Query intent identification. With the collected seed QAs, we collaborate with domain experts to analyze the underlying query requirements in these pairs and common TSDB usage scenarios. Based on this analysis, we identify nine representative query intents, comprising four time-specific query intents (I1–I4) and five time-agnostic query intents (I5–I9). Specifically, the time-specific intents are: I1: Window aggregation & resampling; I2: Temporal change analysis; I3: Time localization; and I4: Relationship analysis. The time-agnostic intents are: I5: Conditional filtering; I6: Statistical computation; I7: Sort & rank; I8: Metadata query; and I9: Domain knowledge dependency. We find that most query requirements can be represented by a single intent or a combination of multiple intents, indicating that the identified intents cover a broad range of TSDB usage needs. These nine query intents provide a common foundation for QA construction across different TSDB systems, regardless of their underlying query syntax. For details, refer to App. A.3.

Human-AI collaborative construction. In TQTS-BENCH, a qualified QA pair should consist of a natural-language question that faithfully reflects query intents in the target TSDB scenario, together with an executable query that correctly answers it under the given schema and TSDB context. To facilitate QA construction, we develop a visual analytics tool that supports a human-centric AIassisted workflow. Specifically, we provide LLMs (e.g., GPT-5.6-Sol) with the identified query intents, database schemas, and TSDB context, requiring them to generate the corresponding questions. Annotators then review these questions to verify they match the query intents and reflect meaningful TSDB scenarios, refining their descriptions when necessary to improve clarity and alignment with the query intents. If the questions are determined, the annotators then use LLMs to generate the corresponding executable queries and construct the generated unverified QA pairs.

QA verification. After construction, all generated unverified QA pairs undergo a final verification step to ensure their correctness and quality. To facilitate this process, we use the developed visual analytics tool. For each unverified QA pair, we conduct a cross-validation procedure. Specifically, two annotators independently hand-write queries based on the question, schema, and provided TSDB context. They can execute the hand-written queries to determine whether the query results match the question’s requirements. If both annotators obtain results consistent with the unverified answer and both determine that the question is answerable on the target TSDB, the QA pair is accepted as verified. Otherwise, the case is escalated to an adjudicator. The adjudicator examines the annotators’ queries, results, and feedback, and revises the QA pairs accordingly. The revised QA pairs then undergo a further round of verification, or are finalized by the adjudicator if the issue is resolved. For the visual analytics tool and verification details, refer to App. B.

Overall, a total of 6,125 QA pairs are verified. During this process, the annotators collectively spent more than 600 hours and executed 30,774 queries, averaging approximately 6 minutes inspection and 5 executions per QA pair. This rigorous verification process facilitates careful inspection and iterative validation, thereby ensuring the quality and correctness of the constructed QA pairs.

## 3.3 DATA STATISTICS

TQTS-BENCH is a multi-syntax benchmark designed for TQTS tasks. Specifically, we summarize the data statistics of TQTS-BENCH from three perspectives: supported query syntaxes, domain coverage, and time-specific query intents.

Multiple query syntaxes. In total, TQTS-BENCH contains 6,125 QA pairs spanning 23 distinct query syntaxes across 97 TSDBs, which differ substantially in their structures and usage patterns. As shown in Fig. 3, for the same question, InfluxDB’s Flux represents the query as a pipeline of transformations, Prometheus’s PromQL uses metric selectors and range-vector functions, and QuestDB SQL extends conventional SQL with dedicated temporal operations such as time sampling. This diversity makes TQTS-BENCH challenging, as models must generate queries that are both semantically correct and syntactically compatible with the target TSDB. For more details, refer to App. A.1.

## Q What was sensor A's average temperature each hour over the last 24 hours?

![](images/c4bacd05432571bd67bdd38b0304e7c456f74ce1c50167828586e00f8ebca97f.jpg)  
Figure 3: A representative question expressed using Flux, PromQL, and QuestDB SQL.

Broad domain coverage. Fig. 4(a) compares the domain coverage of TQTS-BENCH with existing text-to-query benchmarks over RDBs and TSDBs. TQTS-BENCH covers all 22 application domains considered in the comparison, while previous benchmarks typically cover only a subset of them. In particular, many RDB benchmarks emphasize general domains such as finance, health care and education. In contrast, TSDB benchmarks are often concentrated on scenarios involving continuously generated, high-frequency time-series data, such as IoT, AIOps, and cloud services, where efficient ingestion and temporal analysis are critical. By spanning both general-purpose and time-intensive domains, TQTS-BENCH supports a more comprehensive evaluation of text-to-query methods across real-world data settings. For more domain details, refer to App. A.2

More time-specific query intents. Fig. 4(b) reports the query intent statistics across different textto-query benchmarks. Compared with other benchmarks, TQTS-BENCH has the highest average numbers of both query intents and time-specific intents per QA pair, as well as the highest average number of time-specific operations per QA pair. Together, these statistics highlight the temporal complexity of TQTS-BENCH, making it a challenging benchmark for translating diverse timespecific intents into executable queries. For more query intent details, refer to App. A.3.

![](images/a114f421283509a5ddfa21cb0fb712338653c0ea56ae0f71cc4b986ef9f66457.jpg)  
TQTS-Bench BIRD Mini-Dev Spider 2.0-Lite PromCopilot TEFD

![](images/82d36a1a98314516729999da09526b802fc56750f8534ed278b8df086158333a.jpg)  
(b)  
Figure 4: (a) Domain coverage and (b) time-specific query intent statistics of TQTS-BENCH.

## 4 EXPERIMENT

## 4.1 EXPERIMENTAL SETUP

Evaluation metric. We evaluate model performance using the widely adopted metric Execution Accuracy (EX) (Li et al., 2023; Lei et al., 2025), which measures whether the execution result matches the expected answer and thereby reflects the semantic correctness of the generated query.

Difficulty level. Given the diversity of query syntaxes, token count of the gold query for a question is not a suitable measure of difficulty. Therefore, we classify questions by the number of query intents: 1–3 as easy (31.72%), 4–5 as medium (44.33%), and 6–9 as hard (23.95%). This categorization reflects the increasing compositional complexity required to answer the question.

Human evaluation. Given the substantial time and cost involved, we conduct human evaluation on a randomly sampled 10% subset of TQTS-BENCH to establish a human performance baseline for real-world TSDB application scenarios. For details, refer to App. C.2.

LLMs. We evaluated a broad range of advanced LLMs, including both open-source and closedsource models. These models comprise representative and state-of-the-art offerings from major AI developers, enabling a comprehensive comparison between open-source and proprietary frontier models. The open-source models include Qwen3.8-Flash (Qiu et al., 2026), DeepSeek-V4- Pro (DeepSeek-AI et al., 2026), GLM-5.3-Flash (GLM-5-Team et al., 2026), and Kimi-K3 (Kimi Team et al., 2026), while the closed-source models include GPT-6-Sol (OpenAI, 2026), Claude-Opus-5 (Anthropic, 2026), and Gemini-3.7-Flash (Google DeepMind, 2026). Following the experiment setting of Lei et al. (2025), if the input length exceeds the model’s maximum token limit, it will be truncated from the beginning.

Text-to-query methods. We also evaluate current state-of-the-art and representative text-to-query methods for both RDBs and TSDBs to examine their ability to address TSDB-specific chal lenges and transferability across different TSDB management systems. The RDB methods include DeepEye-SQL (Li et al., 2026), OpenSearch-SQL (Xie et al., 2025), RSL-SQL (Cao et al., 2024), DAIL-SQL (Gao et al., 2024), and DIN-SQL (Pourreza & Rafiei, 2023), while the TSDB methods include PromCopilot (Zhang et al., 2026). Although RDB methods are not specifically designed for TSDBs, they provide strong baselines because they address fundamental text-to-query challenges such as schema linking and query generation, which are also critical in TSDB applications. To ensure a controlled comparison and eliminate the impact of different backbone models, we use GPT-4o-mini as the unified LLM backbone for all methods.

## 4.2 QUANTITATIVE EVALUATION RESULTS

Table 1: Evaluation results on TQTS-BENCH. The best result (except human) is in bold, and the runner-up is underlined. Results with (\*) are tested on a randomly sampled 10% subset.
<table><tr><td rowspan="2">Method</td><td colspan="4">EX(↑)</td></tr><tr><td>Easy</td><td>Medium</td><td>Hard</td><td>Overall</td></tr><tr><td>Human performance</td><td>*95.71%</td><td>*85.52%</td><td>*81.94%</td><td>*87.34%</td></tr><tr><td colspan="5">Open-Source Models</td></tr><tr><td>Qwen3.8-Flash</td><td>43.75%</td><td>20.07%</td><td>17.79%</td><td>27.04%</td></tr><tr><td>DeepSeek-V4-Pro</td><td>39.89%</td><td>15.87%</td><td>11.79%</td><td>22.51%</td></tr><tr><td>GLM-5.3-Flash</td><td>54.30%</td><td>28.88%</td><td>19.15%</td><td>34.61%</td></tr><tr><td>Kimi-K3</td><td>58.26%</td><td>37.24%</td><td>34.08%</td><td>43.15%</td></tr><tr><td colspan="3">Closed-Source Models</td><td></td><td></td></tr><tr><td>GPT-6-Sol</td><td>62.74%</td><td>40.63%</td><td>35.38%</td><td>46.38%</td></tr><tr><td>Claude-Opus-5</td><td>62.79%</td><td>42.91%</td><td>41.92%</td><td>48.98%</td></tr><tr><td>Gemini-3.7-Flash</td><td>63.25%</td><td>37.68%</td><td>32.92%</td><td>44.65%</td></tr><tr><td colspan="5">RDB Methods</td></tr><tr><td>DeepEye-SQL</td><td>14.82%</td><td>1.92%</td><td>1.16%</td><td>5.83%</td></tr><tr><td>OpenSearch-SQL</td><td>16.37%</td><td>1.84%</td><td>1.09%</td><td>6.27%</td></tr><tr><td>RSL-SQL</td><td>11.99%</td><td>1.33%</td><td>0.95%</td><td>4.62%</td></tr><tr><td>DAIL-SQL</td><td>1.34%</td><td>0.26%</td><td>0.00%</td><td>0.54%</td></tr><tr><td>DIN-SQL</td><td>12.25%</td><td>2.17%</td><td>1.23%</td><td>5.14%</td></tr><tr><td colspan="5">TSDB Methods</td></tr><tr><td>PromCopilot</td><td>1.80%</td><td>0.22%</td><td>0.07%</td><td>0.69%</td></tr></table>

Existing RDB methods struggle with TQTS tasks and exhibit limited transferability to TS-DBs. Tab. 1 reports the performance of state-of-the-art RDB methods on TQTS tasks. Specifically, DeepEye-SQL, a high-performing RDB method among those compared on the BIRD benchmark<sup>2</sup>, achieves only 5.83% EX on TQTS-BENCH, a notably low score. Moreover, other RDB methods also perform poorly, with most achieving below 10.00% EX and some as low as 0.54%. This performance gap between RDB benchmark and TQTS-BENCH suggests that methods optimized for RDB text-to-query tasks do not necessarily generalize to TSDB scenarios. More detailed analysis of these results is provided in App. C.3.

Although LLMs outperform existing RDB methods, their performance remains unsatisfactory. Table 1 shows that advanced LLMs still achieve limited EX on TQTS tasks. The best-performing model, Claude-Opus-5 achieves the highest EX and consistently outperforms other models at the medium and hard difficulty levels. However, its overall performance remains relatively low, with an EX of only 48.98%, which is far below the human performance of 87.34%. This gap indicates that TQTS tasks remain challenging even for the most advanced models. The gap is more pronounced for some open-source models; for example, DeepSeek-V4-Pro achieves an EX of only 22.51%, leaving substantial room for improvement in practical applications. Compared with explicitly optimized RDB methods for text-to-SQL tasks, LLMs demonstrate stronger generalization ability, but their performance remains far from human-level capability.

## Existing TSDB methods exhibit limited generalizability for

TQTS tasks. Tab. 1 shows that PromCopilot, a state-of-the-art TSDB method for TQTS tasks, still achieves limited performance, with an EX of only 0.69%. PromCopilot is specifically designed for PromQL, the query syntax of Prometheus, and is primarily targeted in cloud services and AIOps application domains. As reported in Tab. 2, its performance decreases from 3.03% to 0.00% when the domain shifts to others. The result indicates that existing TSDB methods are sensitive to changes in application domains, revealing the limited generalizability of existing TSDB methods across different TQTS task settings. For detailed analysis, refer to App. C.4.

Table 2: Performance of Prom-Copilot on different domains.
<table><tr><td>Domain</td><td>EX(↑)</td></tr><tr><td>Targeted</td><td>3.03%</td></tr><tr><td>Others</td><td>0.00%</td></tr></table>

## 4.3 ERROR ANALYSIS

We perform a systematic error analysis by randomly sampling 600 examples and manually examining the incorrect cases. Given its weak performance, we select DeepSeek-V4-Pro for error analysis, as it may expose more representative failure modes in practical settings. Fig. 5 shows three identified representative error types: query syntax errors, intent understanding errors, and schema linking errors.

Query syntax errors (75.29%). Due to the diversity and complexity of TSDB query syntax, the model needs to understand various syntax structural patterns and function invocation rules to generate correct queries. We categorize query syntax errors into two types based on the type of syntax violation: (1) Query structure errors (42.50%). The model fails to construct valid query structures such as improper ordering of operators or functions, resulting in incorrect query results (e.g., Fig. 6(a)). (2) Function/keyword usage errors (32.79%). The model misuses functions or keywords, including incorrectly applying valid functions (i.e., misunderstanding parameters, or usage constraints) and invoking non-existent functions, thereby causing errors (e.g., Fig. 6(b)). For more error cases, refer to App. C.5.

![](images/04488de8f3ce298070a6f3734d232bf78c6ea4705bff3d895192187bd3217e68.jpg)  
Figure 5: Proportions of incorrect cases by error type.

![](images/2208177a6e1f8c151e8c24a64a54359a52a35b945cb72e41ed0297c3c54459f9.jpg)  
Figure 6: Examples of query syntax errors. (a) Query structure error: reversing group and aggregateWindow selects per-series rather than per-group last readings, producing incorrect results. (b) Function usage error: replacing bar(Time, 15m) with floor(Time, 15m) violates the single-argument requirement of floor, causing a compilation error.

Intent understanding errors (61.43%). For the TQTS task, accurately identifying the requirement of a question is particularly challenging due to the complexity of time-specific query intents. TQTS BENCH includes 9 query intents, 4 of which are time-specific and require models to correctly interpret and execute temporal operations. As shown in Fig. 7(a), errors involving time-specific query intents account for 65.41% of all intent understanding errors, substantially exceeding those associated with time-agnostic query intents (34.59%). Specifically, the error rates for the 4 time-specific intents are 26.31%, 25.19%, 7.52%, and 6.39%, respectively. These results suggest that current models struggle particularly with interpreting time-specific query intents. Fig. 7(b) shows one such failure, caused by missing window aggregation and resampling.

Schema linking errors (17.09%). Schema linking is a critical step in text-to-query tasks (Li et al., 2026). However, existing methods are mainly designed for RDBs and cannot be directly applied to TSDBs due to their different schema organizations. TSDBs typically organize data using metrics, labels, and tags rather than tables and columns, making traditional schema linking error types unsuitable for TSDBs. Moreover, TSDBs are widely used in IoT and AIOps domains, where highfrequency data collection leads to large-scale schemas. We identify two major challenges for schema linking in TQTS tasks: (1) Diverse schema structures. The heterogeneous use of metrics, labels, and tags introduces new schema ambiguities. For schema details, refer to App. A.4. (2) Large-scale schema. Cases with schema linking errors have much larger schemas (381k tokens on average) than other erroneous cases (115k tokens on average). Large schemas hinder schema comprehension and relevant element retrieval, and long contexts can further degrade LLM performance due to the “lost-in-the-middle” phenomenon (Liu et al., 2024). Therefore, improving schema linking performance in TQTS tasks remains an important direction for future work.

![](images/05a2a9f53574969fcab1fa6de7c38d36e70f76c3d0f827f1c9cf4e1ac131df36.jpg)

![](images/afff915211a285cdebcfa997d083e14921502cf178182660917d3f292057c905.jpg)  
(b)  
Figure 7: Intent understanding errors. (a) Distribution of primary erroneous intents; time-specific intents account for 65.41%. (b) Window aggregation & resampling error: the predicted query lacks [2h:15m], returning a single top-3 result instead of results at each 15-minute checkpoint.

## 4.4 ABLATION STUDY

To figure out why existing models perform poorly in TQTS-BENCH, we conduct an ablation study. Although domains and query intents may also affect TSDB query generation, we focus on syntax because it is an intuitive and explicit difference between TSDBs and RDBs. Specifically, we convert the original TSDBs into RDBs (e.g., SQLite) while keeping the questions of error cases unchanged, and require the model to generate standard SQL queries. We compare the performance with two representative RDB benchmarks, BIRD and Spider 2.0- lite. As shown in Tab. 3, converting TSDBs into RDBs improves EX from 0.00% to 32.21%, which is lower than BIRD (56.91%) but higher than Spider 2.0-lite (22.94%). It suggests that the model’s poor performance is due not to the difficulty of the questions but to the diverse query syntaxes. The model is capable of understanding the questions and generating effective SQL queries, but it lacks specific syntax knowledge, which leads to incorrect answers. For implementation details, refer to App. C.6.

Table 3: EX across different query syntaxes and benchmarks.
<table><tr><td>Setting</td><td>Syntax</td><td>EX(↑)</td></tr><tr><td>Original</td><td>Diverse</td><td>0.00%</td></tr><tr><td>w/ conversion</td><td>SQL</td><td>32.21%</td></tr><tr><td>BIRD</td><td>SQL</td><td>56.91%</td></tr><tr><td>Spider 2.0-lite</td><td>SQL</td><td>22.94%</td></tr></table>

## 5 CONCLUSION

In this paper, we introduce TQTS-BENCH, a comprehensive benchmark for evaluating text-to-query capabilities over TSDBs. Unlike existing RDB benchmarks, TQTS-BENCH is designed to capture TSDB-specific challenges by covering diverse query syntaxes, application domains, and timespecific query intents. Extensive evaluations of advanced LLMs and representative text-to-query methods show that existing methods still struggle with TQTS tasks, particularly in query syntax handling, intent understanding, and schema linking in TSDBs. Our analysis reveals key limitations of current methods and highlights opportunities for developing more effective and generalizable text-to-query methods for real-world time-series analysis.

## AI USAGE DISCLOSURE

This work used large language models (LLMs) in three limited capacities: (1) to assess the coverage of manually collected related work; (2) to generate candidate QA pairs during benchmark construction, which were subsequently reviewed and finalized by human experts; and (3) to assist with grammar and phrasing refinement. LLMs were not used to generate experimental results, figures, or references. All cited works and final manuscript were verified and approved by the authors, who take full responsibility for the accuracy and integrity of this paper.

## REFERENCES

Anthropic. Introducing claude opus 5. https://www.anthropic.com/news/ claude-opus-5, July 2026.

Andreas Bader, Oliver Kopp, and Michael Falkenthal. Survey and comparison of open source time series databases. In Datenbanksysteme fur Business, Technologie und Web (BTW 2017)-¨ Workshopband, pp. 249–268. Gesellschaft fur Informatik eV, 2017.¨

Zhenbiao Cao, Yuanlei Zheng, Zhihao Fan, Xiaojin Zhang, Wei Chen, and Xiang Bai. RSL-SQL: Robust schema linking in text-to-sql generation. arXiv preprint arXiv:2411.00073, 2024. doi: 10.48550/arXiv.2411.00073.

Shuaichen Chang, Jun Wang, Mingwen Dong, Lin Pan, Henghui Zhu, Alexander Hanbo Li, Wuwei Lan, Sheng Zhang, Jiarong Jiang, Joseph Lilien, Steve Ash, William Yang Wang, Zhiguo Wang, Vittorio Castelli, Patrick Ng, and Bing Xiang. Dr.Spider: A diagnostic evaluation benchmark towards text-to-SQL robustness. In Proceedings of the International Conference on Learning Representations, 2023.

DeepSeek-AI, Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, et al. DeepSeek-V4: Towards highly efficient million-token context intelligence. arXiv preprint arXiv:2606.19348, 2026.

Lacramioara Dranca, Pablo Donate, Julio A. Sanguesa, Piedad Garrido, Vicente Torres-Sanz, and Francisco J. Martinez. LLMs for industrial databases: An agro-food production plant use case. Frontiers in Artificial Intelligence, 9, 2026. doi: 10.3389/frai.2026.1764367.

Dawei Gao, Haibin Wang, Yaliang Li, Xiuyu Sun, Yichen Qian, Bolin Ding, and Jingren Zhou. Text-to-SQL empowered by large language models: A benchmark evaluation. Proceedings ofthe VLDB Endowment, 17(5):1132–1145, 2024. doi: 10.14778/3641204.3641221.

GLM-5-Team, Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, et al. GLM-5: from vibe coding to agentic engineering. arXiv preprint arXiv:2602.15763, 2026.

Google DeepMind. Gemini 3.7 Flash Model Card, August 2026. URL https://deepmind. google/models/model-cards/gemini-3-7-flash/.

Søren Kejser Jensen, Torben Bach Pedersen, and Christian Thomsen. Time series management systems: A survey. IEEE Transactions on Knowledge and Data Engineering, 29(11):2581–2600, 2017. doi: 10.1109/TKDE.2017.2740932.

Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, Jianfeng Cai, Xinyuan Cai, Peizhou Cao, Yuxuan Cao, Ziwei Chai, Y. Charles, et al. Kimi K3: Open frontier intelligence. arXiv preprint arXiv:2607.24653, 2026.

Fangyu Lei, Jixuan Chen, Yuxiao Ye, Ruisheng Cao, Dongchan Shin, Hongjin Su, Zhaoqing Suo, Hongcheng Gao, Wenjing Hu, Pengcheng Yin, Victor Zhong, Caiming Xiong, Ruoxi Sun, Qian Liu, Sida Wang, and Tao Yu. Spider 2.0: Evaluating language models on real-world enterprise text-to-SQL workflows. In Proceedings of the International Conference on Learning Representations, 2025.

Boyan Li, Chong Chen, Zhujun Xue, Yinan Mei, and Yuyu Luo. DeepEye-SQL: A softwareengineering-inspired text-to-sql framework. Proceedings of the ACM on Management of Data, 4 (3):1–28, 2026. doi: 10.1145/3802035.

Jinyang Li, Binyuan Hui, Ge Qu, Jiaxi Yang, Binhua Li, Bowen Li, Bailin Wang, Bowen Qin, Ruiying Geng, Nan Huo, Xuanhe Zhou, Chenhao Ma, Guoliang Li, Kevin Chang, Fei Huang, Reynold Cheng, and Yongbin Li. Can LLM already serve as a database interface? a BIg bench for largescale database grounded text-to-SQLs. In Proceedings of the Advances in Neural Information Processing Systems, volume 36, pp. 42330–42357, 2023. doi: 10.52202/075280-1835.

Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. Transactions of the Associationfor Computational Linguistics, 12:157–173, 2024.

Tao Liu, Xutao Mao, Dixuan Zhang, Yifan Li, Haixin Liu, Lulu Kong, Jiaming Hou, Rui Li, Yunlong Li, Aoze Zheng, Zhiqiang Zhang, Zhewei Luo, Hongying Zan, Kunli Zhang, and Min Peng. LogicCat: A chain-of-thought text-to-SQL benchmark for complex reasoning. Proceedings of the AAAI Conference on Artificial Intelligence, 40(36):29958–29966, 2026. doi: 10.1609/aaai. v40i36.40243.

OpenAI. Introducing gpt-6 sol and luna, September 2026. URL https://openai.com/ index/introducing-gpt-6-sol-and-luna/.

Mohammadreza Pourreza and Davood Rafiei. DIN-SQL: Decomposed in-context learning of textto-sql with self-correction. In Proceedings of the Advances in Neural Information Processing Systems, volume 36, pp. 36339–36348, 2023. doi: 10.52202/075280-1577.

Zihan Qiu, Zekun Wang, Xiao Li, Yanpeng Li, Yang Xu, Yixuan Wang, Huaqing Zhang, Rui Men, Bochao Mao, Chengruidong Zhang, Fan Zhou, Hao Luo, Haofeng Huang, Haoran Lian, Haoyan Huang, Hongqing Chen, Jianwei Zhang, Jing Xu, Junjie Wang, Langshi Chen, Liangyu Wang, Linlang Jiang, Man Yuan, Minmin Sun, Peng Jin, Siqi Zhang, Siyu Wang, Xingzhang Ren, Yakai Wang, Yi Zhang, Yiming Dong, Yizhong Cao, Yubo Ma, Yunfei Mao, Bo Zheng, and Dayiheng Liu. On the design of Qwen3.8-Next architecture: Evaluation, efficiency, and training stability, 2026. URL https://arxiv.org/abs/2608.30320.

Ngoc Phuoc An Vo, Octavian Popescu, Irene Manotas, and Vadim Sheinin. Tackling temporal questions in natural language interface to databases. In Proceedings of the Conference on Empirical Methods in Natural Language Processing: Industry Track, pp. 179–187, 2022. doi: 10.18653/v1/2022.emnlp-industry.18.

Xuefeng Wu, Yuanfeng Song, Jiawei Wen, Dandan Huang, Raymond Chi-Wing Wong, and Haodi Zhang. TEFD: A benchmark for natural language to Flux query generation in time-series databases. In Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2, pp. 10020–10031, 2026. doi: 10.1145/3770855.3817508.

Xiangjin Xie, Guangwei Xu, Lingyan Zhao, and Ruijie Guo. OpenSearch-SQL: Enhancing text-tosql with dynamic few-shot and consistency alignment. Proceedings of the ACM on Management ofData, 3(3):1–24, 2025. doi: 10.1145/3725331.

Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning Yao, Shanelle Roman, Zilin Zhang, and Dragomir Radev. Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-to-SQL task. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, pp. 3911– 3921, 2018. doi: 10.18653/v1/D18-1425.

Tao Yu, Rui Zhang, Heyang Er, Suyi Li, Eric Xue, Bo Pang, Xi Victoria Lin, Yi Chern Tan, Tianze Shi, Zihan Li, Youxuan Jiang, Michihiro Yasunaga, Sungrok Shim, Tao Chen, Alexander Fabbri, Zifan Li, Luyao Chen, Yuwen Zhang, Shreya Dixit, Vincent Zhang, Caiming Xiong, Richard Socher, Walter Lasecki, and Dragomir Radev. CoSQL: A conversational text-to-SQL challenge towards cross-domain natural language interfaces to databases. In Proceedings ofthe Conference on Empirical Methods in Natural Language Processing and the International Joint Conference on Natural Language Processing, pp. 1962–1979, 2019a. doi: 10.18653/v1/D19-1204.

Tao Yu, Rui Zhang, Michihiro Yasunaga, Yi Chern Tan, Xi Victoria Lin, Suyi Li, Heyang Er, Irene Li, Bo Pang, Tao Chen, Emily Ji, Shreya Dixit, David Proctor, Sungrok Shim, Jonathan Kraft, Vincent Zhang, Caiming Xiong, Richard Socher, and Dragomir Radev. SParC: Cross-domain

semantic parsing in context. In Proceedings of the Annual Meeting of the Association for Computational Linguistics, pp. 4511–4523, 2019b. doi: 10.18653/v1/P19-1443.

Chenxi Zhang, Bicheng Zhang, Dingyu Yang, Xin Peng, Miao Chen, Senyu Xie, Gang Chen, Wei Bi, and Wei Li. PromCopilot: Simplifying Prometheus metric querying in cloud native online service systems via large language models. ACM Transactions on Software Engineering and Methodology, 2026. doi: 10.1145/3797910.

Yi Zhang, Jan Deriu, George Katsogiannis-Meimarakis, Catherine Kosten, Georgia Koutrika, and Kurt Stockinger. ScienceBenchmark: A complex real-world benchmark for evaluating natural language to SQL systems. Proceedings ofthe VLDB Endowment, 17(4):685–698, 2023.

## A DETAILS OF TQTS-BENCH

In this section, we provide detailed information about TQTS-BENCH, including which TSDB management systems were selected for use, what the selected TSDBs are, and which domain they belong to. We also describe the query intents covered in TQTS-BENCH, including the identified intent types and their proportions.

## A.1 IDENTIFYING TSDB MANAGEMENT SYSTEMS AND THEIR QUERY SYNTAX

We identified 23 TSDB management systems based on their popularity rankings on the TSDB leaderboard and their accessibility, and used them to build TQTS-BENCH. Note that several highly ranked TSDB management systems, such as kdb+, were not selected because they require users to apply for and obtain a license before they can be used, which limits their accessibility. As shown in Tab. 4, we present syntax usage examples for two query intents in each TSDB management system, along with the number of TSDBs included in our benchmark. Specifically, the examples illustrate how some systems provide specific functions and operators to help answer questions involving window aggregation & resampling (I1) or temporal change analysis (I2). The “ / ” indicates that no suitable example was identified for the query interface considered.

Table 4: Selected TSDB management systems and their temporal functions and operators.
<table><tr><td rowspan="2">TSDB management system</td><td colspan="2">Representative functions and operators</td><td rowspan="2">#TSDB</td></tr><tr><td>Window aggregation &amp; resampling (I1)</td><td>Temporal change analysis (I2)</td></tr><tr><td>InfluxDB OSS v2</td><td>aggregateWindow()</td><td>derivative()</td><td></td></tr><tr><td>InfluxDB 3 Core</td><td>date_bin_gapfill()</td><td>1</td><td>17</td></tr><tr><td>Prometheus</td><td>avg_over_time()</td><td>resets ()</td><td>13</td></tr><tr><td>TimescaleDB</td><td>time_bucket_gapfill()</td><td>delta()</td><td>8</td></tr><tr><td>DolphinDB</td><td>resample()</td><td>ratios()</td><td>5</td></tr><tr><td>Apache Druid</td><td>TIME_FLOOR()</td><td>TIMESTAMPDIFF()</td><td>4</td></tr><tr><td>QuestDB</td><td>SAMPLE BY</td><td>datediff()</td><td>5</td></tr><tr><td>TDengine</td><td>STATE_WINDOW()</td><td>CSUM()</td><td>4</td></tr><tr><td>Apache IoTDB</td><td>GROUP BY SESSION()</td><td>TIME_DIFFERENCE()</td><td>4</td></tr><tr><td>VictoriaMetrics</td><td>rollup()</td><td>deriv_fast()</td><td>3</td></tr><tr><td>Arc</td><td>time_bucket()</td><td>REGR_SLOPE(v, t)</td><td>2</td></tr><tr><td>GridDB</td><td>TIME_SAMPLING()</td><td>TIMESTAMP_DIFF()</td><td>4</td></tr><tr><td>M3DB</td><td>summarize()</td><td>perSecond( )</td><td>1</td></tr><tr><td>CrateDB</td><td>date_bin()</td><td>age ()</td><td>4</td></tr><tr><td>CnosDB</td><td>date_bin()</td><td>idelta_right ()</td><td>3</td></tr><tr><td>ArcadeDB</td><td>ts.timeBucket()</td><td>ts.delta()</td><td>1</td></tr><tr><td>GreptimeDB IBM Db2 Event Store</td><td>avg_over_time()</td><td>increase() 1</td><td>1</td></tr><tr><td></td><td>window()</td><td>1</td><td>4</td></tr><tr><td>Riak TS</td><td>1</td><td></td><td>2</td></tr><tr><td>BangDB</td><td>ROLLUP 5</td><td>1</td><td>3</td></tr><tr><td>Machbase Neo</td><td>timewindow()</td><td>MAP_DIFF()</td><td>5 3</td></tr><tr><td>OpenMLDB</td><td>ROWS_RANGE</td><td>drawdown()</td><td>1</td></tr><tr><td>openGemini</td><td>GROUP BY time()</td><td>ELAPSED()</td><td></td></tr></table>

## A.2 IDENTIFYING TSDBS AND THEIR DOMAINS

In TQTS-BENCH, we manually identify a total of 97 high-quality real-world TSDBs for benchmark construction. Fig. 8 summarizes these TSDBs, along with their application domains, data volumes, and numbers of QA pairs. The inner ring denotes domains, while the outer ring denotes individual TSDBs. Each sector’s angle represents its number of QA pairs, and its color intensity reflects the data volume of the corresponding database: darker shades indicate larger volumes, and lighter shades indicate smaller ones. The inner ring covers domains such as cloud services, energy, and IoT, whereas the outer ring presents different TSDBs.

As shown in Fig. 8, TQTS-BENCH covers a wide range of real-world domain applications, especially some time-series-intensive domains, such as IoT, AIOps and cloud services. These domains typically generate and store high-frequency time-series data continuously, resulting in diverse data characteristics and large data volumes. The total data volume of TQTS-BENCH is 41.68 GB, re flecting the large-scale and diverse nature of real-world time-series workloads collected from various application domains. To effectively and efficiently analyze these time-series data, many TSDB systems, such as InfluxDB, Prometheus, and TimescaleDB, have been widely adopted. The diversity of application domains, together with the different query syntaxes of TSDBs, highlights the significant challenges to real-world TQTS tasks.

![](images/7f626d51fab41e0bc65bff0cde3de63362262acd7e68c6aaf98026ce8d898311.jpg)  
Figure 8: The domain and size of the TSDBs in TQTS-BENCH.

## A.3 DETAILS OF QUERY INTENTS

This subsection details the query intents in TQTS-BENCH, including the identified query intent types, corresponding question examples, and the proportion of each query intent.

## A.3.1 THE IDENTIFIED QUERY INTENTS

We collaborate with domain experts and identify 9 representative query intents in real-world scenarios, including 4 time-specific intents and 5 time-agnostic intents. The definitions of these query intents, along with their corresponding examples, are presented in Tab. 5.

Table 5: The definitions and examples of the nine query intents used in TQTS-BENCH.
<table><tr><td>Query intent</td><td>Definition</td><td>Example</td></tr><tr><td>Time-specific</td><td colspan="2"></td></tr><tr><td>I1: Window aggregation &amp; resampling</td><td>Resample or aggregate a time series over time windows or a new sampling grid, yielding results for each window or grid.</td><td>&quot;What was Hankyung&#x27;s average temperature forecast for each day of June 2017?&quot;</td></tr><tr><td>12: Temporal change analysis</td><td>Analyzing temporal changes in values within a single time series.</td><td>“What was the largest one-second increase in whole-home active power?&quot; 6</td></tr><tr><td>I3: Time localization I4: Relationship analysis</td><td>Locate the time point(s) or interval at which an event or state occurs. Analyzing relationships between two or more independently identifiable time series.</td><td>When was the earliest weather observation in the database?&quot; “On September 12, at which minutes did Apple&#x27;s trading volume exceed</td></tr><tr><td>Time-agnostic I5: Conditional</td><td>Filtering samples or events based</td><td>Amazon&#x27;s?&quot; &quot;Return the full records for matches</td></tr><tr><td>filtering I6: Statistical computation</td><td>on specified conditions. Computing statistical values from samples or events.</td><td>playedafter June 15, 2024 &quot;What was the average fare across all trips ?&quot; “Show the first 100</td></tr><tr><td>I7: Sort &amp; rank I8: Metadata</td><td>Ranking candidates by one or more ordering criteria. Querying metadata about</td><td>sensor readings in chronological order “How iseach column in the</td></tr><tr><td>query I9: Domain knowledge</td><td>time-series databases, such as schema information. Requiring the use of specialized domain knowledge to answer the</td><td>flight-record tabledefined in Db2?&quot; &quot;What were the 15-minute OHLC pricesand traded share volume for</td></tr></table>

## A.3.2 STATISTIC OF QUERY INTENTS

Fig. 9 presents the statistics of query intents. To examine whether the distribution of query intents is unbiased with respect to query difficulty, we analyze the proportion of each intent across different difficulty levels, as shown in Fig. 9(a). Overall, each query intent is relatively evenly distributed across different difficulty levels, suggesting that query difficulty is not dominated by any particular intent type and that the benchmark maintains a balanced intent distribution across difficulty levels. Nevertheless, time-specific intents (I1–I4) occur more frequently in hard questions, and less in easy questions, indicating that temporal reasoning tends to introduce additional complexity.

Fig. 9(b) further shows that the overall distribution of query intents is consistent with real-world information-seeking demands. For example, I5: Conditional Filtering and I6: Statistical Computation account for the largest proportions, reflecting their prevalence in practical querying scenarios. We additionally investigate how the number of query intents varies with query difficulty, as shown in Fig. 9(c). As the difficulty increases, queries tend to involve more intents, with time-specific query intents showing the most pronounced increase.

![](images/37b0b2382be8a7ea0bb6c41621978ddab7cf3eb05183c80b2732561edf0ac198.jpg)  
Figure 9: The statistics of query intents. (a) Distribution of query intents across different difficulty levels; (b) Distribution of query intents in TQTS-BENCH; (c) Average query intent count across different difficulty levels.

## A.4 NON-UNIFIED SCHEMA STRUCTURES ACROSS TSDBS

Here, we illustrate the non-unified schema structures of TSDBs. TSDB schemas describe the logical organization of time-series data and can be broadly grouped into seven categories: Relational tables, Time–dimension–metric tables, Tagged time-series table, Measurement-based organization, Metric–label series, Device tree and Property graphs. Fig. 10 shows how the same timeseries data are organized differently across these seven schema structures. The descriptions of these non-unified seven schema structures are as follows:

• Relational tables store records in named tables with typed columns. Each row contains attributes such as entity identifiers, timestamps, and values. Each table has a primary key that uniquely identifies records, while some columns may serve as foreign keys to reference records in related tables.

• Time–dimension–metric tables store event records, where each row represents an event associated with a datasource. Each row contains a primary timestamp, dimensions describing event attributes, and metrics representing quantitative values for analysis.

• Tagged time-series tables store timestamped observations, where each row represents an observation associated with an entity or series. Each row contains tags for identifying the entity or series and data columns recording observed values.

• Measurement-based organization stores observation points under different measurements within a data container. Each point contains a timestamp, tags describing the observation context, and fields recording measured values.

• Metric–label series represent time-series data as collections of timestamped samples. Each series contains a metric name, a set of labels identifying the series, and samples recording the metric values over time.

• Device trees store device and measurement data in a hierarchical structure. Each root-toleaf path represents a time series, and each point along the path contains a timestamp and an observed value.

• Property graphs represent entities and observations as vertices connected by edges. Each vertex and edge contain properties that describe attributes such as timestamps and observed values.

## Relational tables

Time–dimension–metric tables
<table><tr><td>Table: cpu_readings</td><td></td><td></td><td></td><td></td></tr><tr><td>time</td><td>device</td><td>cpu</td><td>temp_C</td><td>util_pct</td></tr><tr><td>tl</td><td>A</td><td>0</td><td>65</td><td>40</td></tr><tr><td>t2</td><td>A</td><td>0</td><td>67</td><td>55</td></tr><tr><td>tl</td><td>A</td><td>1</td><td>61</td><td>30</td></tr></table>

(a)

![](images/a3b994e2f7cfd025774db0df2bf216bf3f8b87817deb6cd1bc4d713ff42637ad.jpg)  
(b)

## Tagged time-series table

![](images/d190fd2ce8d8fba3241fec60e0d8ef6469d68bf4973a49a872c3df0fe974c36d.jpg)

![](images/115d309fdf0863b2695267cf4972f94af4f07d52a1a89c33702e7b1bd0199626.jpg)  
(c)  
(d)

![](images/3618e468065c32ca0f0b70adfeda3ffa803f246e1a679e5cdde9388fed1d6c7e.jpg)

![](images/d8f3baee74748ed5c831eaad2f813ab87731a6c334e90ab40462c426eaad892b.jpg)

(f)  
(e)  
![](images/12d36688eb300e6322a1b3d8ac375f1992c8c8b6a30517fe9e3e6a4c180bdbfd.jpg)  
(g)  
Figure 10: The seven types of schema structures of the same time-series data.

## A.5 DETAILS OF THE TSDB CONTEXT

For each TSDB included in TQTS-BENCH, we provide a customized context that captures its background information, key characteristics, and concrete usage scenarios. By incorporating this contextual knowledge, the model can gain a deeper understanding of the target TSDB and its application requirements, allowing it to generate questions that better reflect real-world TSDB usage scenarios and business needs.

For example, as shown in Fig. 11, the context of a Prometheus database contains descriptions of its background, including the fixed benchmark time setting, and the real-world cloud-service telemetry scenarios involving Kubernetes, node, application, Loki, PostgreSQL, and service metrics.

```jsonl
Database context used in TQTS-BENCH construction process
{
"name": "Prometheus March 2022 cloud-service snapshot",
"description": "Seven contiguous native Prometheus blocks
containing real Kubernetes, node, application, Loki, PostgreSQL,
and service telemetry from a cloud-service environment.",
"engine": {
"name": "prometheus",
"version": "2.54.1"
},
"query_language": "PromQL: Prometheus Query Language for instant
queries at the fixed benchmark current_time, using declared metric
and label names. Guidance: The query language is PromQL, and
Prometheus has no SQL tables.",
"time": {
"current_time": "2022-03-23T17:59:59Z",
"timezone": "UTC",
"ranges": [
{
"start": "2022-03-18T12:00:00Z",
"end": "2022-03-23T18:00:00Z"
}
]
}
```  
Figure 11: An example of the customized Prometheus database context used in TQTS-BENCH.

## B DETAILS OF QA CONSTRUCTION

## B.1 DESIGN OF VISUAL ANALYTICS ANNOTATION TOOL

Candidate QA pairs constructed with AI assistance may contain incorrect queries, unnatural wording, or mismatches between the intended request and the database context. To support systematic human verification before benchmark inclusion, we developed a visual analytics tool guided by three quality requirements. These requirements concern answer correctness, question naturalness and clarity, and alignment with query intents and TSDB context.

R1: Ensuring the correctness of executable answers. Reliable answers are essential for evaluating model performance, yet successful execution does not establish whether a query correctly answers its associated question. We therefore require independent human interpretations and execution evidence to assess answer correctness beyond execution success.

R2: Ensuring the naturalness and clarity of questions. AI-assisted construction requires explicit review of whether generated questions express meaningful requests in natural, comprehensible language. Awkward phrasing, ambiguity, and mechanical descriptions can obscure the intended meaning. The system must therefore help reviewers identify these issues and support targeted refinement before benchmark inclusion.

R3: Ensuring alignment with query intents and TSDB context. A fluent question and an executable answer do not necessarily constitute a suitable benchmark example. Each pair must faithfully represent the selected query intents, reflect a meaningful TSDB scenario, and be answerable from the supplied context. Human review must therefore identify misalignment, missing information, and unstated assumptions.

![](images/97e56acd0f549c9d9bc3b506b6ab99782e89fffc0bbcec573673826997e1c00f.jpg)  
Figure 12: The visual interface for annotators in human-AI collaboration step.

![](images/e4612f5fa27742e6b8c9785526bc0e0b1f7d62d9112ce5b17f215f2c7e087893.jpg)  
Figure 13: The visual interface for adjudicator in QA verification step.

## B.2 DETAIL OF QA VERIFICATION

The verification workflow begins with two annotators independently reviewing each candidate QA pair. Fig. 12A enables each annotator to select TSDB and inspect current progress. Fig. 12B presents the question, database description, and schema while hiding the candidate answer and the peer annotator’s solution. Annotators manually construct their own queries in the editor (Fig. 12C), providing independent answers to the questions and examining whether the available context supports answering it. The annotation interface in Fig. 12B allows them to flag unclear wording or insufficient information. They can then inspect their queries’ execution results and refine their answers before submission (Fig. 12D).

When verification reveals inconsistent results or an unanswerable question, an adjudicator examines the QA pair together with its intent combination, match policy, and annotator feedback. Fig. 13 shows the interface for adjudicator to review and revise the unverified QA pair. The adjudicator can inspect the QA pair (Fig. 13A), its corresponding query intents and can check the query result (Fig. 13B). The Review evidence view brings together both annotator queries and execution results for comparison (Fig. 13C). These views support assessment of query correctness alongside the question’s wording, intended meaning, and contextual requirements.

The adjudicator revises the QA pair based on the identified issues, after which it undergoes another verification round. This process continues until both annotators independently obtain results consistent with the revised answer. The tool thus supports human review and correction of AI-assisted construction outputs, keeping interpretation, revision, and verification under human control.

The tool also records active review time and review-related actions, including query executions, annotations, review decisions, and arbitration operations. The Human activity dashboard summarizes these records across database packages and displays review progress. These records allow us to document the human effort involved in QA verification and provide the basis for the verification effort reported in the main text.

## C MORE EXPERIMENT DETAIL AND ANALYSIS

## C.1 EVALUATION CRITERIA FOR TQTS-BENCH

Here, we introduce the detailed criteria for the Execution Accuracy (EX) metric used in the experiment for TQTS-BENCH. Inspired by Spider 2.0 (Lei et al., 2025), EX evaluates whether a predicted query produces the expected results, primarily by checking whether its output contains all columns returned by the gold query. However, in TSDBs, this situation becomes more complex, since models can often generate different queries that return the same underlying data in different formats, such as queries involving the pivot function.

As illustrated in Fig. 14, the absence of the pivot function results in a different output format between the predicted and gold results, causing the evaluation to incorrectly mark the prediction as incorrect. To address this issue, inspired by the matching policy in Spider 2.0, we introduce an additional transformation step to establish a more precise and suitable evaluation criterion for TQTS tasks. This policy reduces false negatives in evaluation while avoiding an increase in false positives.

![](images/d93a7f0a0a508acf352279d3ac69380174fabfbace8b67d528f0c4ea74e1801e.jpg)  
Figure 14: The evaluation criteria and an example of the match policy used in the evaluation.

## C.2 DETAILED INFORMATION FOR HUMAN EVALUATION

In the human evaluation, we utilize the developed visual analytics tool as the evaluation interface. Specifically, human experts are provided with the question, database schema, and TSDB context,

and are asked to manually write the corresponding queries. After completion, we evaluate the EX of the queries provided by human experts.

## C.3 WHY THE RDB METHODS STRUGGLE WITH THE TQTS TASK

The experimental results demonstrate that existing RDB methods generally perform poorly on the TQTS task. To investigate the reasons behind their limitations, we select DeepEye-SQL, a representative and high-performing text-to-query method on BIRD, for further analysis.

For text-to-query tasks, DeepEye-SQL consists of four main stages: schema linking, query generation, query verification and refinement, and confidence-based selection. However, these stages are primarily designed for generating SQL queries. Specifically, as shown in Fig. 15, the prompts used in the query generation stage are heavily constrained by SQL syntax. Such SQL-specific guidance limits the model’s ability to generate queries required by TSDBs, which often adopt different query syntaxes and syntactic structures.

As shown in Fig. 16, the generated queries may not conform to the syntax requirements of TSDB query syntax, leading to execution failures. Therefore, the SQL-specific design of existing RDB methods is a key reason why they struggle with the TQTS task, as it prevents them from effectively adapting to the diverse query syntaxes of TSDBs.

![](images/28611811f491aaeff6744fd7d7cf7d15333dc95dfc1e5a3901a2e3b7a1296f2f.jpg)  
Figure 15: An excerpt from the SQL generation prompt of DeepEye-SQL.

![](images/ab4edf510a5071be49519ce4302cbb8a9ae5ad1229289cf948937f429d176b30.jpg)  
Figure 16: An example where DeepEye-SQL generates SQL instead of the required PromQL.

## C.4 WHY THE TSDB METHODS STILL STRUGGLE WITH TQTS TASK

In the experiment, TSDB methods still achieve limited EX on TQTS-BENCH. To understand the reasons behind such poor performance, we conduct a further analysis of the state-of-the-art method, PromCopilot, a text-to-query method designed for Prometheus-based monitoring scenarios.

PromCopilot addresses the TQTS task through two stages. First, it constructs a knowledge graph to describe system contexts and facilitate schema linking. However, this knowledge graph is tightly coupled with Kubernetes cloud-native environments, where entities and relations are defined based on components such as pods, nodes, and services. Such a domain-specific schema makes it difficult to directly transfer the method to other domains. Second, PromCopilot generates queries using LLMs with carefully designed prompts.

As illustrated in Fig. 17(a), the system prompts are primarily tailored for cloud services and AIOps scenarios, further limiting their applicability beyond the original domain. These domain-specific designs hinder the generalization of PromCopilot to broader TSDB application domains, which may explain its limited performance on TQTS-BENCH, where it achieves only 0.69% EX. This observation is also consistent with the authors’ acknowledgment that PromCopilot relies on Kubernetesand Prometheus-specific knowledge (Fig. 17(b)).

![](images/b05e0b6eadd47b3226f92157e52be1c8b7da3d2edc3913bce54ff9a7ce4ebda1.jpg)  
Figure 17: The description of tightly coupled domains as reflected in (a) the prompt for query generation and (b) the statement of external validity in the original paper.

## C.5 MORE EXAMPLES OF ERROR ANALYSIS

We provide additional examples for each error type identified in error analysis. Tab. 6 summarizes the error taxonomy and links each subcategory to representative error cases. The following figures present detailed examples, illustrating how these errors manifest in generated queries and how they affect the correctness of the final results. Here, for intent understanding errors, we only consider the errors with time-specific query intents.

Table 6: Overview of the error types and their representative examples.
<table><tr><td>Error Type</td><td>Subcategory</td><td>Representative Example(s)</td></tr><tr><td rowspan="2">Query syntax errors</td><td>Query structure errors</td><td>Refer to Fig. 6(a) and Fig. 18(a)</td></tr><tr><td>Function/keyword usage errors</td><td>Refer to Fig. 6(b) and Fig. 18(b)</td></tr><tr><td rowspan="4">Intent understanding errors</td><td>I1: Window aggregation &amp; resampling</td><td>Refer to Fig. 7(b)</td></tr><tr><td>I2: Temporal change analysis</td><td>Refer to Fig. 19(a)</td></tr><tr><td>I3: Time localization</td><td>Refer to Fig. 19(b)</td></tr><tr><td>I4: Relationship analysis</td><td>Refer to Fig. 19(c)</td></tr><tr><td>Schema linking errors</td><td>1</td><td>Refer to Fig. 20(a) and Fig. 20(b)</td></tr></table>

![](images/94ed0e475fd4eb3802b0abcc295ede39721472f0716da22b8bce271fb036241d.jpg)  
Figure 18: Examples of query syntax errors. (a) Query structure error: placing LAG() directly in the WHERE clause, rather than computing the preceding reading in a subquery, causes a windowfunction execution error. (b) Operator usage error: replacing subtraction with offset 1h by delta(...[1h]) substitutes an extrapolated change for the difference between two time points; together with a changed metric, this produces incorrect values and rankings.

## C.6 DETAILS OF THE ABLATION STUDY

Here, we introduce the implementation details of the ablation study in Sec. 4.4. Although query syntax errors account for 75.29% of all error cases, most of these cases involve two or more types of errors simultaneously. Therefore, to eliminate the interference from other error sources, we only select cases with syntax errors as the sole error type for the ablation study. For QA pairs containing only syntax errors, we map the TSDBs to RDBs and organize the schemas following the same format as BIRD. We then keep the original questions unchanged, manually annotate each question with a corresponding SQL gold query, and verify that the execution results are consistent with the original gold answers. Based on this converted setting, we evaluate the model and observe that the execution accuracy on these error cases improves from 0% to 32.21%, which falls between the performance levels on BIRD and Spider 2.0-lite. This suggests that the poor performance of the model is primarily due to the model’s inability to handle TSDB query syntax rather than the difficulty of the questions.

![](images/297f3df04a290c968ce8bf9b33c1d2e3d78d7d43eac0e0f984bc05b2664ef035.jpg)  
Figure 19: Examples of intent understanding errors. (a) Temporal change analysis error: ranking signed differences rather than absolute magnitudes interprets the largest changes as the largest increases, selecting incorrect time boundaries. (b) Time localization error: returning availablememory values instead of applying max(timestamp(...)) reports memory quantities rather than the latest sample time. (c) Relationship analysis error: using IN (8, 32) instead of matching timestamps across both communities confuses union with intersection; a global LIMIT 1 further returns a single time rather than the earliest shared time for each day.

![](images/c4351d43766c365735e4753e4335eb21299bf860d4344ee1970d3fa7d80143c7.jpg)  
Figure 20: Two examples of schema linking errors. (a) The model treats the attack number tag as a field and filters on $\scriptstyle { \mathrm { . ~ f ~ i ~ e ~ l ~ d ~ } } \ = = \ { \mathrm { " } } \ a { \mathrm { t } } \ a { \mathrm { c k } } \_ { \mathrm { . } } { \mathrm { n u m b e r } } \ { \mathrm { " } }$ , which removes all relevant records and produces an empty result. (b) The model uses designated timestamp instead of the metadata column designatedTimestamp, causing an invalid-column error and preventing retrieval of the designated timestamp column.

To further demonstrate the difference in the model’s proficiency between SQL and TSDB query syntax, we present a case study. As shown in Fig. 21, the model-generated Flux query fails due to a syntax error, whereas the generated SQL query successfully produces the correct result, despite the SQL gold query, containing 766 tokens, requiring a much more complex implementation than its Flux counterpart, containing 255 tokens. This suggests that the model still lacks sufficient capability in understanding and generating TSDB query syntax, such as Flux.

![](images/8238428f4313b5539d9e80a561281b02fc1322ae354769a10d8e7dbd4666b47c.jpg)  
Figure 21: A case study on the differences in LLM capabilities between generating Flux for TSDBs and SQL for RDBs.

# D PROMPTS USED IN TQTS-BENCH CONSTRUCTION PROCESS

## The system prompt used for question generation in TQTS-BENCH construction process

<table><tr><td></td></tr><tr><td>Task: You are an assistant for constructing natural-language questions over time-series databases. Using the query intents, target intent combination, database schema, and database context below, generate one</td></tr><tr><td>natural-language question (Q) for the target time-series database. In this stage, generate Q only; do not generate a query statement (A) or query results. Inputs:</td></tr><tr><td>Query intents:</td></tr><tr><td>A question may involve one or more intents. I1–I4 are time-specific intents, and I5–I9 are time-agnostic intents. I1 – Window aggregation and resampling: Resample or aggregate a time series over time windows or</td></tr><tr><td>a new sampling grid, yielding results for each window or grid.</td></tr><tr><td>I2 – Temporal change analysis: Analyze temporal changes in values within a single time series. I3 – Time localization: Locate the time point(s) or interval at which an event or state occurs.</td></tr><tr><td>I4 – Relationship analysis: Analyze relationships between two or more independently identifiable time</td></tr><tr><td>series. I5 – Conditional filtering: Filter samples or events based on specified conditions.</td></tr><tr><td>I6 – Statistical computation: Compute statistical values from samples or events.</td></tr><tr><td>I7 – Sort &amp; rank: Rank candidates by one or more ordering criteria.</td></tr><tr><td>I8 – Metadata query: Query metadata about time-series databases, such as schema information.</td></tr><tr><td>I9 – Domain knowledge dependency: Require the use of specialized domain knowledge to answer the</td></tr><tr><td>question. Target intent combination for this generation:</td></tr><tr><td>{TARGET_INTENT_COMBINATION}</td></tr><tr><td>Database schema:</td></tr><tr><td>{DATABASE_SCHEMA}</td></tr><tr><td>Database context:</td></tr><tr><td>{DATABASE_CONTEXT}</td></tr><tr><td>Generation constraints:</td></tr><tr><td>1. The question must express a real, meaningful query need for the target database and accurately re- flect the specified target intent combination. Include an intent only when it is genuinely required by the</td></tr><tr><td>question. 2. Use only entities, data meanings, and domain conventions supported by the schema and database</td></tr><tr><td>context, so that the question can be answered using the target database. 3. Express the user's information need in natural, clear, and unambiguous English. Do not mechanically turn physical field names or a query statement into a question, and do not describe query implementation</td></tr><tr><td>steps. 4. Do not combine unrelated objectives merely to satisfy the target intent combination. Do not assume</td></tr><tr><td>schema elements, data meanings, domain knowledge, or database capabilities that were not provided. 5. Do not generate tasks such as forecasting, anomaly detection, or pattern clustering that require separate</td></tr><tr><td>algorithmic models and cannot be completed by querying the target TSDB alone. Output format:</td></tr><tr><td>Return exactly one JSON object, with no explanation or Markdown code fence. The intents array must list the intents actually used by Q and must match the target intent combination:</td></tr><tr><td>[User]</td></tr><tr><td>Target intent combination: {I1， I5}</td></tr><tr><td>[System] Task:</td></tr><tr><td>You are an assistant for constructing time-series database queries. The natural-language question (Q) and its intent labels below have been finalized. Using Q, the target database type, database schema, and database context, generate a query statement (A) that correctly answers Q. Do not change the meaning of</td></tr><tr><td>Q.</td></tr><tr><td>Inputs:</td></tr><tr><td>Finalized Q:</td></tr><tr><td>{REVIEWED_QUESTION} Confirmed intent labels:</td></tr><tr><td>{REVIEWED_INTENTS}</td></tr><tr><td>Target database type:</td></tr><tr><td>{TARGET_TSDB}</td></tr><tr><td>Database schema:</td></tr><tr><td>{DATABASE_SCHEMA}</td></tr><tr><td>Database context:</td></tr><tr><td>{DATABASE_CONTEXT}</td></tr><tr><td>Generation constraints: 1. Generate one query that can actually be executed in {TARGET_TSDB}. Use the query syntax supported</td></tr><tr><td>by that database type; do not default to the syntax of another database system. The query must fully and accurately express the goal, conditions, and temporal semantics of Q, and its result must answer Q.</td></tr><tr><td>2. Use only objects that actually exist in the schema and data meanings and domain conventions provided in the database context. Do not invent fields, data, or query results. 3. Do not add analysis goals, conditions, or result content that Q does not request, and do not change Q</td></tr><tr><td>to accommodate a particular query formulation. 4. If the target database is accessible, execute A, check that execution succeeds, that the result is non-</td></tr><tr><td>empty and valid, and that the result answers Q; revise A if the check fails. Unless Q explicitly asks for a nonexistent entity or an empty result, an empty result does not count as a verified answer. If the database is inaccessible, do not claim that A was executed or verified.</td></tr><tr><td>5. If Q conflicts with the schema or database context, or if information needed for a correct query is missing, do not guess a query statement. Output format:</td></tr></table>

Database schema: {schema}   
Database context: {context}   
[Assistant]   
{   
"question": "What were the available-memory readings for farm140105,   
farm140108, and farm140109 at each fifteen-minute checkpoint   
during the final hour?",   
"intents": ["I1", "I5"]   
}   
[User]   
Target intent combination: {TARGET INTENT COMBINATION}   
Database schema: {DATABASE SCHEMA}   
Database context: {DATABASE CONTEXT}

The system prompt used for query generation in TQTS-BENCH construction process

Return exactly one JSON object, with no explanation or Markdown code fence. question must   
reproduce Q exactly, intents must match the confirmed intent labels, and answer must contain   
a query for the target database type. Record only evidence obtained from an actual execution in   
execution evidence; if the query was not run, set executed to false and do not invent results.   
[User]   
Finalized Q: {question}   
Confirmed intent labels: {I1, I5}   
Target database type: {TSDB}   
Database schema: {schema}   
Database context: {context}   
[Assistant]   
"target\_tsdb": "Prometheus",   
"question": "Show the total memory allocation for the pods of   
the admin user service on node k8s-node3 or k8s-node4   
over the last hour.",   
"intents": ["I1", "I5"],   
"answer": "<executable query for the target database>",   
"execution\_evidence": {   
"executed": false,   
"non\_empty": null,   
"result\_summary": null   
}   
[User]   
Finalized Q: {REVIEWED QUESTION}   
Confirmed intent labels: {REVIEWED INTENTS}   
Target database type: {TARGET TSDB}   
Database schema: {DATABASE SCHEMA}   
Database context: {DATABASE CONTEXT}