# When Errors Become Memories: Causal Pathway Tracing in Multi-Turn Memory-Augmented LLMs

Shuyao Xiao<sup>1,2</sup> <sup>∗</sup>, Shengling Wang<sup>1†</sup>, Xuan Chen<sup>2</sup>, Ke Chao<sup>1</sup>, Ming Cui<sup>2</sup>, Feifei Qian<sup>1</sup>, Fanlin Meng<sup>2</sup>, Chaoyang Mei<sup>2</sup>, Chaoyong Jiang<sup>1</sup>, Qi Ouyang<sup>3</sup>, Junxi Yin<sup>2</sup>,

<sup>1</sup>College of Artificial Intelligence, Beijing Normal University, Beijing, China <sup>2</sup>Beike Language and Intelligence, Beijing, China

<sup>3</sup>Center for Data Science, New York University, New York, NY, USA xiaoshuyao@mail.bnu.edu.cn, wangshengling@bnu.edu.cn, chenxuan046@ke.com

## Abstract

Long-term memory enables large language models (LLMs) to preserve and reuse information across interactions, but it can also turn localized errors into persistent risks. Existing work mainly evaluates whether memory systems store and retrieve information correctly, leaving limited understanding of how errors propagate across responses, memory states, and future interactions. We propose a structural causal model (SCM)-based framework for cross-turn error propagation in memory-augmented LLMs. We model user questions, model responses, and memory states as a dynamic causal process, and identify two entry pathways: internal memory updating and external question feedback. By intervening on these pathways, we construct four counterfactual trajectories and quantify their downstream efects and interaction. Error influence is evaluated at four levels: memory retention, natural responses, targeted diagnostic probing, and probability-level error preference. Experiments show that error influence generally decays with interaction distance, while the memory-update pathway contributes more persistent efects than question feedback; latent errors may remain even after disappearing from natural responses. Propagation patterns also vary across memory categories and memory mechanisms. Pathway-guided restoration further validates this decomposition: Question Repair reduces residual error by 27.5%, Memory Repair by 70.2%, and Joint Repair by 98.3%, nearly eliminating residual propagation.

## 1 Introduction

As large language models (LLMs) are increasingly deployed in long-term conversational settings, personalized assistants, and persistent interactive systems, external long-term memory has become an important mechanism for preserving and reusing information across sessions (Zhang et al. 2025). Memory systems such as MemoryBank (Zhong et al. 2024), Generative Agents (Park et al. 2023), and MemGPT (Packer et al. 2023) enable models to move beyond the limitations of the context window through continuous memory updates, dynamic retrieval, and hierarchical memory management. Meanwhile, benchmarks such as LongMemEval (Wu et al.

2024), MemBench (Tan et al. 2025), and MemoryAgent-Bench (Hu, Wang, and McAuley 2025) evaluate memory systems in terms of information retrieval, knowledge updating, long-range understanding, selective forgetting, and overall memory capability.

Long-term memory, however, also changes the nature of model errors. In conventional single-turn interactions, an incorrect response typically constitutes a localized output failure. In memory-augmented systems, by contrast, the same error may be incorporated into subsequent memory states and repeatedly retrieved and reused, thereby influencing responses across future interactions (Xiong et al. 2025; Zhang et al. 2023; Kwan et al. 2024). In high-stakes settings that rely on persistent information, such as health management, financial planning, and task execution, these enduring efects may further shape user decisions or agent actions (Dash et al. 2026; Qian 2026). Prior studies have shown that erroneous or contaminated information may persist once incorporated into long-term memory and continue to afect subsequent responses, plans, and behaviors when retrieved (Xiong et al. 2025; Deng et al. 2026). Consequently, an initially isolated error may propagate over time, evolving into a long-term risk that afects both system states and real-world outcomes.

Existing research has primarily focused on the construction, management, and evaluation of long-term memory, typically assessing whether a model can correctly retain and use historical information through final-answer quality or aggregate task performance (Wu et al. 2024; Tan et al. 2025; Hu, Wang, and McAuley 2025). However, long-term memory is not just a static storage component; it is part of a dynamic system where user questions, model responses, and memory states evolve together. At the system level, an error may initially afect future behavior through both internal memory states and external interaction dynamics. Internally, an erroneous response may be written into long-term memory, afecting subsequent generation (Xiong et al. 2025; Zhang et al. 2026; Deng et al. 2026). Externally, it may reshape the user’s follow-up questions, causing the conversation to proceed from an incorrect premise (Kwan et al. 2024). These pathways may later interact, prolonging the influence of the original error. Existing approaches provide limited insight into how errors propagate dynamically across responses, memory states, and subsequent questions, and how the pathway through which an error enters the system shapes its downstream influence.

To address this gap, we introduce a structural causal mode (SCM)-based framework (Pearl 2009) for analyzing crossturn error propagation dynamics. The framework integrates the joint evolution of user questions, model responses, and memory states within a unified causal formulation and employs counterfactual interventions to control whether an error enters through internal memory, external question feedback, or both. We use error influence to refer to the measurable downstream efects of an injected error on memory states and subsequent behavior, including memory retention, occurrence in natural responses, re-elicitation under targeted probes, and shifts in generation probabilities. By tracing these manifestations, our framework reveals how errors persist, decay, and re-emerge across long-term interactions, providing a unified perspective for understanding and mitigating systematic error propagation in memory-augmented LLMs. Our main contributions are as follows:

• We develop a dynamic structural causal model and a fourtrajectory counterfactual intervention framework that estimates the downstream efects of introducing an error through memory updating and question feedback, along with their interaction.

• Our experiments show that error influence generally decreases with interaction distance, while errors introduced through the internal memory pathway have more persistent downstream efects than those from the questionfeedback pathway. Even after an error disappears from natural responses, it may remain as a latent memory trace or a shift in generation preference.

• Error propagation varies across memory categories and mechanisms, showing how memory organization shapes error persistence. We perform pathway-guided restoration to update the internal memory state and the external question-feedback trajectory. Restoring the corrupted memory state suppresses error propagation more efectively than correcting the error-afected question trajectory alone, while restoring both components nearly eliminates the residual efect.

## 2 Related Work

Long-Term Memory-Augmented LLMs. To support long-term dialogue, personalized interaction, and information reuse across sessions, prior studies have augmented large language models with external memory. Memory-Bank (Zhong et al. 2024) maintains continuously updated user memories and incorporates reinforcement and forgetting mechanisms based on recency and importance. Generative Agents (Park et al. 2023) organize experiences into a natural-language memory stream and combine retrieval with reflection to support planning and temporally coherent behavior. MemGPT (Packer et al. 2023) draws inspiration from virtual memory in operating systems and dynamically manages information between the active context and external storage, thereby enabling multi-session dialogue and longdocument processing. LongMem (Wang et al. 2023) adopts a decoupled architecture for memory encoding, retrieval, and reading, allowing language models to cache and utilize information from extended historical contexts.

These studies improve the storage, updating, retrieval, and management of historical information. Their primary objective, however, is to develop more efective memory architectures or improve downstream performance, with comparatively limited attention paid to how information propagates across multiple rounds of interaction.

Memory Evaluation in LLM Systems. As long-term memory mechanisms have evolved, recent work has begun to systematically evaluate memory capabilities in persistent interactions. LongMemEval (Wu et al. 2024) assesses whether conversational assistants can retrieve and integrate historical information through tasks involving information extraction, cross-session reasoning, temporal reasoning, knowledge updating, and abstention. MemBench (Tan et al. 2025) evaluates memory systems along dimensions such as factual and reflective memory, participatory and observational interaction, efectiveness, eficiency, and capacity. MemoryAgentBench (Hu, Wang, and McAuley 2025) transforms longcontext data into incremental, multi-turn interactions and evaluates four core capabilities: accurate retrieval, test-time learning, long-range understanding, and selective forgetting.

These benchmarks have advanced memory evaluation from static long-context question answering to incremental and interactive settings. However, they primarily evaluate a system’s ability to retain, update, and utilize historical information, usually based on final-answer quality or aggregate task performance.

Causal Analysis of LLM Behavior. Causal intervention has become an important approach for understanding the generation mechanisms of large language models. Causal tracing perturbs and restores intermediate activations to identify internal computations that causally contribute to factual predictions (Meng et al. 2022). ROME (Meng et al. 2022), activation patching, and interchange interventions (Zhang and Nanda 2023) further localize and manipulate specific layers, token positions, or model components, while causal mediation analysis decomposes input–output efects transmitted through selected intermediate representations (Vig et al. 2020). Recent work has examined how the causal influence of input questions and generated prefixes evolves during autoregressive generation (Xiao, Wang, and Chao 2026), but its analysis remains confined to a single generation trajectory.

Existing causal studies focus on internal variables within a single generation process, leaving cross-turn system dynamics underexplored. In memory-augmented LLMs, however, questions, responses, and memory states evolve jointly, allowing errors to persist and spread across multiple pathways. The contribution of these pathways to long-term error propagation remains poorly understood.

## 3 Structural Causal Modeling

To characterize the dynamic causal relationships among user questions, model responses, and memory updates in an LLM agent, we abstract the interaction and memory-update processes common to long-term memory systems and formulate the multi-turn interaction as a structural causal model indexed by interaction round (Pearl 2009). As illustrated in Figure 1, at round $r , Q ^ { r } , A ^ { r }$ , and $M ^ { r }$ denote the user question, generated response, and memory state available before response generation, respectively.

![](images/035f9c2d4f4eb6919ad0acc7d15009bec49998b8ca64b3fe0e1ae35e0d531d12.jpg)  
Figure 1: Dynamic structural causal graph of a long-term memory-augmented LLM agent. The memory-update path $A ^ { r - 1 } \stackrel { \cdot } {  } \bar { M } ^ { r }  A ^ { r }$ and question-feedback path $A ^ { r - \bar { 1 } } $ $Q ^ { r }  A ^ { r }$ represent two mechanisms through which an erroneous response can afect subsequent interactions.

We adopt a first-order Markov assumption to model the temporal evolution of the memory system. Variables at round r depend only on those from round $r - 1$ and the current round, while information from earlier rounds may still influence the current interaction through intermediate states. This assumption provides a tractable abstraction while preserving question evolution, response generation, and memory updating. Under this assumption, the causal graph includes four main types of relationships.

Memory persistence. The edge $M ^ { r - 1 } \ \to \ M ^ { r }$ represents the persistence of long-term memory across interaction rounds, capturing the retention and transmission of stored information (Zhong et al. 2024; Packer et al. 2023).

Memory update. The edges $Q ^ { r - 1 } \to M ^ { r }$ and $A ^ { r - 1 } $ $M ^ { r }$ describe memory updating. The former indicates that user queries influence retained information, while the latter captures how model responses may be written into memory and become part of the future internal state (Zhong et al. 2024; Packer et al. 2023; Xiong et al. 2025).

Question continuation. The edge $A ^ { r - 1 } \to Q ^ { r }$ captures how the assistant’s previous response influences the user’s subsequent question. Users may pose follow-up questions, request corrections, or seek elaboration based on previous responses, causing the dialogue trajectory to depend on historical outputs (Kwan et al. 2024; Laban et al. 2025).

Response generation. The edges $Q ^ { r }  A ^ { r }$ and $M ^ { r } $ $A ^ { r }$ represent response generation, in which the current response is conditioned jointly on the current question and the memory state available at that round (Zhong et al. 2024).

Although the causal graph characterizes the normal evolution of a memory agent, an erroneous response may introduce incorrect information into subsequent interactions. Based on the system dynamics, we identify two primary pathways through which an erroneous response $A ^ { \bar { r } - 1 }$ can afect subsequent interactions: the memory-update pathway $A ^ { r - 1 }  \dot { M } ^ { r }  A ^ { r }$ and the question-feedback pathway $A ^ { r - 1 } \to Q ^ { r } \to A ^ { r }$

Memory-update path. This path describes how erroneous information enters subsequent interactions through the agent’s internal memory. Information from an erroneous response may be extracted and stored by the memory module through $\dot { A ^ { r - 1 } } \ \to \ M ^ { r }$ . Because $A ^ { r }$ depends on $M ^ { r }$ , the corrupted memory may bias generation toward the same error (Xiong et al. 2025; Zhang et al. 2026; Deng et al. 2026).

Question-feedback path. This path describes how an error enters subsequent interactions by shaping the user’s next question. An erroneous response may afect the next question through $A ^ { r - 1 } \to Q ^ { r }$ . For example, a user may continue the conversation based on an incorrect premise introduced by the model, causing subsequent responses to follow a trajectory shaped by the original error (Laban et al. 2025).

The two entry pathways afect diferent components of the causal system. The memory-update pathway changes the agent’s internal state, whereas the question-feedback pathway changes the external interaction context. Over long-term interactions, these mechanisms may interact: an erroneous response may alter future questions, trigger additional errors, and cause further incorrect information to be written into memory. Thus, the two pathways represent distinct mechanisms through which an error enters the system at the intervention turn and propagates across subsequent interactions.

To quantify the downstream efects of each entry mechanism, we introduce controlled causal interventions in the following section.

## 4 Causal Intervention and Pathway Efects

Let t denote the error-injection round. Following the two outgoing causal edges from the injected response $A ^ { t }$ to the next-round memory state $M ^ { t + 1 }$ and user question $Q ^ { t + 1 }$ in Figure 1, we intervene on the inputs to the memory updater and the next-question generator. These two successor mechanisms define a $2 \times 2$ joint intervention design.

At round t, we replace the naturally generated response with a matched correct response $A _ { c } ^ { t }$ or erroneous response $A _ { e } ^ { t }$ to the same question. The two responses difer only in the target proposition, while the remaining content and wording are preserved as closely as possible. Passing them separately to the memory updater produces a clean memory state $M _ { c } ^ { t + \mathbf { \breve { 1 } } }$ and an erroneous memory state $M _ { e } ^ { t + 1 }$ , respectively. Combining the correct and erroneous response variants as inputs to the two mechanisms yields four counterfactual trajectories (Shpitser and Tchetgen 2014):

$$
\begin{array} { r l r l } & { \mathcal { T } _ { c c } = \left( M _ { c } ^ { t + 1 } , A _ { c } ^ { t } \right) , } & & { \mathcal { T } _ { e c } = \left( M _ { e } ^ { t + 1 } , A _ { c } ^ { t } \right) , } \\ & { \mathcal { T } _ { c e } = \left( M _ { c } ^ { t + 1 } , A _ { e } ^ { t } \right) , } & & { \mathcal { T } _ { e e } = \left( M _ { e } ^ { t + 1 } , A _ { e } ^ { t } \right) . } \end{array}\tag{1}
$$

The first component specifies the memory state available at round $t + 1$ , while the second specifies the response supplied to the next-question generator. Accordingly, the first subscript denotes the intervention on the memory-update mechanism, and the second denotes the intervention on the question-feedback mechanism; c and e indicate correct and erroneous inputs, respectively. Thus, $\mathcal { T } _ { c c }$ is the fully clean trajectory, $\bar { \mathcal { T } _ { e c } }$ introduces the error through memory updating only, $\mathcal { T } _ { c e }$ introduces it through question feedback only, and $\mathcal { T } _ { e e }$ introduces it through both mechanisms.

After the intervention at round t, the four trajectories evolve independently according to their respective memory states, dialogue histories, and previous responses. Their differences therefore identify the downstream causal contrasts induced by the assigned error-entry interventions, while subsequent memory updating, question evolution, and response generation unfold naturally within each trajectory.

## 4.1 Error-Propagation Outcomes

Let k denote the interaction distance from the error-injection round, where $k = 1$ corresponds to the first post-intervention round. We measure the downstream influence of the injected error at four complementary levels: memory retention, natural responses, targeted diagnostic probing, and generation preference. Within a given trajectory, let $M ( k )$ denote the memory state available immediately before response generation at interaction distance k. The memory-state outcome is defined as

$$
Y _ { M } ( k ) = \mathbb { I } \left[ \mathrm { t h e ~ t a r g e t ~ e r r o r ~ i s ~ d e t e c t e d ~ i n ~ } M ( k ) \right] ,\tag{2}
$$

where $Y _ { M } ( k ) = 1$ indicates that the target error remains in the current memory state. The natural-response outcome is defined as

$Y _ { A } ( k ) = \mathbb { I }$ [the natural response contains the target error] .

(3)

Here, $Y _ { A } ( k ) = 1$ when the response explicitly states, necessarily assumes, or directly relies on the target error.

Because natural follow-up questions may not directly concern the target fact, we also use a targeted diagnostic question $q _ { k }$ that asks about it directly. At each interaction distance k, the same probe $q _ { k }$ is used across all four trajectories to ensure comparability. The controlled-probe outcome is defined as

$$
Y _ { C } ( k ) = \mathbb { I } \left[ \mathrm { t h e  r e s p o n s e \ t o \ } q _ { k } \mathrm { \ c o n t a i n s \ t h e \ t a r g e t \ e r r o r } \right] ,\tag{4}
$$

where $Y _ { C } ( k ) = 1$ indicates that the target error can be directly elicited from the current trajectory state. The outcomes ${ Y _ { M } ( k ) , Y _ { A } ( k ) }$ , and $Y _ { C } ( k )$ are determined by a propositionlevel evaluator. A text is labeled erroneous only if it explicitly states, necessarily assumes, or directly relies on the complete target error; merely quoting, negating, or correcting the erroneous proposition does not count as propagation.

To capture latent shifts that may not appear in the final decoded response, we construct a matched erroneous target answer $a _ { \mathrm { e r r } }$ and correct target answer $a _ { \mathrm { c o r } }$ for $q _ { k }$ , and define the probability-level error preference as

$$
\begin{array} { r } { Y _ { P } ( k ) = \frac { 1 } { \ln 2 } \left[ \frac { \log P ( a _ { \mathrm { e r r } } | q _ { k } , M ( k ) ) } { | a _ { \mathrm { e r r } } | } - \frac { \log P ( a _ { \mathrm { c o r } } | q _ { k } , M ( k ) ) } { | a _ { \mathrm { c o r } } | } \right] . } \end{array}\tag{5}
$$

where |a| denotes the number of tokens in answer a, and the factor 1/ ln 2 converts the diference in natural loglikelihoods into bits. A positive $Y _ { P } ( k )$ indicates a preference for the erroneous target, whereas a negative value indicates a preference for the correct target. The four outcomes capture, respectively, error retention in memory, explicit error occurrence in natural responses, error re-elicitation through targeted probing, and latent shifts in generation preference.

## 4.2 Pathway Efects

Motivated by causal contrasts for pathway and joint interventions (Avin, Shpitser, and Pearl 2005; Shpitser and Tchetgen 2014), we define trajectory-level efects according to the mechanism through which the error initially enters the system. For any outcome $Y \in \{ Y _ { M } , Y _ { A } , Y _ { C } , Y _ { P } \}$ , let $Y _ { x z } ( k )$ denote its value under trajectory $\mathcal { T } _ { x z }$ at interaction distance k, where $x , z \in \{ c , e \}$ , and let $\mathrm { \bar { E } } [ \cdot ]$ denote the average over experimental instances.

Definition 1 (Total Propagation Efect). The total propagation efect at interaction distance k is defined as

$$
\mathrm { T P E } _ { Y } ( k ) = \mathbb { E } [ Y _ { e e } ( k ) ] - \mathbb { E } [ Y _ { c c } ( k ) ] .\tag{6}
$$

It measures the overall downstream efect of introducing the error through both entry mechanisms relative to the fully clean trajectory.

Definition 2 (Memory-Update Pathway Efect). The memory-update pathway efect at interaction distance k is defined as

$$
\mathrm { P E } _ { M , Y } ( k ) = \mathbb { E } [ Y _ { e c } ( k ) ] - \mathbb { E } [ Y _ { c c } ( k ) ] .\tag{7}
$$

It measures the downstream efect associated with introducing the error only through the memory updater.

Definition 3 (Question-Feedback Pathway Efect). The question-feedback pathway efect at interaction distance k is defined as

$$
\mathrm { P E } _ { Q , Y } ( k ) = \mathbb { E } [ Y _ { c e } ( k ) ] - \mathbb { E } [ Y _ { c c } ( k ) ] .\tag{8}
$$

It measures the downstream efect associated with intro ducing the error only through the next-question generator.

Definition 4 (Pathway Interaction Efect). The pathway interaction efect is defined as

$$
\begin{array} { r l } & { { \mathrm { P I E } } _ { Y } ( k ) = { \mathrm { T P E } } _ { Y } ( k ) - { \mathrm { P E } } _ { M , Y } ( k ) - { \mathrm { P E } } _ { Q , Y } ( k ) } \\ & { \qquad = { \mathrm { \mathbb { E } } } [ Y _ { e e } ( k ) ] - { \mathrm { \mathbb { E } } } [ Y _ { e c } ( k ) ] - { \mathrm { \mathbb { E } } } [ Y _ { c e } ( k ) ] + { \mathrm { \mathbb { E } } } [ Y _ { c c } ( k ) ] . } \end{array}\tag{9}
$$

These estimands capture the downstream efects associated with the pathway through which the error is introduced at the injection turn, while each trajectory subsequently evolves according to its own states and dialogue history. The interaction efect measures non-additivity between the two pathway interventions, with positive and negative values indicating super-additive and sub-additive efects, respectively. For example, $\mathrm { T P E } _ { Y _ { M } } ( k ) , \mathrm { P E } _ { M , Y _ { A } } ( k )$ , and $\mathrm { P E } _ { Q , Y _ { C } } ( k )$ denote the total efect on memory retention, the memory-update pathway efect on natural responses, and the question-feedback pathway efect on controlled elicitation, respectively. All trajectories share the same pre-intervention history, and the same diagnostic question is used across all four trajectories for controlled-probe and probability-level evaluation.

## 5 Experimental Evaluation

Dataset Construction. Existing multi-turn dialogue datasets typically use fixed interaction trajectories and do not support the pathway-specific interventions required by our causal design. Therefore, we construct 144 multi-turn interactions across six memory categories, with 24 instances each. Each interaction contains 15 turns; one target error is injected at turn 3 and traced over the following 12 turns, where $\mathbf { \bar { \boldsymbol { k } } } \in \{ 1 , \dots , 1 2 \}$ denotes the interaction distance. Construction details are provided in the supplementary material.

![](images/acfb445ba072862337f31e2407ce29948271da3f454c6dcbde245bbcfa31f8dc.jpg)  
(b) OLMo3-7B-Instruct  
Figure 2: Pathway efects of an injected error across interaction distances for two models. The first and second rows show results for Qwen2.5-7B-Instruct and OLMo3-7B-Instruct, respectively. From left to right, the columns correspond to $Y _ { A } ( k ) , Y _ { M } ( k )$ $Y _ { C } ( k )$ , and $Y _ { P } ( k )$ . The curves show the total propagation efect $\mathrm { T P E } ,$ memory-update pathway efect $\bar { \mathrm { P E } } _ { M }$ , question-feedback pathway efect $\mathrm { \dot { P } E } _ { Q }$ , and pathway interaction efect PIE; shaded regions indicate 95% bootstrap confidence intervals.

Memory Mechanisms and Models. The main setting follows a MemoryBank-style mechanism (Zhong et al. 2024) where the memory state is updated after each interaction round. We additionally evaluate an episodic-retrieval variant inspired by prior retrieval-based memory systems (Park et al. 2023; Packer et al. 2023). This variant stores each interaction as an independent memory entry and retrieves the three most relevant entries using a deterministic, lengthnormalized lexical-overlap score, with recency used to break ties. We use Qwen2.5-7B-Instruct (Qwen et al. 2025) and OLMo3-7B-Instruct (Olmo et al. 2026) as answer models, while Qwen2.5-32B-Instruct (Qwen et al. 2025) is used for question simulation, counterfactual construction, and proposition-level evaluation.

Experimental Settings. Natural responses, memory updates, controlled-probe questions and responses, and proposition-level evaluator outputs use greedy decoding. Subsequent questions are sampled with temperature 0.7 and $\mathrm { t o p } \cdot p = 0 . 9 ;$ failed answer-pair validations are retried with temperature 0.7 and $\mathrm { t o p } \cdot p = 0 . 9 5 ,$ for at most five attempts. Maximum generation lengths are 96, 128, 256, 256, and 192 tokens for questions, answers, memory updates, answer pairs, and evaluator outputs, respectively.

Valid Instances and Statistical Reporting. We retain instances where the correct response yields a clean memory state and the erroneous response successfully writes the target error, resulting in 125 valid instances for Qwen2.5-7B-Instruct and 120 for OLMo3-7B-Instruct. Each valid instance is executed under each of the four counterfactual trajectories, and reported efects are averaged over instances. Shaded regions denote 95% percentile bootstrap confidence intervals computed from 5,000 instance-level resamples. All experiments use 8 NVIDIA H200 GPUs.

## 5.1 Error Propagation and Pathway Efects

We evaluate error propagation using four complementary outcomes defined in Section 4.1: the natural-response outcome $Y _ { A }$ , memory-state outcome $Y _ { M }$ , controlled-probe outcome $Y _ { C }$ , and probability-level error preference $\dot { Y } _ { P }$ . To assess cross-model robustness, we repeat the experiment with Qwen2.5-7B-Instruct and OLMo3-7B-Instruct under identical data, memory, intervention, and evaluation settings.

As shown in Figure 2, both models exhibit strong shortrange propagation followed by gradual decay across all four outcomes. The natural-response efect decreases most rapidly: for Qwen2.5-7B-Instruct and OLMo3-7B-Instruct, the total efect falls from 0.584 and 0.642 at k = 1 to 0.040 and 0.067 at k = 12, respectively. In contrast, memory-state and controlled-probe outcomes reveal longer-lasting traces. $\mathrm { A t } \ k = 1$ , the memory-state total efect reaches 1.0 for both models, while the controlled-probe efect reaches 0.944 and 0.767, showing that, among valid injections, the error remains present in the next-round memory state and can be

![](images/29a111e65e99c76422b8c27148c468359589f92a5401564d7af67f43d8d51d1e.jpg)  
(b) OLMo3-7B-Instruct  
Figure 3: Error propagation across six memory categories for (a) Qwen2.5-7B-Instruct and (b) OLMo3-7B-Instruct. From left to right, the columns show the total propagation efect $\mathrm { T P E } _ { Y _ { C } } ( k )$ , memory-update pathway efect $\mathrm { P E } _ { M , Y _ { C } } ( k )$ , and questionfeedback pathway efect $\mathrm { P E } _ { Q , Y _ { C } } ( k )$ . Colors indicate efect magnitude under a shared scale

<table><tr><td>Comparison</td><td>Evaluation Set</td><td>Agreement</td><td>K</td></tr><tr><td>Qwen-GLM</td><td>Full (35,280)</td><td>91.9%</td><td>0.709</td></tr><tr><td>Qwen-GPT</td><td>Stratified (900)</td><td>89.3%</td><td>0.591</td></tr><tr><td>GLM-GPT</td><td>Stratified (900)</td><td>96.1%</td><td>0.805</td></tr></table>

Table 1: Binary agreement among proposition-level evaluators. GPT results are post-stratified to the full experiment.

reliably re-elicited.

The probability-level outcome provides a more finegrained view of latent propagation. At k = 1, the total efects are 2.726 for Qwen2.5-7B-Instruct and 1.646 for OLMo3- 7B-Instruct, indicating a substantial shift in relative preference toward the erroneous target. Thus, an error may disappear from natural responses while remaining stored in memory, recoverable under targeted probing, or reflected in the underlying generation distribution.

The pathway analysis shows that the memory-update pathway contributes more strongly to $Y _ { M } , \ Y _ { C } ,$ , and $Y _ { P }$ . The question-feedback pathway is generally weaker and relies more on whether later questions reactivate error-related information. The interaction efects are mostly negative or near zero. This may happen because the two pathways have overlapping downstream influences: once one pathway induces the error, redundancy or outcome saturation limits the efect of activating the other. These patterns hold across both answer models, though controlled-probe and probability-level traces generally last longer for Qwen2.5-7B-Instruct, while OLMo3-7B-Instruct shows faster attenuation.

Evaluator robustness. We assess judge dependence by reevaluating all 35,280 proposition-level outcomes with GLM-5.2 (GLM-5-Team et al. 2026) and a stratified sample of 900 outcomes with GPT-5.5 (OpenAI 2026). As shown in

Table 1, Qwen agrees strongly with both external judges. Although GLM yields lower absolute error rates than the primary evaluator, re-judging preserves the direction of all 72 TPE estimates and yields temporal-curve correlations of $r = . 9 7 5  – . 9 9 6$ . The stronger contribution of the memoryupdate pathway and the negative or near-zero interactions also remain unchanged. Thus, evaluator calibration afects efect magnitude but not our temporal or pathway-level conclusions.

## 5.2 Propagation Patterns across Memory Types

We further compare error propagation across six memory categories: Constraint, Decision, Plan, Preference, Profile, and Routine. To reduce variation caused by natural follow-up questions, we use a fixed diagnostic question that directly probes the target fact.

As shown in Figure 3, propagation varies across memory categories and answer models. For Qwen2.5-7B-Instruct, preference errors are the most persistent and receive larger contributions from the memory-update pathway, while plan and routine errors receive greater contributions from question feedback. For OLMo3-7B-Instruct, plan errors are the most persistent and are sustained by both memory retention and changes in subsequent questions. Across both models, preference errors are primarily associated with persistent memory retention, while plan errors significantly alter later interaction trajectories. Routine errors receive contributions from both pathways, while constraint, decision, and profile errors generally decay faster or fluctuate more across rounds. Overall, memory category afects both the duration and mechanism of error propagation, and cross-model diferences indicate that these patterns also depend on how each model stores and utilizes long-term information.

![](images/78a22aa75a65554107ea96975458130abde79e27ea415bb14b85268cd56a1906.jpg)  
Figure 4: Efects of the injected error on the memory-state outcome $Y _ { M } ( k )$ under diferent memory mechanisms. Red and blue curves denote MemoryBank and episodic retrieval, respectively. Line styles indicate the total propagation efect TPE, memory-update pathway efect PE<sub>M</sub>, and questionfeedback pathway efect $\mathrm { P E } _ { Q }$

## 5.3 Propagation Patterns across Memory Mechanisms

To examine whether the identified propagation patterns generalize across memory mechanisms, we compare two longterm memory architectures: continuously updated Memory-Bank (Zhong et al. 2024) and episodic retrieval with independent entries and Top-3 retrieval (Park et al. 2023; Packer et al. 2023). MemoryBank consolidates historical interactions into a dynamic memory state, whereas episodic retrieval preserves interactions as independent entries and retrieves those most relevant to the current query.

As shown in Figure 4, the two mechanisms exhibit different error-persistence patterns. Under MemoryBank, both TPE and $\mathrm { P E } _ { M }$ decrease steadily as the memory state is updated. In contrast, episodic retrieval retains the erroneous entry in memory, resulting in persistently high memory-state efects over long interaction distances. Despite the diference in persistence, both mechanisms exhibit the same pathwaylevel pattern: $\mathrm { P E } _ { M }$ accounts for most of the memory-state efect, whereas $\mathrm { P E } _ { Q }$ remains comparatively small. Thus, memory-state error persistence depends both on the errorentry pathway and the subsequent memory mechanism.

## 5.4 Can Pathway Efects Guide Error Restoration?

To examine whether pathway efects can guide efective interventions, we conduct a pathway-guided state-restoration experiment. The error is injected at turn 3, and a one-time restoration is applied before the turn at interaction distance $k = 3$ . We then evaluate the remaining propagation over subsequent interactions. We conduct this experiment using Qwen2.5-7B-Instruct with the MemoryBank-style mechanism. We consider four strategies: No Repair, Question Repair, Memory Repair, and Joint Repair. Question Repair restores the dialogue history and previous response while retaining the error-afected memory. Memory Repair restores the memory state while retaining the error-afected dialogue context. Joint Repair restores both components.

<table><tr><td>Strategy</td><td>Residual MRE↓</td><td>Reduction ↑</td></tr><tr><td>No Repair</td><td>0.178</td><td></td></tr><tr><td>Question Repair</td><td>0.129</td><td>27.5%</td></tr><tr><td>Memory Repair</td><td>0.053</td><td>70.2%</td></tr><tr><td>Joint Repair</td><td>0.003</td><td>98.3%</td></tr></table>

Table 2: Pathway-guided state-restoration results on controlled probes. Lower mean residual efect (MRE) indicates more efective restoration.

Using fixed controlled probes, we define the residual effect of strategy s at post-restoration distance j as $R _ { s } ( j ) =$ $Y _ { s } ( j ) \mathrm { ~ - ~ } Y _ { \mathrm { c l e a n } } ( j )$ , where $Y _ { s } ( j )$ and $Y _ { \mathrm { c l e a n } } ( \bar { j } )$ denote the controlled-probe outcomes under strategy s and the clean trajectory. We summarize the remaining propagation using the mean residual efect:

$$
\mathrm { M R E } ( s ) = \frac { 1 } { J } \sum _ { j = 0 } ^ { J - 1 } R _ { s } ( j ) ,
$$

where $J$ is the number of post-restoration distances. Lower MRE indicates more efective restoration. We also report the relative reduction compared with No Repair:

$$
{ \mathrm { R e d u c t i o n } } ( s ) = { \frac { \mathrm { M R E } ( { \mathrm { N o R e p a i r } } ) - \mathrm { M R E } ( s ) } { \mathrm { M R E } ( { \mathrm { N o R e p a i r } } ) } } \times 1 0 0 \%
$$

As shown in Table 2, Memory Repair reduces residual MRE by 70.2%, compared with 27.5% for Question Repair, consistent with the stronger memory-update pathway efects identified in our causal analysis. On held-out instances, larger question-feedback pathway efects are associated with greater gains from Question Repair on controlled probes (ρ = 0.335, 95% CI [0.158, 0.493]). Joint Repair further reduces residual propagation by 98.3%, demonstrating that coordinated correction of the memory state and dialogue context is most efective. Overall, pathway efects quantify error propagation and indicate which components are more likely to benefit from intervention.

## 6 Conclusion

We present a dynamic structural causal framework for characterizing cross-turn error propagation in long-term memory-augmented LLMs and decomposing its total effect into memory-update, question-feedback, and interaction components. Experiments across models, memory categories, and memory mechanisms show that errors can persist across multiple turns, with the memory-update pathway typically contributing more strongly and latent errors remaining beyond observable responses. Pathway-guided restoration further shows that repairing the memory-update pathway is more efective than repairing the error-afected questionfeedback pathway alone, while jointly repairing both pathways nearly eliminates residual propagation.

## References

Avin, C.; Shpitser, I.; and Pearl, J. 2005. Identifiability of path-specific efects.

Xu, B.; Huang, M.; Wang, H.; Li, J.; Dong, Y.; and Tang, J. 2026. GLM-5: from Vibe Coding to Agentic Engineering. arXiv:2602.15763.

Dash, P.; Ge, T.; Jain, A.; Shah, T.; and Shang, Z. 2026. From Untrusted Input to Trusted Memory: A Systematic Study of Memory Poisoning Attacks in LLM Agents. arXiv:2606.04329.

Deng, X.; Zhong, R.; Peng, H.; Lu, X.; Wu, Y.; Li, G.; Xu, B.; Yao, Y.; Fang, J.; Cao, H.; et al. 2026. MemTrace: Tracing and Attributing Errors in Large Language Model Memory Systems. arXiv preprint arXiv:2605.28732.

Hu, Y.; Wang, Y.; and McAuley, J. 2025. Evaluating memory in llm agents via incremental multi-turn interactions. arXiv preprint arXiv:2507.05257.

Kwan, W.-C.; Zeng, X.; Jiang, Y.; Wang, Y.; Li, L.; Shang, L.; Jiang, X.; Liu, Q.; and Wong, K.-F. 2024. Mt-eval: A multiturn capabilities evaluation benchmark for large language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 20153–20177.

Laban, P.; Hayashi, H.; Zhou, Y.; and Neville, J. 2025. Llms get lost in multi-turn conversation. arXiv preprint arXiv:2505.06120.

Meng, K.; Bau, D.; Andonian, A. J.; and Belinkov, Y. 2022. Locating and editing factual associations in gpt. In Advances in neural information processing systems.

Olmo, T.; Ettinger, A.; Bertsch, A.; Kuehl, B.; Graham, D.; Heineman, D.; Groeneveld, D.; Brahman, F.; Timbers, F.; Ivison, H.; Morrison, J.; Poznanski, J.; Lo, K.; Soldaini, L.; Jordan, M.; Chen, M.; Noukhovitch, M.; Lambert, N.; Walsh,

P.; Dasigi, P.; Berry, R.; Malik, S.; Shah, S.; Geng, S.; Arora, S.; Gupta, S.; Anderson, T.; Xiao, T.; Murray, T.; Romero, T.; Graf, V.; Asai, A.; Bhagia, A.; Wettig, A.; Liu, A.; Rangapur, A.; Anastasiades, C.; Huang, C.; Schwenk, D.; Trivedi, H.; Magnusson, I.; Lochner, J.; Liu, J.; Miranda, L. J. V.; Sap, M.; Morgan, M.; Schmitz, M.; Guerquin, M.; Wilson, M.; Huf, R.; Bras, R. L.; Xin, R.; Shao, R.; Skjonsberg, S.; Shen, S. Z.; Li, S. S.; Wilde, T.; Pyatkin, V.; Merrill, W.; Chang, Y.; Gu, Y.; Zeng, Z.; Sabharwal, A.; Zettlemoyer, L.; Koh, P. W.; Farhadi, A.; Smith, N. A.; and Hajishirzi, H. 2026. Olmo 3. arXiv:2512.13961.

OpenAI. 2026. GPT-5.5 Model. https://developers.openai. com/api/docs/models/gpt-5.5.

Packer, C.; Fang, V.; Patil, S.; Lin, K.; Wooders, S.; and Gonzalez, J. 2023. MemGPT: towards LLMs as operating systems.

Park, J. S.; O’Brien, J.; Cai, C. J.; Morris, M. R.; Liang, P.; and Bernstein, M. S. 2023. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th annual acm symposium on user interface software and technology, 1–22.

Pearl, J. 2009. Causal inference in statistics: An overview.

Qian, J. 2026. Visual Inception: Compromising Long-term Planning in Agentic Recommenders via Multimodal Memory Poisoning. In Liakata, M.; Moreira, V. P.; Zhang, J.; and Jurgens, D., eds., Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 20846–20862. San Diego, California, United States: Association for Computational Linguistics. ISBN 979-8-89176-390-6.

Qwen; Yang, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.; Yu, B.; Li, C.; Liu, D.; Huang, F.; Wei, H.; Lin, H.; Yang, J.; Tu, J.; Zhang, J.; Yang, J.; Yang, J.; Zhou, J.; Lin, J.; Dang, K.; Lu, K.; Bao, K.; Yang, K.; Yu, L.; Li, M.; Xue, M.; Zhang, P.; Zhu, Q.; Men, R.; Lin, R.; Li, T.; Tang, T.; Xia, T.; Ren, X.; Ren, X.; Fan, Y.; Su, Y.; Zhang, Y.; Wan, Y.; Liu, Y.; Cui, Z.; Zhang, Z.; and Qiu, Z. 2025. Qwen2.5 Technical Report. arXiv:2412.15115.

Shpitser, I.; and Tchetgen, E. T. 2014. Causal inference with a graphical hierarchy of interventions. arXiv preprint arXiv:1411.2127.

Tan, H.; Zhang, Z.; Ma, C.; Chen, X.; Dai, Q.; and Dong, Z. 2025. Membench: Towards more comprehensive evaluation on the memory of llm-based agents. In Findings of the Association for Computational Linguistics: ACL 2025, 19336–19352.

Vig, J.; Gehrmann, S.; Belinkov, Y.; Qian, S.; Nevo, D.; Singer, Y.; and Shieber, S. 2020. Investigating gender bias in language models using causal mediation analysis. Advances in neural information processing systems, 33: 12388–12401.

Wang, W.; Dong, L.; Cheng, H.; Liu, X.; Yan, X.; Gao, J.; and Wei, F. 2023. Augmenting language models with longterm memory. Advances in Neural Information Processing Systems, 36: 74530–74543.

Wu, D.; Wang, H.; Yu, W.; Zhang, Y.; Chang, K.-W.; and Yu, D. 2024. Longmemeval: Benchmarking chat as-

sistants on long-term interactive memory. arXiv preprint arXiv:2410.10813.

Xiao, S.; Wang, S.; and Chao, K. 2026. Bridging Internal Consistency and External Alignment: A Causal and Dynamic Interpretability Framework for LLM Generation. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 20378– 20392.

Xiong, Z.; Lin, Y.; Xie, W.; He, P.; Liu, Z.; Tang, J.; Lakkaraju, H.; and Xiang, Z. 2025. How memory management impacts llm agents: An empirical study of experiencefollowing behavior. arXiv preprint arXiv:2505.16067.

Zhang, D.; Lin, Y.; Wu, Z.; Sun, Y.; Li, B.; Li, D.; and Peng, H. 2026. Useful memories become faulty when continuously updated by llms. arXiv preprint arXiv:2605.12978.

Zhang, F.; and Nanda, N. 2023. Towards best practices of activation patching in language models: Metrics and methods. arXiv preprint arXiv:2309.16042.

Zhang, M.; Press, O.; Merrill, W.; Liu, A.; and Smith, N. A. 2023. How language model hallucinations can snowball. arXiv preprint arXiv:2305.13534.

Zhang, Z.; Dai, Q.; Bo, X.; Ma, C.; Li, R.; Chen, X.; Zhu, J.; Dong, Z.; and Wen, J.-R. 2025. A survey on the memory mechanism of large language model-based agents. ACM Transactions on Information Systems, 43(6): 1–47.

Zhong, W.; Guo, L.; Gao, Q.; Ye, H.; and Wang, Y. 2024. Memorybank: Enhancing large language models with longterm memory. In Proceedings of the AAAI conference on artificial intelligence, volume 38, 19724–19731.