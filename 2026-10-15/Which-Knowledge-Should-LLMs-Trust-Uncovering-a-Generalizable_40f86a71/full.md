# Which Knowledge Should LLMs Trust? Uncovering a Generalizable Steering Direction Across Knowledge Conflicts

Yibin Lou<sup>1,2,4</sup>, Yi Hu<sup>3</sup>, Jianhui Chen<sup>2</sup>, Chenpeng Wang<sup>5</sup>, Muhan Zhang<sup>3</sup>, Liangming Pan<sup>1,2</sup>\* <sup>1</sup>State Key Laboratory of Multimedia Information Processing, Peking University <sup>2</sup>School of Computer Science, Peking University <sup>3</sup>Institute for Artificial Intelligence, Peking University <sup>4</sup>Southern University of Science and Technology <sup>5</sup>YiXin-AI Lab, YIXIN, Beijing, China 12310513@mail.sustech.edu.cn

## Abstract

Large Language Models (LLMs) frequently encounter conflicts between internal parametric knowledge and external contextual knowledge, yet the mechanisms governing this selection remain unclear. We show that the choice between context and parametric knowledge is linearly represented and can be controlled along a single direction in the model’s residual stream. We first demonstrate that this direction generalizes across factual-conflict settings, including context formats, knowledge categories, and datasets, while consistently shifting behavior between parametric and contextual answers. We then transfer a direction learned from simple factual conflicts to three more complex task families: RAG, memory, and rule following, where models must acquire information from retrieved documents, interaction histories, or explicit rules. Across four model families and multiple transfer settings, steering consistently improves the use of prompt-provided information over the unsteered model. These results suggest that the direction captures a transferable preference for external knowledge rather than a mechanism specific to factual conflicts. Code is available at https://github.com/AheadOFpotato/ knowledge\_selection\_steering.

## 1 Introduction

Large Language Models (LLMs) store a vast amount of world knowledge within their parameters, commonly referred to as parametric knowledge, enabling remarkable performance across various knowledge-intensive tasks (Wang et al., 2024). Concurrently, in complex settings involving retrieved evidence, multi-turn memories, and explicit rules, LLMs must reliably acquire external contextual knowledge (Gao et al., 2024). However, these two knowledge sources often contain disparate or contradictory facts, leading to the challenge of knowledge conflict. Such conflicts frequently induce undesired behaviors, where the model indiscriminately relies on inaccurate sources and thus produces incorrect outputs (Xu et al., 2024; Cheng et al., 2024). Consequently, understanding the internal mechanism of how LLMs choose between conflicting knowledge sources, and achieving reliable control over this knowledge selection process, has become a critical open problem (Zhao et al., 2025b; Jin et al., 2024).

![](images/8b5c5973796391fd618ab419b17752d3e88c8b87196739a5b6d399956b1b7d09.jpg)  
Figure 1: Under knowledge conflicts, steering along a single residual-stream direction smoothly shifts outputs between parametric knowledge and contextual knowledge, with intermediate mixed or hesitant responses.

Existing approaches address knowledge conflict primarily from two directions. One line of work controls knowledge source choice at the input level, which can be categorized into two main approaches: explicit prompt or instruction design (Choi et al., 2025; Merth et al., 2024), and structured RAG templates that organize retrieved context into designated slots or label relevance (Cattan et al., 2025). While straightforward to implement, these methods provide limited visibility into the model’s internal arbitration process and offer only indirect control over the reliance on context versus parameters. A complementary line intervenes on internal components, including manipulating attention (Jin et al., 2024; Li et al., 2025), steering sparse autoencoder (SAE) features (Zhao et al., 2025a), and modifying decoding-time token distributions (Bi et al., 2025; Wang et al., 2025a). These methods are primarily evaluated on simple factual datasets, without testing knowledge conflicts involving retrieved evidence, long interaction histories, or explicit rules. They also incur the cost of task-specific attribution, feature search, adaptation, or decoding-time modification, while lacking robust evaluation across such complex application settings.

These limitations motivate us to ask whether internal knowledge arbitration admits a simple, generalizable control mechanism. A growing body of literature in mechanistic interpretability suggests that LLMs represent complex high-level concepts, such as truthfulness (Li et al., 2024), safety (Wolf et al., 2025; Zou et al., 2025), emotions (Sofroniew et al., 2026), or refusal behaviors (Arditi et al., 2024), as linear directions within their activation space (Turner et al., 2024; Elhage et al., 2022; Park et al., 2024). These directions serve as causal mediators that allow fine-grained output steering. This perspective motivates the following research question: Can knowledge selection under conflict be captured and controlled by a low-dimensional steering axis that generalizes across tasks?

In this work, we demonstrate that a robust “knowledge selection direction” exists. Specifically, we extract this vector by contrasting the hidden states of paired prompts designed to isolate parametric reliance from contextual reliance. By identifying the critical intervention layer through cosine similarity analysis, we apply signed activation steering exclusively at this layer to modulate the model’s knowledge selection.

A direction demonstrated on one factual-conflict setting may reflect a local shortcut rather than a reusable control mechanism. We evaluate across four architectures and a progression from controlled factual conflicts to complex tasks. Within the controlled setting, we vary factual format, relation category, and dataset while retaining short prompts with competing answers. We then apply the direction learned from these conflicts to RAG, memory, and rule-following tasks, where retrieved documents, multi-turn histories, and naturallanguage rules supply information to be acquired. Together, these experiments test whether different tasks share a common direction for acquiring inputprovided knowledge. Our contributions are:

• We identify a single residual-stream direction on controlled factual-conflict benchmarks that produces bidirectional, continuous control of knowledge selection across four model families.

• We demonstrate that the same direction works across diverse knowledge-conflict settings, including changes in context format, knowledge category, and factual-conflict dataset.

• More importantly, we show that it improves complex RAG, memory, and rule-following tasks. These improvements suggest that different task families may share an underlying direction for acquiring external knowledge.

## 2 Related Work

Knowledge conflict. Knowledge conflict arises when parametric and contextual knowledge disagree (Xu et al., 2024; Mallen et al., 2023). Existing interventions target attention heads (Jin et al., 2024; Li et al., 2025), SAE features (Zhao et al., 2025a), or decoding (Bi et al., 2025; Wang et al., 2025a). Our clean–conflict contrast instead models explicit source competition and enables bidirectional knowledge selection.

Residual Geometry of Steering Vectors. Localization-plus-intervention studies trace model behaviors to layers, heads, or neurons (Meng et al., 2023; Geva et al., 2023), while activation steering modifies hidden states at inference time (Turner et al., 2024; Panickssery et al., 2024). These approaches build on the view that high-level concepts can correspond to approximately linear directions in representation space (Marks and Tegmark, 2024; Park et al., 2024). Closely related to our setting, (Zhang and Lin, 2026) argue that in-context conflict resolution may involve angular displacement in the residual stream rather than simple magnitude dilution. Motivated by this geometric view, we localize layers by angular separation between paired clean and conflict prompts, and then steer along a difference-in-means direction.

Transfer of steering directions. Another question is whether a direction learned from one concept captures a local shortcut or a general behavioral axis. Prior work shows truthfulness-related directions can generalize across datasets and causally affect outputs (Marks and Tegmark, 2024), but that steering vectors can be brittle under distribution shift (Tan et al., 2025). Recent cross-model studies suggest that concept directions can be aligned across model families (Huang et al., 2025). Closest steering methods increase reliance on input information through layer search (Anand et al., 2026), proxy models (Wang et al., 2025b), contextsensitivity directions (Minder et al., 2025), or instruction vectors (Stolfo et al., 2025). We test a stricter transfer: whether a direction estimated from explicit source competition generalizes to other formats, properties, and benchmarks while retaining bidirectional control.

![](images/e234b0ee71377e77d55c0f212620c1c067e7c376d1cb431014d99648ec69b9d6.jpg)  
Figure 2: Overview of knowledge-selection steering. For each question $q ,$ the clean prompt is q alone and the conflict prompt is the supplied passage followed by q. We localize the layer ℓ of maximal angular separation between their mean residual activations, estimate direction v by a difference-in-means contrast at ℓ, and steer the conflict prompt at its own final token with $h ^ { \prime } = h + \alpha v$ to trade parameter-stored versus context-embedded answers.

## 3 Methodology

## 3.1 Background

Transformers. We study decoder-only Transformers. Given input of N tokens $t \_ =$ $( t _ { 1 } , t _ { 2 } , \ldots , t _ { N } )$ , the model maps them to hidden states across L layers. Let $h ^ { \bar { l } } \_ i \in \mathbb { R } ^ { d _ { - } \mathrm { m o d e l } }$ denote residual stream at layer l and token position i. The residual stream is initialized by token embedding, $h ^ { 1 } \_ i = \mathrm { E m b e d } ( t _ { i } )$ , and is iteratively updated through self-attention and MLP blocks:

$$
\tilde { h } _ { i } ^ { l } = h _ { i } ^ { l } + \mathrm { A t t n } ^ { l } ( h _ { 1 : i } ^ { l } ) , \quad h _ { i } ^ { l + 1 } = \tilde { h } _ { i } ^ { l } + \mathrm { M L P } ^ { l } ( \tilde { h } _ { i } ^ { l } ) .\tag{1}
$$

The final output probabilities are decoded from the hidden states at the last layer via the unembedding matrix and a softmax function.

## 3.2 Models and Datasets

Models. We evaluate four instruction-tuned models: Llama-3.1-8B-Instruct, Qwen3-8B (Team, 2025), Gemma-2-9B-it (Team, 2024), and Mistral-7B-Instruct-v0.3 (Jiang et al., 2023).

Benchmarks. We evaluate steering on five benchmarks spanning factual conflict, conflict format, knowledge type, and memorization fallback: NQ-Swap (Longpre et al., 2022), MacNoise (Hong et al., 2024), ParaConflict (Li et al., 2025), Counter-Fact (Meng et al., 2023), and MemoTrap (McKenzie et al., 2022). NQ-Swap and MacNoise serve as the primary settings, while ParaConflict, CounterFact, and MemoTrap test transfer across conflict formats, factual relations, and memorizationbased behavior, respectively. Appendix A gives the dataset construction and splits.

To evaluate transfer beyond simple conflicts, we further consider three complex task families: RAG (RGB (Chen et al., 2023); CRAG (Yang et al., 2024)), memory (LoCoMo, LoCoMo-Plus (Maharana et al., 2024); LongMemEval (Wu et al., 2025)), and rule following (IFEval (Zhou et al., 2023); FollowBench (Jiang et al., 2024)). In these tasks, retrieved documents, interaction histories, and explicit instructions provide external information that the model must correctly incorporate.

What is the capital of France?

## 3.3 Extracting a Knowledge Selection Direction

Paired prompts. As the first step in Figure 2, for each question q, we construct two prompts: a clean prompt, which consists of $q$ alone, and a conflict prompt, which combines $q$ with a passage inducing knowledge conflict.

For example, the clean prompt is:

The conflict prompt concatenates the conflicting passage with the question:

The capital ofFrance is Lyon. What is the capital   
of France?

We retain items answered correctly on the clean question and four paraphrases, but answered with the injected alternative under conflict, yielding examples with reliable parametric knowledge and demonstrated context acquisition.

Layer localization. We localize the intervention layer by angular separation between clean and conflict representations, rather than by a performance sweep. We run both prompts through the model and average activations, yielding $\bar { h } _ { T _ { \mathrm { c l e a n } } } ^ { ( \mathrm { c l e a n } ) }$ and $\bar { h } _ { T _ { \mathrm { c o n f i c t } } } ^ { \mathrm { ( c o n f l i c t ) } i }$ , where the final token positions may differ because the conflict passage changes the prompt length. At each layer i, we compute

$$
s _ { T _ { \mathrm { c l e a n } } , T _ { \mathrm { c o n f i c t } } } ^ { i } = \frac { \bar { h } _ { T _ { \mathrm { c l e a n } } } ^ { ( \mathrm { c l e a n } ) i } \cdot \bar { h } _ { T _ { \mathrm { c o n f i c t } } } ^ { ( \mathrm { c o n f i c t } ) i } } { \| \bar { h } _ { T _ { \mathrm { c l e a n } } } ^ { ( \mathrm { c l e a n } ) i } \| \ \| \bar { h } _ { T _ { \mathrm { c o n f i c t } } } ^ { ( \mathrm { c o n f i c t } ) i } \| } .\tag{2}
$$

where $s _ { T _ { \mathrm { c l e a n } } , T _ { \mathrm { c o n f l i c t } } } ^ { i }$ is the cosine similarity between the two final positions. We select the layer with the lowest similarity:

$$
\ell = \arg \operatorname* { m i n } _ { i } s _ { T _ { \mathrm { c l e a n } } , T _ { \mathrm { c o n f l i c t } } } ^ { i } .\tag{3}
$$

This criterion requires no sweep over downstream task scores. The final layer is also excluded when its separation reflects the already-decoded output distribution rather than a reusable sourceselection representation; only an earlier angular minimum is eligible for intervention.

The layer-wise curves in Appendix B.2 verify that the selected layer is the point of greatest clean– conflict separation across the evaluated models.

Extracting the direction. At the selected layer ℓ, we define the steering direction as a differencein-means contrast between the conflict and clean activations at the last token:

$$
v _ { T _ { \mathrm { c l e a n } } , T _ { \mathrm { c o n f i c t } } } ^ { \ell } = \bar { h } _ { T _ { \mathrm { c l e a n } } } ^ { ( \mathrm { c l e a n } ) \ell } - \bar { h } _ { T _ { \mathrm { c o n f i c t } } } ^ { ( \mathrm { c o n f i c t } ) \ell } .\tag{4}
$$

Thus $v _ { \ell , T _ { \mathrm { c l e a n } } , T _ { \mathrm { c o n f l i c t } } }$ points from the conflict regime toward stronger parameter-stored reliance.

## 3.4 Applying the Steering Vector

Inference-time intervention. We edit the resid post stream only at the angular-separation-selected layer ℓ and the conflict prompt’s final token. Appendix B reports representation-hook, layer, and token-position ablations. Let $h _ { T _ { \mathrm { c o n f l i c t } } } ^ { \ell }$ denote this activation. We apply the steering vector as

$$
h _ { \ T _ { \mathrm { c o n f l i c t } } } ^ { \prime \ell } \ = \ h _ { T _ { \mathrm { c o n f l i c t } } } ^ { \ell } + \alpha v _ { T _ { \mathrm { c l e a n } } , T _ { \mathrm { c o n f i c t } } } ^ { \ell } .\tag{5}
$$

Here, α is a signed control coefficient: positive values favor parametric reliance, negative values favor contextual reliance, and $\alpha = 0$ recovers the unsteered conflict pass.

## 3.5 Evaluation

We measure steering by matching the first 256 generated tokens to the parametric and contextual answers. Each item receives one of four labels: parametric-only, contextual-only, both, or \*neither\*. Let N be the number of evaluated items; the reported unconditional rates are

$$
\begin{array} { r } { \mathrm { A c c } _ { \mathrm { p a r } } = \frac { N _ { \mathrm { p a r a m e t r i c - o n l y } } } { N } , } \\ { \mathrm { A c c } _ { \mathrm { c t x } } = \frac { N _ { \mathrm { c o n t e x t u a l - o n l y } } } { N } . } \end{array}
$$

The main table retains these two directional rates. Appendix C details normalization, the fixed answer window, the match rules, and the blinded-judge validation of the inexpensive string-match metric. We use train/validation/test splits for direction estimation, coefficient selection, and final reporting, with a 70/15/15 split for NQ-Swap.

## 4 Continuous Control Along a Single Direction

We first evaluate whether the model’s knowledge selection behavior can be captured and continuously modulated along a single steering direction v. Here, the scalar α serves as a continuous control coefficient, with positive values shifting the model toward parametric reliance and negative values toward contextual reliance. Implementation details are provided in Appendix D.

Table 1: In-domain source-match rate (%) on NQ-Swap and MacNoise. The upper block reports parametric-source match rate $\mathbf { A c c _ { p a r } } ,$ , and the lower block reports contextual-source match rate $\operatorname { A c c } _ { \operatorname { c t x } } .$ . Bold marks the best result in each column; green arrows denote absolute gains over the unsteered baseline. Ours and Random split use the same validation-selected coefficient. Random split values are mean ± standard deviation over 10 seeds.
<table><tr><td rowspan=1 colspan=8>NQ-Swap                                    MacNoiseMethodLlama-3.1               Gemma-2  Mistral-7B  Llama-3.1               Gemma-2  Mistral-7B8B-Instruct  Qwen3-8B    9B-IT   Instruct-v0.3 8B-Instruct  Qwen3-8B    9B-IT   Instruct-v0.3Steer toward parametric knowledge (Accpar)</td></tr><tr><td rowspan=1 colspan=8>Unsteered                           33.7      46.5      32.8      42.2      46.1      50.8      45.5      50.3</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=7>32.2 ± 2.6 45.7 ± 1.9 33.5 ± 2.3 41.6 ± 2.1 45.8 ± 3.1 50.2 ± 2.4 44.9 ± 2.0 49.7 ± 2.5ContextFocus (Anand et al., 2026) 44.6 (↑10.9) 49.1 (↑2.6) 45.2 (↑12.4) 47.7 (↑5.5) 50.2 (↑4.1) 52.5 (↑1.7) 51.3 (↑5.8) 54.2 (↑3.9)CCS (Minder et al., 2025)        53.5 (↑19.8) 50.8 (↑4.3) 58.6 (↑25.8) 52.1 (↑9.9) 55.2 (↑9.1) 53.1 (↑2.3) 59.4 (↑13.9) 56.8 (↑6.5)</td></tr><tr><td rowspan=1 colspan=8>CSKS (Wang et al., 2025b)       52.4 (↑18.7) 51.7 (↑5.2) 66.3 (↑33.5) 54.3 (↑12.1) 53.7 (↑7.6) 53.8 (↑3.0) 63.6 (↑18.1) 58.4 (↑8.1)CK-PLUG (Zhao et al., 2025a)   54.5 (↑20.8) 52.9 (↑6.4) 74.7 (↑41.9) 56.2 (↑14.0) 55.7 (↑9.6) 53.4 (↑2.6) 69.5 (↑24.0) 59.7 (↑9.4)PH3 (Li et al., 2025)              48.3 (↑14.6) 47.8 (↑1.3) 61.5 (↑28.7) 49.5 (↑7.3) 49.5 (↑3.4) 51.7 (↑0.9) 54.7 (↑9.2) 52.6 (↑2.3)IRCAN (Shi et al., 2024)         49.5 (↑15.8) 51.4 (↑4.9) 59.3 (↑26.5) 50.8 (↑8.6) 50.0 (↑3.9) 52.4 (↑1.6) 63.2 (↑17.7) 53.9 (↑3.6)</td></tr><tr><td rowspan=1 colspan=8>JUICE (Jin et al., 2024)           53.4 (↑19.7) 53.3 (↑6.8) 78.4 (↑45.6) 57.1 (↑14.9) 53.2 (↑7.1) 56.1 (↑5.3) 74.2 (↑28.7) 60.5 (↑10.2)</td></tr><tr><td rowspan=1 colspan=8>SpARE (Zhao et al., 2025a)      58.7 (↑25.0) 49.2 (↑2.7) 77.3 (↑44.5) 55.4 (↑13.2) 59.5 (↑13.4) 52.3 (↑1.5) 72.3 (↑26.8) 57.6 (↑7.3)</td></tr><tr><td rowspan=1 colspan=8>Ours                             56.8 (↑23.1) 54.6 (↑8.1) 80.9 (↑48.1) 59.0(↑16.8) 57.5 (↑11.4) 54.5 (↑3.7) 76.9（↑31.4) 61.6 (↑11.3)</td></tr><tr><td rowspan=1 colspan=8>Steer toward contextual knowledge (Accctx)</td></tr><tr><td rowspan=1 colspan=4>Unsteered                           49.4      36.7      52.6      45.8</td><td rowspan=1 colspan=4>27.3      22.2      25.4      23.8</td></tr><tr><td rowspan=1 colspan=4>Random split                     49.2 ± 2.8 37.1 ± 2.2 51.9 ± 2.6 45.3 ± 1.8 2</td><td rowspan=1 colspan=1>6.1 ± 0.7 2</td><td rowspan=1 colspan=1>1.8 ± 1.4</td><td rowspan=1 colspan=1>24.9 ± 1.6 2</td><td rowspan=1 colspan=1>3.4 ± 1.3</td></tr><tr><td rowspan=1 colspan=4>ContextFocus (Anand et al., 2026) 56.8 (↑7.4) 42.3 (↑5.6) 58.1 (↑5.5) 51.2 (↑5.4)</td><td rowspan=1 colspan=1>31.5 (↑4.2)</td><td rowspan=1 colspan=1>25.7 (↑3.5)</td><td rowspan=1 colspan=1>29.3 (↑3.9) 2</td><td rowspan=1 colspan=1>7.1 (↑3.3)</td></tr><tr><td rowspan=1 colspan=4>CCS (Minder et al., 2025)         52.3 (↑2.9) 46.1 (↑9.4) 60.7 (↑8.1) 54.6 (↑8.8) 3</td><td rowspan=1 colspan=1>0.1 (↑2.8) 2</td><td rowspan=1 colspan=1>7.4 (↑5.2) 3</td><td rowspan=1 colspan=1>2.5 (↑7.1) 2</td><td rowspan=1 colspan=1>8.6 (↑4.8)</td></tr><tr><td rowspan=2 colspan=4>CSKS (Wang et al., 2025b)       57.2 (↑7.8) 49.7 (↑13.0) 62.9 (↑10.3) 56.8 (↑11.0)CK-PLUG (Zhao et al., 2025a)    53.4 (↑4.0) 55.2 (↑18.5) 64.3 (↑11.7) 57.7 (↑11.9)</td><td rowspan=1 colspan=1>32.2 (↑4.9)</td><td rowspan=1 colspan=1>30.1 (↑7.9) 3</td><td rowspan=1 colspan=1>5.4 (↑10.0)</td><td rowspan=1 colspan=1>29.7 (↑5.9)</td></tr><tr><td rowspan=1 colspan=1>30.9 (↑3.6)</td><td rowspan=1 colspan=1>29.5 (↑7.3) 3</td><td rowspan=1 colspan=1>7.7 (↑12.3) 3</td><td rowspan=1 colspan=1>0.2 (↑6.4)</td></tr><tr><td rowspan=1 colspan=4>PH3 (Li et al., 2025)              52.4 (↑3.0) 51.5 (↑14.8) 54.3 (↑1.7) 49.1 (↑3.3)</td><td rowspan=1 colspan=1>28.9 (↑1.6)</td><td rowspan=1 colspan=1>28.5 (↑6.3)</td><td rowspan=1 colspan=1>32.4 (↑7.0) 2</td><td rowspan=1 colspan=1>5.7 (↑1.9)</td></tr><tr><td rowspan=1 colspan=2>IRCAN (Shi et al., 2024)          55.2 (↑5.8) 4</td><td rowspan=1 colspan=1>7.4 (↑10.7) 57.3 (↑4.7)</td><td rowspan=1 colspan=1>52.4 (↑6.6) 3</td><td rowspan=1 colspan=1>1.7 (↑4.4) 32</td><td rowspan=1 colspan=1>.4 (↑10.2) 3</td><td rowspan=1 colspan=1>6.1 (↑10.7) 2</td><td rowspan=1 colspan=1>8.8 (↑5.0)</td></tr><tr><td rowspan=1 colspan=2>JUICE (Jin et al., 2024)           54.5 (↑5.1) 4</td><td rowspan=1 colspan=2>8.4 (↑11.7) 66.9 (↑14.3) 58.3 (↑12.5)</td><td rowspan=1 colspan=1>34.0 (↑6.7) 3</td><td rowspan=1 colspan=1>4.2 (↑12.0)</td><td rowspan=1 colspan=1>38.5 (↑13.1)</td><td rowspan=1 colspan=1>30.4 (↑6.6)</td></tr><tr><td rowspan=1 colspan=2>SpARE (Zhao et al., 2025a)       52.7 (↑3.3) 5</td><td rowspan=1 colspan=2>4.4 (↑17.7) 68.3 (↑15.7) 55.9 (↑10.1)</td><td rowspan=1 colspan=1>30.9 (↑3.6) 3</td><td rowspan=1 colspan=1>4.3 (↑12.1)</td><td rowspan=1 colspan=1>42.4 (↑17.0)</td><td rowspan=1 colspan=1>29.2 (↑5.4)</td></tr><tr><td rowspan=1 colspan=4>Ours                             55.5 (↑6.1) 56.0 (↑19.3) 65.4 (↑12.8) 60.2 (↑14.4)</td><td rowspan=1 colspan=1>32.8 (↑5.5) 3</td><td rowspan=1 colspan=1>5.1 (↑12.9)</td><td rowspan=1 colspan=2>40.6 (↑15.2) 31.1 (↑7.3)</td></tr></table>

## 4.1 Improving Binary Source Choice

Table 1 first shows that Ours improves the desired source-selection rate over the unsteered model across the four models and both benchmarks, establishing control at a validation-selected operating point. The steering-strength sensitivity analysis in Appendix E further shows stable transfer performance at neighboring α values around each selected operating point. Beyond these discrete endpoints, sweeping α reveals how the same direction changes source selection continuously.

## 4.2 Continuous Control at Population and Instance Levels

Having established discrete source control at a validation-selected operating point, we next examine whether the same direction supports a continuous transition as α varies.

At the population level, Figure 3 shows smooth, opposing changes in parametric and contextual selection as α varies across models and datasets. This aggregate trade-off is also visible within individual fixed examples, and Appendix F shows that it extends to larger instruction-tuned models.

At the instance level, Figure 1 traces a response from one source through mixed outputs to the other as α changes. Thus, the aggregate trend reflects graded individual source selection rather than only a shift in proportions. Appendix G.1 provides trajectories for every model and cases that do not reach the requested source.

## 4.3 Robustness to External-Knowledge Strength and Question Difficulty

We test two factors: external-knowledge strength and question difficulty. As Table 2 shows, stronger contextual evidence raises unsteered contextual selection, while steering remains effective. Table 3 shows gains at every difficulty level, with the largest parametric gain on hard questions.

![](images/29f2b2dd91e6463549f120f70aff604926ceb91ea9c0098786afaa1590534981.jpg)  
Figure 3: Bidirectional continuous modulation on NQ-Swap and MacNoise (15% test split). We sweep the full α grid directly on test items and plot aggregate source-match proportions at each α; the curves describe population-level averages and do not imply that every individual instance changes smoothly (no α tuning on test).

Table 2: Robustness to external-knowledge strength on NQ-Swap with Llama-3.1-8B-Instruct (%).
<table><tr><td>Context occurrences</td><td>Unsteered parametric</td><td>Parametric steering</td><td>Unsteered contextual</td><td>Contextual steering</td></tr><tr><td>1</td><td>33.7</td><td>56.8</td><td>49.4</td><td>55.5</td></tr><tr><td>2</td><td>30.5</td><td>55.2</td><td>54.2</td><td>58.1</td></tr><tr><td>4</td><td>25.4</td><td>53.6</td><td>61.8</td><td>62.4</td></tr><tr><td>8</td><td>17.1</td><td>47.9</td><td>73.5</td><td>74.8</td></tr></table>

Table 3: Robustness to question difficulty on NQ-Swap with Llama-3.1-8B-Instruct (%).
<table><tr><td>Difficulty</td><td>Unsteered parametric</td><td>Parametric steering</td><td>Unsteered contextual</td><td>Contextual steering</td></tr><tr><td>Easy</td><td>42.6</td><td>62.4</td><td>45.8</td><td>56.2</td></tr><tr><td>Medium</td><td>33.1</td><td>56.1</td><td>49.6</td><td>57.3</td></tr><tr><td>Hard</td><td>12.4</td><td>49.8</td><td>64.2</td><td>74.3</td></tr></table>

## 4.4 Knowledge Selection Is More Than Context Sensitivity

The preceding results establish source control, but not whether it reflects explicit knowledge conflict or generic context sensitivity. We therefore compare source competition with matched prompts that add answer-aligned or irrelevant context without the same parametric–contextual conflict.

Table 4 compares no-context contrasts with conflicting, aligned, and irrelevant contexts. The conflicting-context contrast yields the strongest source-selection effects across datasets, whereas matched controls are weaker, indicating more than generic context sensitivity; Appendix B.3 further shows that the directions are similar but not identical (cosine 0.875–0.899).

Table 4: Matched controls for knowledge selection versus generic context sensitivity (%). The three steered columns contrast no context with conflicting, aligned, and irrelevant context.

<table><tr><td>Dataset</td><td>Unsteered</td><td>Conflicting Aligned context</td><td>context</td><td>Irrelevant context</td></tr><tr><td colspan="5">Panel A: Parametric-answer selection (positive α)</td></tr><tr><td>ParaConflict</td><td>15.9</td><td>86.5</td><td>72.6</td><td>74.0</td></tr><tr><td>MacNoise</td><td>46.2</td><td>73.5</td><td>67.1</td><td>62.1</td></tr><tr><td>MemoTrap</td><td>67.1</td><td>84.1</td><td>71.7</td><td>75.3</td></tr><tr><td>CounterFact</td><td>36.0</td><td>54.4</td><td>46.1</td><td>49.1</td></tr><tr><td colspan="5">Panel B: Context/distractor-answer selection (negative α)</td></tr><tr><td>ParaConflict</td><td>81.7</td><td>96.5</td><td>84.4</td><td>83.2</td></tr><tr><td>MacNoise</td><td>27.4</td><td>44.5</td><td>31.2</td><td>34.0</td></tr><tr><td>MemoTrap</td><td>29.7</td><td>57.0</td><td>49.9</td><td>47.5</td></tr><tr><td>CounterFact</td><td>60.8</td><td>64.1</td><td>57.6</td><td>55.1</td></tr></table>

## 4.5 Preserving Performance Without Conflict

Finally, we test the cost of this source preference when no conflict is present. Table 5 shows that, with the same fixed directions and coefficients, both directions retain substantial original-answer accuracy across all four datasets, although contextual steering incurs a larger trade-off on some datasets.

## 5 Generalization of the Knowledge-Selection Direction

We next test whether this fixed direction survives distribution shifts. Section 5.1 tests simple factualconflict tasks; Section 5.2 turns to complex tasks with external knowledge to acquire.

## 5.1 Across Controlled Simple Factual-Conflict Tasks

These tests vary format, semantic relation, or benchmark construction while retaining factual conflicts: prompts with competing answers. Thus, this level establishes generalization within the task family while retaining bidirectional steering.

![](images/02f42be52e0806fa19dda0d5257fb9fa5f11ec92c9ed22e2ccd910135093b3aa.jpg)  
Figure 5: Controlled cross-relation transfer on CounterFact (Llama-3.1-8B-Instruct). Each cell evaluates a direction learned from one factual relation on another relation.

Table 5: Preservation of the original answer without a conflicting context. Values are target-answer accuracy (%); the same fixed parametric and contextual steering directions used in transfer are applied.
<table><tr><td>Dataset</td><td>Unsteered</td><td>Parametric direction</td><td>Contextual direction</td></tr><tr><td>ParaConflict</td><td></td><td>98.0</td><td>85.5</td></tr><tr><td>MacNoise</td><td>100.0 100.0</td><td>80.8</td><td>86.7</td></tr><tr><td>MemoTrap</td><td>100.0</td><td>84.2</td><td>91.1</td></tr><tr><td>CounterFact</td><td>100.0</td><td>88.8</td><td>91.7</td></tr></table>

Cross-format transfer. We begin by varying the conflict format while holding the fact and distractor fixed. ParaConflict separates a short substitution conflict from a richer coherent conflict for the same fact; Appendix H gives the full prompt forms. Both prompts inject the same distractor, “football,” for the correct answer “basketball”:

Table 6: Controlled factual-conflict cross-dataset transfer on Llama-3.1-8B-Instruct (%). The direction and coefficients are selected only on NQ-Swap.
<table><tr><td></td><td colspan="2">Parametric steering</td><td colspan="2">Contextual steering</td></tr><tr><td>Test data</td><td>Unsteered</td><td>Ours</td><td>Unsteered</td><td>Ours</td></tr><tr><td>MacNoise</td><td>46.2</td><td>60.0</td><td>27.0</td><td>34.1</td></tr><tr><td>ParaConflict</td><td>15.8</td><td>83.9</td><td>81.4</td><td>90.8</td></tr><tr><td>CounterFact</td><td>35.7</td><td>38.9</td><td>60.0</td><td>60.7</td></tr><tr><td>MemoTrap</td><td>66.7</td><td>75.4</td><td>30.0</td><td>53.3</td></tr></table>

![](images/7eb1991ec50da51e9014e3da6c4b9fb8aca2865af05118b734c9354e2938e771.jpg)  
Figure 5: Controlled cross-format transfer on ParaConflict (Llama-3.1-8B-Instruct). The direction is fit on substitution-format conflicts and evaluated on coherentformat test items.

Coherent: Recognized by peers andfans alike, Michael Jordan’s journey in football has been highlighted in sports publications and athlete profiles. . . . The impact of Michael Jordan in football is celebrated in community events and athletic programs. . . .

Figure 5 shows that a direction fitted on the short format also changes source selection in the coherent format, so the effect is not tied to the estimation template. Appendix I.1 reports the corresponding results for the other models.

Cross-relation transfer. Having shown that the direction is not tied to one prompt template, we next hold the conflict structure fixed and vary the semantic relation. CounterFact lets us learn directions from country of citizenship, manufacturer, occupation, religion, sport, and original broadcaster, then evaluate each direction on the other relations.

Short: Michael Jordan plays the sport of football.

Table 7: Generalization to complex tasks through external-knowledge steering (%). Models are grouped vertically and tasks horizontally. ContextFocus, JUICE, and CK-PLUG are included as additional Llama-3.1-8B-Instruct baselines.
<table><tr><td>Model</td><td>Method</td><td colspan="2">RAG</td><td colspan="3">Memory</td><td colspan="2">Rule</td></tr><tr><td></td><td></td><td>RGB</td><td>CRAG</td><td>LoCoMo</td><td>LoCoMo-Plus</td><td>LongMemEval</td><td>IFEval</td><td>FollowBench</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>Unsteered</td><td>62.0±1.2</td><td>23.7±0.8</td><td>8.5±0.4</td><td>32.7±1.1</td><td> $4 3 . 2 { \pm } 0 . 9 $ </td><td>33.6±1.0</td><td>14.5±0.6</td></tr><tr><td></td><td>Random split</td><td> $6 4 . 0 { \pm } 1 . 4 $ </td><td> $2 4 . 7 { \pm } 0 . 9$ </td><td>8.5±0.4</td><td> $3 2 . 7 { \pm } 1 . 0 $ </td><td> $4 2 . 8 { \pm } 0 . 8 $ </td><td> $3 3 . 9 { \pm } 1 . 1 $ </td><td> $1 2 . 9 { \pm } 0 . 5 $ </td></tr><tr><td></td><td>ContextFocus</td><td> $6 9 . 0 { \pm } 0 . 3 $ </td><td> $2 4 . 5 { \pm } 1 . 8 $ </td><td> $1 0 . 6 { \pm } 1 . 8 $ </td><td> $3 4 . 9 { \pm } 2 . 1 $ </td><td> $4 4 . 1 \pm 0 . 9$ </td><td> $3 3 . 4 \pm 1 . 7$ </td><td> $1 3 . 1 { \pm } 1 . 3 $ </td></tr><tr><td></td><td>JUICE</td><td> $6 2 . 0 { \pm } 1 . 1 $ </td><td> $2 3 . 7 { \pm } 0 . 7 $ </td><td> $7 . 6 { \pm } 0 . 3 $ </td><td> $3 4 . 2 \pm 1 . 9$ </td><td> $4 2 . 5 { \pm } 2 . 8 $ </td><td> $3 1 . 0 { \pm } 0 . 9 \ $ </td><td>13.8±1.4</td></tr><tr><td></td><td>CK-PLUG</td><td> $7 1 . 0 { \pm } 1 . 4 $ </td><td> $2 5 . 1 { \pm } 0 . 9 $ </td><td> $1 2 . 9 { \pm } 1 . 6 $ </td><td> $3 5 . 3 { \pm } 1 . 2 $ </td><td> $4 6 . 8 \pm 1 . 0$ </td><td> $3 4 . 1 { \pm } 1 . 1$ </td><td>14.6±0.6</td></tr><tr><td></td><td>Ours</td><td>82.0±0.5</td><td>25.8±1.0</td><td>13.2±0.9</td><td> ${ \bf 3 6 . 7 \pm 1 . 3 }$ </td><td> $4 5 . 4 \pm 1 . 1$ </td><td>35.0±1.2</td><td>14.9±0.7</td></tr><tr><td>Qwen3-8B</td><td>Unsteered</td><td>72.0±1.4</td><td>27.5±1.0</td><td>9.8±0.5</td><td>28.2±0.9</td><td> $4 2 . 8 { \pm } 0 . 9 \ $ </td><td>34.9±1.1</td><td>15.7±0.5</td></tr><tr><td></td><td>Random split</td><td>72.0±1.1</td><td>27.5±0.5</td><td>10.1±0.5</td><td>28.5±1.0</td><td> $4 2 . 1 { \pm } 0 . 4 $ </td><td>34.9±1.0</td><td>15.7±1.3</td></tr><tr><td></td><td>ContextFocus</td><td> $7 2 . 1 { \pm } 0 . 8 $ </td><td>27.6±1.3</td><td> $9 . 9 { \pm } 0 . 6 $ </td><td> $2 8 . 9 { \pm } 1 . 1 $ </td><td> $4 2 . 9 { \pm } 0 . 4 $ </td><td>34.8±1.1</td><td>16.6±1.9</td></tr><tr><td></td><td>JUICE</td><td> $6 8 . 5 { \pm } 1 . 2 $ </td><td>25.4±1.8</td><td> $9 . 0 { \pm } 0 . 4 $ </td><td>26.8±0.8</td><td> $4 1 . 6 { \pm } 0 . 8 $ </td><td> $3 3 . 1 \pm 0 . 9$ </td><td>14.2±1.2</td></tr><tr><td></td><td>CK-PLUG</td><td> $7 4 . 3 { \pm } 0 . 5 $ </td><td>28.3±1.1</td><td> $1 0 . 6 { \pm } 0 . 6 $ </td><td> $3 3 . 5 { \pm } 1 . 2 $ </td><td> $4 4 . 1 \pm 1 . 0$ </td><td> $3 5 . 4 \pm 1 . 2$ </td><td>16.6±0.8</td></tr><tr><td></td><td>Ours</td><td>79.0±1.6</td><td>30.0±1.2</td><td>13.1±0.7</td><td> $4 5 . 4 \pm 1 . 5$ </td><td> $4 3 . 8 \pm 1 . 1$ </td><td> ${ \bf 3 6 . 2 \pm 1 . 3 }$ </td><td>18.3±0.9</td></tr><tr><td>Gemma-2-9B-IT</td><td>Unsteered</td><td>71.0±1.3</td><td>26.6±0.9</td><td>8.8±0.4</td><td>9.2±0.3</td><td>42.0±0.7</td><td>29.2±0.9</td><td>16.1±0.1</td></tr><tr><td></td><td>Random split</td><td>72.0±0.7</td><td>26.9±0.9</td><td>8.8±0.5</td><td>11.0±0.4</td><td> $4 1 . 7 { \pm } 0 . 9$ </td><td>29.8±1.2</td><td>15.7±0.7</td></tr><tr><td></td><td>ContextFocus</td><td>77.0±1.1</td><td>26.9±1.2</td><td>8.7±0.2</td><td> $1 1 . 1 { \pm } 1 . 2 $ </td><td> $4 2 . 1 \pm 1 . 3$ </td><td>29.7±0.7</td><td>16.1±1.8</td></tr><tr><td></td><td>JUICE</td><td>75.0±1.2</td><td>25.0±0.6</td><td> $8 . 2 \pm 0 . 7$ </td><td> $9 . 6 { \pm } 0 . 3 $ </td><td> $4 1 . 1 { \pm } 0 . 7 $ </td><td>27.9±0.3</td><td>15.0±0.6</td></tr><tr><td></td><td>CK-PLUG</td><td>81.0±0.9</td><td>27.0±1.0</td><td> $9 . 3 { \pm } 0 . 5 $ </td><td> $1 4 . 4 \pm 0 . 9$ </td><td> $4 3 . 6 { \pm } 1 . 4 $ </td><td>30.0±1.1</td><td>16.5±2.3</td></tr><tr><td></td><td>Ours</td><td>88.0±1.1</td><td>27.7±1.1</td><td>10.5±0.6</td><td>17.2±0.6</td><td> $4 4 . 2 \pm 1 . 2$ </td><td>30.5±0.9</td><td>17.0±0.8</td></tr><tr><td>Mistral-7B-Instruct-v0.3</td><td>Unsteered</td><td>81.0±1.5</td><td>27.3±1.0</td><td>9.4±0.5</td><td>32.7±1.1</td><td> $4 0 . 2 { \pm } 0 . 8 $ </td><td>25.1±0.8</td><td>14.9±0.4</td></tr><tr><td></td><td>Random split</td><td> $8 1 . 0 { \pm } 1 . 4 $ </td><td>27.1±0.9</td><td> $9 . 1 \pm 0 . 4$ </td><td> $3 3 . 2 \pm 1 . 2$ </td><td> $3 9 . 4 \pm 0 . 9$ </td><td> $2 5 . 1 \pm 0 . 9$ </td><td> $1 4 . 9 { \pm } 0 . 6 $ </td></tr><tr><td></td><td>ContextFocus</td><td> $8 2 . 0 { \pm } 0 . 5 $ </td><td>27.8±1.0</td><td> $9 . 5 { \pm } 0 . 7 $ </td><td> $3 4 . 6 { \pm } 1 . 7 $ </td><td> $4 0 . 1 { \pm } 0 . 8 $ </td><td> $2 3 . 3 { \pm } 1 . 3$ </td><td> $1 4 . 8 { \pm } 1 . 3 $ </td></tr><tr><td></td><td>JUICE</td><td> $7 7 . 0 { \pm } 1 . 7 $ </td><td>25.8±0.8</td><td> $8 . 7 { \pm } 1 . 4 $ </td><td> $3 1 . 9 { \pm } 1 . 0 \ $ </td><td> $3 9 . 3 { \pm } 0 . 7 \ $ </td><td> $2 3 . 6 { \pm } 0 . 7 \ $ </td><td> $1 3 . 7 { \pm } 0 . 5 $ </td></tr><tr><td></td><td>CK-PLUG</td><td> $8 2 . 0 { \pm } 0 . 6 $ </td><td>28.2±1.1</td><td> $9 . 9 { \pm } 0 . 4 $ </td><td> $3 3 . 1 \pm 1 . 3$ </td><td> $4 1 . 6 { \pm } 1 . 5 $ </td><td> $2 6 . 4 \pm 1 . 0$ </td><td> $1 5 . 3 { \pm } 1 . 1 $ </td></tr><tr><td></td><td>Ours</td><td>86.0±1.2 28.5±0.6</td><td></td><td> ${ \bf 1 0 . 7 \pm 0 . 6 }$ </td><td> ${ \bf 3 5 . 9 2 1 . 3 }$ </td><td> ${ \pm 1 . 6 \pm 1 . 0 }$ </td><td> ${ \bf 2 9 . 4 \pm 1 . 1 }$ </td><td> ${ \bf 1 5 . 9 2 0 . 8 }$ </td></tr></table>

This tests whether the direction encodes source selection rather than lexical or semantic features of one relation. The off-diagonal effects in Figure 5 show transfer across all six knowledge categories; Appendix I.2 provides the other-model heatmaps. Cross-dataset transfer. We apply the direction learned on NQ-Swap to distinct factual-conflict datasets without target-side refitting. These dataset shifts introduce variations in conflict construction, factual content, and other nuisance factors, reducing the possibility that the observed effect is tied to dataset-specific properties. Table 6 demonstrates bidirectional transfer across datasets, while contextonly controls achieve consistently weaker effects.

Memory. Memory tasks provide multi-turn histories. In LoCoMo, LoCoMo-Plus, and Long-MemEval, the relevant fact may lie many turns from the question and compete with a continuation, stale association, or another event. These tasks test retrieval, localization, and temporal ordering of knowledge rather than passage matching.

Rule-following. Rule-following tasks provide explicit behavioral constraints. IFEval and Follow-Bench require the model to retain rules about output structure, length, required content, and exclusions throughout generation; they serve as instruction-following controls for whether the direction improves use of contextual constraints.

## 5.2 To Complex, Application-Relevant Tasks

We next evaluate whether a direction learned from explicit factual conflicts transfers to broader inputgrounded tasks without target-task direction fitting. Unlike the controlled benchmarks, these tasks do not necessarily involve direct competition between parametric and contextual answers.

Results. Table 7 reports held-out unsteered and steered scores using source matching for controlled factual tasks and each benchmark’s native evaluator otherwise. The corresponding item-level changes and error-stratified cases are reported in Appendix G, including the case tables.

RAG. RAG tasks provide retrieved documents: RGB supplies deliberately counterfactual evidence, whereas CRAG uses retrieved passages that can be long, stale, noisy, or only partly relevant. Success requires selecting the supplied evidence when it conflicts with a plausible parametric answer.

## 6 Conclusion

We show that a single residual-stream direction enables smooth, bidirectional selection between parametric knowledge and external context. Changing its signed coefficient at one carefully selected layer shifts both aggregate source-match rates and

fixed-instance responses.

The control generalizes across conflict formats, factual relations, and datasets, and transfers to RAG, multi-turn memory, and rule-following tasks without target-task direction fitting. Weaker context-attention controls suggest that transfer reflects source competition. These results provide evidence for a transferable control direction underlying knowledge-source selection.

## Limitations

First, our method requires white-box access to model activations. Since the direction is estimated and applied in the residual stream, it is not directly applicable to closed-source or API-only models.

Second, the learned direction should be viewed as a useful control axis rather than a complete mechanism of knowledge selection. Although a single direction enables effective steering, source selection may still involve broader circuits, including attention heads, MLP features, and decoding dynamics.

Third, our evaluation is based mainly on controlled factual-conflict benchmarks and sourcechoice accuracy. This setup isolates parametric versus contextual reliance, but may not capture more nuanced outputs such as hedging, partial agreement, or answers that combine both sources.

Finally, the direction improves the complex tasks we evaluate, but the gains are often smaller than in controlled factual conflicts and depend on the underlying capabilities of each model. Establishing how model competence, task difficulty, and evidence quality jointly determine these effects remains future work.

## Acknowledgments

This work is supported by CCF-Tencent Rhino-Bird Open Research Fund (Grant No. CCF-Tencent RAGR20260112).

## References

Nikhil Anand, Shwetha Somasundaram, Anirudh Phukan, Apoorv Saxena, and Koyel Mukherjee. 2026. Contextfocus: Activation steering for contextual faithfulness in large language models. Preprint, arXiv:2601.04131.

Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda. 2024. Refusal in language models is mediated by a single direction. Preprint, arXiv:2406.11717.

Baolong Bi, Shenghua Liu, Yiwei Wang, Yilong Xu, Junfeng Fang, Lingrui Mei, and Xueqi Cheng. 2025. Parameters vs. context: Fine-grained control of knowledge reliance in language models. Preprint, arXiv:2503.15888.

Arie Cattan, Alon Jacovi, Ori Ram, Jonathan Herzig, Roee Aharoni, Sasha Goldshtein, Eran Ofek, Idan Szpektor, and Avi Caciularu. 2025. Dragged into conflicts: Detecting and addressing conflicting sources in search-augmented llms. Preprint, arXiv:2506.08500.

Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun. 2023. Benchmarking large language models in retrieval-augmented generation. Preprint, arXiv:2309.01431.

Sitao Cheng, Liangming Pan, Xunjian Yin, Xinyi Wang, and William Yang Wang. 2024. Understanding the interplay between parametric and contextual knowledge for large language models. Preprint, arXiv:2410.08414.

Eunseong Choi, June Park, Hyeri Lee, and Jongwuk Lee. 2025. Conflict-aware soft prompting for retrievalaugmented generation. Preprint, arXiv:2508.15253.

Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, and Christopher Olah. 2022. Toy models of superposition. Preprint, arXiv:2209.10652.

Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and Haofen Wang. 2024. Retrieval-augmented generation for large language models: A survey. Preprint, arXiv:2312.10997.

Mor Geva, Jasmijn Bastings, Katja Filippova, and Amir Globerson. 2023. Dissecting recall of factual associations in auto-regressive language models. Preprint, arXiv:2304.14767.

Giwon Hong, Jeonghwan Kim, Junmo Kang, Sung-Hyon Myaeng, and Joyce Jiyoung Whang. 2024. Why so gullible? enhancing the robustness of retrieval-augmented models against counterfactual noise. Preprint, arXiv:2305.01579.

Youcheng Huang, Chen Huang, Duanyu Feng, Wenqiang Lei, and Jiancheng Lv. 2025. Cross-model transferability among large language models on the platonic representations of concepts. Preprint, arXiv:2501.02009.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7b. Preprint, arXiv:2310.06825.

Yuxin Jiang, Yufei Wang, Xingshan Zeng, Wanjun Zhong, Liangyou Li, Fei Mi, Lifeng Shang, Xin Jiang, Qun Liu, and Wei Wang. 2024. Follow-Bench: A multi-level fine-grained constraints following benchmark for large language models. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 4667–4688, Bangkok, Thailand. Association for Computational Linguistics.

Zhuoran Jin, Pengfei Cao, Hongbang Yuan, Yubo Chen, Jiexin Xu, Huaijun Li, Xiaojian Jiang, Kang Liu, and Jun Zhao. 2024. Cutting off the head ends the conflict: A mechanism for interpreting and mitigating knowledge conflicts in language models. Preprint, arXiv:2402.18154.

Gaotang Li, Yuzhong Chen, and Hanghang Tong. 2025. Taming knowledge conflicts in language models. Preprint, arXiv:2503.10996.

Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. 2024. Inference-time intervention: Eliciting truthful answers from a language model. Preprint, arXiv:2306.03341.

Shayne Longpre, Kartik Perisetla, Anthony Chen, Nikhil Ramesh, Chris DuBois, and Sameer Singh. 2022. Entity-based knowledge conflicts in question answering. Preprint, arXiv:2109.05052.

Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. 2024. Evaluating very long-term conversational memory of llm agents. Preprint, arXiv:2402.17753.

Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. 2023. When not to trust language models: Investigating effectiveness of parametric and non-parametric memories. Preprint, arXiv:2212.10511.

Samuel Marks and Max Tegmark. 2024. The geometry of truth: Emergent linear structure in large language model representations of true/false datasets. Preprint, arXiv:2310.06824.

Ian McKenzie, Alexander Lyzhov, Alicia Parrish, Ameya Prabhu, Aaron Mueller, Najoung Kim, Sam Bowman, and Ethan Perez. 2022. The inverse scaling prize.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. 2023. Locating and editing factual associations in gpt. Preprint, arXiv:2202.05262.

Thomas Merth, Qichen Fu, Mohammad Rastegari, and Mahyar Najibi. 2024. Superposition prompting: Improving and accelerating retrieval-augmented generation. Preprint, arXiv:2404.06910.

Julian Minder, Kevin Du, Niklas Stoehr, Giovanni Monea, Chris Wendler, Robert West, and Ryan Cotterell. 2025. Controllable context sensitivity and the knob behind it. In International Conference on Learning Representations, volume 2025, pages 81073–81102.

Nina Panickssery, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Matt Turner. 2024. Steering llama 2 via contrastive activation addition. Preprint, arXiv:2312.06681.

Kiho Park, Yo Joong Choe, and Victor Veitch. 2024. The linear representation hypothesis and the geometry of large language models. Preprint, arXiv:2311.03658.

Dan Shi, Renren Jin, Tianhao Shen, Weilong Dong, Xinwei Wu, and Deyi Xiong. 2024. Ircan: Mitigating knowledge conflicts in llm generation via identifying and reweighting context-aware neurons. Preprint, arXiv:2406.18406.

Nicholas Sofroniew, Isaac Kauvar, William Saunders, Runjin Chen, Tom Henighan, Sasha Hydrie, Craig Citro, Adam Pearce, Julius Tarng, Wes Gurnee, Joshua Batson, Sam Zimmerman, Kelley Rivoire, Kyle Fish, Chris Olah, and Jack Lindsey. 2026. Emotion concepts and their function in a large language model. Transformer Circuits Thread.

Alessandro Stolfo, Vidhisha Balachandran, Safoora Yousefi, Eric Horvitz, and Besmira Nushi. 2025. Improving instruction-following in language models through activation steering. In International Conference on Learning Representations, volume 2025, pages 55790–55823.

Daniel Tan, David Chanin, Aengus Lynch, Dimitrios Kanoulas, Brooks Paige, Adria Garriga-Alonso, and Robert Kirk. 2025. Analyzing the generalization and reliability of steering vectors. Preprint, arXiv:2407.12404.

Gemma Team. 2024. Gemma.

Qwen Team. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, and Monte MacDiarmid. 2024. Steering language models with activation engineering. Preprint, arXiv:2308.10248.

Han Wang, Archiki Prasad, Elias Stengel-Eskin, and Mohit Bansal. 2025a. Adacad: Adaptively decoding to balance conflicts between contextual and parametric knowledge. Preprint, arXiv:2409.07394.

Mengru Wang, Yunzhi Yao, Ziwen Xu, Shuofei Qiao, Shumin Deng, Peng Wang, Xiang Chen, Jia-Chen Gu, Yong Jiang, Pengjun Xie, Fei Huang, Huajun Chen, and Ningyu Zhang. 2024. Knowledge mechanisms in large language models: A survey and perspective. Preprint, arXiv:2407.15017.

Yilin Wang, Heng Wang, Yuyang Bai, and Minnan Luo. 2025b. Continuously steering llms sensitivity to contextual knowledge with proxy models. Preprint, arXiv:2508.19720.

Yotam Wolf, Noam Wies, Dorin Shteyman, Binyamin   
Rothberg, Yoav Levine, and Amnon Shashua. 2025.   
Tradeoffs between alignment and helpfulness in lan   
guage models with steering methods. Preprint,   
arXiv:2401.16332.   
Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang,   
Kai-Wei Chang, and Dong Yu. 2025. Longmemeval:   
Benchmarking chat assistants on long-term interac   
tive memory. Preprint, arXiv:2410.10813.   
Rongwu Xu, Zehan Qi, Zhijiang Guo, Cunxiang Wang,   
Hongru Wang, Yue Zhang, and Wei Xu. 2024.   
Knowledge conflicts for llms: A survey. Preprint,   
arXiv:2403.08319.   
Xiao Yang, Kai Sun, Hao Xin, Yushi Sun, Nikita Bhalla,   
Xiangsen Chen, Sajal Choudhary, Rongze Daniel   
Gui, Ziran Will Jiang, Ziyu Jiang, Lingkun Kong,   
Brian Moran, Jiaqi Wang, Yifan Ethan Xu, An Yan,   
Chenyu Yang, Eting Yuan, Hanwen Zha, Nan Tang,   
and 8 others. 2024. Crag – comprehensive rag bench  
mark. Preprint, arXiv:2406.04744.   
Long Zhang and Fangwei Lin. 2026. Simulated   
adoption: Decoupling magnitude and direction   
in llm in-context conflict resolution. Preprint,   
arXiv:2602.04918.   
Yu Zhao, Alessio Devoto, Giwon Hong, Xiaotang   
Du, Aryo Pradipta Gema, Hongru Wang, Xuanli   
He, Kam-Fai Wong, and Pasquale Minervini. 2025a.   
Steering knowledge selection behaviours in llms   
via sae-based representation engineering. Preprint,   
arXiv:2410.15999.   
Yu Zhao, Xiaotang Du, Giwon Hong, Aryo Pradipta   
Gema, Alessio Devoto, Hongru Wang, Xuanli   
He, Kam-Fai Wong, and Pasquale Minervini.   
2025b. Analysing the residual stream of lan  
guage models under knowledge conflicts. Preprint,   
arXiv:2410.16090.   
Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha   
Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and   
Le Hou. 2023. Instruction-following evaluation for   
large language models. Preprint, arXiv:2311.07911.   
Andy Zou, Long Phan, Sarah Chen, James Campbell,   
Phillip Guo, Richard Ren, Alexander Pan, Xuwang   
Yin, Mantas Mazeika, Ann-Kathrin Dombrowski,   
Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan   
Wang, Alex Mallen, Steven Basart, Sanmi Koyejo,   
Dawn Song, Matt Fredrikson, and 2 others. 2025.   
Representation engineering: A top-down approach   
to ai transparency. Preprint, arXiv:2310.01405.

## List of appendices

A Dataset Details 13   
B Intervention Location and Representa  
tion Validation 13   
B.1 Representation Hook 13   
B.2 Layer Selection 14   
B.3 Context-Sensitivity Direction Sim   
ilarity 14   
B.4 Token Position 14   
C Judge-Based Evaluation Protocol 14   
C.1 Answer Normalization 14   
C.2 Answer Window 14   
C.3 Match Rules 15   
C.4 LLM-Judge Validation 15   
C.5 Complex-Task Evaluators . 15   
D Implementation Details 16   
E Steering-Strength Sensitivity Analysis 16   
F Scaling to Larger Models 16   
G Instance-Level Changes and Failure   
Analysis 16   
G.1 Controlled Factual-Conflict   
Instance-Level Changes . 18   
G.2 Complex-Task Instance-Level   
Changes 18   
G.3 Failure Analysis 18   
H Prompt Templates 23   
I Generalization Within Simple Factual-  
Conflict Tasks 25   
I.1 Cross-Format Transfer 25   
I.2 Cross-Knowledge-Type Transfer . 25

## Appendices

## A Dataset Details

We briefly describe how each benchmark is converted into our source-choice setting. For all datasets, we first keep examples where the model answers the clean prompt correctly. We then evaluate the same examples under a conflict prompt that supports an alternative answer. The original answer is treated as the parametric answer, while the answer supported by the conflict prompt is treated as the contextual answer. Prompt examples are provided in Appendix H.

NQ-Swap. NQ-Swap pairs Natural Questions examples with substituted passages. We use the original answer as the parametric answer and the substituted answer as the contextual answer. The clean prompt contains only the question, while the conflict prompt prepends the substituted context before the same question. We use the questiondeduplicated split without tables or list items.

MacNoise. MacNoise provides QA examples with perturbed retrieved contexts. The original answer is used as the parametric answer, and the replacement answer is used as the contextual answer. For each example, we select the first perturbed context marked by pert\_lbls\_100 == 1 and prepend its text\_pert field to form the conflict prompt.

ParaConflict. ParaConflict contains synthetic factual conflicts across multiple categories. The gold answer is treated as the parametric answer, and the distracted token is treated as the contextual answer. We use the coherent conflict format in the main experiments because it provides a natural-language passage supporting the distractor. For cross-format transfer, we train on substitution-format items and evaluate on coherentformat items.

CounterFact. CounterFact contains factual editing examples grouped by Wikidata relation. The true target is treated as the parametric answer, and the edited target is treated as the contextual answer. The conflict prompt is constructed by inserting the edited target into a paraphrased prefix and appending the original factual query. For crossknowledge-type transfer, we evaluate six relations: country of citizenship, manufacturer, occupation, religion, sport, and original broadcaster.

MemoTrap. MemoTrap creates conflicts between memorized phrase completions and explicit prompt instructions. The familiar phrase continuation is treated as the parametric answer, while the instruction-satisfying completion is treated as the contextual answer. The clean prompt removes the explicit ending constraint, whereas the conflict prompt keeps the original instruction.

RGB and CRAG. RGB supplies retrieved, deliberately counterfactual passages whose answer conflicts with a familiar factual answer; we evaluate whether the output follows the supplied evidence. CRAG supplies retrieved web-style passages for factual questions, including cases with stale, competing, noisy, or partially relevant evidence. We use each benchmark’s held-out evaluation items and its factual source-match protocol.

LoCoMo, LoCoMo-Plus, and LongMemEval. These datasets require answers grounded in multiturn personal histories. LoCoMo and LoCoMo-Plus test retrieval and linkage across earlier dialogue turns, while LongMemEval additionally stresses temporal ordering and aggregation across events. We preserve each benchmark’s native heldout split and exact-match evaluator.

IFEval and FollowBench. These instructionfollowing datasets specify explicit requirements on output structure, length, content, or lexical exclusions. We use their native constraint-compliance evaluators; they serve as controls for retaining and applying external instructions across the generated response.

## B Intervention Location and Representation Validation

We ablate each modeling choice underlying our main pipeline (Llama-3.1-8B-Instruct, NQ-Swap test unless noted). For each variant we re-estimate vˆ and ℓ when required, re-tune $( \alpha ^ { + } , \alpha ^ { - } )$ on validation, and report $\operatorname { A c c } _ { \operatorname { p a r } }$ and $\operatorname { A c c } _ { \operatorname { c t x } }$ on the test split.

## B.1 Representation Hook

Table 8 shows that resid post yields stronger bidirectional control; resid mid generally underperforms, indicating that the MLP sublayer contributes to the separable knowledge-selection signal.

Table 8: Steered source-choice accuracy (%) on NQ-Swap test across all evaluated models. $\operatorname { A c c } _ { \operatorname { p a r } }$ and $\operatorname { A c c } _ { \operatorname { c t x } }$ report source-choice accuracy when steering toward parametric and contextual knowledge, respectively. Each steered cell reports resid-mid / resid-post (best α per direction).
<table><tr><td>Model</td><td> $\operatorname { A c c } _ { \operatorname { p a r } }$ </td><td> $\mathbf { A c c _ { c t x } }$ </td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>50.4/56.8</td><td>59.7/55.5</td></tr><tr><td>Qwen3-8B</td><td>62.4/54.6</td><td>51.1/56.0</td></tr><tr><td>Gemma-2-9B-IT</td><td>33.8/80.9</td><td>53.2/65.4</td></tr><tr><td>Mistral-7B-Instruct-v0.3</td><td>59.0/59.0</td><td> $7 1 . 1 / 6 0 . 2 $ </td></tr></table>

![](images/9e659953dceb0d86a732ddd2b1966c3083adfb3d0f9bd00c2463ac217f19a5de.jpg)

![](images/6a1bfcf3974d7c1dfe4d1f5c5f3bbe8cf86fc5b7a3cfe1cb6e8fee4500a32420.jpg)  
Qwen3-8B

Llama-3.1-8B-Instruct  
![](images/e9c857242fb5bf26332b9b66c765e5ae6f19ac48448929cbdce43432b4b71640.jpg)  
Gemma-2-9B-Instruct

![](images/b4c1db5f3fff0d8381250f9564502725ea26c237b1a1b4a7bd42450e82769663.jpg)  
Mistral-7B-Instruct-v0.3  
Figure 6: Layer-wise cosine similarity $s _ { T _ { \mathrm { c l e a n } } , T _ { \mathrm { c o n f l i c t } } } ^ { i }$ between the respective final-token activations of clean and conflict prompts ((2)) on NQ-Swap.

## B.2 Layer Selection

Instead of steering only at the selected layer $\ell ,$ we inject the intervention at each layer i separately and examine the resulting performance across all four models on NQ-Swap. Figure 6 reports the cosine curves used for localization, while Figure 7 shows $\mathsf { A c c } _ { \mathrm { p a r } }$ and $\operatorname { A c c } _ { \operatorname { c t x } }$ as a function of the intervention layer. Across models, the layer-wise steering performance follows a similar trend to the angular separation scores: layers with larger separation generally yield stronger steering effects, while shallow and very late layers show weaker or less stable control. This alignment supports our layer-selection strategy and suggests that the angular separation score is a useful indicator of where knowledge selection is most steerable.

Across the completed sweeps, peaks align with the angular-separation layer; shallow and late layers show weaker or asymmetric control.

## B.3 Context-Sensitivity Direction Similarity

Table 9 reports cosine similarities among the three no-context contrasts used in the matched contextsensitivity controls. A denotes no context; B, C,

Table 9: Cosine similarity between the three matched no-context contrasts on NQ-Swap.
<table><tr><td>Contrast pair</td><td>Cosine similarity</td></tr><tr><td>A-C vs. A-B</td><td>0.899399</td></tr><tr><td> $\mathbf { A { \mathrm { - } } C \ v s . \ A { \mathrm { - } } D }$ </td><td>0.878706</td></tr><tr><td> $\mathbf { A { - } B \ v s . \ A { - } D }$ </td><td>0.874608</td></tr></table>

Table 10: Token-position ablation on NQ-Swap test. $\operatorname { A c c } _ { \operatorname { p a r } }$ and $\operatorname { A c c } _ { \operatorname { c t x } }$ report source-choice accuracy when steering toward parametric and contextual knowledge, respectively. Each entry is question-final / answer-final token, using validation-selected coefficients at that position.
<table><tr><td>Model</td><td> $\operatorname { A c c } _ { \operatorname { p a r } }$ </td><td> $\operatorname { A c c } _ { \operatorname { c t x } }$ </td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>56.8 / 65.6</td><td>55.5 / 54.0</td></tr><tr><td>Qwen3-8B</td><td>54.6 / 47.1</td><td>56.0 / 44.5</td></tr><tr><td>Gemma-2-9B-IT</td><td>80.9 / 75.3</td><td>65.4 / 62.3</td></tr><tr><td>Mistral-7B-Instruct-v0.3</td><td>59.0 / 63.9</td><td>60.2 / 57.8</td></tr></table>

and D denote aligned, conflicting, and irrelevant context.

## B.4 Token Position

As shown in Table 10, the question-final token generally yields stronger bidirectional control than the answer-final token, although answer-final intervention can perform better for parametric steering in a few models.

## C Judge-Based Evaluation Protocol

$\operatorname { \mathrm { A c c } } _ { \operatorname { \mathrm { p a r } } }$ and $\operatorname { A c c } _ { \operatorname { c t x } }$ depend on how free-form outputs are mapped to discrete source labels. This appendix makes the protocol fully explicit.

## C.1 Answer Normalization

Each gold parametric answer $a _ { \mathrm { p a r } }$ and contextual answer $a _ { \mathrm { c t x } }$ is normalized before matching:

1. lowercase;

2. remove punctuation except intra-token hyphens;

3. remove leading articles (a, an, the);

4. collapse whitespace;

## C.2 Answer Window

We take the first 256 generated tokens of the raw continuation and normalize this fixed answer window. This avoids making the metric depend on sentence segmentation or on whether steering changes the placement of a sentence boundary. Outputs shorter than 256 tokens are matched in full.

![](images/2dda0cf0f7549468faf7185bbd5a64804631d21be2065d051982e3723fb1f6db.jpg)  
(a) Llama-3.1-8B-Instruct

![](images/e3b043b18098c1c156f4b04d0dd1f6b7ee278398ab74df54006984bcaea2558d.jpg)  
(c) Gemma-2-9B-IT

![](images/3b5f699f1f6d4d348580f4969db9131f950a5b40d80b76b358ddc4aaa70bbbde.jpg)  
(b) Qwen3-8B

![](images/f4d328689e88432bb6dd0411228658d191ac03743bff058b2ae0eb159b978da7.jpg)  
(d) Mistral-7B-Instruct-v0.3  
Figure 7: Layer-selection analysis on NQ-Swap. Each panel shows performance when the intervention is applied at different layers for one model. The layer-wise performance trend is consistent with the cosine-based separation pattern: layers with lower clean–conflict cosine similarity, corresponding to larger angular separation, tend to produce stronger steering effects.

## C.3 Match Rules

Let s be the normalized 256-token answer window. The four labels below are mutually exclusive and exhaustive; both/neither outputs are retained in the denominator rather than discarded.

• Parametric-only: s contains $a _ { \mathrm { p a r } }$ as a substring, or an alias thereof, and does not contain $a _ { \mathrm { c t x } }$

• Contextual-only: s contains $a _ { \mathrm { c t x } }$ and does not contain $a _ { \mathrm { p a r } }$

• Both: s contains both answers.

• Neither: s contains neither answer, or the output is empty/refusal/off-topic.

Long answers are handled by substring search within the normalized answer window only; tokens after position 256 are not searched. This operational metric measures answer-string source matching, not full semantic entailment; Appendix C.5 therefore describes the benchmark-native semantic or constraint evaluators used for complex tasks.

## C.4 LLM-Judge Validation

String matching permits deterministic, low-cost evaluation across the full alpha sweep. To audit its construct validity, we sample 100 OOD outputs per dataset. Given the prompt, both candidate answers, and the complete generation, a blinded LLM judge assigns one of the same four labels without seeing the string-match result. Table 11 compares the resulting four-way distributions; the close correspondence supports using string matching for the

Table 11: Four-way label distributions (%) from normalized string matching and a blinded LLM judge on 100 OOD outputs per dataset. The order is correct answer / conflicting distractor / both / neither.
<table><tr><td>Dataset</td><td>String matching</td><td>LLM judge</td></tr><tr><td>ParaConflict</td><td>57 / 9 / 34 / 0</td><td>59 / 10 / 30 / 1</td></tr><tr><td>MacNoise</td><td>77 / 5 / 1 / 17</td><td>74 / 5 / 8 / 13</td></tr><tr><td>MemoTrap</td><td>48 / 15 / 32 / 5</td><td>50 / 17 / 30 / 3</td></tr><tr><td>CounterFact</td><td>53 / 20 / 23 / 4</td><td>54 / 24 / 18 / 4</td></tr></table>

full sweeps.

## C.5 Complex-Task Evaluators

For each model, all complex-task experiments use its direction learned on NQ-Swap, injected at that model’s selected layer with coefficients fixed from NQ-Swap validation; no target-task direction, coefficient, or layer is estimated or relocalized. For each task, all coefficients are evaluated on the same deterministic sample subset, and we report the mean official per-item score times 100. These downstream scores assess input-grounded behavior: except for RGB, a correct output alone does not establish whether the unsteered error arose from factual priors, evidence localization, reasoning, or response generation.

Sampling and prompts. We evaluate at most 500 processed items per task, sampling without replacement with seed 43 when a pool exceeds 500 items; smaller pools are evaluated in full. The model receives only the task-supplied context, and credit is assigned against the contextual target or benchmark constraint. Thus, complex-task scores do not use the 256-token source-match protocol above.

RAG evaluators. RGB contains 100 counterfactual retrieval items. Up to three supplied passages support a fakeanswer, and its official binary judge scores that contextual target rather than the worldfactual answer. For CRAG, we sample 500 of 2,698 development items; ranked snippets are HTMLstripped, title-prefixed when available, concatenated in retrieval order, and capped at 12,000 characters. Its released evaluator handles exact, invalid, and semantic-answer cases.

Memory evaluators. For LoCoMo, 500 of 1,542 QA rows are sampled and the official categoryaware multi-hop or token-F1 evaluator is averaged. LoCoMo-Plus evaluates all 401 cognitive items with its released binary constraint-consistency judge. LongMemEval evaluates its 500-item oracle split in full, caps dated relevant sessions at 12,000 characters, and uses its released answercheck judge conditioned on question type.

Rule-following evaluators. For IFEval, 500 of 541 prompts are sampled and its released strict evaluator gives one only when every instruction is satisfied. For FollowBench, 500 of 820 items are sampled; released template, rule-based, or discriminative evaluators give hard per-item satisfaction, and we report the macro average across six constraint categories.

## D Implementation Details

All experiments are inference-only; we do not fine-tune any model. We run each model on a single NVIDIA A100 GPU with 80GB memory using bfloat16 precision and the Hugging Face transformers library. Unless otherwise specified, decoding is greedy with do\_sample=False and max\_new\_tokens=200. The base evaluation cohort consists of examples where the model answers the clean prompt correctly and is then evaluated under the conflict prompt, denoted as stage1\_correct\_use\_wrong\_context. When subsampling is required, we use random seed 42.

## E Steering-Strength Sensitivity Analysis

For the Llama-3.1-8B-Instruct direction transferred from NQ-Swap, we report the selected coefficient and the two nearest evaluated values on either side when available. Table 14 gives parametric-answer rate for positive steering and distractor/contextanswer rate for negative steering. No target-side coefficient is reselected.

Table 12: Model checkpoints and selected intervention layers.
<table><tr><td>Model</td><td>Size</td><td>Local key</td><td>Layer</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>8B</td><td>1lama3-8b-it</td><td>10</td></tr><tr><td>Gemma-2-9B-IT</td><td>9B</td><td>gemma2-9b-it</td><td>16</td></tr><tr><td>Qwen3-8B</td><td>8B</td><td>qwen3-8b</td><td>21</td></tr><tr><td>Mistral-7B-Instruct-v0.3</td><td>7B</td><td>mistral-7b-instruct-v0.3</td><td>16</td></tr></table>

Table 13: Inference and tokenization settings.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Framework</td><td>Hugging Face transformers</td></tr><tr><td>Precision</td><td>bfloat16</td></tr><tr><td>Hardware</td><td>1× NVIDIA A100 80GB</td></tr><tr><td>Attention implementation</td><td>eager</td></tr><tr><td>Decoding</td><td>Greedy, do_sample=False</td></tr><tr><td>max_new_tokens</td><td>200</td></tr><tr><td>Tokenizer max_length</td><td>8192</td></tr><tr><td>Padding side</td><td>Left padding</td></tr><tr><td>pad_token</td><td>eos_token if unset</td></tr><tr><td>Batch size</td><td>128</td></tr><tr><td>Batch size for NQ-Swap</td><td>64</td></tr><tr><td>GPUs per job</td><td>1, set by CUDA_VISIBLE_DEVICES</td></tr><tr><td>Random seed</td><td>42 when subsampling is used</td></tr></table>

## F Scaling to Larger Models

We further examine whether the same onedimensional steering behavior holds for larger instruction-tuned models. Figure 8 reports sourcechoice rates on NQ-Swap and MacNoise for four larger instruction-tuned models. Across most models and datasets, increasing α raises the internal, or parametric, source-choice rate while reducing the external, or contextual, source-choice rate. This consistent trend indicates that the learned direction remains effective beyond the smaller models used in the main experiments, suggesting that knowledge selection can be continuously modulated along a similar residual-stream direction even at larger model scales.

## G Instance-Level Changes and Failure Analysis

This audit classifies baseline context-conflict errors that an available positive-α sweep changed into an officially correct output. Here, knowledge conflict is used broadly: a model default (factual prior, continuation, event ordering, or output habit) can conflict with information or constraints supplied by the context. The alpha shown for each example is the first observed correct positive-alpha output, not an item-level selection rule. “Parametric knowledge” is an inference from the baseline text, not a claim about an unobserved model state. We use three mechanism labels: P for an explicit override of supplied evidence by a real-world or general-knowledge correction; L for missed, misordered, or underspecified supplied context; and I for a requested format or constraint that the context requires.

Table 14: Cross-dataset operating-point stability for Llama-3.1-8B-Instruct. Each cell gives α: held-out sourcematch rate (%) followed by the number of valid Stage-2 generations in parentheses. The selected transfer coefficient is bold; “–” indicates that a farther neighboring grid point was not evaluated.
<table><tr><td>Setting</td><td>ParaConflict</td><td>MacNoise</td><td>MemoTrap</td><td>CounterFact</td></tr><tr><td colspan="5">Positive direction: parametric-answer rate</td></tr><tr><td>α = 0</td><td>15.9 (1711)</td><td>46.7 (1047)</td><td>67.0 (624)</td><td>36.1 (2160)</td></tr><tr><td>Lower 2</td><td>1.2: 82.3 (1701)</td><td>1.2: 71.1 (849)</td><td>2.4: 83.7 (509)</td><td>2.6: 43.7 (1854)</td></tr><tr><td>Lower 1</td><td>1.4: 84.7 (1697)</td><td>1.4: 72.1 (795)</td><td>2.6: 82.7 (490)</td><td>2.8: 43.6 (1803)</td></tr><tr><td>Selected</td><td>1.6: 85.6 (1687)</td><td>1.6: 72.4 (749)</td><td>2.8: 85.6 (465)</td><td>3.0: 48.1 (1565)</td></tr><tr><td>Higher 1</td><td>1.8: 84.3 (1674)</td><td>1.8: 70.4 (689)</td><td>3.0: 83.1 (450)</td><td></td></tr><tr><td>Higher 2</td><td>2.0: 81.6 (1662)</td><td>2.0: 62.3 (159)</td><td></td><td></td></tr><tr><td colspan="5">Negative direction: distractor/context-answer rate</td></tr><tr><td>α = 0</td><td>81.8 (1711)</td><td>27.3 (1047)</td><td>30.1 (624)</td><td>60.8 (2160)</td></tr><tr><td>Lower 2</td><td>-1.8: 98.3 (1264)</td><td>-1.8: 23.4 (124)</td><td></td><td>-1.4: 61.6 (1914)</td></tr><tr><td>Lower 1</td><td>-1.6: 98.6 (1280)</td><td>-1.6: 21.1 (147)</td><td>-1.8: 57.6 (580)</td><td>—1.2: 62.9 (1957)</td></tr><tr><td>Selected</td><td>-1.4: 98.8 (1292)</td><td>-1.4: 43.8 (706)</td><td>-2.0: 58.0 (571)</td><td>-1.0: 63.0 (2005)</td></tr><tr><td>Higher 1</td><td>-1.2: 96.8 (1528)</td><td>-1.2: 43.1 (796)</td><td></td><td>-0.8: 62.8 (2038)</td></tr><tr><td>Higher 2</td><td>-1.0: 96.9 (1578)</td><td>-1.0: 42.2 (855)</td><td></td><td>-0.6: 62.8 (2065)</td></tr></table>

![](images/6e391bf9d32d647983f342e25bf3d1e9ab67f180276eea429c2a86a81e5c2e1c.jpg)  
Figure 8: Source-choice trajectories on larger models. Blue curves denote internal, or parametric, source-choice rates, while red curves denote external, or contextual, source-choice rates. Across NQ-Swap and MacNoise, increasing α generally shifts the model from external reliance toward internal reliance, showing that the steering direction remains effective for larger models.

## G.1 Controlled Factual-Conflict Instance-Level Changes

Each trajectory contains an unsteered output, an intermediate output containing both answer strings, and an output selecting the requested source. Llama, Qwen, and Gemma cases use saved NQ-Swap test alpha grids. Mistral retained only three final NQ-Swap coefficients, so its cases use the saved complete CounterFact within-relation grid. Outputs are answer-bearing excerpts; BOTH denotes a string-match label, not a claim that the full generation provides a coherent answer.

Llama-3.1-8B-Instruct. Table 15 gives three parametric and three contextual trajectories for the main model, illustrating the transition from an initial source choice through a mixed string-match output to the requested source.

Qwen3-8B. Table 16 repeats the same trajectory audit for Qwen3-8B, showing that both source directions can be expressed through intermediate mixed outputs.

Gemma-2-9B-IT. Table 17 reports the corresponding Gemma trajectories; the examples include both clean source switches and cases where the model retains a competing answer in the intermediate generation.

Mistral-7B-Instruct-v0.3. Table 18 reports Mistral’s within-relation CounterFact trajectories, using the saved CounterFact grid because a complete NQ-Swap alpha sweep was unavailable for this model.

## G.2 Complex-Task Instance-Level Changes

The audit is reported in Table 20 and Table 21, with one row per representative item. The labels distinguish explicit parametric overrides (P), localization or reasoning over supplied context (L), and instruction constraints (I). RGB provides synthetic counterfactual conflicts; CRAG mixes evidence selection with temporal, arithmetic, and comparison cases; the memory benchmarks distribute the relevant fact across earlier turns; and IFEval/Follow-Bench test contextual constraints on output form.

## G.2.1 Gain stratification and limitations

The current sweeps support a broad context-conflict account spanning factual evidence, distributed memory, and instruction constraints, while the mechanism-level comparison remains qualitative. A stronger test would sample 50–100 CRAG baseline errors before inspecting steered outputs, independently label evidence spans and confidence, freeze α on validation data, and report permechanism test gains with bootstrap confidence intervals.

## G.3 Failure Analysis

We analyze representative failure cases of steering. Here, PARAMETER denotes the gold or internally stored answer, CONTEXT denotes the distractor supported by the conflict prompt, BOTH denotes outputs containing both answers, and OTHER denotes outputs containing neither.

## G.3.1 Failure to Reach the Target Source

In this category, steering does not move the first answer entity to the intended knowledge source. For positive steering, the output remains contextual; for negative steering, the output remains parametric.

## Example 1.

Question: George V Coast is a part of the continent of

<table><tr><td>α</td><td>Output Label</td></tr><tr><td>0 “Africa.&quot;</td><td>CONTEXT</td></tr><tr><td>+0.2 “Africa.&quot;</td><td>CONTEXT</td></tr><tr><td>+1.0 “Africa.&quot;</td><td>CONTEXT</td></tr><tr><td></td><td>+2.4 “Africa, and it is a region that is CONTEXT rich in natural resources.&quot;</td></tr><tr><td></td><td>+3.0 “Africa, and it is a region of great CONTEXT scientific interest due to its unique geology and geography.&quot;</td></tr></table>

The conflict prompt anchors the first answer to Africa; even strong positive steering fails to recover the parametric answer Antarctica.

## Example 2.

Table 15: Instance-level continuous trajectories for Llama-3.1-8B-Instruct. P and C denote the requested parametric and contextual sources.
<table><tr><td>Question</td><td>Generation trajectory</td></tr><tr><td colspan="2">P: requested parametric source</td></tr><tr><td>When did the first Ninja Turtles come out?</td><td>Initial: 1999. Intermediate: The first comic was published in 1984, not 1999.</td></tr><tr><td>Who designed the Earth Day flag in 1969?</td><td>Target: 1984. Initial: Andreas Vesalius. Intermediate: Vesalius .. . likely John McConnell. Target: John McConnell.</td></tr><tr><td>Who was in the Don&#x27;t Worry, Be Happy video?</td><td>Initial: CeCe Drake. Intermediate: CeCe Drake. Bobby McFerrin ... Target: Bobby McFerrin.</td></tr><tr><td colspan="2">C: requested contextual source When was Edward Tulane pub- Initial: The novel is from 2006.</td></tr><tr><td>lished?</td><td>Intermediate: 15th century ... a 21st-century novel. Target: 15th century.</td></tr><tr><td>How many episodes is season 4 of The Flash?</td><td>Initial: 23 episodes. Intermediate: The correct answer should be 23, not 360. Target: 360 episodes.</td></tr><tr><td>When was the $1,000 bill discontin- ued?</td><td>Initial: 1969. Intermediate: 2017 ... discontinued in 1969. Target: 2017.</td></tr></table>

Table 16: Instance-level continuous trajectories for Qwen3-8B. P and C denote the requested parametric and contextual sources.
<table><tr><td>Question</td><td>Generation trajectory</td></tr><tr><td colspan="2">P: requested parametric source When did The Karate Kid come</td></tr><tr><td>out?</td><td>Initial: The text says the remake was in 1835. Intermediate: The original was 1984, and the remake was 2010. Target: The remake was released in 2010.</td></tr><tr><td>How many grams of alcohol are in one beer?</td><td>Initial: The text says a standard drink contains 1612 grams. Intermediate: The correct value should be 14 grams, not 1612. Target: A standard drink contains about 14 grams.</td></tr><tr><td>Industrial city on Germany&#x27;s Rhine— Herne Canal?</td><td>Initial: The canal connects near Hawaii. Intermediate: It connects near Hawaii ... Duisburg Target: It connects the Rhine River near Duisburg.</td></tr><tr><td colspan="2">C: requested contextual source Total presidential electoral votes?</td></tr><tr><td>When was Harry Potter and the Philosopher&#x27;s Stone published?</td><td>Target: 201. Initial: It was first published in 1997. Intermediate: The answer is 11 November 1918; the correct date should be 1997. Target: The text says it was first published in 11 November 1918.</td></tr><tr><td>When was the first Star Wars film released?</td><td>Initial: The first film was released in 1977. Intermediate: The trilogy was released from 1889 .. . the first film was 1977. Target: The original trilogy was released between 1889 and 1983.</td></tr></table>

Table 17: Instance-level continuous trajectories for Gemma-2-9B-IT. P and C denote the requested parametric and contextual sources.
<table><tr><td>Question</td><td>Generation trajectory</td></tr><tr><td colspan="2">P: requested parametric source</td></tr><tr><td>Who overthrew the Mongols and es- tablished the Ming dynasty?</td><td>Initial: Cliff Williams. Intermediate: Cliff Williams (Zhu Yuanzhang).</td></tr><tr><td>When was the first Ford F-150</td><td>Target: Zhu Yuanzhang. Initial: 1994. Intermediate: 1994. This is incorrect; it was made in 1975.</td></tr><tr><td>made? When did the Right to Buy scheme</td><td>Target: 1975. Initial: March 31. Intermediate: March 31, 1980.</td></tr><tr><td colspan="2">C: requested contextual source</td></tr><tr><td>Where did the Titanic sink?</td><td>Initial: The Titanic sank in the North Atlantic Ocean. Intermediate: It sank in Saint Petersburg in the North Atlantic Ocean.</td></tr><tr><td>Nuclear plant that blew up in Rus- sia?</td><td>Target: The Titanic sank in Saint Petersburg. Initial: Chernobyl. Intermediate: Chernobyl .. . the text does not describe a Colorado disaster. Target: Colorado Nuclear Power Plant.</td></tr><tr><td>Which country joined Mexico and the US to form NAFTA?</td><td>Initial: Canada. Intermediate: Brooklyn. This is incorrect; the answer is Canada. Target: Brooklyn.</td></tr></table>

Table 18: Instance-level continuous trajectories for Mistral-7B-Instruct-v0.3 on CounterFact. P and C denote the requested parametric and contextual sources.
<table><tr><td>Question</td><td>Generation trajectory</td></tr><tr><td colspan="2">P: requested parametric source</td></tr><tr><td>Abd Allah ibn al-Zubayr&#x27;s religion?</td><td>Initial: Buddhism. Intermediate: Buddhism. Abd Allah ibn al-Zubayr ... Target: Islam.</td></tr><tr><td>Kasey Keller is a professional?</td><td>Initial: Football player, specifically a goalkeeper. Intermediate: Footballer .. . one of the greatest in American soccer. Target: Soccer player who played as a goalkeeper.</td></tr><tr><td>Manolache Costache Epureanu citi- zenship?</td><td>Initial: Cambodia. Intermediate: Romania and Cambodia. Target: Romania.</td></tr><tr><td colspan="2">C: requested contextual source What sport does Corey Perry play?</td></tr><tr><td></td><td>Initial: Ice hockey. Intermediate: Corey Perry is an ice hockey player, not a soccer player. Target: Soccer.</td></tr><tr><td>BMW M52, developed by?</td><td>Initial: BMW, a straight-6 engine. Intermediate: Sony, a 2.5-liter inline-six engine used in BMW automobiles. Target: Sony, produced by Sony.</td></tr><tr><td>Dokka Umarov&#x27;s religion?</td><td>Initial: Islam. Intermediate: Umarov is a follower of Judaism. Islam.</td></tr></table>

## Example 2.

Table 19: Audit-supported interpretation of the reported Llama-3.1-8B-Instruct gains. “N/A” denotes an unobserved stratum, not zero gain.
<table><tr><td>Dataset</td><td>Conflict mecha- nism in audit</td><td>Observed gain</td><td>Interpretation</td></tr><tr><td>RGB</td><td>Dominant P by construction</td><td>+20.0 (62.0→82.0)</td><td>Synthetic counterfactual adherence; no matched non-conflict</td></tr><tr><td>CRAG</td><td>Mixed P/L can- didates</td><td>+2.6 (23.7→25.8)</td><td>cohort Both kinds may occur; audit is illustrative, not random</td></tr><tr><td>LoCoMo</td><td>Distributed- context re- trieval (L)</td><td>+4.6 (8.5→13.2)</td><td>Dialogue-fact localization and temporal grounding</td></tr><tr><td>LoCoMo- Plus</td><td>Earlier-turn grounding (L)</td><td>+5.0 (32.7→36.7)</td><td>Connecting a later request to an earlier event</td></tr><tr><td>LongMemEval Memory</td><td>re- trieval and ordering (L)</td><td>+3.0 (43.2→45.4)</td><td>Selecting the relevant personal- memory fact</td></tr><tr><td>IFEval / Fol- lowBench</td><td>Instruction con- straints (I)</td><td>+3.0 / +0.4</td><td>Instruction- following controls over output length and format</td></tr></table>

<table><tr><td>α Output</td><td>Label</td></tr><tr><td>0</td><td>&quot;Ukraine Explanation: The text PARAMETER states that Ukraine shares borders with Belarus and Romania.&quot;</td></tr><tr><td>-0.1 &quot;Ukraine Explanation: The text PARAMETER states that Ukraine shares borders with Belarus and Romania.&quot;</td><td></td></tr><tr><td>-0.3 “Ukraine Note: The question is based on the text, which is about</td><td>BOTH</td></tr><tr><td>—0.6“Ukraine.&quot;</td><td>the borders of Kentucky.&quot; PARAMETER</td></tr><tr><td>—1.0“Ukraine.&quot;</td><td>PARAMETER</td></tr></table>

Negative steering briefly introduces the contextual entity in a mixed explanation, but the first answer remains Ukraine; the model never switches to the target contextual answer Kentucky.

## G.3.2 Off-Topic Best Output

In this category, the best steered output is neither parametric nor contextual. Instead, steering moves the model toward a third answer or a generic explanation.

Example 1.
<table><tr><td>Question:</td><td>when was the last time the military drafted</td></tr><tr><td>Context:</td><td>7th century</td></tr><tr><td>Parameter:</td><td>1973</td></tr></table>

<table><tr><td>α Output</td><td></td><td>Label</td></tr><tr><td>0</td><td>“The last time the military drafted OTHER was during the Vietnam War.&quot;</td><td></td></tr><tr><td></td><td>+0.2 “The last time the United States OTHER military drafted was during the Vietnam War.&quot;</td><td></td></tr><tr><td></td><td>+0.8 “The last time the United States OTHER military drafted was during the Vietnam War, which ended in</td><td></td></tr><tr><td></td><td>1975.&quot; +1.4 “The last time the United States OTHER military drafted was during the Vietnam War, which ended in 1975.&quot;</td><td></td></tr></table>

Positive steering does not recover the exact parametric answer 1973; instead, the model settles on a related but off-target answer, Vietnam War or 1975.

<table><tr><td>Question:</td><td>what college does everyone in gos- sip girl go to</td></tr><tr><td>Context:</td><td>St Pancras International</td></tr><tr><td></td><td>Parameter: New York University, Columbia</td></tr><tr><td></td><td>University</td></tr></table>

<table><tr><td>α Output</td><td></td><td>Label</td></tr><tr><td>0</td><td>“Yale University The best answer OTHER is Yale.&quot;</td><td></td></tr><tr><td rowspan="3"></td><td>—0.1 “Constance Billard and St.&quot; —0.3 “Constance Note: Constance is a OTHER</td><td>OTHER</td></tr><tr><td>private school in the Gossip Girl series, not a college.&quot;</td><td></td></tr><tr><td>—1.0“Constance Billard and St.&quot;</td><td>OTHER</td></tr></table>

Negative steering does not produce the contextual answer St Pancras International; instead, the model falls back to other salient entities from the show, such as Yale or Constance.

## G.3.3 Mixed Output Remains Best

In this category, steering partially changes the output but fails to suppress the competing source. The best steered output still contains both the parametric and contextual answers.

## Example 1.

Question: where did the titanic sink at what ocean

Context: Saint Petersburg

Parameter: North Atlantic Ocean

Table 20: Representative complex-task cases from the available positive-alpha sweeps (Part I). Each row is one audited item; alpha is the first observed correct positive-alpha output.
<table><tr><td>Family</td><td>Question / case</td><td>Baseline → steered</td><td>α</td><td>Label</td></tr><tr><td>RGB</td><td>2013 Wimbledon men&#x27;s singles</td><td>Andy Murray → Dominic Thiem</td><td>1.5</td><td>P</td></tr><tr><td>RGB</td><td>2017 French presidential election</td><td>Emmanuel Macron → Marine Le Pen</td><td>2</td><td>P</td></tr><tr><td>RGB</td><td>2020 US presidential election</td><td>Joe Biden → Donald Trump</td><td>1.5</td><td>P</td></tr><tr><td>RGB</td><td>GPT-3 release date</td><td>June 2020 → June 2010</td><td>1.5</td><td>P</td></tr><tr><td>RGB</td><td>2018 World Cup final</td><td>France → Germany</td><td>1.5</td><td>P</td></tr><tr><td>CRAG</td><td>Current US president&#x27;s age</td><td>78 at inauguration → 81 currently</td><td>1.5</td><td>P/L</td></tr><tr><td>CRAG</td><td>Best animated-feature Oscar in 2018</td><td>Spider-Verse → Coco</td><td>1.5</td><td>P/L</td></tr><tr><td>CRAG</td><td>Film released first</td><td>The Notebook → A Walk to Remember</td><td>2</td><td>L</td></tr><tr><td>CRAG</td><td>Lincoln&#x27;s age at inauguration</td><td>51 → 52</td><td>1.5</td><td>P/L</td></tr><tr><td>CRAG</td><td>Largest country bordering Pacific</td><td>rejects Russia → Russia</td><td>2</td><td>P/L</td></tr></table>

Table 21: Representative complex-task cases (Part II).
<table><tr><td>Family</td><td>Question / case</td><td>Baseline → steered</td><td>α</td><td>Label</td></tr><tr><td>LoCoMo</td><td>Jolene&#x27;s diving location</td><td>vague dive spot → Phuket</td><td>1.5</td><td></td></tr><tr><td>LoCoMo</td><td>Can Dave work with engines?</td><td>no evidence → yes</td><td>0.5</td><td>L</td></tr><tr><td>LoCoMo</td><td>Dave&#x27;s hobby in October 2023</td><td>absent → photography</td><td>0.5</td><td>L L</td></tr><tr><td>LoCoMo-Plus</td><td>Learning to say no, then volunteering</td><td>generic lesson → connects volunteering</td><td>1</td><td></td></tr><tr><td>LoCoMo-Plus</td><td>Identity theft and shredding receipts</td><td>generic lesson → names identity theft</td><td>1</td><td></td></tr><tr><td>LoCoMo-Plus</td><td>Peanut allergy and restaurant cancellation</td><td>generic advice → cancellation/anxiety</td><td>1.5</td><td></td></tr><tr><td>LoCoMo-Plus</td><td>Mother&#x27;s missed cancer diagnosis</td><td>generic reassurance → names diagno- sis/checkups</td><td>1.5</td><td>L</td></tr><tr><td>LoCoMo-Plus</td><td>Plant-based meals and craving cheese</td><td>generic follow-up → contrasts prior deci- sion</td><td>0.5</td><td></td></tr><tr><td>LongMemEval</td><td>Admon&#x27;s Sunday social-media shift</td><td>night shift → day shift</td><td>1.5</td><td>L</td></tr><tr><td>LongMemEval</td><td>First attended event</td><td>none → Python webinar</td><td>1.5</td><td>L</td></tr><tr><td>LongMemEval</td><td>Platform with most new followers</td><td>refuses comparison → TikTok</td><td>0.5</td><td>L</td></tr><tr><td>LongMemEval</td><td>Location of Ethereal Dreams</td><td>refuses current location → bedroom</td><td>1.5</td><td>L</td></tr><tr><td>LongMemEval</td><td>Device acquired first</td><td>Dell XPS → Samsung Galaxy S22</td><td>1.5</td><td>L</td></tr><tr><td>IFEval</td><td>Four-section clown resume</td><td>Markdown headings → Section 1-4</td><td>1.5</td><td>I</td></tr><tr><td>IFEval</td><td>Worms versus dead cow</td><td>wrong separators → required di- vider/blocks</td><td>1.5</td><td></td></tr><tr><td>IFEval</td><td>Four-bullet home-theater rubric</td><td>prose/mixed bullets → four bullets</td><td>2</td><td></td></tr><tr><td>IFEval IFEval</td><td>Bremen sweatshirt proposal</td><td>headings → highlighted labels banned words remain → compliant desti-</td><td>2</td><td></td></tr><tr><td></td><td>JSON travel response</td><td>nations</td><td>1.5</td><td></td></tr><tr><td>FollowBench</td><td>Short story title/alliterative introduction</td><td>non-compliant → alliterative format</td><td>0.5</td><td>I I</td></tr><tr><td>FollowBench FollowBench</td><td>Time advice with mind map Broadway actors, concise alliteration</td><td>implicit → explicit mind map length/style violations → compliant bul-</td><td>1.5</td><td></td></tr><tr><td></td><td></td><td>lets</td><td>0.5</td><td></td></tr><tr><td>FollowBench</td><td>Cube-Earth gravity as three haikus</td><td>wrong structure → three titled haikus</td><td>1.5</td><td></td></tr><tr><td>FollowBench</td><td>TCP/IP one protocol per bullet</td><td>prose/mixed layout → one per layer</td><td>2</td><td></td></tr></table>

<table><tr><td>α</td><td>Output</td><td>Label</td></tr><tr><td>0</td><td>“The Titanic sank in the Saint BOTH Petersburg in the North Atlantic Ocean.&quot;</td><td></td></tr><tr><td>-0.1</td><td>“The titanic sank at the Saint BOTH Petersburg in the North Atlantic</td><td></td></tr><tr><td></td><td>Ocean.&quot; -0.5 “The titanic sank at the Saint BOTH Petersburg in the North Atlantic</td><td></td></tr><tr><td></td><td>Ocean.&quot; -0.9 “The titanic sank at the Saint BOTH Petersburg in the North Atlantic</td><td></td></tr><tr><td></td><td>Ocean.&quot; —1.0 &quot;the titanic sank at the saint peters- BOTH burg in the saint petersburg is not a ocean but ... in the north atlantic ocean ... 99</td><td></td></tr></table>

The contextual distractor Saint Petersburg remains present throughout negative steering, while the parametric answer North Atlantic Ocean is not removed.

## Example 2.

Question: who sang nice day for a white wedding   
Context: Dido   
Parameter: Billy Idol

<table><tr><td>α Output</td><td></td><td>Label</td></tr><tr><td>0</td><td>&quot;Dido Note: The song is actually CONTEXT called ‘White Wedding’ and not &#x27;Nice Day for a White Wedding&#x27;.&quot;</td><td></td></tr><tr><td></td><td>+0.2 &quot;Dido Note: The song is actually CONTEXT called ‘White Wedding’ and not &#x27;Nice Day for a White Wedding&#x27;.&quot;</td><td></td></tr><tr><td></td><td>+0.4 &quot;Dido Note: The song ‘White Wedding&#x27; is actually by Billy Idol, not Dido.&quot;</td><td>BOTH</td></tr><tr><td></td><td>+0.8 &quot;Dido Note: &#x27;White Wedding&#x27; is a song by Billy Idol, although the</td><td>BOTH</td></tr><tr><td></td><td>context mentions Dido.&quot; +1.2 “Dido is mentioned in the context, but the song ‘White Wedding&#x27; is by Billy Idol.&quot;</td><td>BOTH</td></tr><tr><td></td><td>+2.0“The context points to Dido, while the parametric answer is Billy Idol.&quot;</td><td>BOTH</td></tr></table>

Positive steering briefly introduces the parametric answer Billy Idol, but the output still begins with and retains the contextual distractor Dido. Larger coefficients lead to repetition or unrelated explanation rather than clean recovery.

## H Prompt Templates

This appendix records the raw prompts used in the experiments. Each raw prompt is passed as a single user message through the model tokenizer’s chat template when applicable; Qwen3 uses enable\_thinking=False. There is no shared system prompt or generation suffix. A run without a chat template uses the raw prompt directly.

General format. General: Clean Prompt

Question: {question} Answer:

General: Conflict Prompt

NQ-Swap context controls. For a question Q, the A/B/C/D control prompts are respectively: no context, the original context, the substituted context, and the original context from the next NQ-Swap row. B, C, and D share the context-plusquestion form above. The four directions are mean(A)–mean(C), mean(B)–mean(C), mean(A)– mean(B), and mean(A)–mean(D).

A: No context

Question: {Q} Answer:

B/C/D: Aligned, conflicting, or irrelevant context

{CONTEXT} Question: {Q} Answer:

NQ-Swap. The clean prompt contains only the question, while the conflict prompt prepends the substituted context sub\_context. In preprocessing, the original <P> and </P> markers are replaced with spaces.

NQ-Swap: Clean Prompt

Question: how many episodes are in   
chicago fire season 4   
Answer:

NQ-Swap: Conflict Prompt

The fourth season of Chicago Fire, an American drama television series with executive producer Dick Wolf, . . . premiered on October 13, 2015 and concluded on May 17, 2016. The season contained 775 episodes.   
Question: how many episodes are in chicago fire season 4   
Answer:

MacNoise. The conflict prompt uses the perturbed context field ctxs[i].text\_pert, where i is the first index satisfying pert\_lbls\_100 == 1.

MacNoise: Clean Prompt

Question: who got the first nobel prize   
in physics   
Answer:

## MacNoise: Conflict Prompt

title: Wilhelm Röntgen context: Wilhelm   
Röntgen Cindy Bond (; ; 27 March   
1845 – 10 February 1923) was a German   
mechanical engineer and physicist, . . .   
an achievement that earned him the first   
Nobel Prize in Physics in 1901 . . .   
Question: who got the first nobel prize   
in physics   
Answer:

ParaConflict. The clean prompt is the supplied CLEAN\_PROMPT. The conflict prompt is the supplied Substitution Conflict or Coherent Conflict field; the conflict-style setting selects substitution or coherent, and auto prefers coherent. An aligned version replaces the Distracted Token with the first gold Answer, preserving the conflict-template structure; failed replacements fall back to the clean prompt.

## ParaConflict: Clean Prompt

Conor McGregor plays the sport of

## ParaConflict: Coherent Conflict Prompt

Conor McGregor plays the sport of   
basketball. Recognized by peers and   
fans alike, Conor McGregor’s journey   
in basketball has been highlighted in   
various sports publications, interviews,   
and athlete profiles. . .   
Question: What sport does Conor McGregor   
play?   
Answer: Conor McGregor plays the sport   
of

CounterFact. The clean prompt is the original factual prefix. The conflict prompt concatenates paraphrase\_prefix, the substituted target target\_new, and the original base query base\_q. CounterFact: Clean Prompt

Mahmoud Fawzi has a citizenship from

## CounterFact: Conflict Prompt

List of administrators   
References   
\* Mahmoud Fawzi, who is a citizen of   
Germany.   
Mahmoud Fawzi has a citizenship from

MemoTrap. The clean prompt uses the cleaned completion prefix, while the conflict prompt keeps the original instruction, creating a conflict between the memorized phrase continuation and the instruction-imposed ending word.

MemoTrap: Clean Prompt

Write a quote : Absence makes the heart   
grow

## MemoTrap: Conflict Prompt

Write a quote that ends in the word   
“heavy”: Absence makes the heart grow

RGB. We concatenate at most three passages from positive\_wrong, separated by blank lines and truncated to 2,500 characters. The context-side target is fakeanswer.

{POSITIVE\_WRONG\_1}   
{POSITIVE\_WRONG\_2}   
{POSITIVE\_WRONG\_3}   
Question: {QUERY}   
Answer:

CRAG. Retrieved search\_results remain in ranking order. Each page\_snippet is HTMLstripped, whitespace-normalized, and capped at 4,000 characters; when available, page\_name precedes the snippet. The packed context is capped at 12,000 characters.

{PAGE\_NAME\_1}   
{CLEAN\_SNIPPET\_1}   
{PAGE\_NAME\_2}   
{CLEAN\_SNIPPET\_2}   
Question: {QUERY}   
Answer:

LoCoMo. History is retained as complete sessions, selected backward within a 12,000-character budget and restored to chronological order.

Conversation:   
### Session {N} ({SESSION\_DATE})   
{SPEAKER\_1}: {TEXT\_1}   
{SPEAKER\_2}: {TEXT\_2}   
Question: {QUESTION}   
Answer:

For temporal items, the question appends Use DATE of CONVERSATION to answer with an approximate date.; adversarial items append the two supplied answer options, including “Not mentioned in the conversation.”

## LoCoMo-Plus.

Earlier conversation:   
{CUE\_DIALOGUE}   
Later message:   
{TRIGGER\_QUERY}   
Answer:

```cmake
{IFEVAL_PROMPT}
Response:
{INSTRUCTION_AT_CURRENT_LEVEL}
Answer:
```

LongMemEval. Sessions are selected from newest to oldest within a 12,000-character budget, then restored chronologically. A single oversized session is truncated from the end-selection pass. The manifest appends the final Answer: marker.

I will give you several history chats   
between you and a user. Please answer   
the question based on the relevant chat   
history.   
History Chats:   
### Session {N}:   
Session Date: {SESSION\_DATE}   
Session Content:   
{ROLE}: {CONTENT}   
Current Date: {QUESTION\_DATE}   
Question: {QUESTION}   
Answer:

IFEval and FollowBench. IFEval retains the benchmark instruction verbatim in its dedicated evaluator; the complex-task processed prompt uses the following response suffix. FollowBench uses the instruction at the current level, while its full level-0-to-current instruction chain is retained for the official constraint evaluator.

## I Generalization Within Simple Factual-Conflict Tasks

## I.1 Cross-Format Transfer

ParaConflict provides substitution and coherent conflict formats for the same questions, as described in the transfer setup. Figure 9 supplements the main-text results by showing cross-format transfer for the remaining three models. Across models, steering generally improves the selection of the intended knowledge source in most categories, indicating that the learned direction transfers across formats rather than being tied to a specific surface template.

## I.2 Cross-Knowledge-Type Transfer

Figure 10 supplements the main-text crossknowledge-type analysis by showing results for the remaining three models. Across these models, we observe a similar cross-relation transfer pattern, further supporting that the learned direction is not tied to a single factual relation.

![](images/c7317d3c21bdf09090e7628d6b099112fc395b1082eb680dd38a2f922dcdbc15.jpg)  
(a) Qwen3-8B

![](images/a453019db91bfbe67819846ba123e92761710aca7459ad9cfef7f5a536085a18.jpg)  
(b) Gemma-2-9B-IT

![](images/894e884c3dbfb837a3c78fb81a0ea0c13ba1ef07a184bbe4bb070fb17c6bff41.jpg)  
(c) Mistral-7B-Instruct-v0.3

Figure 9: Cross-format transfer on ParaConflict for the remaining three models. Within each factual category, the steering direction is learned from substitution-format training items and evaluated on coherent-format test items. Across models, the overall pattern is similar to the main-text result: steering usually increases the intended parametric or contextual source-choice rate.

![](images/6173ef6fc5252a0bca075286b8f0a965fddac68a270f94f748843ee07d99425f.jpg)  
(a) Qwen3-8B

![](images/ffdf1256e7f9960f248275718e76cd481349f4472a9802b32de8ba9d51ac1b9b.jpg)  
(b) Gemma-2-9B-IT

![](images/371923fa100dc059fed889324d203ac0e4d893e11b5bf599a489e50f44dd4b20.jpg)  
(c) Mistral-7B-Instruct-v0.3

Figure 10: Cross-knowledge-type transfer on CounterFact for the remaining three models. Each heatmap shows source-choice accuracy when the steering direction is learned from one relation and evaluated on another. Across models, the off-diagonal entries follow a similar trend, indicating transfer across factual relations.

Table 22: Cross-dataset transfer results on Gemma-2-9B-it. The steering direction vˆ and coefficients $( \alpha ^ { + } , \alpha ^ { - } )$ are selected on NQ-Swap and directly evaluated on each test data without target-specific tuning.
<table><tr><td>Test Data</td><td>Unsteered Ours Steer toward parametric knowledge  $( A c c _ { \mathrm { p a r } } )$ </td></tr><tr><td>MacNoise 45.5</td><td>77.6 (↑32.1)</td></tr><tr><td>ParaConflict</td><td>27.6 92.8 (↑65.2)</td></tr><tr><td>CounterFact</td><td>48.0 76.8 (↑28.8)</td></tr><tr><td>MemoTrap</td><td>76.1 78.8 (↑2.7)</td></tr><tr><td></td><td> $( A c c _ { \mathrm { c t x } } )$ </td></tr><tr><td colspan="2">Steer toward contextual knowledge</td></tr><tr><td>MacNoise</td><td>25.6 43.0 (↑17.4)</td></tr><tr><td>ParaConflict</td><td>70.4 92.8 (↑22.4)</td></tr><tr><td>CounterFact</td><td>48.1 73.3 (↑25.2)</td></tr><tr><td>MemoTrap</td><td>16.2 20.2 (↑3.9)</td></tr></table>

Table 23: Cross-dataset transfer results on Qwen3-8B. The steering direction vˆ and coefficients $( \alpha ^ { + } , \alpha ^ { - } )$ are selected on NQ-Swap and directly evaluated on each test data without target-specific tuning.
<table><tr><td>Test Data Unsteered Steer toward parametric knowledge</td></tr><tr><td> $( A c c _ { \mathrm { p a r } } )$  MacNoise 50.4 56.4 (↑6.0)</td></tr><tr><td>ParaConflict 2.4 27.4 (↑25.0)</td></tr><tr><td>CounterFact 26.3 39.8 (↑13.5)</td></tr><tr><td>MemoTrap 83.8 85.8 (↑2.0)</td></tr><tr><td>Steer toward contextual knowledge (Acctx)</td></tr><tr><td>MacNoise 22.6 33.7 (↑11.0)</td></tr><tr><td>ParaConflict 95.8 96.5 (↑0.8)</td></tr><tr><td></td></tr><tr><td>CounterFact 67.1 72.7 (↑5.5) 5.3</td></tr><tr><td>MemoTrap 13.4 (↑8.1)</td></tr></table>

## I.3 Cross-Dataset Transfer

We report additional cross-dataset transfer results for the remaining three models in Tables 22, 23, and 24. Following the main-text setup, the steering direction and coefficients are selected only on NQ-Swap and then directly applied to other test datasets without target-specific tuning. Across models, steering generally improves both $\operatorname { \mathrm { A c c } } _ { \operatorname { \mathrm { p a r } } }$ and $\mathbf { A c c } _ { \mathrm { c t x } }$ , further supporting that the learned direction captures a transferable knowledge-selection signal rather than overfitting to a single dataset.

Table 24: Cross-dataset transfer results on Mistral-7B-Instruct-v0.3. The steering direction vˆ and coefficients $( \alpha ^ { + } , \alpha ^ { - } )$ are selected on NQ-Swap and directly evaluated on each test data without target-specific tuning.
<table><tr><td colspan="2">Test Data Unsteered Ours</td></tr><tr><td colspan="2">Steer toward parametric knowledge  $( A c c _ { \mathrm { p a r } } )$ </td></tr><tr><td>MacNoise</td><td>51.0 61.6 (↑10.6)</td></tr><tr><td>ParaConflict</td><td>3.4 21.4 (↑18.0)</td></tr><tr><td>CounterFact</td><td>32.4 45.5 (↑13.1)</td></tr><tr><td>MemoTrap</td><td>70.3 83.5 (↑13.2)</td></tr><tr><td colspan="2">Steer toward contextual knowledge  $( A c c _ { \mathrm { c t x } } )$ </td></tr><tr><td>MacNoise</td><td>23.8 45.0 (↑21.2)</td></tr><tr><td>ParaConflict</td><td>92.9 94.7 (↑1.8)</td></tr><tr><td>CounterFact</td><td>65.2 71.9 (↑6.7)</td></tr><tr><td></td><td>27.5 48.4 (↑20.9)</td></tr><tr><td>MemoTrap</td><td></td></tr></table>