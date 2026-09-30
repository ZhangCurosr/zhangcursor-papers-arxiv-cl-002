# WHAT MAKES RECURRENCE EFFECTIVE IN LOOPED LANGUAGE MODELS?

Xinlin Zhuang<sup>1,2</sup> Siyuan Wang<sup>1</sup> Imran Razzak<sup>2</sup> Weiyang Liu<sup>1,\*</sup>

<sup>1</sup>The Chinese University of Hong Kong <sup>2</sup>MBZUAI

## ABSTRACT

Looped language models (LoopLMs) increase computational depth through parameter sharing, offering a path to scale inference computation without adding parameters. However, it remains unclear when additional recurrence is beneficial and how architectural choices affect its effectiveness. Through controlled experiments, we systematically examine (1) when recurrence helps, (2) where it should be applied, and (3) how its conditioning affects performance. Our evaluation covers inference budgets below, within, and beyond the training horizon under knowledge and reasoning tasks. (1) We find that recurrence can improve reasoning beyond the training horizon while degrading knowledge performance, but harder reasoning instances do not consistently benefit more. (2) Performance also depends on how distinct layers and recurrent iterations are allocated, showing that effective depth alone is insufficient to predict behavior. Non-recurrent output layers improve robustness to under-unrolling, while the preferred placement of input and output layers varies with inference budget. (3) Finally, we find that conventional initial-state injection offers limited robustness to varying recurrence depth. We therefore propose history-state injection as an alternative, and show that channel-wise history-state injection combined with timestep conditioning offers a low-cost and more effective design, better preserving knowledge under extended unrolling while improving robustness across inference budgets. Overall, our results clarify when recurrent computation helps, where it fails, and offer practical guidelines for designing LoopLMs across variable inference budgets.

## 1 INTRODUCTION

Scaling large language models (LLMs) has driven substantial capability gains, but at the cost of rapidly growing parameter and memory requirements, making further scaling increasingly costly to train and deploy. Looped language models (LoopLMs) have emerged as a parameter-efficient alternative by repeatedly executing a shared stack of Transformer blocks (Geiping et al., 2025; Saunshi et al., 2025; Zhu et al., 2025; Jeddi et al., 2026). By reusing parameters across iterations, LoopLMs enable deeper computation with a smaller memory footprint. More importantly, their recurrent structure allows inference depth to be flexibly adjusted by varying the loop count, providing a natural mechanism for scaling test-time computation (Alabdulmohsin & Zhai, 2025).

However, the potential of LoopLMs for test-time scaling remains underexplored. Existing work primarily evaluates models at the fixed recurrent depth during training (hereafter referred to as the training horizon) or focuses on early exiting, leaving it unclear whether additional depth beyond the training horizon continue to provide useful computation and performance gains, particularly for computation-intensive tasks. Meanwhile, recent LoopLM architectures explores a growing design space, varying the size and count of recurrent blocks, where recurrence is placed within the network, and how recurrent computation is conditioned across iterations. While these designs can improve performance within the training horizon, they do not necessarily guarantee improvement during continued test-time unrolling. This raises a fundamental question: can recurrence remain effective as inference computation scales beyond its training horizon, and whatfactors determine this behavior?

We begin by characterizing test-time recurrent scaling using a minimalist LoopLM without specialized architectural designs (Sec. 3). We pre-train variants configured with different combinations of physical depth (the number of distinct layers) and loop count under a fixed effective training depth (the product of physical depth and loop count), and vary their inference depth. Surprisingly, even this simple design scales substantially beyond its training horizon, with continued test-time unrolling further improving performance and certain configurations even outperforming non-recurrent counterparts under equivalent training compute. However, this scaling behavior is not universal. Reasoning tasks can benefit from additional loops while knowledge-oriented tasks degrade, and greater reasoning depth does not consistently lead to larger gains from further unrolling. Moreover, different combinations of physical depth and loop count exhibit markedly different scaling behaviors despite sharing the same effective training depth. These observations reveal both the potential and instability of recurrent test-time scaling: additional loops can unlock extra computation, their efficacy is strongly conditioned on task semantics and how recurrence is configured.

Motivated by this variability, we systematically investigate two factors that shape test-time recurrent scaling (Sec. 4, 5): which computations should be recurrent, and how recurrent computation should be conditioned. For the former, we find that independently parameterized blocks surrounding the recurrent core play distinct roles across inference regimes. Output-side non-recurrent blocks improve knowledge robustness to under-unrolling, whereas allocating more non-recurrent computation to the input side tend to better support reasoning extrapolation; near the training horizon, performance is less sensitive to their allocation. For the latter, we observe that initial-state injection degrades knowledge robustness during loop extrapolation when over-parameterized. We therefore propose history-state injection, which conditions on relative differences from intermediate states to capture dynamic trajectories and substantially rescues deep extrapolation performance. Timestep conditioning further improves reasoning extrapolation by allowing the shared computation to vary across recurrent iterations, although its effectiveness depends on the recurrent configuration.

Building on these insights, we introduce a lightweight conditioning framework that jointly incorporates history-state injection and timestep conditioning through channel-wise parameterization. This design alleviates the limitations of either conditioning scheme alone, yielding complementary gains and consistently outperforming both individual variants and the unconditioned BaseLoop baseline under extended inference budgets (Fig. 7(c)). In summary, our contributions are threefold:

• Characterizing recurrent scaling. We present a systematic empirical study characterizing LoopLMs beyond their training horizon, showing substantial test-time scaling potential but also strong dependence on specific tasks, reasoning depth, and recurrent configuration.

• Understanding effective recurrence. We identify key factors, including non-recurrent block allocation, dynamic state history, and timestep conditioning, that govern extrapolation behavior across knowledge and reasoning tasks.

• Designing a synergistic LoopLM. We introduce a lightweight joint conditioning mechanism combining our proposed history-state injection and timestep conditioning, achieving robust performance gains on both knowledge and reasoning tasks across varied inference budgets.

## 2 PRELIMINARIES AND EVALUATION SETTINGS

## 2.1 FORMULATION

Following Geiping et al. (2025), a LoopLM causal decoder typically comprises three groups of Transformer blocks: a p-block Prelude $P ,$ , an s-block parameter-shared recurrent core $F .$ , and a c-block Coda C. Omitting the token embedding $e ( \cdot )$ , the final normalization, and the languagemodel head $W$ , which are identical across all models we study, a looped decoder can be written as $\mathcal { M } _ { p , s , c , K } \ : = \ : C \circ F ^ { K } \circ P$ . Here, $F ^ { K }$ denotes $K$ successive applications of the shared core and K is the loop count during training, termed the training horizon. At inference, the model can execute a varying number of loops $r ,$ corresponding to under-unrolling $( r < K )$ , evaluation at the training horizon $( r = K )$ , or loop extrapolation $( r > K )$ ). Given an input sequence $x ,$ the Prelude initializes the recurrent state as $\tilde { \boldsymbol { x } } \overset { \cdot } { = } P ( e ( \boldsymbol { x } ) ) \in \mathbb { R } ^ { n \times d }$ . The recurrent computation then evolves as

$$
h ^ { 0 } = \tilde { x } ; \qquad h ^ { t + 1 } = F \bigl ( h ^ { t } ; \gamma _ { t } \bigr ) , \quad 0 \le t < r ; \qquad y = W C ( h ^ { r } ) ;\tag{1}
$$

where $\gamma _ { t }$ denotes auxiliary signals that condition the recurrent trajectory, such as the initial state, past recurrent states, or timestep information (Geiping et al., 2025; Fein-Ashley & Rashidinejad, 2026; Xu & Sato, 2025). $\gamma _ { t }$ is optional and setting $\gamma _ { t } = \mathcal { D }$ recovers a standard recurrence $\boldsymbol { h ^ { t + 1 } } = \boldsymbol { F } ( \boldsymbol { h ^ { t } } )$ We discuss different forms of recurrent conditioning strategies and their impacts in Sec. 5.

![](images/1bb1e738a7bbd10d5eed0aa52d0ddab81ee9031e89ad66622f96e29d0e22c4f6.jpg)

![](images/9339443286b7dacac11a71132e96fb597dadbc0e59ae52a87f18b718bc2a51f6.jpg)

![](images/45d2456b651fba83a2d7297d6a2cfb7233a664e89d1631c71ae4cc7dd68751c2.jpg)

![](images/021d25c3a21ea5c09425af3f988af9699afcfcee246440bf78f653b4f6b52e8f.jpg)  
Figure 1: Task- and architecture-dependent benefits of recurrence. (a) Score changes relative to $L ( r ) = 2 0 $ for BaseLoop $4 \times 5$ across knowledge and reasoning splits. (b,c) Absolute accuracy (%) for BaseLoop $2 \times 1 0$ across inference depths, stratified by ProofWriter proof depth and CLUTRR chain length. (d) Reasoning scores across varying configurations of physical depth and loop count. In (a,d), dashed vertical lines mark the training depth, and shaded parts indicate the depth extrapolation.

We distinguish the model’s physical depth, $L _ { \mathrm { p h y s } } = p + s + c _ { \mathrm { i } }$ from its effective depth, $L ( r ) =$ $p + r \cdot s + c .$ . The former determines the number of independently parameterized Transformer layers, while the latter characterizes the real per-token compute. Accordingly, $L ( K )$ denotes the training effective depth and varying r controls the inference effective depth $\bar { L ( r ) }$ . A special case of LoopLM is when $p = c = 0$ , i.e. $\mathbf { \bar { \mathcal { M } } } _ { 0 , s , 0 , K } = F ^ { K }$ (Saunshi et al., 2025), where the entire network acts as the shared recurrent core, with physical depth s, effective depth $K \cdot s ,$ and no independently parameterized blocks surrounding the recurrence. We term this configuration BaseLoop and refer to LoopLMs with $p + c > 0 .$ , which places non-recurrent boundary blocks on one or both sides of the core, as CoreLoop for later comparisons. Detailed comparisons between them are discussed in Sec. 4.

## 2.2 CONTROLLED EXPERIMENTAL SETUP

As recurrence trades parameters against compute, comparing architectural variants is meaningful only under controlled computation budgets. Every comparison in this paper matches four quantities: (i) physical depth $L _ { \mathrm { p h y s } } = p + s + c ; \mathrm { ( i i ) }$ training effective depth $L ( K ) = p + K \cdot s + c ;$ (iii) inference effective depth $L ( r ) ;$ ; and (iv) the training pipeline, including the token budget, optimizer, etc. Further details of model architectures, data composition, training and evaluation are provided in App. C.

Models and Data. We study LoopLMs in a controlled pre-training-from-scratch setup built on two dense backbone families: Llama3.1-1B (Grattafiori et al., 2024) and Qwen3-0.6B (Yang et al., 2025). We adopt their layer configurations, widths, and tokenizers, while randomly initializing all parameters. Models are trained on FineWeb-Edu (Penedo et al., 2024) with the standard next-token prediction objective, packing documents into 2048-token sequences without padding. Unless otherwise specified, the training token budget follows the Chinchilla ratio of 20 tokens per parameter, with parameter counts estimated from the model’s effective training depth $L ( K )$

Training. Each model is trained with a fixed loop count K, executing exactly K recurrent iterations per sample without adaptive halting or early exits (Bae et al., 2025). Gradients are backpropagated through all K iterations without truncation, and the causal language modeling loss is applied. We use the Muon optimizer (Jordan et al., 2024; Liu et al., 2025) for hidden matrix parameters, and AdamW for embeddings, output heads, biases, and other non-matrix parameters. All other optimization settings, including the learning-rate scheduler, warmup, and weight decay are held consistent for fair comparison. A detailed optimizer ablation for LoopLM training is provided in App. D.1.

## 2.3 EVALUATION ACROSS COMPUTATIONAL DEMANDS

To investigate whether and when additional recurrent computation benefits distinct task capabilities, we evaluate LoopLMs along two complementary axes: the computational demand ofthe task and the executed depth at inference. This reveals capability-specific dynamics that are otherwise obscured by the aggregate metrics in prior LoopLM evaluations (Jeddi et al., 2026; Geiping et al., 2025).

Knowledge vs. Reasoning. We partition downstream benchmarks into knowledge and reasoning groups according to the computation required to produce an answer. Knowledge tasks primarily rely on facts stored within model weights and require limited multi-step composition, whereas Reasoning tasks require composing information through multiple inference steps and may therefore benefit more from additional recurrent computation. The knowledge group includes SciQ (Welbl et al., 2017), ARC-Easy (Clark et al., 2018), and PIQA (Bisk et al., 2020); the Reasoning group includes ARC-Challenge (Clark et al., 2018), WinoGrande (Sakaguchi et al., 2020), OpenBookQA (Mihaylov et al., 2018), HellaSwag (Zellers et al., 2019), CommonsenseQA (Talmor et al., 2019), ProofWriter (Tafjord et al., 2021), CLUTRR (Sinha et al., 2019), and BBH (Suzgun et al., 2023). We report the unweighted mean within each group, with the aggregate across groups as a summary. Beyond this binary split, ProofWriter and CLUTRR provide controlled reasoning-depth axes, defined by proof depth and relation-chain length, respectively. This allows us to examine whether the utility of additional recurrent computation changes with the amount of reasoning required by the task.

Different Inference Budgets: Under-Unrolling, Training Horizon, and Loop Extrapolation. For each model trained with a fixed loop count K, we vary the inference loop count r at test time without any additional training to evaluate three computational regimes: under-unrolling $( r < K )$ , the training horizon $( r = K )$ , and loop extrapolation $( r > K )$ . We report performance as a function of the resulting effective depth $L ( r )$ , allowing us to track how different capabilities respond as recurrent computation is reduced, matched to training, or extended beyond the training horizon.

## 3 WHEN DOES RECURRENCE HELP?

To investigate whether and under what conditions additional recurrent depth provides useful test-time compute, we begin by analyzing BaseLoop models following the Llama3.1-1B architecture. We train several variants with different loop configurations, $K \times s \in \{ 2 \times 1 0 , 4 \times 5 , 5 \times 4 , 1 0 \times 2 \}$ , while fixing the effective training depth at $L ( \bar { K } ) = 2 0$ , where $K \times s$ denotes s recurrent unrolls over a physical core of depth K. All models share identical training configurations and are evaluated across under-unrolling, training horizon, and extrapolation settings. A standard non-recurrent baseline $( 2 0 \times 1$ , termed NonLoop) whose physical depth matches the effective depth of the loop models $( L ( K ) = 2 0 )$ serves as an upper-bound performance reference under equal compute.

Additional recurrence can improve reasoning beyond the training horizon. Fig. 1(a) illustrates how inference depth scaling affects knowledge and reasoning performance for BaseLoop $2 \times 1 0$ Within the training horizon $( L ( r ) \leq 2 0 , r \leq 4 )$ , increasing inference depth improves both task groups. During extrapolation $( L ( r ) > 2 0 , r > 4 )$ , however, the two groups diverge. Reasoning performance continues to improve, rising from 28.52 at $L ( r ) = 2 0 $ to 31.49 at $L ( r ) = 4 0 , \mathrm { { a } } 2 . 9 7$ percentage point gain achieved purely at test-time without parameter updates. In contrast, knowledge performance decreases, dropping from 62.80 to 52.11. Thus, the training horizon does not impose a strict ceiling on effective recurrence and test-time depth extrapolation selectively benefits tasks with higher computational demands while failing to scale static knowledge memorization.

Gains vary across reasoning demands and complexities. The stratified results in Fig. $^ { 1 ( \mathbf { b } , \mathbf { c } ) }$ reveal that the additional recurrence also varies across reasoning benchmarks and complexity levels. On ProofWriter, extending to the extrapolation range $L ( r ) = \bar { 2 } 4$ consistently boosts accuracy across all proof depths (0–5), yielding absolute gains of around 7.5 percentage points at deeper proof depths (depth 3-5). Conversely, the gains on CLUTRR vary significantly across relation-chain lengths. Extrapolating to $L ( r ) = 2 4 $ yields large accuracy jumps on short chains, reaching 31.6 for length 2 and 23.4 for length 3, whereas longer chains exhibit substantially smaller gains or a decline (depths 4-10). This shows that while extra recurrence helps reasoning, harder instances with long relation chains do not automatically benefit as much from simply adding inference loops.

Physical depth and recurrent loops require a balanced allocation. Models trained at the same effective training depth $( L ( K ) = 2 0 )$ exhibit markedly different scaling behaviors depending on how depth is allocated between physical layers and recurrent iterations (Fig. 1d). BaseLoop $2 \times 1 0$ , with a shallow physical core and many recurrent iterations, peaks early and degrades under further unrolling, whereas $1 0 \times 2$ , with a deep physical core but few recurrent iterations, remains relatively flat during extrapolation, gaining little from additional loops. More balanced configurations exhibit stronger test-time scaling: $5 \times 4$ peaks at $L ( r ) = 3 0$ , while $4 \times 5$ continues improving up to $L ( r ) = 4 0$ . These results suggest that effective recurrent scaling requires a balanced allocation between physical depth and recurrent iterations, rather than being determined by effective depth alone.

Additional computation enables recurrent models to surpass the non-recurrent upper-bound. With extra test-time compute, several recurrent configurations exceed the NonLoop reasoning score of 30.25 (Fig. 1d). Specifically, BaseLoop $5 \times 4$ reaches 31.88 at $L ( r ) = 3 0$ , while $4 \times 5$ achieves

Base 4×5  
(%)  
(a) Physical Layers = 2  
![](images/635861b64f1f524b492f4d3ad68a58ba3de7ddd336f8eb98de84275967f64410.jpg)

(b) Physical Layers = 4  
![](images/e87fbf2ed8427f3f4b3db1d8214434b87972fa184b14c0eb55590c67f290e103.jpg)

(c) Physical Layers = 5  
![](images/31efdce249e21c6b78d15feac718b385a5939f390580e9efe1c4585374156671.jpg)

(d) Physical Layers = 10  
![](images/2f3871f38215999bfcac2394406e050e2bc9f8277b192b2f10df7354943d771b.jpg)  
Effective Layer Depth  
Figure 2: BaseLoop and CoreLoop performance based on the Llama3.1-1B architecture configuration on Overall, Knowledge, and Reasoning benchmarks across inference depths. Dashed vertical lines mark the effective training depth $L ( K ) = 2 0 .$ , and shaded regions denote loop extrapolation.

(a) Layer Angular Distance  
![](images/290b43d8b0d57aa9927d51add5b81aff0558ff9592b4e3fe54d3c2bd9d017a2d.jpg)

(b) Loop Angular Distance  
![](images/44e66f95b89bcb0676d137752cf7cd88b1f7918693011ee59bd0995e7fe53ae3.jpg)

(c) Relative Update Norm  
![](images/88d8bdf3d15d6d2e933168e1ce51e7589e73425a70f13cfe096d0ee686cabf52.jpg)

(d) Normalized State Variance  
![](images/1ae49a9824860e1a7f6ddeb50553fd2809c452d377edb0fc13e6dfc920a4e00f.jpg)  
Figure 3: Geometric dynamics of BaseLoop and CoreLoop models with 4 physical layers from the Llama3.1-1B architecture. The horizontal axes in (b)-(d) are normalized by each configuration’s training iterations. (a) Angular distance between consecutive layer states, with crosses marking Coda layers. (b) Angular distance between recurrent states before and after each complete loop iteration. (c) Per-loop update magnitude relative to the preceding state. (d) Recurrent-state variance at the training horizon. (c-d) use logarithmic scales.

31.49 at $L ( r ) = 4 0 $ . Crucially, these recurrent variants are trained under the exact same effective depth and training FLOPs $( L ( K ) = 2 0 )$ as NonLoop $( 2 0 \times 1 )$ , yet rely on significantly fewer distinct physical Transformer layers. This comparison demonstrates that parameter-efficient recurrent models can effectively trade additional test-time computation for superior reasoning performance.

## 4 WHICH COMPUTATIONS SHOULD BE RECURRENT?

We next investigate whether all Transformer blocks should participate in recurrent weight sharing, or whether some computations are better implemented by independently parameterized boundary layers. To this end, we compare BaseLoop, which recurrently applies the entire Transformer stack, with CoreLoop, which reserves non-recurrent Prelude and/or Coda blocks around the shared core. Under matched effective training depth and token budget, we evaluate Llama3.1-1B and Qwen3-0.6B architecture configurations. We present the Llama3.1-1B results in this section, considering physical depths of {2, 4, 5, 10} with the effective training depth fixed at $L ( K ) = 2 0$ , while varying the Prelude and Coda allocation. Results for Qwen3-0.6B are provided in App. D.2.

At inference, we vary the loop count r and evaluate knowledge, reasoning, and overall performance across under-unrolling, training horizon, and loop extrapolation. We further characterize recurrent representation dynamics using geometry metrics, including Angular Distance, Relative Update Norm, and Normalized State Variance, as defined in App. C.4. Full results and comparisons are in App. D.2.

As shown in Fig. 2, the optimal allocation of non-recurrent boundary layers varies with the inference budget. Across physical depths, three compute regimes (under-unrolling, the training horizon, and loop extrapolation) exhibit distinct trade-offs closely linked to representation dynamics (Fig. 3). These metrics capture how much recurrent states change, but not how strongly later computation depends on earlier computation, a distinction we examine in Sec. 6.

(%)  
![](images/c2eda23009efbce23b079c1bcbeb75f25854c11cb2e7108c86274c8278b527cf.jpg)  
BaseLoop Scalar Channel-wise Residual Channel-wise Dense NonLoop 28x1  
Figure 4: Initial-state injection results on Qwen3-0.6B BaseLoop configurations. BaseLoop provides the reference without state conditioning, and NonLoop $2 8 \times 1$ provides a non-recurrent reference.

Coda alleviates knowledge decay during under-unrolling. BaseLoop’s performance degrades rapidly with fewer inference loops, especially on knowledge tasks. Allocating non-recurrent layers to the Coda $( \mathbf { e . g . , 0 + 2 \times 9 + 2 }$ in Fig. 2(b), $0 + 3 \times 6 + 2$ in Fig. 2(c)) substantially mitigates this degradation. Geometrically, CoreLoop configurations with a Coda block exhibit smaller angular changes and more stable recurrent states during under-unrolling than BaseLoop and the Prelude-only variant, as shown in Fig. ${ 3 ( \mathbf { a } , \mathbf { b } , \mathbf { c } ) }$ . This suggests that separating the output-side transformation from the recurrent core helps align under-executed recurrent states with the final readout.

Performance is strong and robust to boundary allocation at the training horizon. Around the effective training depth $L ( K ) = 2 0$ , models generally achieve strong and stable performance across overall, knowledge, and reasoning tasks, while different Prelude-Coda allocations become substantially smaller (Fig. 2). This is the computation regime directly encountered during training, where the recurrent core produces representations well aligned with the trained output pathway.

Prelude-heavy allocations better support reasoning extrapolation. Beyond the training horizon, knowledge and overall performance generally deteriorate, whereas reasoning exhibits stronger and more configuration-dependent scaling. Notably, allocating more non-recurrent capacity to the Prelude can better sustain or further improve reasoning performance under extended unrolling. For example, the Prelude-heavy configuration $2 + 2 \times 9 + 0$ continues to improve in the four-layer setting, and $5 + 5 \times 3 + 0$ and $4 + 5 \times 3 + 1$ show strong reasoning extrapolation in the ten-layer setting (Fig. 2(b,d)). Although not universal, this trend suggests that dedicated input transformations can improve recurrent computation beyond the training horizon.

Convergence alone does not explain useful extrapolation. The representation dynamics reveal a notable discrepancy. Beyond the training horizon, both loop angular distance and relative update norm progressively decrease (Fig. $^ { 3 ( \mathbf { b } , \mathbf { c } ) ) }$ , indicating increasingly small changes between recurrent iterations. Meanwhile, recurrent-state variance continues to grow relative to its value at the training horizon (Fig. 3(d)), even as downstream performance can deteriorate. This suggests that small periteration updates can still accumulate, gradually drifting recurrent states away from the distribution encountered during training. Thus, increasingly small recurrent updates do not necessarily indicate that additional iterations remain useful; effective extrapolation also depends on how recurrent states evolve and how the shared core operates along this trajectory. These observations motivate us to examine whether additional conditioning can help sustain useful recurrent computation as the trajectory evolves beyond the training horizon.

## 5 HOW SHOULD RECURRENCE BE CONDITIONED?

We further investigate whether additional conditioning can mitigate performance degradation under mismatched training and inference budgets (Sec. 4). We consider State Conditioning, which injects hidden-state information, and Timestep Conditioning, which encodes the current recurrent iteration.

## 5.1 STATE CONDITIONING

Initial-state injection and its limitations. We first study initial-state injection, which supplies the initial representation $h _ { 0 }$ as a fixed reference at every recurrent iteration (Geiping et al., 2025): $h _ { \ell + 1 } =$ $F _ { \theta } ( \mathcal { T } _ { \phi } ( \dot { h _ { \ell } } , h _ { 0 } ) )$ . Huginn implements $\mathcal { T } _ { \phi }$ by concatenating $h _ { \ell }$ and $h _ { 0 }$ followed by a linear projection

(%)

![](images/7a277a0260f7e02f780d33899c4eaaebfd77c3a63f78888255114dcd471b0ce4.jpg)  
BaseLoop Initial-state History (w=1) History (w=2) History (w=4) NonLoop 28x1  
Figure 5: History-state injection on the Qwen3-0.6B BaseLoop 4×7 configuration. BaseLoop gives the reference without injection, and NonLoop 28×1 is a non-recurrent reference. The dashed vertical line marks the effective training depth $L ( K ) = 2 8$ , and shading region indicates loop extrapolated up to 3× the training depth.

(Dense). To systematically examine how the form and capacity of this injection affect recurrent scaling, we additionally consider Scalar, Channel-wise, and Residual Channel-wise parameterizations:

$$
\begin{array} { r l } & { \mathcal { Z } _ { \phi } ^ { \mathrm { s c a l a r } } ( h , h _ { 0 } ) = h + \alpha h _ { 0 } , } \\ & { \quad \mathcal { T } _ { \phi } ^ { \mathrm { r e s } } ( h , h _ { 0 } ) = \left( { \bf 1 } + \delta _ { a } \right) \odot h + b \odot h _ { 0 } , \qquad \mathcal { T } _ { \phi } ^ { \mathrm { d e n s e } } ( h , h _ { 0 } ) = W _ { h } h + W _ { 0 } h _ { 0 } . } \end{array}\tag{2}
$$

Here, $\alpha \in \mathbb { R } , a , b , \delta _ { a } \in \mathbb { R } ^ { d }$ , and $W _ { h } , W _ { 0 } \in \mathbb { R } ^ { d \times d }$ . The maps act on the hidden dimension at every token position and are applied before each execution of the shared stack. The injection parameters are learned jointly with the backbone and initialized such that $\mathcal { T } _ { \phi } ( h , h _ { 0 } ) = h$

As evaluated in Fig. 4, initial-state injection exhibits severe limitations, particularly on knowledgeintensive tasks. During loop extrapolation, the high-capacity Dense variant causes a catastrophic performance collapse in overall and knowledge accuracy across configurations, indicating that strong parametric conditioning overfits to the training loop count and disrupts knowledge retention during deep unrolling. In contrast, on reasoning tasks, performance remains largely flat across injection variants, showing minimal sensitivity to initial-state conditioning. Meanwhile, lightweight variants (Scalar, Channel-wise) avoid the severe breakdown on knowledge, but offer negligible net gains over the unconditioned BaseLoop baseline. Because Transformer architectures naturally retain input semantics via internal residual streams, forcibly re-injecting a static $h _ { 0 }$ provides redundant semantics rather than dynamic trajectory guidance.

History-state injection: conditioning on trajectory dynamics. While initial-state injection provides a static semantic anchor to where the recurrent trajectory originates, it fails to capture the evolving local dynamics during deep execution. Driven by this limitation, we propose history-state injection, which dynamically conditions on recent intermediate states. To maintain numerical stability and smooth computation during inference extrapolation, we condition each update on relative state differences rather than absolute historical vectors:

$$
h _ { \ell + 1 } = F _ { \theta } \biggl ( h _ { \ell } + \sum _ { j = 1 } ^ { m _ { \ell } } \mathcal { B } _ { j } ( h _ { \ell - j } - h _ { \ell } ) \biggr ) , \qquad m _ { \ell } = \operatorname* { m i n } \{ w , \operatorname* { m a x } ( \ell - 1 , 0 ) \} .\tag{3}
$$

This difference-based formulation explicitly models the local velocity of the recurrent path while preserving baseline recurrence when the history branch is initialized to zero. The history window includes only completed recurrent states $h _ { 1 } , \ldots , h _ { \ell - 1 }$ , explicitly excluding $h _ { 0 }$ to isolate history-state conditioning from static input injection. Each lag-specific operator $B _ { j }$ is shared across recurrent iterations and instantiated as a scalar, channel-wise vector, or dense linear map.

We evaluate history-state injection across different window sizes $( w \in \{ 1 , 2 , 4 \} )$ on the $4 \times 7$ BaseLoop configuration in Fig. 5. We find that history conditioning acts as a powerful dynamic regularizer for high-capacity mappings. Under the dense parameterization (Fig. 5(c)), history conditioning with a minimal window $( w = 1 )$ rescues the model from static injection collapse, sustaining robust accuracy near the NonLoop reference. Two observations underpin this stabilizing effect:

• Window-size sensitivity under Dense mapping: Expanding the memory window to $w = 2 \mathrm { o r } w = 4$ progressively diminishes extrapolation performance, indicating that a minimal one-step difference $\bar { ( \Delta h _ { k - 1 } ) }$ provides sufficient velocity cues without introducing redundant temporal noise.

(%)

![](images/5676961e6dfd156c4635434082657073f5acaf1a6bd0c075bc1d0871c43fafea.jpg)  
BaseLoop AdaLN Branch Gating Loop Gating NonLoop 28x1  
Figure 6: Timestep conditioning results of the Qwen3-0.6B backbone. All timestep variants are compared using rescaled time grids at inference. NonLoop $2 8 \times 1$ provides an unshared reference.

• Negligible gains under low-capacity mappings: For Scalar and Channel-wise variants (Fig. 5(a,b)), history injection provides limited gains over BaseLoop and slightly hurts reasoning performance.

## 5.2 TIMESTEP CONDITIONING

Inspired by Xu & Sato (2025); Jeddi et al. (2026), we condition the recurrent computation on normalized timestep size using three variants. Loop Gating (LG) uses a time-dependent scalar to scale the difference between the shared stack’s output and its input. Branch Gating (BG) applies separate time-dependent scalar gates to the attention and MLP residual branches within each layer. AdaLN adopts LoopFormer’s modulation structure (Jeddi et al., 2026), applying channel-wise scaling to the RMSNorm outputs and channel-wise gating to the corresponding residual branches. All three variants use the same continuous time features. At inference, we rescale the time grid to the requested number of recurrent iterations: for $L _ { \mathrm { i n f e r } }$ iterations, iteration ℓ receives normalized time $t _ { \ell } = \ell / L _ { \mathrm { i n f e r } }$ and step size $\Delta t = 1 / L _ { \mathrm { i n f e r } } ( \ell = 0 , \ldots , L _ { \mathrm { i n f e r } } - 1 )$ . Increasing the inference budget therefore divides the same normalized time interval [0, 1] into more, smaller steps. Full results are provided in App. D.5.

Timestep conditioning enhances recurrent scaling, with efficacy tied to architecture design. As shown in Fig. 6, LG and BG variants can improve overall performance at extrapolation parts, most notably in the 7 × 4 setup. Specifically at $\bar { L } ( r ) = 5 6$ , BG and LG achieve overall scores of 38.46 and 36.91, respectively, outperforming BaseLoop (36.59, Fig. 6c). However, the same schemes provide much smaller or negative gains at this budget for 4 × 7 and 2 × 14 (Fig. 6(a,b)). The 14 × 2 results likewise show that improvements at the training depth do not guarantee a consistent advantage at larger budgets (Fig. 6(d)). Timestep-dependent gating therefore interacts with the allocation of shared-stack depth and recurrent iterations. Its effectiveness must be assessed jointly with the recurrent architecture and the intended inference budget.

Loop and Branch gating provide lightweight alternatives to AdaLN. Our proposed LG and BG use scalar modulation at the loop or residual-branch level, requiring fewer conditioning parameters than channel-wise AdaLN modulation. For 7 × 4 at $L ( r ) = 5 6$ , BG outperforms AdaLN in both overall (38.46 vs 38.01) and reasoning (30.88 vs 29.12) (Fig. 6(c)). LG also achieves a higher Reasoning score for $1 4 \times 2 \mathrm { a t } L ( r ) = 5 \bar { 6 } ( 2 9 . 6 1 \mathrm { v s } 2 8 . 7 8 ) ( \mathrm { F i g . } \bar { 6 } ( \bf { d } ) )$ . These results demonstrate that simple scalar modulation can compete with more expressive channel-wise conditioning, and that the preferred modulation granularity depends on the recurrent architecture. AdaLN (Jeddi et al., 2026) better preserves Knowledge in these comparisons, indicating a task-dependent trade-off. LG and BG therefore provide lightweight design options for controlling recurrent computation. These results motivate combining timestep conditioning with input injection to test whether knowledge retention and reasoning gains can be maintained jointly.

## 6 WHEN ARE CONDITIONING MECHANISMS COMPOSABLE?

The preceding results show that state and timestep conditioning improve recurrent scaling through distinct signals, motivating us to examine whether their benefits are complementary. Using Qwen3- 0.6B BaseLoop 4×7, we evaluate all pairwise combinations of initial-state, history-state, and timestep conditioning, as well as the combination of all three, up to an effective depth of 84 (3x training depth).

(a) Initial-state + History-state  
![](images/ce4d92e491a77f81dbd9c1238f06bdcafb461d6446fa0360a639512d3fe33670.jpg)  
Effective Layer Depth

(b) Initial-state + Timestep  
![](images/cafcf971143e4356c70fbfc7b6abc0f427943b5372d8ffb6e71ecd9c9329fb79.jpg)  
Effective Layer Depth

(c) History-state + Timestep  
![](images/fe7cbbb2975b6d9bfdcb7b9c1363cc0832d428029a6a5a6b1249350eace8ae57.jpg)  
Effective Layer Depth

(d) All  
![](images/06d722c16e557aa76475946bc072d55b0baa80d57c92bf0ae7fe568e5ad6c0ae.jpg)  
Effective Layer Depth  
Figure 7: Performance of combined conditioning mechanisms for Qwen3-0.6B BaseLoop $4 \times 7 .$ All state conditioning uses the channel-wise parameterization, with history window w = 2. In the legends, I and H denote initial-state and history-state injection, LG denotes loop gating, and BG denotes branch gating.

![](images/b1ad4240bd0525fae9f4a955c8c8da30f3f74e5d982a4b8de732d67033d24a9d.jpg)

![](images/859f99c5550ae0be269024f0941030fe5a9200eac6ba70b91f25c0c8c783b1c7.jpg)  
Loop Lag

![](images/a11d659a00a22b5174c013ec5cf2747c21c524ed4fa5919579dbe019f3c19b16.jpg)  
Loop Lag

![](images/8775b07e576feee506199b458d3519aeaf582abed334d7c6eff34825f2b79e66.jpg)  
Loop Lag  
Figure 8: Results of relative update responses to skipping an earlier block occurrence. (a) Llama BaseLoop/CoreLoop at $D \ = \ 4 0 !$ responses averaged over separate skips at positions 21-24, divided by each model’s mean response in its final recurrent loop. Crosses mark coda blocks. (b,c) Qwen BaseLoop $4 \times 7 { : }$ mean C grouped by loop lag for initial-state injection and pairwise conditioning. (d) Three-mechanism responses divided by the corresponding H+T responses, with the timestep and history settings matched. All state injections in (c,d) are channel-wise, with $w = 2$ for history. In (b-d), dashed curves with open circles denote $D = 5 6$ and solid curves with filled triangles denote $D = 8 4 ;$ all measured lags are shown.

History-state and timestep conditioning form the strongest pair under deep extrapolation. At $D = 8 4$ , H+LG reaches 37.30 Overall accuracy, exceeding I+H (36.91) and I+LG (36.33), as well as its stronger individual component by 1.16 percentage points (Fig. $^ { 7 ( \mathbf { a } , \mathbf { b } , \mathbf { c } ) ) }$ . Adding initialstate conditioning provides no further benefit, with the three-way combinations remaining below H+LG (Fig. 7(d)). These results identify history-state and timestep conditioning as the strongest complementary pair for recurrent scaling, with no additional gains from initial-state conditioning.

Geometry analysis. To understand this complementarity beyond performance, we examine how earlier computation influences later recurrent updates. We measure computational interaction C, defined as the relative change in a downstream block’s update after skipping an earlier block occurrence (App. C.4). Unlike angular distance or update magnitude, which quantify how much a representation changes, C captures how strongly later computation depends on earlier computation.

• Computational interaction reveals information beyond update magnitude. Large state transformations need not imply strong computational interaction (Fig. 8(a,b)): Coda layers reduce the terminal-to-core response ratio from 0.85-0.89 to 0.48-0.50, and channel-wise initial-state injection shows nearly 7× the cross-loop response of Dense injection at $D = 8 4$ . Thus, transformation magnitude alone does not characterize effective iterative refinement.

• History-state and timestep complementarity emerges through cross-loop interaction. At $D = 8 4$ H+LG exhibits 18.6% and 10.4% higher mean interaction over lags 1–6 than I+H and I+LG, respectively, consistent with its higher accuracy (Fig. 7). This advantage does not come from uniformly larger responses: H+LG shows a smaller within-loop response than I+H, with its gain appearing mainly across loops (Fig. 8(c)). This suggests complementary roles across iterations: history-state conditioning exposes how previous states evolve, while timestep conditioning modulates the shared update at each iteration. Together, they allow later iterations to build more strongly on earlier computation, whereas pairings with the static initial state provide weaker cross-loop coupling.

• Initial-state conditioning provides no additional cross-loop benefit. Adding initial-state conditioning to H+LG preserves local interactions but reduces the mean interaction over lags 2-6 by 7.4% and 4.2% at $D = 5 6$ and 84, respectively (Fig. 8(d)), alongside lower accuracy for the three-way combination. The same trend appears in the unnormalized responses, indicating that it is not solely a normalization effect. Although the pattern is not universal across all lags and gating variants, it provides a possible explanation for the lack of additive gains from initial-state conditioning once history-state and timestep conditioning are combined.

Overall, these results suggest that effective recurrent scaling depends not only on the magnitude of state transformations, but also on how computation interacts across recurrent iterations.

## ACKNOWLEDGEMENT

The work described in this paper was supported by the General Research Fund and Early Career Scheme by the Research Grants Council of Hong Kong (Project Number: 24211626).

## REFERENCES

Ibrahim Alabdulmohsin and Xiaohua Zhai. Recursive inference scaling: A winning path to scalable inference in language and multimodal systems. In NeurIPS, 2025. URL https: //openreview.net/forum?id=cLbGkINOLP. 1, 15

Sangmin Bae, Yujin Kim, Reza Bayat, Sungnyun Kim, Jiyoun Ha, Tal Schuster, Adam Fisch, Hrayr Harutyunyan, Ziwei Ji, Aaron Courville, and Se-Young Yun. Mixture-of-recursions: Learning dynamic recursive depths for adaptive token-level computation. In NeurIPS, 2025. URL https: //openreview.net/forum?id=QuqsEIVWIG. 3, 15

Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. Piqa: Reasoning about physical commonsense in natural language. In AAAI, 2020. URL https://ojs.aaai.org/ index.php/AAAI/article/view/6239. 4, 21, 26

Hugh Blayney, Alvaro Arroyo, Johan Obando-Ceron, Pablo Samuel Castro, Aaron Courville,<sup>´</sup> Michael M Bronstein, and Xiaowen Dong. A mechanistic analysis of looped reasoning language models. arXiv preprint arXiv:2604.11791, 2026. URL https://arxiv.org/abs/ 2604.11791. 16

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018. 4, 21, 26

Chunyuan Deng, Yizhe Zhang, Rui-Jie Zhu, Yuanyuan Xu, Jiarui Liu, TS Ng, and Hanjie Chen. Lt2: Linear-time looped transformers. arXiv preprint arXiv:2605.20670, 2026. URL https: //arxiv.org/abs/2605.20670. 15

Jacob Fein-Ashley and Paria Rashidinejad. Solve the loop: Attractor models for language and reasoning. arXiv preprint arXiv:2605.12466, 2026. URL https://arxiv.org/abs/2605. 12466. 2, 15

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. The language model evaluation harness, 07 2024. URL https://zenodo.org/records/12608602. 21

Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, and Tom Goldstein. Scaling up test-time compute with latent reasoning: A recurrent depth approach. In NeurIPS, 2025. URL https://openreview. net/forum?id=S3GhJooWIC. 1, 2, 3, 6, 15, 26

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024. URL https://arxiv.org/abs/2407. 21783. 3

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In ICLR, 2021. URL https: //openreview.net/forum?id=d7KBjmI3GmQ. 26

Jaber Jaber and Osama Jaber. Ouroboros: Dynamic weight generation for recursive transformers via input-conditioned lora modulation. arXiv preprint arXiv:2604.02051, 2026. URL https: //arxiv.org/abs/2604.02051. 15

Ahmadreza Jeddi, Marco Ciccone, and Babak Taati. Loopformer: Elastic-depth looped transformers for latent reasoning via shortcut modulation. In ICLR, 2026. URL https://openreview. net/forum?id=RzYXb5YWBs. 1, 3, 8, 15, 26

Keller Jordan, Yuchen Jin, Vlado Boza, You Jiacheng, Franz Cesista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks, 2024. URL https: //kellerjordan.github.io/posts/muon/. 3, 20

Harsh Kohli, Srinivasan Parthasarathy, Huan Sun, and Yuekun Yao. Loop, think, & generalize: Implicit reasoning in recurrent-depth transformers. In CoLM, 2026. URL https: //openreview.net/forum?id=8fz7WRThKL. 16

Yeskendir Koishekenov, Aldo Lipani, and Nicola Cancedda. Encode, think, decode: Scaling testtime reasoning with recursive latent thoughts. arXiv preprint arXiv:2510.07358, 2025. URL https://arxiv.org/abs/2510.07358. 15

Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. RACE: Large-scale ReAding comprehension dataset from examinations. In EMNLP, 2017. URL https://aclanthology. org/D17-1082. 26

Ryan Lee, Jacob Biloki, Edward J Hu, and Jonathan May. Sparse layers are critical to scaling looped language models. arXiv preprint arXiv:2605.09165, 2026. URL https://arxiv.org/abs/ 2605.09165. 15

Shuzhen Li, Yifan Zhang, Jiacheng Guo, Quanquan Gu, and Mengdi Wang. Deeploop: Depth scaling for looped transformers. arXiv preprint arXiv:2607.13491, 2026a. URL https://arxiv. org/abs/2607.13491. 15

Ziyue Li, Yang Li, and Tianyi Zhou. Skip a layer or loop it? learning program-of-layers in LLMs. In ICML, 2026b. URL https://openreview.net/forum?id=pl10b6EQAN. 15

Ruhai Lin, Yiyang Guo, Rui-Jie Zhu, Hao Ye, and Jason K Eshraghian. Allocating recurrent compute in looped language models. arXiv preprint arXiv:2608.18230, 2026. URL https: //arxiv.org/abs/2608.18230. 15

Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu, Junjie Yan, et al. Muon is scalable for llm training. arXiv preprint arXiv:2502.16982, 2025. URL https://arxiv.org/abs/2502.16982. 3, 20

Joe Logan. Per-token fixed-point convergence in depth-recurrent transformers. arXiv preprint arXiv:2607.14427, 2026. URL https://arxiv.org/abs/2607.14427. 16

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2019. URL https://openreview.net/forum?id=Bkg6RiCqY7. 20

Sean McLeish, Ang Li, John Kirchenbauer, Dayal Singh Kalra, Brian R Bartoldson, Bhavya Kailkhura, Avi Schwarzschild, Jonas Geiping, Tom Goldstein, and Micah Goldblum. Teaching pretrained language models to think deeper with retrofitted recurrence. arXiv preprint arXiv:2511.07384, 2025. URL https://arxiv.org/abs/2511.07384. 15

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In EMNLP, 2018. URL https: //aclanthology.org/D18-1260. 4, 21, 26

Sajad Movahedi, Vera Milovanovic, Shlomo Libo Feigin, Alexander Theus, Thomas Hofmann,´ Valentina Boeva, T Konstantin Rusch, and Antonio Orvieto. Fixed-point reasoners: Stable and adaptive deep looped transformers. arXiv preprint arXiv:2606.18206, 2026. URL https: //arxiv.org/abs/2606.18206. 15

Guilherme Penedo, Hynek Kydl´ıcek, Loubna Ben allal, Anton Lozhkov, Margaretˇ Mitchell, Colin Raffel, Leandro Von Werra, and Thomas Wolf. The fineweb datasets: Decanting the web for the finest text data at scale. In NeurIPS, 2024. URL https://proceedings.neurips.cc/paper\_files/paper/2024/file/ 370df50ccfdf8bde18f8f9c2d9151bda-Paper-Datasets\_and\_Benchmarks\_ Track.pdf. 3, 17

Andrei Cristian Popescu, Haitz Saez de Oc´ ariz Borde, and Pietro Li´ o. Adaptive depth in\` looped transformers: Diagnosing learned halting gates and trajectory readouts. arXiv preprint arXiv:2607.20519, 2026. URL https://arxiv.org/abs/2607.20519. 16

Hayden Prairie, Zachary Novack, Taylor Berg-Kirkpatrick, and Daniel Y Fu. Parcae: Scaling laws for stable looped language models. arXiv preprint arXiv:2604.12946, 2026. URL https: //arxiv.org/abs/2604.12946. 16

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. In AAAI, 2020. 4, 21, 26

Nikunj Saunshi, Nishanth Dikkala, Zhiyuan Li, Sanjiv Kumar, and Sashank J Reddi. Reasoning with latent thoughts: On the power of looped transformers. In ICLR, 2025. URL https: //openreview.net/forum?id=din0lGfZFd. 1, 3, 15

Kristian Schwethelm, Daniel Rueckert, and Georgios Kaissis. How much is one recurrence worth? iso-depth scaling laws for looped language models. arXiv preprint arXiv:2604.21106, 2026. URL https://arxiv.org/abs/2604.21106. 16

Mark Shapiro. Retrofitting recurrent depth into a pretrained language model: Installation, extrapolation, transfer, and retention at two parameter budgets. arXiv preprint arXiv:2608.11233, 2026. URL https://arxiv.org/abs/2608.11233. 15

Koustuv Sinha, Shagun Sodhani, Jin Dong, Joelle Pineau, and William L. Hamilton. CLUTRR: A diagnostic benchmark for inductive reasoning from text. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 4506–4515, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1458. URL https: //aclanthology.org/D19-1458/. 4, 21

Yutao Sun, Li Dong, Tianzhu Ye, Shaohan Huang, Jianyong Wang, and Furu Wei. Universal yoco for efficient depth scaling. arXiv preprint arXiv:2604.01220, 2026. URL https://arxiv.org/ abs/2604.01220. 15

Mirac Suzgun, Nathan Scales, Nathanael Scharli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung,¨ Aakanksha Chowdhery, Quoc Le, Ed Chi, Denny Zhou, and Jason Wei. Challenging BIGbench tasks and whether chain-of-thought can solve them. In Findings of ACL, 2023. URL https://aclanthology.org/2023.findings-acl.824/. 4, 21

Oyvind Tafjord, Bhavana Dalvi, and Peter Clark. ProofWriter: Generating implications, proofs, and abductive statements over natural language. In Findings of ACL, 2021. URL https: //aclanthology.org/2021.findings-acl.317/. 4, 21

Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. CommonsenseQA: A question answering challenge targeting commonsense knowledge. In NAACL-HLT, 2019. URL https: //aclanthology.org/N19-1421. 4, 21, 26

Shaowen Wang, Bingrui Li, Ge Zhang, Wenhao Huang, Shen Yan, and Jian Li. On the residual scaling of looped transformers: Stability and transferability. arXiv preprint arXiv:2606.18524, 2026a. URL https://arxiv.org/abs/2606.18524. 16

Shaowen Wang, Ge Zhang, Kairong Luo, Yuhao Wu, Shaofan Liu, Jiaheng Liu, Wenhao Huang, Shen Yan, and Jian Li. Smelt: Scaling laws for compute-matched moe looped transformers. arXiv preprint arXiv:2609.01343, 2026b. URL https://arxiv.org/abs/2609.01343. 16

Johannes Welbl, Nelson F. Liu, and Matt Gardner. Crowdsourcing multiple choice science questions. In Proceedings of the 3rd Workshop on Noisy User-generated Text, 2017. URL https:// aclanthology.org/W17-4413. 4, 21, 26

Kevin Xu and Issei Sato. On expressive power of looped transformers: Theoretical analysis and enhancement via timestep encoding. In ICML, 2025. URL https://openreview.net/ forum?id=H4BuhRezCV. 2, 8, 16

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. URL https://arxiv.org/abs/2505.09388. 3

Xiao-Wen Yang, Ziyu Han, Xi-Hua Zhang, Wen-Da Wei, Jie-Jing Shao, Lan-Zhe Guo, and Yu-Feng Li. Stabilizing recurrent dynamics for test-time scalable latent reasoning in looped language models. arXiv preprint arXiv:2605.26733, 2026. URL https://arxiv.org/abs/2605.26733. 15

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: Can a machine really finish your sentence? In ACL, 2019. URL https://aclanthology.org/P19-1472. 4, 21, 26

Rui-Jie Zhu, Zixuan Wang, Kai Hua, Tianyu Zhang, Ziniu Li, Haoran Que, Boyi Wei, Zixin Wen, Fan Yin, He Xing, et al. Scaling latent reasoning via looped language models. arXiv preprint arXiv:2510.25741, 2025. URL https://arxiv.org/abs/2510.25741. 1, 15, 26

## Appendix

A Limitations 15   
B Extended Literature Review 15   
C Experimental Settings 17   
C.1 Models and Data . 17   
C.2 Training 20   
C.3 Evaluation 21   
C.4 Geometry Probing 22   
D Full Experimental Results 26   
D.1 Optimizer Experiment 26   
D.2 BaseLoop and CoreLoop 26   
D.3 Initial-state Input Injection 27   
D.4 History-state Input Injection . 27   
D.5 Timestep Conditioning . 28   
D.6 Combination Experiments 29   
E Further Analysis 33   
E.1 Input Noise Initialization Ablation 33   
E.2 Geometry Analysis 36

## A LIMITATIONS

Our study has several limitations. First, due to computational constraints, our experiments are conducted on models of up to roughly 1B parameters (Llama3.1-1B and Qwen3-0.6B configurations) under a fixed token budget. Second, we evaluate extrapolation up to a fixed multiple of the training horizon with a manually specified loop count, and do not explore adaptive mechanisms that determine the number of iterations per input. Finally, our geometry analyses, including computational interaction, are intended to offer complementary insights into recurrent dynamics. Further study is needed to more fully establish their relationship with downstream performance.

## B EXTENDED LITERATURE REVIEW

LoopLM variants. Looped language models (LoopLMs) increase effective depth by repeatedly applying a shared set of Transformer layers, thereby decoupling sequential computation from the number of unique parameters. Existing LoopLM variants can be broadly divided into two categories according to how recurrence is introduced: models that are pre-trained with recurrent depth from scratch, and methods that retrofit recurrence into pretrained Transformers through continual pretraining.

## Strategy 1: Pre-training from scratch.

Early works such as BaseLoop Saunshi et al. (2025) and Huginn Geiping et al. (2025) establish recurrent depth as a viable alternative to conventional depth scaling, showing that repeatedly applying shared layers can provide additional latent computation and improve reasoning tasks without proportionally increasing parameter count. Subsequent methods focus on making recurrence more adaptive and scalable. Ouro Zhu et al. (2025), MoR Bae et al. (2025), and LoopFormer Jeddi et al. (2026) introduce dynamic or elastic recurrence, allowing computation depth to vary across inputs, tokens, or inference budgets. Other works address the stability of deep recurrence: FPRM Movahedi et al. (2026), DeepLoop Li et al. (2026a), and STARS Yang et al. (2026) study fixed-point behaviors, residual scaling, and recurrent dynamical stability, respectively, with the goal of enabling reliable extrapolation to larger loop counts than those seen during training.

A parallel line explores which computations should be repeated and how recurrent architectures can be made more efficient or expressive. RINS Alabdulmohsin & Zhai (2025) studies alternative recursive execution patterns and identifies A<sup>r</sup>B as an effective topology, while YOCO-U Sun et al. (2026) and LT2 Deng et al. (2026) combine recurrent depth with efficient attention or KV-cache designs. Attractor models Fein-Ashley & Rashidinejad (2026) formulate iterative latent refinement as convergence toward an equilibrium, providing an implicit alternative to finite-depth recurrence. More recent works further examine the granularity of looping. Looped-MoE Lee et al. (2026) shows that sparse expert routing can increase functional diversity across repeated passes, while MixerLoop Lin et al. (2026) demonstrates that recurrent compute can be concentrated on selected Transformer submodules. PoLar Li et al. (2026b) similarly broadens the design space by learning flexible programs that can skip or repeat different layers instead of using a fixed recurrent block.

## Strategy 2: Retrofitting through continual pre-training.

Instead of introducing recurrence during initial pre-training, another line of work converts existing pretrained Transformers into recurrent models. ETD (Koishekenov et al., 2025) identifies a small subset of reasoning-relevant intermediate layers and trains the model to repeatedly apply these layers as a latent stage, enabling additional inference-time computation while preserving the original parameter count and overall architecture. McLeish et al. (2025) more systematically studies the conversion of standard pretrained Transformers into depth-recurrent models, showing that a curriculum that gradually increases the number of recurrences can preserve pretrained capabilities while efficiently adapting the model to repeated weight reuse. Ouroboros Jaber & Jaber (2026) addresses a limitation of such weight sharing, namely that identical recurrent weights repeatedly apply the same transformation, by using an input-conditioned controller to dynamically modulate LoRA parameters at each recurrence step, allowing shared layers to implement step- and input-dependent transformations. More recently, Shapiro (2026) explores the installation and persistence of recurrent computation in pretrained models in greater detail, showing that continual training can induce reusable iterative latent procedures that extrapolate beyond the supervised recurrence depth and persist under subsequent outcome-only training.

Understanding LoopLMs. A growing body of work analyzes the internal dynamics and computational properties of LoopLMs. Blayney et al. (2026) shows that repeated computation gives rise to structured recurrent trajectories, where hidden states and attention patterns progressively stabilize across loops and exhibit fixed-point-like behaviors. Building on this perspective, Logan (2026) finds that convergence is strongly token-dependent, with different tokens reaching stable representations after different numbers of recurrent steps. Popescu et al. (2026) further examines how such convergence can be exploited for adaptive computation, showing that the effectiveness of learned halting depends not only on the stopping criterion itself but also on how training shapes the recurrent trajectory and final readout.

Other studies investigate the theoretical and optimization properties induced by recurrence. Xu & Sato (2025) characterizes the expressive limitations of weight-shared Transformers and show that timestep encoding can increase the functional diversity across recurrent steps. From a generalization perspective, Kohli et al. (2026) demonstrates on synthetic controlled compositional tasks that recurrence promotes systematic generalization and depth extrapolation, although excessive looping can lead to overthinking. Finally, Wang et al. (2026a) analyzes signal propagation under repeated weight reuse and derive loop-specific residual scaling rules, highlighting that stability conditions for recurrent depth differ fundamentally from those of conventional Transformers with independent layers.

Scaling laws of LoopLMs. Recent work has begun to characterize recurrent depth as a distinct scaling dimension in LoopLMs. Parcae (Prairie et al., 2026) derives scaling laws over model size, training data, and recurrence depth, showing that recurrence should scale jointly with data under fixed compute and that test-time depth exhibits diminishing returns. Complementarily, Schwethelm et al. (2026) quantify the value of recurrence relative to unique depth, estimating a recurrence-equivalence exponent of $\varphi = 0 . 4 6$ , which indicates that repeated layers provide additional capacity but are substantially less effective than independent layers. More recently, SMELT (Wang et al., 2026b) extends this analysis to MoE-based LoopLMs under a stricter budget-matched setting, jointly controlling per-token FLOPs, non-embedding parameters, and KV cache. By fitting separate Chinchilla-style scaling laws up to 54B non-embedding parameters, SMELT shows that looped MoE models scale more favorably than their unlooped counterparts, requiring 6.8–18.0% fewer training FLOPs along the compute-optimal frontier.

## C EXPERIMENTAL SETTINGS

## C.1 MODELS AND DATA

Backbone architectures. We construct all models from scratch using two decoder-only Transformer architectures: a 20-layer Llama 3.1-style model with approximately one billion parameters, denoted Llama3.1-1B, and the 28-layer Qwen3-0.6B architecture. These names refer to the corresponding nonrecurrent reference configurations. The recurrent models introduced below preserve their block-level architectural dimensions while instantiating fewer physical Transformer blocks. Both architectures use pre-normalization (Pre-LN), causal self-attention, grouped-query attention (GQA), rotary positional embeddings (RoPE), and SwiGLU feed-forward networks. Each Transformer block contains an RMSNorm before the attention sublayer and another RMSNorm before the feed-forward sublayer. Attention and MLP projections are bias-free, and we use zero attention dropout.

The Llama3.1-1B reference configuration contains 20 Transformer blocks with model dimension $d _ { \mathrm { m o d e l } } = 1 5 3 6$ and feed-forward dimension $d _ { \mathrm { f f } } = 5 3 7 6$ . Its attention module has 12 query heads and 3 key-value heads, each with head dimension 128; thus, four query heads share each key-value head. It uses the Llama 3 RoPE scaling rule with base frequency $\theta = 5 \times 1 0 ^ { 5 } ,$ , an original context length of 8,192, and a configured maximum context length of 131,072. RMSNorm uses $\epsilon = 1 0 ^ { - 5 }$ . The input embedding and output language-model head are not tied. The resulting non-recurrent 20-layer model contains exactly 1,007,482,368 parameters.

The Qwen3-0.6B reference configuration contains 28 Transformer blocks with $d _ { \mathrm { m o d e l } } = 1 0 2 4$ and $d _ { \mathrm { f f } } = 3 0 7 2$ . It uses 16 query heads and 8 key-value heads with head dimension 128. Consequently, its query projection has inner dimension $1 6 \times 1 2 8 = 2 0 4 8$ , whereas its key and value projections have inner dimension $8 \times 1 2 8 = 1 0 2 4$ . In addition to the RMSNorms surrounding the two sublayers, Qwen3-0.6B applies RMSNorm to the query and key vectors before RoPE (QK-Norm). It uses standard RoPE with $\theta = 1 0 ^ { 6 }$ , and a configured maximum context length of 40,960. RMSNorm uses $\epsilon = 1 0 ^ { - 6 }$ , and the input embedding and language-model head are tied. The complete 28-layer reference model contains 596,049,920 parameters. Tab. 1 summarizes the principal architectural details of these two backbones.

<table><tr><td>Architecture</td><td> $\pmb { L } _ { \mathbf { r e f } }$ </td><td>Parameters</td><td>|2|</td><td> $d _ { \mathrm { m o d e l } }$ </td><td> $d _ { \mathrm { f f } }$ </td><td>Q/KV heads</td><td> $d _ { \mathbf { h e a d } }$ </td></tr><tr><td>Llama3.1-1B</td><td>20</td><td>1,007,482,368</td><td>128,256</td><td>1,536</td><td> $^ { 5 , 3 7 6 }$ </td><td>12/3</td><td>128</td></tr><tr><td>Qwen3-0.6B</td><td>28</td><td>596,049,920</td><td>151,936</td><td>1,024</td><td>3,072</td><td>16/8</td><td>128</td></tr></table>

Table 1: Non-recurrent reference architectures. $L _ { \mathrm { r e f } }$ denotes the number of Transformer blocks in the corresponding reference configuration. Parameter counts include token embeddings, the final RMSNorm, and the language-model head.

Pretraining data and tokenization. All models are pretrained on the same raw FineWeb-Edu-350BT corpus Penedo et al. (2024), with a separate held-out FineWeb-Edu validation split. We stream the training data and shuffle examples using a buffer of 10,000 documents and random seed 42. No data curriculum strategy is applied.

We use the tokenizer of Llama3.1-8B for the Llama3.1-1B models and the tokenizer of official Qwen3- 0.6B for the Qwen3-0.6B models, yielding vocabulary sizes of 128,256 and 151,936, respectively. Documents are tokenized without a beginning-of-sequence token and are terminated with an endof-sequence token. The resulting token stream is concatenated and packed into non-overlapping sequences of length 2,048, and incomplete final sequences are discarded. Unless otherwise specified, the Llama3.1-1B experiments process approximately 20.15B packed tokens, while the Qwen3-0.6B experiments process approximately 12B packed tokens, which follows 1x Chinchilla optimal ratio.

BaseLoop and CoreLoop models. We describe a recurrent architecture by

$$
p + s \times K + c ,
$$

where $p$ is the number of non-recurrent Prelude blocks, s is the number of parameterized blocks in the recurrent core, K is the number of training-time loop count of that core, and c is the number of non-recurrent Coda blocks. Its physical depth and training effective depth are therefore

$$
L _ { \mathrm { p h y s } } = p + s + c , \qquad L _ { \mathrm { e f f } } = p + K s + c .
$$

The Prelude and Coda blocks have independent parameters and are each executed once, whereas the same s core blocks are reused at every recurrent iteration.

A BaseLoop model sets $p = c = 0$ , so that all physical Transformer blocks belong to the recurrent stack. A CoreLoop model places one or more independently parameterized blocks before or after a smaller recurrent core. This construction allows us to vary where the parameterized blocks are allocated while holding both $L _ { \mathrm { p h y s } }$ and $L _ { \mathrm { e f f } }$ fixed. We train the Llama3.1-1B variants at $L _ { \mathrm { e f f } } = 2 0$ matching the depth of the corresponding non-recurrent reference model, and the Qwen3-0.6B variants at $L _ { \mathrm { e f f } } = 2 8$ . Notably, although recurrent models match their reference architecture in effective depth, their parameter counts depend on the number of physical blocks $L _ { \mathrm { p h y s } } .$ rather than on $L _ { \mathrm { e f f } }$ The complete configurations of BaseLoop and CoreLoop models are given in Tab. 2.
<table><tr><td>Backbone</td><td> $\scriptstyle { { \mathbf { L } } _ { \mathrm { e f f } } }$ </td><td> $\underline { { \mathbf { L _ { p h y s } } } }$ </td><td>Parameters</td><td>BaseLoop</td><td>CoreLoop</td></tr><tr><td>Llama3.1-1B</td><td>20</td><td>2</td><td>455,351,808</td><td> $\overline { { 2 \times 1 0 } }$ </td><td> $\overline { { 0 + 1 \times 1 9 + 1 , 1 + 1 \times 1 9 + 0 } }$ </td></tr><tr><td>Llama3.1-1B</td><td>20</td><td>4</td><td>516,699,648</td><td> $_ \textrm { 4 } \times 5$ </td><td> $0 + 2 \times 9 + 2 , 1 + 2 \times 9 + 1 , 2 + 2 \times 9 + 0$ </td></tr><tr><td>Llama3.1-1B</td><td>20</td><td>5</td><td>547,373,568</td><td> $5 \times 4$ </td><td> $0 + 3 \times 6 + 2 , 1 + 3 \times 6 + 1 , 2 + 3 \times 6 + 0$ </td></tr><tr><td>Llama3.1-1B</td><td>20</td><td>10</td><td>700,743,168</td><td> $1 0 \times 2$ </td><td> $0 + 5 \times 3 + 5 , 1 + 5 \times 3 + 4 , 2 + 5 \times 3 + 3 ,$ </td></tr><tr><td>Qwen3-0.6B</td><td>28</td><td>4</td><td>218,507,264</td><td> $4 \times 7$ </td><td> $3 + 5 \times 3 + 2 , 4 + 5 \times 3 + 1 , 5 + 5 \times 3 + 0$   $0 + 2 \times 1 3 + 2 , 1 + 2 \times 1 3 + 1 , 2 + 2 \times 1 3 + 0$ </td></tr></table>

Table 2: Details of BaseLoop and CoreLoop models. An allocation is written as $p + s \times K + c .$ Architectures within a row have the same physical depth, total parameter count, and training effective depth. The reported parameter counts correspond to the unconditioned naive-loop models, without adding any input-injection or timestep-conditioning parameters.

At inference time, the recurrent core can instead be applied r times, resulting in inference depth of

$$
L _ { \mathrm { i n f e r e n c e } } ( r ) = p + r s + c .
$$

Consequently, comparisons at different r account for the resulting operator-level compute rather than equating models solely by their raw loop counts.

Initial-state injection models. The initial-state injection experiments use a decoder-only BaseLoop architecture consisting of token embeddings, a shared stack of K Transformer layers, a final RM-SNorm, and a vocabulary projection. The K layers have distinct parameters, but the entire stack is reused at every recurrent iteration. The prelude and coda Transformer depths are both zero, so $h _ { 0 }$ is directly the token embedding and the effective depth is $D = K L$ . The initial-state injection adapter is placed at the entrance to the shared stack. Its parameters are shared over all iterations and token positions. Training uses next-token cross-entropy on the final recurrent output and full backpropagation through the fixed number of training iterations.

The Qwen3-0.6B corresponding BaseLoop parameter counts, before adding injection, are approximately 187.05M, 218.51M, 265.70M, and 375.82M. Thus, Qwen3-0.6B identifies the backbone configuration; the recurrent models have fewer distinct parameters because of layer sharing. The additional Llama experiments use the Llama3.1-1B backbone. We use $K \times \mathsf { \bar { L } } _ { \mathrm { t r a i n } } \in$ $\{ \bar { 2 \times 1 0 } , 4 \times 5 , 5 \times 4 , 1 0 \times 2 \}$ , all with effective training depth 20. Their BaseLoop parameter counts are approximately 455.35M, 516.70M, 547.37M, and 700.74M, respectively. All four initial-state injection variants are evaluated for every Qwen and Llama configuration, alongside the corresponding BaseLoop without injection. The non-loop references have 28 layers for Qwen3-0.6B and 20 layers for Llama3.1-1B.

For Scalar injection, we initialize $\alpha = 0$ . For Channel-wise injection, we initialize $a = \mathbf { 1 }$ and $b = \mathbf { 0 } ;$ for Residual Channel-wise injection, we initialize $\delta _ { a } = b = \mathbf { 0 }$ . Dense injection implements a bias-free projection from the concatenated state and initial representation, with weight $\mathsf { \bar { [ } } W _ { h } \ W _ { 0 } ] \in \mathbb { R } ^ { d \times 2 d }$ initialized to $\left[ I _ { d } \left( \boldsymbol { 0 } \right] \right.$ . These initializations make the injection maps initially equal to the identity on the current state. The additional parameter counts are 1, 2d, 2d, and $2 d ^ { 2 }$ , respectively. All coefficients are unconstrained learned parameters. The two channel-wise variants have the same function class but different parameterizations. With the configured weight decay, regularizing a favors a state scale of zero, whereas regularizing $\delta _ { a }$ favors a state scale of one.

History-state injection models. All reported history-state experiments use the Qwen3-0.6B BaseLoop $4 \times 7$ architecture and the same 12B-token training recipe as the corresponding initial-state injection models. We instantiate Eq. 3 with the scalar form of the lag-specific operators, $B _ { j } ( x ) = { \beta } _ { j } x$ with $\beta _ { j } \in \mathbb { R }$ , so that each recurrent transition becomes

$$
h _ { \ell + 1 } = F _ { \theta } \biggl ( h _ { \ell } + \sum _ { j = 1 } ^ { m _ { \ell } } \beta _ { j } \left( h _ { \ell - j } - h _ { \ell } \right) \biggr ) , \qquad m _ { \ell } = \mathrm { m i n } \{ w , \mathrm { m a x } ( \ell - 1 , 0 ) \} .\tag{4}
$$

History window. For the transition $h _ { \ell } \to h _ { \ell + 1 }$ , the history window contains the $m _ { \ell }$ most recent completed recurrent states $h _ { \ell - 1 } , \ldots , h _ { \ell - m _ { \ell } } ,$ , indexed by relative lag $j \colon$ the state at lag $j , h _ { \ell - j }$ , is weighted by $\beta _ { j }$ . The window excludes the current state $h _ { \ell } ,$ which serves as the reference point of every difference, and the embedding state $h _ { 0 } .$ , which is reserved for initial-state injection. Consequently, $m _ { 0 } = m _ { 1 } = 0$ , and the first two transitions reduce to plain BaseLoop recurrence, $h _ { 1 } = F _ { \theta } ( h _ { 0 } )$ and $h _ { 2 } = F _ { \theta } ( h _ { 1 } )$ . The first history-dependent transition is

$$
h _ { 3 } = F _ { \theta } \big ( h _ { 2 } + \beta _ { 1 } ( h _ { 1 } - h _ { 2 } ) \big ) .\tag{5}
$$

Once more than w completed states are available $( \ell - 1 > w )$ , only the w most recent ones are retained. Buffered states are not detached, so gradients propagate through them during training.

Parameterization and initialization. The coefficients $\beta _ { 1 } , \ldots , \beta _ { w }$ are unconstrained, initialized to zero, and depend only on the relative lag $j ;$ they are shared across all recurrent iterations and token positions. History-only injection therefore adds exactly w trainable scalars to BaseLoop and recovers BaseLoop recurrence exactly at initialization.

Combined variant. When history-state injection is combined with initial-state injection, each transition becomes

$$
h _ { \ell + 1 } = F _ { \theta } \bigg ( h _ { \ell } + \alpha h _ { 0 } + \sum _ { j = 1 } ^ { m _ { \ell } } \beta _ { j } \left( h _ { \ell - j } - h _ { \ell } \right) \bigg ) ,\tag{6}
$$

with $\alpha = 0$ and $\beta _ { j } = 0$ for all $j$ at initialization. The differences are always taken with respect to the recurrent state $h _ { \ell } .$ , not the injected input $h _ { \ell } + \alpha h _ { 0 }$ , so the two branches enter additively and do not interact.

Timestep-conditioning models. Let $T$ denote the number of steps in the conditioning grid. At recurrent iteration ℓ, we set $t _ { \ell } = \ell / T$ and $\Delta t = 1 / T$ . The conditioning vector is $\psi _ { \ell } = \psi ( t _ { \ell } , \Delta t ) \in$ $\mathbb { R } ^ { 8 }$ , where

$$
\begin{array} { r } { \psi ( t , \Delta t ) = \big ( t , \Delta t , t ^ { 2 } , ( \Delta t ) ^ { 2 } , t \Delta t , \big . \qquad } \\ { \left. \sin ( \pi t ) , \cos ( \pi t ) - 1 , \sin ( 2 \pi t ) \right) ^ { \top } . } \end{array}\tag{7}
$$

For Loop Gating, a learned vector $q \in \mathbb { R } ^ { 8 }$ determines the update:

$$
\begin{array} { r } { \begin{array} { c } { g _ { \ell } = 1 + q ^ { \top } \psi _ { \ell } , } \\ { h _ { \ell + 1 } = h _ { \ell } + g _ { \ell } \big ( F _ { \theta } ( h _ { \ell } ) - h _ { \ell } \big ) . } \end{array} } \end{array}\tag{8}
$$

For the other two variants, let $k \in \{ 1 , \ldots , K \}$ index a layer and $b \in \{ \mathrm { a t t n } , \mathrm { m l p } \}$ index its residual branch. Writing $B _ { k } ^ { b }$ for the branch operation and $N _ { k } ^ { b }$ for its preceding RMSNorm, the branch update is

$$
u ^ { + } = u + g _ { \ell , k } ^ { b } \odot \mathcal { B } _ { k } ^ { b } \big ( ( { \mathbf 1 } + s _ { \ell , k } ^ { b } ) \odot N _ { k } ^ { b } ( u ) \big ) .\tag{9}
$$

Each layer applies this update first to the attention branch and then to the MLP branch, with u denoting the current branch input. The backbone RMSNorm parameters are retained.

Branch Gating uses

$$
g _ { \ell , k } ^ { b } = 1 + ( q _ { k } ^ { b } ) ^ { \top } \psi _ { \ell } , \qquad s _ { \ell , k } ^ { b } = { \bf 0 } ,\tag{10}
$$

where $q _ { k } ^ { b } \in \mathbb { R } ^ { 8 }$ and the scalar gate is broadcast over hidden channels. AdaLN instead uses

$$
\begin{array} { r } { \left[ \boldsymbol { g } _ { \ell , k } ^ { b } - \mathbf { 1 } \right] = W _ { k } ^ { b } \psi _ { \ell } , \qquad W _ { k } ^ { b } \in \mathbb { R } ^ { 2 d \times 8 } . } \end{array}\tag{11}
$$

Thus, each layer generates four d-dimensional vectors, with no timestep-dependent additive shift.   
These linear maps implement our parameterization of the AdaLN modulation.

All conditioning weights are learned jointly with the backbone and shared across token positions and recurrent iterations. The weights indexed by k and b are distinct across layers and branches. We initialize $q , q _ { k } ^ { b } .$ , and $W _ { k } ^ { b }$ to zero, so every gate initially equals one and every additional normalization scale equals zero, recovering the unconditioned BaseLoop computation. The gates and scales are unconstrained. Loop Gating, Branch Gating, and AdaLN add 8, 16K, and 32Kd trainable parameters, respectively.

During training, $T = L _ { \mathrm { t r a i n } }$ . For an inference budget of $L _ { \mathrm { i n f e r } }$ iterations, we consider two conditioning grids. Prefix inference retains $T = L _ { \mathrm { t r a i n } }$ and executes the first $L _ { \mathrm { i n f e r } }$ steps, requiring $L _ { \mathrm { i n f e r } } \leq L _ { \mathrm { t r a i n } }$ . Rescaled inference sets $T = L _ { \mathrm { i n f e r } } ,$ , so that $t _ { \ell } = \ell / L _ { \mathrm { i n f e r } }$ and $\Delta t = 1 / L _ { \mathrm { i n f e r } } ;$ this supports budgets both below and above the training budget. In both cases, $\ell = 0 , \ldots , L _ { \mathrm { i n f e r } } - 1$ . The step size enters through the conditioning features; the updates above have no additional $\Delta t$ multiplier.

## C.2 TRAINING

Recurrent backpropagation. All LoopLM models are trained from scratch using the standard autoregressive language-modeling objective. Given a token sequence $( x _ { 1 } , \ldots , x _ { T } )$ , we minimize the mean next-token cross-entropy

$$
\mathcal { L } _ { \mathrm { L M } } = - \frac { 1 } { T - 1 } \sum _ { t = 1 } ^ { T - 1 } \log p _ { \theta } ( x _ { t + 1 } \mid x _ { \le t } ) ,
$$

where the loss is averaged over all non-masked target tokens.

During training, each recurrent core is executed using the fixed loop count K specified by its architecture. Gradients are propagated through all K loops of the recurrent core using full backpropagation through time. Consequently, the gradient of each shared core block aggregates its contributions from every recurrent iteration. No recurrent iteration is detached from the computation graph.

Optimizer. We optimize all models using Muon Jordan et al. (2024); Liu et al. (2025). Specifically, Muon is used for the matrix-valued attention and feed-forward projection weights. Token embeddings, the language-model head, normalization parameters, and other vector- or scalar-valued parameters are optimized using AdamW Loshchilov $\&$ Hutter (2019). For Muon parameter group, we use momentum 0.95, Nesterov momentum, and five-step Newton-Schulz iterations per optimizer update. For AdamW parameter group, we use

$$
( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 5 ) , \qquad \epsilon _ { \mathrm { A d a m W } } = 1 0 ^ { - 8 } .
$$

A decoupled weight decay of 0.1 is applied. The configured peak learning rates are

$$
\eta _ { \mathrm { m a x } } ^ { \mathrm { L l a m a } } = 1 . 3 3 8 4 7 \times 1 0 ^ { - 3 } , \qquad \eta _ { \mathrm { m a x } } ^ { \mathrm { Q w e n } } = 1 . 8 0 1 1 1 \times 1 0 ^ { - 3 } .
$$

Learning-rate scheduler. All LoopLM experiments use a linear warmup-stable-decay (WSD) scheduler. The learning rate increases linearly from zero to $\eta _ { \mathrm { m a x } }$ during the first 5% of optimizer updates, remains at $\eta _ { \mathrm { m a x } }$ for the next 85%, and then decreases linearly during the final 10%. The terminal learning rate is set as $0 . 1 \eta _ { \mathrm { m a x } }$ . More precisely, for warmup, stable, and decay lengths $T _ { \mathrm { w } }$ $T _ { \mathrm { s } }$ , and $T _ { \mathrm { d } }$ , respectively, the learning-rate multiplier is

$$
\lambda ( t ) = \left\{ \begin{array} { l l } { t / T _ { \mathrm { w } } , } & { 0 \le t \le T _ { \mathrm { w } } , } \\ { 1 , } & { T _ { \mathrm { w } } < t \le T _ { \mathrm { w } } + T _ { \mathrm { s } } , } \\ { 1 - 0 . 9 \frac { t - T _ { \mathrm { w } } - T _ { \mathrm { s } } } { T _ { \mathrm { d } } } , } & { T _ { \mathrm { w } } + T _ { \mathrm { s } } < t \le T _ { \mathrm { w } } + T _ { \mathrm { s } } + T _ { \mathrm { d } } . } \end{array} \right.
$$

The learning rate at update t is $\eta ( t ) = \eta _ { \mathrm { m a x } } \lambda ( t )$ . Exact schedule lengths and batch sizes are reported in Tab. 3.

Gradient accumulation and clipping. The Llama3.1-1B runs with global batch size 1,824 use a perdevice micro-batch size of 38 and accumulate gradients over six micro-batches. The Qwen3-0.6B runs use a per-device micro-batch size of 32 and accumulate gradients over eight micro-batches. Losses are divided by the number of accumulation steps before backpropagation. After the accumulated gradients have been synchronized and before each optimizer update, we clip the global $\ell _ { 2 }$ norm of all model gradients to 1.0.

<table><tr><td>Models</td><td>ηmax</td><td>Global batch</td><td>Steps</td><td>W/S/D updates</td><td>Tokens</td></tr><tr><td>Llama3.1-1B BaseLoop and CoreLoop variants</td><td> $\begin{array} { c } { { \overline { { { 1 . 3 3 8 4 7 \times 1 0 ^ { - 3 } } } } } } \\ { { 1 . 8 0 1 1 1 \times 1 0 ^ { - 3 } } } \end{array}$ </td><td>1,824</td><td>5,394</td><td>270/4,585/539</td><td>20.15B</td></tr><tr><td>Qwen3-0.6B BaseLoop and CoreLoop variants</td><td></td><td>2,048</td><td>2,860</td><td>143/2,431/286</td><td>12.00B</td></tr></table>

Table 3: Training settings for LoopLM models. Global batch size is measured in packed sequences of length 2,048. W/S/D gives the exact numbers of warmup, stable, and decay updates. Token counts are computed before the one-token shift used by the next-token objective.

Precision, initialization, and reproducibility. Training uses bfloat16 model parameters and bfloat16 mixed-precision computation. Linear and embedding weights are initialized independently from a zero-mean Gaussian distribution with standard deviation 0.02. All experiments use random seed 42.

## C.3 EVALUATION

All benchmarks are evaluated in zero-shot mode with lm-evaluation-harness Gao et al. (2024). Notably, we use length-normalized accuracy for multiple-choice tasks whose options differ in length, and plain accuracy for tasks with fixed-form options. Every task is evaluated at each inference loop count r independently, so a full evaluation sweep yields one accuracy value per task per r.

Task groups. The Knowledge group (Tab. 4) contains three tasks and the Reasoning group (Tab. 5) eight tasks. Group scores are unweighted averages,

$$
S _ { \mathrm { K n o w } } = \textstyle { \frac { 1 } { 3 } } \sum _ { i \in \mathcal { T } _ { \mathrm { K n o w } } } a _ { i } , \qquad S _ { \mathrm { R e a s } } = \textstyle { \frac { 1 } { 8 } } \sum _ { i \in \mathcal { T } _ { \mathrm { R e a s } } } a _ { i } ,\tag{12}
$$

where $a _ { i }$ is the headline metric of task i. Equal weighting is deliberate: it prevents a group score from being dominated by whichever benchmark happens to have the largest dynamic range at this model scale, at the cost of giving each task equal influence regardless of test-set size. Notably, we assign PIQA to the knowledge group even though it is commonly described as physical commonsense reasoning, because its instances are resolved via stored knowledge of object affordance and material properties.

<table><tr><td>Task</td><td>Brief Description</td></tr><tr><td>ScIQ Welbl et al. (2017)</td><td>Science facts; locating evidence in a support passage</td></tr><tr><td>ARC-EASY Clark et al. (2018)</td><td>Elementary science knowledge and simple causal attribution</td></tr><tr><td>PIQA Bisk et al. (2020)</td><td>Object affordances, material properties, outcomes of actions</td></tr></table>

Table 4: Three tasks in the Knowledge group.

<table><tr><td>Task</td><td>Brief Description</td></tr><tr><td>ARC-CHALLENGE Clark et al. (2018)</td><td>Multi-step inference over harder science items</td></tr><tr><td>WINOGRANDE Sakaguchi et al. (2020)</td><td>Coreference resolution requiring commonsense</td></tr><tr><td>OPENBOOKQA Mihaylov et al. (2018)</td><td>Combining a retrieved science fact with additional knowledge</td></tr><tr><td>HELLASWAG Zellers et al. (2019)</td><td>Plausible event continuation; temporal and causal structure</td></tr><tr><td>COMMONSENSEQA Talmor et al. (2019)</td><td>ConceptNet-style relational reasoning over use, location</td></tr><tr><td>PROOFWRITER Tafjord et al. (2021)</td><td>Entailment under natural-language facts and rules</td></tr><tr><td>CLUTRR Sinha et al. (2019)</td><td>Kinship inference along a relation chain in a short story</td></tr><tr><td>BBH Suzgun et al. (2023)</td><td>Logical deduction, temporal ordering, object-state tracking</td></tr></table>

Table 5: Eight tasks in the Reasoning group. The last three tasks are the ones whose difficulty is controlled by an explicit compositional parameter.

Selected BBH subtasks. These 14 subtasks cover complementary dimensions of reasoning, as shown in Tab. 6. disambiguation qa tests pronoun resolution and the recognition of genuine ref erential ambiguity; salient translation error detection requires classifying semantic errors in German-to-English translations; reasoning about colored objects evaluates attribute retrieval, spatial relations, and counting over described objects. causal judgement assesses commonsense causal attribution and intentionality, whereas date understanding tests calendar arithmetic. hyperbaton evaluates knowledge of English adjective ordering. The two logical deduction variants require recovering an ordering from relational constraints. penguins in a table requires structured table lookup, comparison, counting, and sorting.

<table><tr><td>Subtask</td><td>Brief Description</td></tr><tr><td>disambiguation_qa</td><td>Resolves pronoun antecedents and identifies cases that remain genuinely ambiguous.</td></tr><tr><td>salient_translation_error_detection</td><td>Classifies salient semantic errors in German-to-English translations, such as altered entities, numbers, negation, or omitted content.</td></tr><tr><td>reasoning_about_colored_objects</td><td>Answers attribute, spatial-relation, and counting questions about colored objects described in natural language.</td></tr><tr><td>causal_judgement</td><td>Determines commonsense causal attribution and whether an action or outcome was intentional.</td></tr><tr><td>date_understanding</td><td>Infers calendar dates from relative temporal expressions and performs date arithmetic.</td></tr><tr><td>hyperbaton</td><td>Selects the sentence exhibiting the grammatically natural ordering of English adjectives.</td></tr><tr><td>logical_deduction_three_objects</td><td>Infers the ordering of three objects from a set of logically consistent relational constraints.</td></tr><tr><td>logical_deduction_seven_objects</td><td>Infers the ordering of seven objects from relational constraints, requiring a larger reasoning state.</td></tr><tr><td>penguins_in_a_table</td><td>Performs lookup, comparison, counting, sorting, and update operations over a semi-structured table.</td></tr><tr><td>snarks</td><td>Identifies which of two statements is sarcastic, testing pragmatic and</td></tr><tr><td>temporal_sequences</td><td>contextual language understanding. Finds a feasible time interval by reasoning over schedules, event durations, and temporal constraints.</td></tr><tr><td>tracking_shuffled_objects_three_objects</td><td>Tracks a sequence of pairwise swaps among three entities to determine the final object assignment.</td></tr><tr><td>tracking_shuffled_objects_five_objects</td><td>Tracks a sequence of pairwise swaps among five entities, increasing the</td></tr><tr><td>tracking_shuffled_objects_seven_objects</td><td>required state-tracking capacity. Tracks a sequence of pairwise swaps among seven entities, providing the most demanding state-tracking variant.</td></tr></table>

Table 6: The selected 14 BBH subtasks and their evaluated capabilities.

snarks probes pragmatic reasoning through sarcasm detection. temporal sequences tests temporal-constraint reasoning. Finally, the three tracking shuffled objects variants require maintaining entity–object assignments through a sequence of swaps. The variants with different numbers of objects provide a controlled measure of how performance changes as the amount of relational state to be tracked increases. Notably, the remaining BBH subtasks were excluded because they proved excessively challenging for models at the evaluated scale. Across model variants, performance on most of these tasks fluctuated around their task-specific random-guessing baselines and showed no consistent separation between models. Consequently, they provided little discriminative signal for meaningful model comparison and were not included in the aggregate score.

## C.4 GEOMETRY PROBING

Execution and sampling. We probe frozen checkpoints in evaluation mode on held-out validation set. The reported Qwen3-0.6b and Llama3.1-1B comparisons each use 49 packed sequences of maximum length 2,048, with exactly 100,000 selected next-token prediction positions. Within each model family, the tokenized examples and selection masks are identical across configurations and inference depths. Let T contain these selected positions, N be the number of nonempty sequences, and $p _ { n }$ be the last selected prediction position in sequence n. Angular distance and Relative update norm use one position $p _ { n }$ per sequence; Variance and update-response scores use all positions in $\tau$ The selection follows the next-token objective, including the partially selected final sequence at the token budget boundary.

Let $h _ { \ell , p } ^ { \mathrm { i n } } , h _ { \ell , p } ^ { \mathrm { o u t } } \in \mathbb { R } ^ { m }$ denote the actual input and output of the block executed at effective layer position $\ell ,$ for token position $p .$ Repeated executions of a shared block have distinct effective positions, even though they use the same parameters. For prelude depth $^ { a , }$ shared depth $s ,$ coda depth $b ,$ and L inference loops, the evaluated layouts have effective depth $\bar { \boldsymbol { D } } = \boldsymbol { a } + s \boldsymbol { L } + \bar { \boldsymbol { b } }$ . We also record $x _ { p } ^ { ( t ) }$ , the recurrent state after t complete loops, for $t = 0 , \ldots , L ; x _ { p } ^ { ( 0 ) }$ is the state entering the first loop after any prelude. A complete loop includes state injection, the shared blocks, and any loop-level mixing or gating. Block probes use each block’s own boundaries, whereas loop probes include these additional operations. In particular, injection can make a block’s input differ from the preceding block’s output. Recurrent-state probes exclude the coda and final output normalization.

Angular distance. For two state vectors $u , v ,$ , we compute normalized angular distance

$$
c _ { \epsilon } ( \boldsymbol { u } , \boldsymbol { v } ) = \mathrm { c l i p } _ { [ - 1 , 1 ] } \bigg ( \frac { \boldsymbol { u } ^ { \top } \boldsymbol { v } } { \operatorname* { m a x } ( \| \boldsymbol { u } \| _ { 2 } \| \boldsymbol { v } \| _ { 2 } , \epsilon ) } \bigg ) ,\tag{13}
$$

$$
a ( u , v ) = \frac { \operatorname { a r c c o s } c _ { \epsilon } ( u , v ) } { \pi } , \qquad \epsilon = 1 0 ^ { - 1 2 } .\tag{14}
$$

The block and loop measurements are

$$
A _ { \ell } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } a ( h _ { \ell , p _ { n } } ^ { \mathrm { i n } } , h _ { \ell , p _ { n } } ^ { \mathrm { o u t } } ) ,\tag{15}
$$

$$
A _ { \mathrm { l o o p } } ( t ) = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } a ( x _ { p _ { n } } ^ { ( t ) } , x _ { p _ { n } } ^ { ( t + 1 ) } ) , \quad 0 \le t < L .\tag{16}
$$

For nondegenerate vectors, values $0 , 1 / 2 ,$ , and 1 indicate aligned, orthogonal, and opposite directions, respectively. We average the individual angular distances, rather than applying arccos to an averaged cosine similarity. Clipping prevents numerical overshoots; the denominator floor assigns distance $1 / 2$ when either vector is zero. The metric measures directional change and is insensitive to positive rescaling away from the numerical floor.

Relative update norm. We measure the update magnitude relative to the incoming state:

$$
\rho _ { \ell } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \frac { \lVert h _ { \ell , p _ { n } } ^ { \mathrm { o u t } } - h _ { \ell , p _ { n } } ^ { \mathrm { i n } } \rVert _ { 2 } } { \operatorname* { m a x } ( \lVert h _ { \ell , p _ { n } } ^ { \mathrm { i n } } \rVert _ { 2 } , \epsilon ) } ,\tag{17}
$$

$$
\rho _ { \mathrm { l o o p } } ( t ) = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \frac { \| x _ { p _ { n } } ^ { ( t + 1 ) } - x _ { p _ { n } } ^ { ( t ) } \| _ { 2 } } { \operatorname* { m a x } ( \| x _ { p _ { n } } ^ { ( t ) } \| _ { 2 } , \epsilon ) } .\tag{18}
$$

These are means of per-sequence ratios, not ratios of mean norms. A value of 0.1 corresponds to a mean relative update magnitude of 10%. The metric can exceed one. Angular distance and relative update norm capture different changes: multiplying a nonzero state by a positive scalar leaves its angle unchanged but can yield a substantial relative update. Small loop updates indicate little movement under the measured iteration, without establishing convergence or task usefulness.

State variance and plot normalization. For a state vector $\boldsymbol { h _ { p } } \in \mathbb { R } ^ { m }$ , we first compute variance across its hidden coordinates, using correction one, and then average over tokens:

$$
\bar { h } _ { p } = \frac { 1 } { m } \sum _ { k = 1 } ^ { m } h _ { p , k } ,\tag{19}
$$

$$
V ( h ) = \frac { 1 } { | T | } \sum _ { p \in \mathcal { T } } \frac { 1 } { m - 1 } \sum _ { k = 1 } ^ { m } ( h _ { p , k } - \bar { h } _ { p } ) ^ { 2 } .\tag{20}
$$

The block-output variance is $V ( h _ { \ell } ^ { \mathrm { o u t } } )$ , and the recurrent-state variance is $V ( \boldsymbol { x } ^ { ( t ) } )$ ; block-input variances are also recorded. This measures within-token feature dispersion, rather than variance across examples or the rank of a representation covariance. All selected tokens receive equal weight.

The BaseLoop/CoreLoop trajectory figure normalizes each model by its own state variance at the training loop count $L _ { \mathrm { t r } } .$

$$
\widetilde V ( t ) = \frac { V ( x ^ { ( t ) } ) } { V ( x ^ { ( L _ { \mathrm { t r } } ) } ) } .\tag{21}
$$

Thus, $\widetilde { V } ( L _ { \mathrm { t r } } ) = 1 ;$ a value above one indicates growth relative to that model’s state at the training horizon. The block-angle horizontal axis uses effective depth ℓ. Loop-angle and loop-update transitions are located at $( t + 1 ) / L _ { \mathrm { t r } }$ , whereas state variances are located at $t / L _ { \mathrm { t r } }$ and include $t = 0$ The reference at one marks the training loop count. Logarithmic axes change the display scale only.

To locate amplification within a loop, let $z ^ { ( t ) }$ be the input to its first shared block, after any state injection, and $y ^ { ( t ) }$ the output of its last shared block, before loop-level mixing. Define

$$
g _ { \mathrm { i n j } } ( t ) = \frac { V ( z ^ { ( t ) } ) } { V ( x ^ { ( t ) } ) } ,
$$

$$
g _ { \mathrm { c o r e } } ( t ) = \frac { V ( y ^ { ( t ) } ) } { V ( z ^ { ( t ) } ) } ,\tag{22}
$$

$$
g _ { \mathrm { m i x } } ( t ) = { \frac { V ( x ^ { ( t + 1 ) } ) } { V ( y ^ { ( t ) } ) } } .\tag{23}
$$

For positive variances, these factors telescope exactly:

$$
{ \frac { V ( x ^ { ( L ) } ) } { V ( x ^ { ( L _ { \mathrm { t r } } ) } ) } } = \prod _ { t = L _ { \mathrm { t r } } } ^ { L - 1 } g _ { \mathrm { i n j } } ( t ) g _ { \mathrm { c o r e } } ( t ) g _ { \mathrm { m i x } } ( t ) .\tag{24}
$$

This accounting identifies the stage at which variance grows. The factors are ratios of token-averaged variances, not additive contribution fractions or means of per-token variance ratios.

Computational interaction. For effective positions $i < j .$ we run the same examples twice: once normally and once replacing the output of block occurrence i by its input. Only that occurrence is skipped; other executions of the shared parameters remain active, and the remaining computation proceeds normally. At target position j, define

$$
u _ { j , p } = h _ { j , p } ^ { \mathrm { o u t } } - h _ { j , p } ^ { \mathrm { i n } } ,\tag{25}
$$

$$
u _ { j , p } ^ { ( - i ) } = h _ { j , p } ^ { \mathrm { o u t } , ( - i ) } - h _ { j , p } ^ { \mathrm { i n } , ( - i ) } .\tag{26}
$$

Each update uses the input and output from its own execution. The response numerator and baseline update magnitude are

$$
N _ { i  j } = \frac { 1 } { | T | } \sum _ { p \in { \cal T } } \| u _ { j , p } ^ { ( - i ) } - u _ { j , p } \| _ { 2 } ,\tag{27}
$$

$$
B _ { j } = \frac { 1 } { | T | } \sum _ { p \in T } \| u _ { j , p } \| _ { 2 } ,\tag{28}
$$

and the relative response is

$$
C _ { i  j } = \frac { N _ { i  j } } { B _ { j } } .\tag{29}
$$

Unlike $\rho , C$ is a ratio of token-averaged norms. For example, $C = 0 . 1$ means that the mean change in the downstream update vector is one tenth of its baseline mean norm. It measures the change in the update, including its direction, rather than only the change in update magnitude or the difference between output states. The implementation records zero when $\dot { B } _ { j } \leq 1 0 ^ { - 8 } $ ; all pairs in the main response figure exceed this threshold. This fallback should not be interpreted as a measured absence of influence.

Aggregation by loop lag. For Qwen BaseLoop $4 \times 7$ , the loop containing effective position $j$ is $r ( \bar { j } ) = \bar { 1 } + \lfloor ( j - 1 ) \bar { / 4 } \rfloor$ . We include only pairs whose source and target both lie beyond the seven training loops. For lag d, define

$$
{ \mathcal { P } } _ { d } = \{ ( i , j ) : 1 \leq i < j \leq 4 L , \ r ( i ) > 7 , \ r ( j ) > 7 ,
$$

$$
r ( j ) - r ( i ) = d \} ,\tag{30}
$$

$$
C ( d ) = \frac { 1 } { | \mathcal { P } _ { d } | } \sum _ { ( i , j ) \in \mathcal { P } _ { d } } C _ { i \to j } .\tag{31}
$$

Here $d = 0$ compares different blocks in the same loop, $d = 1$ compares adjacent loops, and larger values compare more widely separated loops. The pair counts are $\bigr \vert \mathcal { P } _ { 0 } \bigr \vert ^ { \circ } = 6 ( L - \mathsf { \bar { 7 } } )$ and $| \mathcal { P } _ { d } | = 1 6 ( L - 7 - d )$ for $1 \leq d \leq L - 8$ . Accordingly, the full lag ranges are $_ { 0 - 6 }$ at $D = 5 6$ $( L = 1 4 )$ and 0–13 at $D = 8 4 \left( L = 2 1 \right)$ ). Panels (b,c) plot these $C ( d )$ values.

For a lag set $\mathcal { D } ,$ the numerical summaries in the main text use

$$
C _ { \mathcal { D } } = \frac { \sum _ { d \in \mathcal { D } } | \mathcal { P } _ { d } | C ( d ) } { \sum _ { d \in \mathcal { D } } | \mathcal { P } _ { d } | } .\tag{32}
$$

Thus, individual block pairs receive equal weight. The common cross-loop range is $\mathcal { D } = \{ 1 , \ldots , 6 \}$ the triple comparison also uses $\mathcal { D } = \{ 2 , \ldots , \bar { 6 } \}$ . Percent differences between configurations A and B are $\overset { \cdot } { 1 0 0 } ( C _ { \mathcal { D } } ^ { A } / C _ { \mathcal { D } } ^ { B } - 1 )$ ). Panel (d) instead plots the pointwise ratio

$$
R _ { T } ( d ) = \frac { C _ { \mathrm { I + H + T } } ( d ) } { C _ { \mathrm { H + T } } ( d ) } , \qquad T \in \{ \mathrm { L G } , \mathrm { B G } \} ,\tag{33}
$$

matching history parameterization, window size, gate type, and inference depth. Ratios below one indicate a smaller mean response in the triple. These are ratios of configuration-level means, not means of matched pairwise ratios. Different lags contain different absolute positions and numbers of pairs; the curves do not follow one fixed perturbation as it travels through the network.

## D FULL EXPERIMENTAL RESULTS

## D.1 OPTIMIZER EXPERIMENT

Setup. We conducted a pilot study to determine the optimizer used in our main pre-training experiments. We compare AdamW and Muon under the same pre-training setup across six architectures, including standard non-recurrent Transformers with 12 unique layers (nonloop 12×1) and 3 unique layers (nonloop 3×1), as well as four LoopLM architectures: BaseLoop (3×4), Loop-Former (3×4) Jeddi et al. (2026), Ouro $( 3 \times 4 )$ Zhu et al. (2025), and Huginn (1+2×5+1) Geiping et al. (2025). Here, the notation describes the arrangement of unique and recurrent layers for each architecture. After pre-training, we evaluate all successfully trained models on ten commonly used language understanding and commonsense reasoning benchmarks: ARC-Easy (ARC-E), ARC-Challenge (ARC-C) Clark et al. (2018), SciQ Welbl et al. (2017), MMLU Hendrycks et al. (2021), HellaSwag (HELLA) Zellers et al. (2019), OpenBookQA (OBQA) Mihaylov et al. (2018), PIQA Bisk et al. (2020), RACE Lai et al. (2017), WinoGrande (WINO) Sakaguchi et al. (2020), and CommonsenseQA (CSQA) Talmor et al. (2019). The average score across these benchmarks is reported as the overall performance.

<table><tr><td>Model</td><td>ARC-E</td><td>ARC-C</td><td>SciQ</td><td>MMLU</td><td>HELLA</td><td>OBQA</td><td>PIQA</td><td>RACE</td><td>WINO</td><td>CSQA</td><td>Avg.</td></tr><tr><td>AdamW</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>nonloop (12× 1)</td><td>59.30</td><td>25.68</td><td>81.50</td><td>26.06</td><td>33.64</td><td>22.00</td><td>67.52</td><td>31.67</td><td>51.46</td><td>20.64</td><td>41.95</td></tr><tr><td>nonloop (3×1)</td><td>53.41</td><td>20.90</td><td>74.30</td><td>23.10</td><td>29.04</td><td>18.40</td><td>63.38</td><td>27.56</td><td>50.99</td><td>20.31</td><td>38.14</td></tr><tr><td>BaseLoop (3×4)</td><td>56.23</td><td>23.81</td><td>77.00</td><td>23.00</td><td>31.05</td><td>21.40</td><td>64.20</td><td>30.33</td><td>53.28</td><td>19.49</td><td>39.98</td></tr><tr><td>LoopFormer (3×4)</td><td>56.73</td><td>23.89</td><td>77.90</td><td>24.59</td><td>30.97</td><td>20.00</td><td>64.64</td><td>29.28</td><td>51.62</td><td>19.98</td><td>39.96</td></tr><tr><td>Ouro (3×4)</td><td>55.85</td><td>22.87</td><td>77.90</td><td>23.03</td><td>30.74</td><td>18.00</td><td>63.82</td><td>29.95</td><td>52.49</td><td>19.57</td><td>39.42</td></tr><tr><td colspan="2">Huginn (1+2×5+1) Failed</td><td colspan="10"></td></tr><tr><td>Muon nonloop (12×1)</td><td>59.76</td><td>25.43</td><td>81.20</td><td>24.90</td><td>34.07</td><td>22.80</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>53.41</td><td>22.61</td><td>74.00</td><td>22.93</td><td>29.50</td><td>18.80</td><td>67.90</td><td>32.06</td><td>51.93</td><td>20.31</td><td>42.04</td></tr><tr><td>nonloop (3× 1)</td><td>57.41</td><td>26.19</td><td></td><td></td><td></td><td></td><td>64.98</td><td>26.99</td><td>50.12</td><td>19.57</td><td>38.29</td></tr><tr><td>BaseLoop (3×4)</td><td></td><td></td><td>78.50</td><td>25.30</td><td>31.48</td><td>20.80</td><td>65.18</td><td>30.62</td><td>52.88</td><td>19.08</td><td>40.74</td></tr><tr><td>LoopFormer (3×4)</td><td>56.14</td><td>23.89</td><td>80.90</td><td>23.22</td><td>31.68</td><td>17.60</td><td>65.18</td><td>31.10</td><td>50.59</td><td>20.15</td><td>40.05</td></tr><tr><td>Ouro (3×4)</td><td>56.99</td><td>25.09</td><td>77.60</td><td>23.81</td><td>31.72</td><td>19.80</td><td>63.93</td><td>30.33</td><td>50.51</td><td>19.74</td><td>39.95</td></tr><tr><td>Huginn (1+2×5+1)</td><td>58.16</td><td>24.83</td><td>80.70</td><td>25.22</td><td>33.19</td><td>22.60</td><td>66.10</td><td>32.34</td><td>53.59</td><td>20.48</td><td>41.72</td></tr></table>

Table 7: Pilot study comparing AdamW and Muon for pre-training different non-recurrent and recurrent architectures. Results are reported on ten downstream evaluation benchmarks together with their average. Failed indicates that the corresponding pre-training run did not successfully converge.

Results. As shown in Tab. 7, Muon consistently provides stronger overall performance than AdamW across all architectures successfully trained with both optimizers. In particular, the average score improves from 41.95 to 42.04 for the 12-layer non-recurrent Transformer, from 38.14 to 38.29 for the 3-layer non-recurrent Transformer, from 39.98 to 40.74 for BaseLoop, from 39.96 to 40.05 for LoopFormer, and from 39.42 to 39.95 for Ouro. More importantly, the Huginn $1 + 2 \times 5 + 1$ configuration fails to train with AdamW, whereas Muon successfully trains the model and achieves an average score of 41.72. Based on both its consistently better downstream performance and training stability, we therefore adopt Muon as the default optimizer for all main experiments.

## D.2 BASELOOP AND CORELOOP

Fig. 9 shows that the preferred CoreLoop layout on Qwen depends on inference depth. The coda-only configuration $0 + 2 \times 1 3 + 2$ achieves higher Overall scores at reduced depths, but its Knowledge score declines more strongly during extrapolation. The balanced configuration $1 + 2 \times 1 3 + 1$ achieves the highest Overall and Knowledge scores at the training depth D = 28. At D = 56, however, BaseLoop achieves higher Overall and Reasoning scores than all three CoreLoop variants. Thus, the gains from introducing fixed layers depend on both their placement and the inference budget.

Figs 10, 11, and 12 extend the Llama geometry analysis to two, five, and ten physical layers. At the training loop count, all displayed CoreLoop variants exhibit smaller loop angular distances and relative update norms than their BaseLoop counterparts. During extrapolation, small angular changes coexist with continued growth in normalized state variance. The allocation of fixed layers also changes the trajectories. For example, with two physical layers, the prelude-only configuration

(c) Reasoning

Base 4×7 Core 1+2×13+1 Core 0+2×13+2 Core 2+2×13+0  
![](images/b1a52f168ead95f426928e239547b63c80947a2fce7ff18f1f5cfbb14b7cc6a6.jpg)

![](images/2592fc3db74ae350c566793c268e5cc06fccbff49b9818c37d5ae1ec58888676.jpg)

![](images/564f94350f2d5a07eeb2a654679e068a1957147baf134010996630c634403fdd.jpg)

Figure 9: Average results of BaseLoop and CoreLoop models from Qwen3-0.6B with physical layers of 4 on Overall, Knowledge, and Reasoning benchmarks. The dashed vertical line marks the effective training depth of 28 layers. Results at or below this depth correspond to evaluation within the training depth budget, while the yellow-shaded region denotes extrapolation beyond it.  
![](images/4697fcae1c5035b4aaf3200526d63e1e8a80bb9c618b41a4aeec6a5019ae599c.jpg)

![](images/bdac7650b889517d01d331ac9adc4f772f80723ede283e2289dd328d2da2d9ae.jpg)

![](images/8a9cf4b716e3a5e362e6f708c077f7776ad163a3bb0cfaa3b4ce680d56a0983d.jpg)

![](images/e8dfb61369e1c4f2188a59db81c1ee36b4c39bc7891840fc11599ee58deb297c.jpg)  
Figure 10: Representation dynamics of BaseLoop and CoreLoop models with two physical layers from Llama3.1-1B. The horizontal axes in (b)-(d) are normalized by each configuration’s training iteration count. (a): The angular distance between consecutive layer states. Crosses mark coda layers at the ends of the recorded trajectories. (b): The same angular measure between recurrent states before and after each complete loop iteration. (c): The per-loop update magnitude relative to the preceding state, plotted on a logarithmic scale. (d): Recurrent-state variance relative to its value at the training iteration count, plotted on a logarithmic scale.

$1 + 1 \times 1 9 + 0$ ends with smaller recurrent updates and less normalized variance growth than the coda-only configuration $0 + 1 \times 1 9 + 1$ . For configurations with a coda, the terminal layer-angle increases correspond to the coda transformations following the recurrent trajectory.

## D.3 INITIAL-STATE INPUT INJECTION

The Llama3.1-1B experiments in Fig. 13 exhibit the similar qualitative pattern as the Qwen results in the main content: initial-state injection provides localized improvements, but does not consistently mitigate degradation beyond the training depth. For the 4 × 5 configuration, all four injection variants improve Overall at $D = 2 0$ , from 37.87 to between 38.21 and 38.42, yet all fall below BaseLoop at $\mathbf { \bar { \boldsymbol { D } } = 4 0 \left( T a b . 8 \right) }$ . Dense injection exhibits pronounced Knowledge degradation across all four configurations, while the simpler parameterizations also fail to consistently prevent this decline. The effects are metric-dependent: for $\mathbf { \bar { 4 } } \times 5 \mathbf { a t } D = 4 0 \mathbf { \bar { . } }$ , Scalar and Channel-wise injection retain higher Knowledge scores than BaseLoop, but achieve lower Reasoning and Overall scores. Tabs. 8 and 9 report results for all evaluated parameterizations at the training depth and twice that depth.

## D.4 HISTORY-STATE INPUT INJECTION

Tab. 10 complements the history-state injection curves in the main content with numerical results on Qwen3-0.6B under the $4 \times 7$ shared-stack configuration. We compare BaseLoop, initial-state injection, and history-state injection with windows $\bar { w } \in \{ 1 , 2 , 4 \}$ at effective depths $\bar { D ^ { } } = 2 8 , 5 6 , 8 4 .$ corresponding to the training depth and 2× and 3× depth extrapolation.

(a) Layer Angular Distance  
![](images/0f2392b5ad0d63438900d5b1c7a98e201ba33d7c9b37c1b7cb6bbd4f659bf225.jpg)

(b) Loop Angular Distance  
![](images/5a9a56992b733bc950daf9b0aabf8431d903742c3b44da7686ed7d2eb26bde91.jpg)  
Normalized Loop Iteration Core 0+3×6+2

(c) Relative Update Norm  
![](images/12b31b4ec7c4b189df0e5ac8ed83b6f77c36ba6280036648d38d91404660bb87.jpg)  
Normalized Loop Iteration Core 1+3×6+1 -- (

(d) Normalized State Variance  
![](images/567a50f7476616bf98a024e7d0b72cc8e673b3248001786d9a455e9362e74099.jpg)  
Core 2+3×6+0  
Normalized Loop Iteration  
Figure 11: Representation dynamics of BaseLoop and CoreLoop models with five physical layers from Llama3.1-1B. BaseLoop $5 \times 4$ is compared with three CoreLoop variants that allocate two fixed layers between the prelude and coda.

(a) Layer Angular Distance  
![](images/cdafdd86e387862e4a459f8214e3c5810330fcccebb7f8225fd8044856cc25ec.jpg)

(b) Loop Angular Distance  
![](images/bac07fbc186e149b3bf46427cb67f5d5296f4de072de6ccb19d35e9738b75af1.jpg)  
Normalized Loop Iteration

![](images/d4d47476597cf4ddcac3367aa0d23877f74824c734aa14e59dcfe77b88324a3b.jpg)  
Normalized Loop Iteration

(d) Normalized State Variance  
![](images/7822c07cb3354a53fea438cbaf901076b0b6bf8dfffbac5b4d6c21a0b1cf7f23.jpg)  
Normalized Loop Iteration  
Base 10×2 Core 0+5×3+5 Core 1+5×3+4 Core 2+5×3+3 Core 3+5×3+2 Core 4+5×3+1 Core 5+5×3+0  
Figure 12: Representation dynamics of BaseLoop and CoreLoop models with ten physical layers from Llama3.1-1B. BaseLoop $1 0 \times 2$ is compared with six CoreLoop variants $p + 5 \times 3 + ( 5 - p )$ $p = 0 , \ldots , 5 .$ , covering all allocations of five fixed layers between the prelude and coda.

The preferred history window depends on the parameterization. At D = 84, Scalar history injection with $w = 4$ achieves the highest Overall and Knowledge scores among the evaluated configurations, exceeding BaseLoop by 1.64 and 6.89 percentage points, respectively. Its Knowledge score remains close to its training-depth value (58.90 at $D = 2 8$ vs. 59.18 at $D = 8 4 )$ , although its Reasoning score remains below BaseLoop (29.01 vs. 29.34). For Channel-wise injection, $w = 2$ gives the highest Overall score among the history variants at all three reported depths, but does not outperform initial-state injection at $D \bar { = } 8 4$ (36.14 vs. 36.20).

Dense history injection exhibits a different window preference. With $w = 1 ,$ it achieves the highest Overall score at D = 56 (37.36) and the highest Reasoning score at $D = 8 4 ( 3 0 . 7 7 )$ while its Knowledge score decreases from 59.72 to 54.16. Larger windows substantially weaken depth extrapolation: Dense w = 4 has the highest Overall score at the training depth (37.09), but falls to 30.38 at $D = 8 4$ . These results show that training-depth performance does not reliably predict extrapolation performance, and that improvements in Overall can reflect different Knowledge– Reasoning trade-offs.

## D.5 TIMESTEP CONDITIONING

Tabs 11 and 12 show that the benefits of timestep conditioning depend on the recurrent configuration and task category. On $\mathrm { Q w e n } 3 { \cdot } 0 . 6 \mathrm { B }$ , Branch Gating provides the largest Overall gain at twice the training depth for $7 \times 4 ( 3 8 . 4 6 \mathrm { v s . } 3 6 . 5 9$ for BaseLoop), driven by improved Reasoning, whereas AdaLN achieves the highest Knowledge score (61.72). However, no conditioning variant improves Overall for $2 \times$ 14 at either reported depth, and the training-depth gains of both scalar gating schemes for $1 4 \times 2$ disappear at $D = 5 6$ . On Llama3.1-1B, all three variants improve Overall and Knowledge at $D = 4 0$ , with AdaLN performing best on both metrics, but none matches BaseLoop in Reasoning. Thus, timestep conditioning can improve performance beyond the training depth, but no variant consistently dominates across configurations and metrics.

![](images/c50a50d0a12663c2126895b8d7613ebd6771027719a825e9c463a4b3facbce0f.jpg)  
BaseLoop Scalar Channel-wise Residual Channel-wise Dense NonLoop 20x1

Figure 13: Initial-state input injection on Llama3.1-1B. Columns correspond to four shared-stack configurations; rows report Overall, Knowledge, and Reasoning scores. Curves include all available evaluation depths. Vertical dashed lines mark the training depth $D = 2 0$ , and shading denotes extrapolation. BaseLoop uses no injection; NonLoop $2 0 \times 1$ is the unshared reference.
<table><tr><td rowspan="2">Configuration</td><td rowspan="2">Method</td><td colspan="3">D = 20</td><td colspan="3">D = 40</td></tr><tr><td>Overall</td><td>Knowledge</td><td>Reasoning</td><td>Overall</td><td>Knowledge</td><td>Reasoning</td></tr><tr><td>2 × 10</td><td>BaseLoop</td><td>36.83</td><td>61.10</td><td>27.73</td><td>34.58</td><td>50.37</td><td>28.66</td></tr><tr><td rowspan="4"></td><td>Scalar</td><td>37.26</td><td>62.33</td><td>27.85</td><td>34.90</td><td>53.64</td><td>27.87</td></tr><tr><td>Channel-wise</td><td>37.54</td><td>60.98</td><td>28.75</td><td>35.01</td><td>53.41</td><td>28.10</td></tr><tr><td>Residual Channel-wise</td><td>37.02</td><td>60.46</td><td>28.23</td><td>32.52</td><td>41.39</td><td>29.19</td></tr><tr><td>Dense</td><td>36.66</td><td>59.23</td><td>28.19</td><td>29.96</td><td>32.78</td><td>28.90</td></tr><tr><td rowspan="5">4× 5</td><td>BaseLoop</td><td>37.87</td><td>62.80</td><td>28.52</td><td>37.11</td><td>52.11</td><td>31.49</td></tr><tr><td>Scalar</td><td>38.34</td><td>63.13</td><td>29.05</td><td>35.70</td><td>54.90</td><td>28.50</td></tr><tr><td>Channel-wise</td><td>38.21</td><td>62.53</td><td>29.09</td><td>36.15</td><td>56.39</td><td>28.56</td></tr><tr><td>Residual Channel-wise</td><td>38.42</td><td>63.46</td><td>29.03</td><td>36.77</td><td>53.01</td><td>30.68</td></tr><tr><td>Dense</td><td>38.36</td><td>63.43</td><td>28.96</td><td>31.16</td><td>37.51</td><td>28.78</td></tr><tr><td rowspan="5">5× 4</td><td>BaseLoop</td><td>39.32</td><td>64.79</td><td>29.77</td><td>37.26</td><td>53.64</td><td>31.12</td></tr><tr><td>Scalar</td><td>38.36</td><td>63.06</td><td>29.10</td><td>37.67</td><td>55.03</td><td>31.16</td></tr><tr><td>Channel-wise</td><td>39.34</td><td>63.88</td><td>30.14</td><td>37.35</td><td>53.99</td><td>31.10</td></tr><tr><td>Residual Channel-wise</td><td>38.67</td><td>64.06</td><td>29.15</td><td>35.26</td><td>46.96</td><td>30.88</td></tr><tr><td>Dense</td><td>37.96</td><td>62.71</td><td>28.67</td><td>32.79</td><td>41.12</td><td>29.67</td></tr><tr><td rowspan="5">10 × 2</td><td>BaseLoop</td><td>39.82</td><td>65.21</td><td>30.30</td><td>37.80</td><td>57.73</td><td>30.33</td></tr><tr><td>Scalar</td><td>39.33</td><td>64.92</td><td>29.73</td><td>36.60</td><td>55.18</td><td>29.64</td></tr><tr><td>Channel-wise</td><td>39.42</td><td>65.48</td><td>29.64</td><td>36.97</td><td>56.68</td><td>29.58</td></tr><tr><td>Residual Channel-wise</td><td>40.00</td><td>64.94</td><td>30.64</td><td>37.38</td><td>53.48</td><td>31.34</td></tr><tr><td>Dense</td><td>39.17</td><td>64.69</td><td>29.59</td><td>34.14</td><td>47.08</td><td>29.29</td></tr><tr><td>20 × 1</td><td>NonLoop</td><td>39.83</td><td>65.38</td><td>30.25</td><td></td><td>一</td><td></td></tr></table>

Table 8: Full initial-state injection results on Llama3.1-1B at the training depth $D = 2 0$ and twice that depth $D = 4 0$ . Configuration denotes the shared-stack depth and training loop count. Bold marks the best result within each recurrent configuration for each metric and evaluation depth. NonLoop is evaluated only at its native depth.

## D.6 COMBINATION EXPERIMENTS

Initial-state and history-state. Full results of Initial-state injection plus history-state injection are provided in Figs. 14, 15, and 16. At the training depth of 28, most combined runs have overall scores near 37. Beyond this depth, the scalar and channel-wise combinations remain substantially more stable than the dense combinations. For scalar injection with $w = 4 ,$ , combined knowledge reaches 57.92 at depth 84, compared with 52.10 for input injection alone, although history injection alone reaches 59.18. The clearest gain over both components occurs for channel-wise injection with $w = 2 \mathrm { : }$ at depth 56, its overall score is 37.74, vs. 36.67 and 36.77 for initial-state and history-state injection alone. Dense combinations instead lose substantial overall and knowledge accuracy beyond depth 28. For $w = 1$ at depth 84, combined knowledge falls to 32.77, while history injection alone retains 54.16.

D = 28

<table><tr><td rowspan="2">Configuration</td><td rowspan="2">Method</td><td colspan="3">D = 28</td><td colspan="3">D = 56</td></tr><tr><td>Overall</td><td>Knowledge</td><td>Reasoning</td><td>Overall</td><td>Knowledge</td><td>Reasoning</td></tr><tr><td> $2 \times 1 4$ </td><td>BaseLoop</td><td>36.85</td><td>58.70</td><td>28.66</td><td>37.31</td><td>57.30</td><td>29.81</td></tr><tr><td rowspan="5"></td><td>Scalar</td><td>36.27</td><td>58.00</td><td>28.12</td><td>35.71</td><td>56.89</td><td>27.77</td></tr><tr><td>Channel-wise</td><td>35.87</td><td>58.01</td><td>27.57</td><td>35.59</td><td>56.56</td><td>27.73</td></tr><tr><td>Residual Channel-wise</td><td>36.77</td><td>57.88</td><td>28.85</td><td>32.73</td><td>42.49</td><td>29.07</td></tr><tr><td>Dense</td><td>35.18</td><td>55.95</td><td>27.39</td><td>30.30</td><td>33.15</td><td>29.24</td></tr><tr><td>BaseLoop</td><td>36.80</td><td>59.44</td><td>28.31</td><td>36.59</td><td>57.33</td><td>28.81</td></tr><tr><td rowspan="5"></td><td>Scalar</td><td>36.16</td><td>58.35</td><td>27.84</td><td>36.34</td><td>56.05</td><td>28.95</td></tr><tr><td>Channel-wise</td><td>36.33</td><td>59.39</td><td>27.68</td><td>36.67</td><td>57.15</td><td>29.00</td></tr><tr><td>Residual Channel-wise</td><td>36.96</td><td>58.74</td><td>28.79</td><td>36.09</td><td>51.39</td><td>30.35</td></tr><tr><td>Dense</td><td>36.85</td><td>58.77</td><td>28.63</td><td>31.42</td><td>36.35</td><td>29.57</td></tr><tr><td>BaseLoop</td><td>37.29</td><td>61.26</td><td>28.31</td><td>36.59</td><td>60.33</td><td>27.68</td></tr><tr><td rowspan="5">7×4</td><td>Scalar</td><td>36.98</td><td>61.90</td><td>27.64</td><td>36.64</td><td>59.55</td><td>28.04</td></tr><tr><td>Channel-wise</td><td>37.94</td><td>61.46</td><td>29.12</td><td>37.05</td><td>59.05</td><td>28.81</td></tr><tr><td>Residual Channel-wise</td><td>37.10</td><td>61.22</td><td>28.05</td><td>36.19</td><td>58.30</td><td>27.89</td></tr><tr><td>Dense</td><td>37.04</td><td>60.74</td><td>28.15</td><td>34.19</td><td>46.49</td><td>29.58</td></tr><tr><td>BaseLoop</td><td>37.99</td><td>61.93</td><td>29.02</td><td>37.87</td><td>59.95</td><td>29.59</td></tr><tr><td rowspan="5">14 × 2</td><td>Scalar</td><td>38.04</td><td>61.81</td><td>29.12</td><td>37.23</td><td>58.41</td><td>29.29</td></tr><tr><td>Channel-wise</td><td>38.05</td><td>62.17</td><td>29.00</td><td>37.43</td><td>59.24</td><td>29.24</td></tr><tr><td>Residual Channel-wise</td><td>38.27</td><td>63.63</td><td>28.76</td><td>37.37</td><td>60.11</td><td>28.83</td></tr><tr><td>Dense</td><td>37.75</td><td>62.52</td><td>28.46</td><td>35.66</td><td>51.78</td><td>29.61</td></tr><tr><td></td><td></td><td>64.89</td><td>30.39</td><td></td><td></td><td></td></tr><tr><td>28 × 1</td><td>NonLoop</td><td>39.80</td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 9: Full initial-state injection results on Qwen3-0.6B at the training depth $D = 2 8$ and twice that depth $D = 5 6$

<table><tr><td colspan="2"></td><td colspan="3"> $D = 2 8$ </td><td colspan="3"> ${ \bf { \mathit { D } } } = { \bf { 5 6 } }$ </td><td colspan="3"> $D = 8 4$ </td></tr><tr><td>Parameterization</td><td>Injection</td><td>Overall</td><td>Knowledge</td><td>Reasoning</td><td>Overall</td><td>Knowledge</td><td>Reasoning</td><td>Overall</td><td>Knowledge</td><td>Reasoning</td></tr><tr><td></td><td>BaseLoop</td><td>36.80</td><td>59.44</td><td>28.31</td><td>36.59</td><td>57.33</td><td>28.81</td><td>35.60</td><td>52.29</td><td>29.34</td></tr><tr><td rowspan="4">Scalar</td><td>Initial-state</td><td>36.16</td><td>58.35</td><td>27.84</td><td>36.34</td><td>56.05</td><td>28.95</td><td>35.69</td><td>52.10</td><td>29.53</td></tr><tr><td>History  $( w = 1 )$ </td><td>36.45</td><td>59.41</td><td>27.84</td><td>35.74</td><td>57.67</td><td>27.51</td><td>34.69</td><td>52.23</td><td>28.12</td></tr><tr><td>History  $( w = 2 )$ </td><td>36.46</td><td>58.83</td><td>28.07</td><td>36.78</td><td>59.18</td><td>28.38</td><td>35.75</td><td>55.83</td><td>28.22</td></tr><tr><td>History (w = 4)</td><td>36.61</td><td>58.90</td><td>28.25</td><td>36.80</td><td>59.41</td><td>28.32</td><td>37.24</td><td>59.18</td><td>29.01</td></tr><tr><td rowspan="4">Channel-wise</td><td>Initial-state</td><td>36.33</td><td>59.39</td><td>27.68</td><td>36.67</td><td>57.15</td><td>29.00</td><td>36.20</td><td>52.16</td><td>30.22</td></tr><tr><td>History (w = 1)</td><td>36.44</td><td>59.57</td><td>27.77</td><td>36.30</td><td>57.98</td><td>28.18</td><td>35.27</td><td>53.57</td><td>28.41</td></tr><tr><td>History (w = 2)</td><td>36.64</td><td>59.47</td><td>28.08</td><td>36.77</td><td>58.90</td><td>28.48</td><td>36.14</td><td>53.86</td><td>29.49</td></tr><tr><td>History (w = 4)</td><td>35.99</td><td>58.54</td><td>27.53</td><td>35.94</td><td>57.35</td><td>27.92</td><td>34.79</td><td>53.20</td><td>27.88</td></tr><tr><td rowspan="4">Dense</td><td>Initial-state</td><td>36.85</td><td>58.77</td><td>28.63</td><td>31.42</td><td>36.35</td><td>29.57</td><td>29.88</td><td>32.28</td><td>28.98</td></tr><tr><td>History (w = 1)</td><td>36.92</td><td>59.72</td><td>28.38</td><td>37.36</td><td>59.22</td><td>29.17</td><td>37.15</td><td>54.16</td><td>30.77</td></tr><tr><td>History (w = 2)</td><td>35.96</td><td>58.84</td><td>27.38</td><td>35.99</td><td>51.96</td><td>30.00</td><td>31.90</td><td>38.99</td><td>29.24</td></tr><tr><td>History (w = 4)</td><td>37.09</td><td>58.17</td><td>29.18</td><td>30.54</td><td>34.23</td><td>29.16</td><td>30.38</td><td>32.73</td><td>29.50</td></tr></table>

Table 10: Full history-state input injection results on $\mathrm { Q w e n } 3 { \cdot } 0 . 6 \mathrm { B }$ with the $4 \times 7$ shared-stack configuration. D = 28 is the training depth; $\bar { D } = 5 6$ and D = 84 correspond to 2× and 3× depth extrapolation. Initial-state and History denote initial-state-only and history-state-only injection, respectively; w denotes the history window size. BaseLoop uses no injection. All scores are percentages, with higher values indicating better performance. Bold marks the best result in each column across all listed configurations.

Initial-state and timestep. Fig. 17 compares initial-state injection combined with timestep conditioning against the individual components. Combining the two provides almost no consistent benefit under either BG or LG. At the training depth $D = 2 8$ , the combined Overall scores are 35.89 with BG and 35.99 with LG, below the corresponding gating-only scores of 37.16 and 36.68, respectively. At depth 56, LG exceeds initial-state injection by only 0.07 percentage points, while BG remains below both components. This pattern persists at depth 84: LG gains just 0.13 points over initial-state injection, whereas BG trails it by 0.52 points. Knowledge and reasoning likewise show no reliable joint gain. Thus, timestep conditioning does not provide a meaningful complementary improvement when added to initial-state injection in these runs.

History-state and timestep. Figs 18, 19, and 20 compare history-state injection with timestep conditioning. It is clear that several history-state combinations show clear gains at greater effective depths. At depth 84, channel-wise history with $w = 1$ and BG scores 36.96 overall, compared with 35.68 for initial-state plus BG. With $w = 2$ and LG, the corresponding scores are 37.30 and 36.33; the history-state combination also improves knowledge from 52.92 to 56.32. A channel-wise history signal is already effective on its own: at depth 84, history-only $w = 2$ scores 36.14 overall, versus 31.90 for dense history-only $w = 2 .$ Adding LG raises the channel-wise result to 37.30, while dense history with the same window and gate scores 31.68. Thus, a dense history transformation is unnecessary for the strongest results here. Channel-wise history uses wd injection weights, compared with $w d ^ { 2 }$ for dense history; at $d = 1 0 2 4$ and $w = 2 .$ , this is 2,048 vs. 2,097,152 history-injection weights.

D = 20  
D = 40
<table><tr><td>Configuration</td><td>Method</td><td>Overall</td><td>Knowledge</td><td>Reasoning</td><td>Overall</td><td>Knowledge</td><td>Reasoning</td></tr><tr><td> $4 \times 5$ </td><td>BaseLoop</td><td>37.87</td><td>62.80</td><td>28.52</td><td>37.11</td><td>52.11</td><td>31.49</td></tr><tr><td></td><td>Loop Gating</td><td>37.99</td><td>63.32</td><td>28.49</td><td>37.43</td><td>56.17</td><td>30.41</td></tr><tr><td></td><td>Branch Gating</td><td>37.99</td><td>62.39</td><td>28.84</td><td>37.23</td><td>56.60</td><td>29.97</td></tr><tr><td></td><td>AdaLN</td><td>38.58</td><td>62.65</td><td>29.56</td><td>37.91</td><td>59.68</td><td>29.75</td></tr><tr><td>20 × 1</td><td>NonLoop</td><td>39.83</td><td>65.38</td><td>30.25</td><td>一</td><td>一</td><td>一</td></tr></table>

Table 11: Timestep conditioning results on Llama3.1-1B at the training depth $D = 2 0$ and twice that depth $D = 4 0$ . All timestep variants use rescaled time grids at inference. Configuration denotes the shared-stack depth and training loop count. Bold marks the best result within the recurrent configuration for each metric and evaluation depth. NonLoop is evaluated only at its native depth.

D = 28  
D = 56
<table><tr><td>Configuration</td><td>Method</td><td>Overall</td><td>Knowledge</td><td>Reasoning</td><td>Overall</td><td>Knowledge</td><td>Reasoning</td></tr><tr><td> $2 \times 1 4$ </td><td>BaseLoop</td><td>36.85</td><td>58.70</td><td>28.66</td><td>37.31</td><td>57.30</td><td>29.81</td></tr><tr><td rowspan="4"></td><td>Loop Gating</td><td>36.16</td><td>57.51</td><td>28.16</td><td>35.18</td><td>57.07</td><td>26.97</td></tr><tr><td>Branch Gating</td><td>36.62</td><td>58.20</td><td>28.53</td><td>36.64</td><td>58.01</td><td>28.63</td></tr><tr><td>AdaLN</td><td>35.97</td><td>58.44</td><td>27.54</td><td>35.35</td><td>56.54</td><td>27.41</td></tr><tr><td>BaseLoop</td><td>36.80</td><td>59.44</td><td>28.31</td><td>36.59</td><td>57.33</td><td>28.81</td></tr><tr><td rowspan="4"></td><td>Loop Gating</td><td>36.68</td><td>59.66</td><td>28.06</td><td>36.43</td><td>57.71</td><td>28.46</td></tr><tr><td>Branch Gating</td><td>37.16</td><td>59.98</td><td>28.61</td><td>36.62</td><td>58.60</td><td>28.38</td></tr><tr><td>AdaLN</td><td>36.18</td><td>58.48</td><td>27.81</td><td>36.83</td><td>58.59</td><td>28.67</td></tr><tr><td>BaseLoop</td><td>37.29</td><td>61.26</td><td>28.31</td><td>36.59</td><td>60.33</td><td>27.68</td></tr><tr><td rowspan="4"></td><td>Loop Gating</td><td>37.12</td><td>60.85</td><td>28.23</td><td>36.91</td><td>59.24</td><td>28.54</td></tr><tr><td>Branch Gating</td><td>37.27</td><td>60.73</td><td>28.48</td><td>38.46</td><td>58.67</td><td>30.88</td></tr><tr><td>AdaLN</td><td>37.73</td><td>61.50</td><td>28.81</td><td>38.01</td><td>61.72</td><td>29.12</td></tr><tr><td>BaseLoop</td><td>37.99</td><td>61.93</td><td>29.02</td><td>37.87</td><td>59.95</td><td>29.59</td></tr><tr><td rowspan="4">14 × 2</td><td>Loop Gating</td><td>38.31</td><td>63.27</td><td>28.95</td><td>37.58</td><td>59.85</td><td>29.22</td></tr><tr><td>Branch Gating</td><td>38.44</td><td>62.52</td><td>29.41</td><td>37.38</td><td>59.47</td><td>29.10</td></tr><tr><td>AdaLN</td><td>38.37</td><td>63.25</td><td>29.04</td><td>38.05</td><td>61.24</td><td>29.36</td></tr><tr><td>NonLoop</td><td>39.80</td><td>64.89</td><td>30.39</td><td>一</td><td>一</td><td>一</td></tr></table>

Table 12: Timestep conditioning results on Qwen3-0.6B at the training depth $D = 2 8$ and twice that depth $D = 5 6$ . All timestep variants use rescaled time grids at inference. Configuration denotes the shared-stack depth and training loop count. Bold marks the best result within each recurrent configuration for each metric and evaluation depth. NonLoop is evaluated only at its native depth.

Initial-state, history-state, and timestep. Finally, Fig. 21 evaluates all three mechanisms together. Within the evaluated range, adding a third mechanism does not produce a consistent additive gain. The channel-wise initial-state and history-state combination with LG reaches an Overall score of 37.02 at the training depth $D = 2 8$ , exceeding the best individual component, LG (36.68), by 0.34 percentage points. However, this advantage does not persist at greater depths. $\mathbf { A } \mathbf { t } \ D = 5 6$ , the four three-mechanism configurations achieve Overall scores of $3 5 . 6 9 \mathrm { - } \bar { 3 } 6 . 6 0 ;$ none exceeds its best matched individual component. Their scores are also below the reported two-mechanism results of 37.74 for initial-state plus history-state injection with $w = 2 .$ , and 37.49 for channel-wise history-state injection plus LG with $w = 2 .$ Knowledge and reasoning show no consistent compensating gain. Among the evaluated configurations, combining all three mechanisms does not consistently outperform the two-mechanism alternatives.

![](images/25b6f25c6be5ad3ce1f970bef42f04ec6361d1135970ad59ef438928d5271ebc.jpg)  
Figure 14: Results of Scalar initial plus history injection combination and their individual components across history windows $w = 1 , 2 ,$ 4 with the Qwe $\mathrm { n } 3 \mathrm { - } 0 . 6 \mathrm { B } \ 4 \times 7$ configuration.

![](images/3c2c8adce698f823672c81cfc1154a3e7bc6d086370c240564ced74dc6cb8898.jpg)  
Figure 15: Results of Channel-wise initial plus history injection combination and their individual components across history windows $w = 1 , 2 ,$ 4 with the $\mathrm { \Delta Q w e n 3 - 0 . 6 B ~ 4 ~ } \times 7$ configuration.

(%)  
![](images/e34d409d61ba4453ab947224d130f7cba06f72d4205ae6566594f8ea5d2cd7ed.jpg)  
Figure 16: Results of Dense initial plus history injection combination and their individual components across history windows $w = 1 , 2 ,$ 4 with the Qwe $\mathrm { n } 3 \mathrm { - } 0 . 6 \mathrm { B } \ 4 \times 7$ configuration.

(%)  
![](images/e439c837f89ca03068e3ff92defef9b66e75cf14a39bc9d33cf623f849f96e14.jpg)  
Figure 17: Results of Dense initial plus timestep combination and their individual components with the ${ \mathrm { Q w e n 3 - 0 . 6 B ~ 4 ~ } } \times 7$ configuration.

## E FURTHER ANALYSIS

## E.1 INPUT NOISE INITIALIZATION ABLATION

We study the effect of initialization noise in initial-state injection using Qwen3-0.6B with a BaseLoop $4 \times 7$ configuration, comprising four shared layers trained for seven loops. We compare initialization without noise against a noise scale of 0.03 for Scalar, Channel-wise, and Residual Channel-wise input injection, evaluating overall, knowledge, and reasoning performance across inference loops. As shown in Fig. 22, adding noise has only a minor effect on downstream performance and both initialization settings exhibit similar trends across evaluation depths during an extended inference of

![](images/2257d9c7ea74a9b7983fa8826d4556b32b86fded7b78454d0a34eeaa6a54a899.jpg)  
Figure 18: Results of Dense history injection plus timestep combination and their individual components for history windows $w = 1 , 2$ with the Qwe $1 3  – 0 . 6 \dot { \mathrm { B } } \ : 4 \times 7$ configuration.

![](images/483085a15719bc87cf604984ebb802b49c35ac748af9c6407eba6a43527d9495.jpg)  
Figure 19: Results of Channel-wise history injection combined with BG and their individual components across history windows $w = 1 , 2 ,$ 4 with the Qwen ${ } 1 3 \mathrm { - } 0 . 6 \mathrm { B } \mathrm { ~ 4 ~ } \mathrm { \times ~ } 7$ configuration.

BaseLoop H+LG H LG NonLoop 28x1

![](images/fa9b0d418b01e16e8229fe4e6f663c007eefafe868830b7d16100461518266f0.jpg)  
Figure 20: Results of Channel-wise history injection combined with LG and their individual components across history windows $w = 1 , 2 ,$ 4 with the ${ \mathrm { Q w e n 3 - 0 . 6 B ~ 4 ~ } } \times 7$ configuration.

![](images/afca16b0a57ea3e76548d4f21a95e8b4103fb17f9196c3bbc05174a83c922a7c.jpg)  
Figure 21: Results of three-mechanism combination and their individual components across history windows $w = 1 , 2$ with the ${ \mathrm { Q w e n 3 - 0 . 6 B ~ 4 ~ } } \times 7$ configuration.

84 effective layers. The small differences do not indicate a consistent advantage from noise across variants and metrics. We therefore adopt initialization without noise as the default choice.

![](images/8c5015f7daa0ccbe99e3a91f02e1eb5a30d4575cd42faa782db914fcc49b52a8.jpg)

![](images/e7871dfff7f0588344cbbd24689d26d989c3f037053c7eebdc69526f32922a8e.jpg)

(c) Residual Channel-wise  
![](images/792613111260515c2492771122d06d33e6bf7712b11f2012705ccb771e82aa48.jpg)  
No init noise Init noise = 0.03 NonLoop 28x1

Figure 22: Average results for Qwen3-0.6B BaseLoop $4 \times 7$ with Scalar, Channel-wise, and Residual Channel-wise initial-state input injection under an extended effective layer of 84 (3x training budget). The vertical boundary marks the training depth of 28 layers, with shading indicating depth extrapolation. Initialization noise has a limited effect on performance, supporting our default choice of initialization without noise.  
![](images/1cba3625e84086321b5148344daeecc8d2cb8f96789ffbdda9caff1d583c64aa.jpg)

![](images/4aa00176b068d2bea715ede69c0084345f31e80bd03f3cfe9295689c75247648.jpg)

![](images/e58aeac0e9ed9fb001453c0f27c01daa8531642ce7d155906445bff7e8268f3d.jpg)

![](images/c09309c58c2b0ab650d91e6980f6310a0e1e1eae612a59efb71b8e2c28e80055.jpg)  
Figure 23: Additional computational interaction results for Qwen3-0.6B BaseLoop $4 \times 7 ,$ using the probes defined in App. C.4. All state conditioning is channel-wise, with history window $w = 2 .$ . (a) Mean unnormalized response $N ( d )$ for pairwise conditioning, showing all measured lags. (b) Ratios of mean C for H+LG versus I+H and I+ ${ \mathcal { G } } ,$ separately for each of the 16 shared source/target block pairs over lags 1-6. (c) Triple/double ratios of mean $C ,$ mean $\dot { N }$ , and mean $B$ over lags 2-6, with the gate type and history settings matched. (d) Triple/double ratios of mean C for each shared-block pair over lags 2-6. All means weight occurrence pairs equally on the indicated lag support; $B _ { j }$ is repeated for each included source paired with target j. Configuration ratios divide these means, and mean $C$ is the mean of $N _ { i \to j } / B _ { j }$ , not the ratio of mean N to mean B. In (b,d), s:q denotes one-based source and target block indices; vertical separators distinguish source blocks. Lines between block pairs or metrics are visual guides. Dashed curves with open circles denote $D = 5 6 ;$ solid curves with filled triangles denote $D = 8 4$ . Horizontal references at one indicate equal values.

## E.2 GEOMETRY ANALYSIS

Complementing the geometry analysis in Fig. 8(c,d), Fig. 23 examines unnormalized update responses and variation across shared-block pairings. At D = 84, H+LG has 6.4% and 10.2% higher mean raw response over lags 1-6 than I+H and I+LG, respectively, while its mean relative response is higher in all 16 shared-block pairings for both comparisons (a,b). Thus, the cross-loop interaction advantage identified in the main text also appears in the response numerator and extends across shared-block pairings. Adding initial-state conditioning to H+LG reduces the mean raw response over lags 2-6 by 10.5% at $D = 5 6$ and $6 . 0 \%$ at $D = 8 4$ , supporting the main text’s observation that the lower relative response is not solely a normalization effect (c). This reduction is heterogeneous: at $D = 8 4$ , the LG triple has lower mean C in $9 / 1 6$ shared-block pairings, while the remaining pairings increase; the BG triple’s aggregate responses remain close to those of H+BG $( \mathbf { c } , \mathbf { d } )$ . These results further support the interpretation of history-state and timestep complementarity through cross-loop interaction, and provide a possible explanation for the lack of additive gains from initial-state conditioning, without implying a uniform reduction across gates or block pairings.