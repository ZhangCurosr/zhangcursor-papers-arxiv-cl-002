# Toward a Graded Measure of Belief Stability in Large Language Models

Samantha Dies<sup>1♯</sup>, Branden Fitelson<sup>1</sup>, and Tina Eliassi-Rad<sup>1,2</sup>

<sup>1</sup>Northeastern University, 360 Huntington Ave, Boston, MA 02115 USA <sup>2</sup>Santa Fe Institute, 1399 Hyde Park Road, Santa Fe, NM 87501 USA <sup>♯</sup>dies.s@northeastern.edu

## ABSTRACT

Large language models (LLMs) increasingly mediate how people access and reason with information, yet factual reliability is usually evaluated one judgment at a time. We introduce graded belief stability, a relational measure of how well a belief persists within an LLM’s broader belief system. Unlike individual belief probability, it asks whether support for a claim persists when that claim is considered alongside the model’s other epistemic commitments. We operationalize this idea with a Direct Conditional estimator that uses internal model representations to estimate conditional belief probabilities. Across 12 LLMs and three domains, lower-stability beliefs exhibit greater mean behavioral movement under conversational challenge in 83.3% of model–domain settings after matching on individual belief probability. Graded belief stability therefore extends reliability assessment beyond how strongly an LLM supports a claim to how robustly that belief is supported within its broader system of beliefs.

## 1 Introduction

Large language models (LLMs) increasingly impact how people access and reason with information, making the reliability of their factual judgments central to their use [1, 2]. Existing evaluations characterize several dimensions of this reliability, including accuracy, uncertainty, and calibration [3, 4, 5, 6]. In interactive settings, however, factual claims are rarely considered in isolation. New claims, corrections, and contextual information accumulate over a conversation, and models may revise answers they previously supported. Reliability therefore also depends on a diferent question: which of a model’s beliefs persist as the informational context around them changes?

Such belief revisions are well documented. LLM factual judgments can vary with paraphrasing and changes in semantic context [7, 8, 9], as well as under user disagreement, contradictory feedback, and multi-turn interaction [10, 11, 12, 13, 14]. Work on systematic belief consistency and knowledge editing likewise suggests that factual commitments do not behave as isolated entries in a knowledge base: accepting or revising one proposition can afect related propositions [15, 16]. Consider an LLM that assigns similar support to “Paris is in France” and “Oslo is in Norway.” If it also considers “Oslo is in Sweden” plausible, the Oslo claim sits alongside an incompatible proposition the model has not ruled out. The Paris and Oslo claims may therefore look equally strong on their own even though one is more fragile within the model’s broader belief system. We refer to this diference as belief stability.

Existing work captures important pieces of this picture without directly measuring it. Representation-based studies find that veracity and epistemic uncertainty are systematically reflected in LLM hidden states [17, 18, 19, 20, 21], while LLMs do not reliably distinguish epistemic concepts such as belief, knowledge, and fact [22]. Behavioral evaluations, meanwhile, reveal when outputs change under particular external perturbations [7, 10, 11, 12]. What these approaches do not capture is how a belief is situated relative to the model’s other epistemic commitments. A belief may receive strong support on its own yet become fragile in light of other propositions the model regards as plausible. Stability is therefore a relational property of a belief within a broader belief system.

Formal epistemology provides a principled basis for making this distinction precise. The Lockean Thesis connects graded credence to categorical belief through a probability threshold, with suficiently probable propositions counting as beliefs [23, 24]. Leitgeb’s Humean Thesis strengthens this threshold-based approach by requiring a belief to remain suficiently probable when conditioned on propositions an agent does not disbelieve [25]. We adapt this idea to LLMs by defining the graded belief stability of a believed proposition P as the fraction of the model’s non-disbeliefs x for which Pr(P | x) remains above its belief threshold. Graded stability thus distinguishes how strongly a model supports P on its own from how broadly that support persists across its broader epistemic state.

Estimating graded stability requires eliciting conditional belief probabilities from LLM representations. Supervised probes can recover veracity-related information from hidden states [17, 18, 20, 26], but Pr(P | x) is not directly observable. We therefore introduce a trivalent Direct Conditional estimator grounded in established accounts of indicative conditionals and non-bivalent probability [27, 28]. For each belief–non-disbelief pair $( P , x )$ , we construct a conditional statement of the form “Given $x , P ^ { \ast }$ and probe its hidden representation to estimate the conditional probabilities needed to compute graded stability.

We evaluate graded belief stability across 12 instruction-tuned LLMs from four model families, ranging from 3B to 72B parameters, and across three domains: City Locations, Medical Indications, and Word Definitions. We test whether graded stability captures structure beyond individual belief probability and whether the underlying probability estimates are approximately probabilistically coherent. We then characterize its variation across semantic domains and evaluate whether it is associated with resistance to conversational challenge.

Our work makes five core contributions:

1. Formalization: We introduce graded belief stability as a statement-level measure of how well an LLM’s belief persists across its broader belief system.

2. Measurement: We develop a representation-based Direct Conditional estimator of conditional belief probability for computing graded stability.

3. Validation: We show that graded stability captures systematic information beyond individual belief probability, where 99.0% of model-pair residual correlations are positive. The measured probability systems are also approximately coherent, with median distance from coherence ranging from 0.015 to 0.062 across domains.

4. Characterization: We find that City Locations is consistently the most stable domain, with a median across models of mean graded stability of 0.88, compared with 0.71 for Medical Indications and 0.74 for Word Definitions.

5. Behavioral validity: Among beliefs matched on individual belief probability, lower-stability beliefs show greater mean movement under repeated conversational challenge for 83.3% of model–domain settings.

This work captures a dimension of LLM reliability that individual belief probability alone cannot. Two factual judgments can receive similar support while occupying very diferent positions within the model’s broader belief system, making one substantially more fragile than the other. Graded belief stability quantifies this relational property, extending reliability assessment beyond how strongly an LLM supports a claim to how robustly that belief is supported within the model’s broader system of beliefs.

## 2 Results

We begin by formalizing graded belief stability γ and introducing the Direct Conditional representation-based estimator for conditional belief. We then validate the measure, showing that it captures structure beyond individual belief probability and is approximately probabilistically coherent. We next characterize variation in γ across domains, before testing whether graded stability distinguishes behavioral resilience among beliefs matched on individual belief probability.

## 2.1 Formalizing graded belief stability

We define graded belief stability by adapting the stability theory of belief from formal epistemology, in which a belief is stable when it persists after conditioning on propositions the agent does not reject [24]. We extend this binary notion to a statement-level measure of the degree to which an LLM’s belief persists across conditioning contexts.

Let M denote an LLM and let $\mathrm { P r } _ { \mathcal { M } } ( s )$ denote the probability that M assigns to a statement s being true. Following the Lockean Thesis, we take categorical belief to correspond to suficiently high probability relative to a belief threshold $t _ { \mathcal { M } } ~ [ 2 3 , 2 4 ]$ . In our empirical implementation below, we first identify categorical belief from a trivalent probe and then instantiate $t _ { \mathcal { M } }$ from the resulting belief set rather than imposing a universal cutof. We denote the set of beliefs held by $\mathcal { M }$ as $p _ { { \mathcal { M } } }$ and the larger set of propositions that M does not disbelieve, including both beliefs and suspended beliefs, as $\mathcal { X } _ { \mathcal { M } }$

The Humean Thesis strengthens this static notion of belief by requiring beliefs to remain suficiently probable when considered alongside the agent’s other non-disbeliefs [24, 25]. Intuitively, a stable belief should remain above the belief threshold even when considered alongside propositions the model itself does not reject. Specifically, a belief $P \in B _ { \mathcal { M } }$ is considered stable if conditioning on any $x \in \mathcal { X } _ { \mathcal { M } } \backslash \{ P \}$ does not lower its probability below the belief threshold:

$$
P { \mathrm { ~ i s ~ s t a b l e } } \iff \operatorname* { P r } _ { \mathcal { M } } ( P \mid x ) > t _ { \mathcal { M } } \quad \forall x \in \mathcal { X } _ { \mathcal { M } } \setminus \{ P \} .\tag{1}
$$

Thus, propositions with similar $\mathrm { P r } _ { M } ( P )$ may nevertheless difer substantially in stability.

![](images/a330719be67e5bca180a3da3174436a0479027d0b9acd2e0810a85e39b71fa38.jpg)

(b) From Probabilities to Belief Sets

![](images/5ab0a06a4d7e7cafb68fa6b2e383e7b0b82ef0fad796bf3946f2618914fc0ef1.jpg)

![](images/2f8daf74d3d78efcab13f8b6d33076c8b92f8bb4d5ae501a13ee0ed03f21ec89.jpg)

![](images/466b8dad99968070ceca82d926884b408e132c2ead8d0c1b8f1ed4000a7f9f60.jpg)  
Figure 1. Measuring graded belief stability in LLMs. (a) For each statement s, we extract an internal LLM representation and use a probe to estimate a distribution over True (T, orange), False (F, olive), and Neither (N, purple). (b) The probe’s predicted states yˆ determine whether each statement is treated as a belief $( \hat { y } = T )$ disbelief $( \hat { y } = F )$ , or suspended belief $( { \hat { y } } = N )$ , yielding the belief set $_ { B _ { \mathcal { M } } }$ and non-disbelief set $\mathcal { X } _ { \mathcal { M } }$ . We define the empirical belief threshold $t _ { \mathcal { M } }$ as the minimum True-class probability among statements classified as beliefs. We then estimate $\operatorname* { P r } _ { \mathcal { M } } ( \boldsymbol { P } \mid \boldsymbol { x } )$ for each $P \in B _ { \mathcal { M } }$ and $x \in { \mathcal { X } } _ { \mathcal { M } }$ . (c) The Direct Conditional approach probes representations of conditional statements of the form “Given x, $P . \stackrel { \triangledown } { \ }$ and (d) we compute graded stability $\gamma _ { \mathcal { M } } ( P )$ as the fraction of conditioning propositions for which the conditional probability of $P$ remains above $t _ { \mathcal { M } }$

Binary stability, however, treats all violations identically. A proposition that falls below the belief threshold under a single conditioning statement is indistinguishable from one that does so under nearly all of them. We therefore define the graded belief stability of $P$ as

$$
\gamma _ { \mathcal M } ( P ) : = \frac { | \{ x \in \mathcal { X } _ { \mathcal M } \backslash \{ P \} : \mathrm { P r } _ { \mathcal M } ( P \mid x ) > t _ { \mathcal M } \} | } { | \mathcal { X } _ { \mathcal M } \backslash \{ P \} | } .\tag{2}
$$

Thus, $\gamma _ { \mathcal { M } } ( P ) \in [ 0 , 1 ]$ is the proportion of non-disbeliefs under which belief in P is preserved. Higher values indicate beliefs that persist more broadly across the model’s belief system, distinguishing graded stability from the probability

assigned to P in isolation.

## 2.2 Measuring graded belief stability in LLMs

Measuring graded stability requires identifying an LLM’s beliefs and non-disbeliefs and estimating $\operatorname* { P r } _ { \mathcal { M } } ( \boldsymbol { P } \mid \boldsymbol { x } )$ for belief–non-disbelief pairs. We use calibrated supervised probes over LLM hidden representations to obtain these probability estimates, since unconstrained prompted judgments need not yield calibrated probability estimates and can be sensitive to prompt formulation and conversational context [7, 10, 12, 29].

For each statement s , we extract hidden representations across layers ℓ of model M and train the sparse-aware multiple-instance learning (sAwMIL) probe [20], which models True (T), False (F), and Neither (N) as distinct veracity classes.<sup>1</sup> After calibration, the probe returns the trivalent distribution

$$
\pi _ { \boldsymbol { \mathcal M } } ( s _ { i } ) = \left[ \pi _ { T } ( s _ { i } ) , \pi _ { F } ( s _ { i } ) , \pi _ { N } ( s _ { i } ) \right] ,\tag{3}
$$

where $\pi _ { c } ( s _ { i } )$ is the probe-assigned probability of class c. The predicted epistemic state is then

$$
\hat { y } _ { i } = \arg \operatorname* { m a x } _ { c \in \{ T , F , N \} } \pi _ { c } ( s _ { i } ) .\tag{4}
$$

We interpret ${ \hat { y } } _ { i } = T$ as belief, ${ \hat { y } } _ { i } = F$ as disbelief, and ${ \hat { y } } _ { i } = N$ as suspension of belief (Fig. 1(a–b)). The probe therefore determines the empirical belief and non-disbelief sets directly:

$$
\mathcal { B } _ { \mathcal { M } } = \{ s _ { i } : \hat { y } _ { i } = T \} , \qquad \mathcal { X } _ { \mathcal { M } } = \{ s _ { i } : \hat { y } _ { i } \in \{ T , N \} \} .\tag{5}
$$

This trivalent operationalization retains an explicit state for suspended belief rather than forcing every proposition into a binary believed/not-believed distinction.

We then associate the probe-defined categorical belief set with an empirical Lockean-style threshold. For each model and domain, we define

$$
t _ { \mathcal { M } } : = \operatorname* { m i n } _ { s _ { i } \in \mathcal { B } _ { \mathcal { M } } } \pi _ { T } ( s _ { i } ) .\tag{6}
$$

Importantly, $t _ { \mathcal { M } }$ does not determine which propositions belong to $_ { B _ { \mathcal { M } } }$ . Rather, categorical belief is already defined by the probe’s trivalent prediction in Equation (5). Instead, the threshold translates this categorical belief set into a scalar criterion for belief persistence. Specifically, $t _ { \mathcal { M } }$ is the most stringent True-class support threshold satisfied by every proposition that the probe identifies as a belief. This anchors the persistence criterion to the model’s own empirical belief boundary rather than imposing an externally chosen probability cutof.

Because $\pi _ { T }$ is one component of a calibrated three-class distribution over True, False, and Neither, $t _ { \mathcal { M } }$ will not necessarily exceed 0.50, but is instead lower-bounded by 0.33. The resulting threshold should therefore be understood as a model-, domain-, and probe-specific support floor induced by the empirical belief set, rather than as a universal normative probability threshold (Fig. 1(b), Sec. 4.2).

We next estimate conditional belief probability for every belief $P \in B _ { \mathcal { M } }$ and non-disbelief $x \in \mathcal { X } _ { \mathcal { M } } \backslash \{ P \}$ . The Direct Conditional approach represents $P \mid x$ using the conditional statement “Given x, $P . ^ { \textit { s } }$ and probes its internal representation directly (Fig. 1(c)). A three-class sAwMIL probe predicts whether the resulting conditional is True, False, or Neither using Cooper–Cantwell training labels [28, 31, 32]. For each pair $( P , x )$ , the probe returns the trivalent distribution

$$
\pi _ { \mathcal M } ^ { \mathrm { d i r e c t } } ( P , x ) = \Bigl [ \pi _ { T } ^ { \mathrm { d i r e c t } } ( P , x ) , \pi _ { F } ^ { \mathrm { d i r e c t } } ( P , x ) , \pi _ { N } ^ { \mathrm { d i r e c t } } ( P , x ) \Bigr ] .\tag{7}
$$

We use the conditional’s True-class mass to define our Direct Conditional estimator

$$
{ \widehat { \operatorname* { P r } } } { \overset { \mathrm { d i r e c t } } { \operatorname { M } } } ( P \mid x ) : = \pi _ { T } ^ { \mathrm { d i r e c t } } ( P , x ) .\tag{8}
$$

We use ${ \widehat { \operatorname { P r } } } { \overset { \underset { \mathrm { d i r e c t } } { } } { \operatorname { M } } } ( P \mid x )$ as an empirical operational estimator of conditional belief probability rather than as an algebraic reconstruction of the normalized Cooper–Cantwell probability of a trivalent conditional. In particular, we do not renormalize over the True and False states. This keeps conditional support on the same three-class scale used to define the empirical belief threshold $t _ { \mathcal { M } }$ and allows probability mass assigned to Neither to count against persistence of the original belief. Section 4.3 motivates this choice and contrasts it with our complementary

![](images/7660498453b2777d003d9a501c8d377e85616354f9536dbe2e603de9a6aed2aa.jpg)  
Figure 2. Relationship between individual belief probability and graded stability. We show the held-out $R ^ { 2 }$ for predicting graded stability $\gamma _ { \mathcal { M } } ( P )$ from individual belief probability $\pi _ { T } ( P )$ for (a) City Locations, (b) Medical Indications, and (c) Word Definitions across Llama (red), Gemma (blue), Mistral (yellow), and Qwen (green) model families. We also report (d) distributions of pairwise Spearman correlations $\rho$ between LLMs’ residual graded stability within each domain, where black lines denote the median correlation across model pairs. Individual belief probability predicts a meaningful proportion of graded stability, but graded stability captures proposition-level variation beyond individual belief probability that is systematically shared across models.

Joint-to-Conditional estimator, which instead normalizes over determinate conditional states. Results on the secondary Joint-to-Conditional estimator are reported in Section I.2.2.

Replacing $\operatorname* { P r } _ { \mathcal { M } } ( \boldsymbol { P } \mid \boldsymbol { x } )$ in Eq. (2) with the Direct Conditional estimate in Eq. (8) yields our empirical Direct Conditional stability score $\gamma _ { \mathcal { M } } ^ { \mathrm { d i r e c t } } ( P )$ (Fig. 1(d); Sec. 4.4). We apply this probing-based framework to 12 instructiontuned<sup>2</sup> LLMs spanning four model families and ranging from 3B to 72B parameters. We evaluate three domains with difering epistemic characteristics: City Locations, consisting of comparatively clear-cut factual relations; Medical Indications, requiring specialized factual knowledge; and Word Definitions, containing comparatively ambiguous lexical relations [20].

## 2.3 Validating graded stability

We validate graded stability along two complementary dimensions: whether it captures information beyond individual belief probability and whether the probability estimates underlying it are approximately probabilistically coherent. Main-text results use the Direct Conditional estimator and the sAwMIL probe. Joint-to-Conditional results and analyses using the SVM and Mass Mean probes appear in Section I.

## 2.3.1 Graded stability captures information beyond belief probability

A central motivation for graded stability is that it captures how a belief behaves in the context of an LLM’s broader belief system rather than simply how strongly that belief is held in isolation. We therefore first ask how well the individual belief probability $\pi _ { T } ( P )$ predicts $\gamma _ { \mathcal { M } } ( P )$ for beliefs $P \in B _ { \mathcal { M } }$

If graded stability were simply a transformation of belief probability, then a suficiently flexible function of $\pi _ { T } ( P )$ should accurately predict $\gamma _ { \mathcal { M } } ( P )$ for beliefs not used to fit that function. For each model and domain, we attempt

![](images/8fcbb5ac6f069475a7e0d1650b661d50874eb8d8f906825a168c1d3406490b85.jpg)  
Figure 3. Distance of measured probabilities from probabilistic coherence. Violins show the distribution across $( P , x )$ pairs of the root-mean-square (RMS) adjustment $d _ { \mathrm { C C K } }$ required to project the measured probability distributions for individual statements and conditionals onto the nearest CCK-coherent probability system for (a) City Locations, (b) Medical Indications, and (c) Word Definitions for Llama (red), Gemma (blue), Mistral (yellow), and Qwen (green) model families. Horizontal black lines denote within-model medians. Smaller values indicate greater probabilistic coherence, with $d _ { \mathrm { C C K } } = 0$ corresponding to exact coherence. Although exact coherence is uncommon, only modest adjustments are generally required to reconcile the independently measured probabilities with a coherent trivalent probability system.

to capture such a function using a natural cubic spline,

$$
\begin{array} { r } { \widehat { \gamma } _ { \mathcal { M } } ( P ) = f _ { \mathcal { M } } ( \pi _ { T } ( P ) ) , } \end{array}\tag{9}
$$

with three degrees of freedom. The spline allows belief probability and stability to have a smooth nonlinear relationship without assuming a linear relationship. We use five-fold cross-validation to generate out-of-sample predictions $\widehat { \gamma } _ { { \mathcal M } } ( P )$ and evaluate performance with $R ^ { 2 }$ (Sec. 4.5.1). We fix the spline complexity across all models and domains and show that the results are qualitatively unchanged across alternative spline complexities in Section F.1.

Individual belief probability consistently predicts a meaningful but incomplete proportion of the observed variation in graded stability (Fig. 2(a–c)). The median held-out $R ^ { 2 }$ across models is 0.46 for City Locations, 0.39 for Medical Indications, and 0.47 for Word Definitions. Thus, although belief probability is informative about stability, more than half of the proposition-level variation in $\gamma _ { \mathcal { M } } ( P )$ remains unexplained by the probability assigned to P in isolation. Graded stability therefore captures substantial variation beyond individual belief strength.

We next ask whether this unexplained variation is systematic. For each proposition P, we define its residual stability as

$$
r _ { \mathcal { M } } ( P ) : = \gamma _ { \mathcal { M } } ( P ) - \widehat \gamma _ { \mathcal { M } } ( P ) ,\tag{10}
$$

where positive values identify propositions that are more stable than predicted from their individual belief probability and negative values identify propositions that are less stable than predicted. We align pairs of LLMs on propositions believed by both models and compute the Spearman correlation between their residual stability scores (Fig. 2(d)).

Across domains, 99.0% of model-pair residual correlations are positive, with median correlations of $\rho = 0 . 2 3$ for City Locations, $\rho = 0 . 3 0$ for Medical Indications, and $\rho = 0 . 3 1$ for Word Definitions. All three medians exceed a proposition-shufled permutation null $\left( p = 0 . 0 0 1 ; \mathrm { S e c . ~ } 4 . 5 . 1 \right)$ . Models therefore tend to find the same propositions unusually stable or fragile even after accounting for how strongly they believe those propositions in isolation. This suggests that the variation unique to graded stability contains reproducible proposition-level structure rather than only model-specific measurement noise. Because models are evaluated on the same benchmark propositions using a common probing framework, however, this analysis does not by itself distinguish model-internal relational structure from proposition-level or measurement factors shared across models.

## 2.3.2 Graded stability probability estimates are approximately coherent

Because graded stability is defined within a probabilistic framework, we next ask whether the independently measured individual and conditional probability distributions are pairwise compatible. Under Cooper–Cantwell–Kleene (CCK) trivalent probability, the distributions for $P , x ,$ and $P \mid$ x are coherent when they can all be generated by a single latent $3 \times 3$ probability distribution over the trivalent states of $( P , x )$ . For each measured pair, we collect these probabilities into the nine-dimensional vector

$$
\begin{array} { r } { \mathbf { v } _ { P , x } = \left[ \pi _ { \mathcal { M } } ( P ) , \pi _ { \mathcal { M } } ( x ) , \pi _ { \mathcal { M } } ^ { \mathrm { d i r e c t } } ( P , x ) \right] . } \end{array}\tag{11}
$$

Under CCK semantics, such a latent joint distribution exists if and only if the measured probabilities satisfy

$$
\begin{array} { r } { \pi \mathrm { d i r e c t } ( P , x ) \leq \pi _ { T } ( P ) , \qquad \pi _ { F } ^ { \mathrm { d i r e c t } } ( P , x ) \leq \pi _ { F } ( P ) , \qquad \pi _ { F } ( x ) \leq \pi _ { N } ^ { \mathrm { d i r e c t } } ( P , x ) , \qquad \pi _ { N } ^ { \mathrm { d i r e c t } } ( P , x ) \leq \pi _ { N } ( P ) + \pi _ { F } ( x ) . } \end{array}\tag{12}
$$

We use these constraints to test exact coherence separately for every $( P , x )$ pair (Sec. 4.5.2); a proof of the coherence conditions is provided in Sec. F.2.1. Exact coherence is extremely rare: across all model–domain combinations, no more than 0.21% of measured pairs satisfy all four constraints exactly. Full exact-coherence rates and individual constraint violations are reported in Section F.2.

Because exact coherence is a binary criterion that treats arbitrarily small and large violations identically, we additionally quantify each pair’s distance from the set of coherent probability systems. Following prior work that characterizes approximate coherence in terms of distance to the nearest coherent credence function [33], we extend this distance-based approach to the CCK trivalent setting and define

$$
d _ { \mathrm { C C K } } ( \mathbf { v } _ { P , x } ) : = \sqrt { \frac { 1 } { 9 } \left. \mathbf { v } _ { P , x } - \Pi _ { \mathcal { C } } ( \mathbf { v } _ { P , x } ) \right. _ { 2 } ^ { 2 } } ,\tag{13}
$$

where $\mathcal { C }$ is the set of CCK-coherent probability vectors and $\Pi _ { C } ( \mathbf { v } )$ is the closest vector in this set to the observed probability vector $\mathbf { v . ^ { 3 } }$ Thus, $d _ { \mathrm { C C K } } = 0$ corresponds to exact coherence, while a value such as $d _ { \mathrm { C C K } } = 0 . 0 5$ means that reaching the nearest coherent probability system requires a root-mean-square (RMS) adjustment of 0.05 per probability component.

Across models, the median model-level $d _ { \mathrm { C C K } }$ is 0.015 for City Locations, 0.052 for Medical Indications, and 0.062 for Word Definitions, corresponding to RMS adjustments of 1.5, 5.2, and 6.2 percentage points per probability component, respectively. Thus, despite near-zero exact-coherence rates, the measured probability systems typically require quantitatively modest adjustments to reach the nearest coherent CCK representation. Because graded stability thresholds these probabilities at $t _ { \mathcal { M } }$ , however, a small component-wise RMS adjustment does not imply that the corresponding absolute stability score would be unchanged after projection. We therefore interpret $d _ { \mathrm { C C K } }$ as a diagnostic of the measured probability system rather than as a robustness bound on $\gamma _ { \mathcal { M } } ( P )$

## 2.4 Graded belief stability varies with domain

Having established that graded stability captures information beyond individual belief probability and is approximately probabilistically coherent, we next ask how stability varies across beliefs and models by characterizing the distribution of $\gamma _ { \mathcal { M } } ( P )$ across each model’s beliefs (Fig. 4). Stability difers substantially across semantic domains. Across models, the median of model-level mean graded stability is 0.88 for City Locations, compared with 0.71 for Medical Indications and 0.74 for Word Definitions. Comparing the within-model median stabilities across domains, City Locations has the highest median stability for 11 of the 12 LLMs. The only exception is Llama-3.2-3b, whose median stability is highest for Word Definitions. Within these benchmark-specific belief systems, City Locations beliefs are therefore consistently more stable than beliefs in the Medical Indications and Word Definitions domains. This pattern is consistent with diferences in the semantic structure of the domains, as City Locations contains comparatively clear-cut factual relations, whereas Medical Indications requires more specialized factual knowledge and Word Definitions contains comparatively ambiguous lexical relations. The same broad domain structure is visible in matched pretrained base models (Sec. I.1.1).

These diferences extend beyond shifts in average stability to the proposition-level distributions themselves. In City Locations, stability is concentrated near the upper end of the range for most models, although long lower tails show that even this comparatively stable domain contains individual beliefs that are substantially more fragile (Fig. 4(a)). Medical Indications and Word Definitions exhibit broader distributions and considerably greater variation across models. In Medical Indications in particular, several of the smallest models, including Llama-3.2-3b,

![](images/27314d16e60cea9db8b261b178cf0346c3d72fb4b411a6c0c074f466a8165835.jpg)  
Figure 4. Variation in graded belief stability across domains. Violin plots show the proposition-level distribution of graded stability $\gamma _ { \mathcal { M } } ^ { \mathrm { d i r e c t } } ( P )$ for each of the 12 instruction-tuned LLMs in (a) City Locations, (b) Medical Indications, and (c) Word Definitions for Llama (red), Gemma (blue), Mistral (yellow), and Qwen (green) model families. Horizontal black lines denote within-model medians. Stability distributions difer substantially across the three benchmark domains, with City Locations exhibiting the highest median stability for 11 of the 12 models.

Gemma-7b, and Qwen-2.5-7b, have markedly lower median stability than larger models from the same families (Fig. 4(b)).

Some of the between-model variation in Figure 4 also suggests a possible relationship with model scale. In City Locations and Medical Indications, several of the smallest models within a family have lower median stability than their larger counterparts (Fig. 4(a–b)). This pattern is not consistent across all families, however, and is largely absent in Word Definitions. We examine model scale more directly in Section H, where we demonstrate that neither parameter count nor model depth shows a consistent relationship with graded stability across domains. Thus, model scale may contribute to some of the variation observed within families, but it does not provide a general explanation for diferences in $\gamma _ { \mathcal { M } } ( P )$

## 2.5 Graded stability is associated with behavioral movement beyond belief probability

Graded stability is defined from relationships among internal representations, but its usefulness depends in part on whether those diferences correspond to independently measured model behavior. We therefore test whether $\gamma _ { \mathcal { M } } ( P )$ distinguishes resistance to conversational challenge among beliefs assigned similar individual probabilities. Importantly, the challenge prompts are not conditioning propositions $x \in { \mathcal { X } } _ { \mathcal { M } }$ and do not instantiate the formal operation used to compute graded stability. They instead provide an external behavioral perturbation, allowing us to ask whether representation-level stability generalizes to a qualitatively diferent form of belief pressure.

We construct pairs of believed propositions with similar $\pi _ { T } ( P )$ within each model and domain using sequenceblocked matching (Sec. 4.6, G). We match propositions by maximizing the number of non-overlapping pairs while minimizing their total absolute diference in belief probability. We then identify the propositions $P _ { H }$ and $P _ { L }$ with higher and lower graded stability within each pair. Our primary analysis retains pairs for which both propositions initial behavioral judgments agree with their individual-statement probe classifications.

We evaluate each proposition in a four-stage interaction consisting of an initial binary True/False judgment followed by three natural-language challenges. Propositions are randomly assigned one of six possible challenge orderings to control for order efects (Sec. 4.6). The challenges model conversational pressure from a disagreeing user rather than literal factual conditioning on another proposition x. They therefore provide external behavioral validation of graded stability rather than an empirical realization of $\operatorname* { P r } _ { \mathcal { M } } ( P \mid x )$

We record the normalized probability $u _ { 0 } ( P )$ assigned to the model’s initial judgment and denote the probability assigned to that same response after k challenges by $u _ { k } ( P )$ . We quantify behavioral movement as

$$
M ( P ) : = \frac { 1 } { 3 } \sum _ { k = 1 } ^ { 3 } \left| u _ { k } ( P ) - u _ { k - 1 } ( P ) \right| ,\tag{14}
$$

Higher γ(P) is associated with greater resistance to challenge

![](images/cfb35451ded7e3320c21c30efe67bdaea25678d38df4e706313752b0e5cdc7de.jpg)  
Figure 5. Behavioral resilience among beliefs matched on individual belief probability. Bars show the mean diference in behavioral movement ∆M between matched lower- and higher-stability beliefs for each of the 12 instruction-tuned LLMs, with error bars denoting ±1 standard error. Positive values indicate that lower-stability beliefs exhibit greater movement under conversational challenge. Results are shown for (a) City Locations, (b) Medical Indications, and (c) Word Definitions for Llama (red), Gemma (blue), Mistral (yellow), and Qwen (green) model families. Across domains, 83.3% of model–domain settings show greater behavioral movement for lower-stability beliefs.

where larger $M ( P )$ indicates greater fluctuation in support for the model’s initial response. Because the diferences are absolute, increases and decreases in support both contribute to $M ( P )$ . It therefore measures behavioral movement rather than specifically loss of support or answer flipping.

For each probability-matched pair, we then compare the behavioral movement of the lower- and higher-stability propositions:

$$
\Delta M : = M ( P _ { L } ) - M ( P _ { H } ) .\tag{15}
$$

Thus, $\Delta M > 0$ indicates that the lower-stability belief exhibits greater behavioral movement under challenge. Because $P _ { H }$ and $P _ { L }$ are identified only after matching on individual belief probability, this comparison tests whether graded stability distinguishes behavioral resilience beyond diferences in $\pi _ { T } ( P )$

Mean ∆M is positive for 10/12 models in City Locations, 10/12 in Medical Indications, and 10/12 in Word Definitions. Median efects are 0.011, 0.026, and 0.022, respectively. However, uncertainty is substantial for many individual estimates: the 95% bootstrap confidence intervals are entirely positive in 11/36 settings, entirely negative in one, and include zero in the remainder (Sec. G.2). Thus, graded stability shows a broadly consistent directional association with behavioral resilience, although the magnitude and precision of the efect vary across models and beliefs.

## 3 Discussion

This work asks whether the stability of an LLM judgment can be characterized by its relationship to the model’s other beliefs rather than by individual belief probability alone. Across 12 instruction-tuned LLMs and three semantic domains, individual belief probability explains a meaningful but incomplete fraction of graded stability. The remaining variation is also not arbitrary. After accounting for individual belief probability, diferent models tend to identify the same propositions as unusually stable or fragile, and among beliefs deliberately matched on individual belief probability, lower graded stability is generally associated with greater movement under conversational challenge. These results support the central distinction motivating graded stability: how strongly an LLM supports a proposition in isolation and how robustly that support persists in the context of its other beliefs are related, but non-equivalent, properties.

A central implication of our results is that this relational structure depends strongly on what is believed. City Locations is the most stable domain for 11 of the 12 instruction-tuned models, and the same broad domain structure appears in matched pretrained base models and under the alternative SVM probe (Secs. I.1.1, I.3). By contrast, neither parameter count nor model depth shows a consistent relationship with graded stability across domains (Sec. H). Instruction tuning, however, is associated with greater stability in most matched base versus instruction-tuned comparisons, but the magnitude and even direction of this diference vary across models and domains (Sec. I.1.2). These results argue against treating graded stability as a generic property that simply increases with model capability. Instead, stability appears to reflect an interaction among the semantic structure of the belief, the model in which it is represented, and the post-training regime applied to that model. A key next step is therefore to identify the semantic properties that systematically confer stability or fragility, and to determine whether those properties generalize beyond the controlled domains studied here.

Our robustness analyses further suggest that the most reproducible feature of graded stability is its relative structure rather than its absolute numerical scale. The Direct Conditional and Joint-to-Conditional estimators operationalize conditional belief probability in diferent ways, yet they produce positively correlated stability rankings in every setting (Sec. I.2.3). At the same time, their absolute stability magnitudes difer substantially, with Joint to-Conditional scores compressed toward 1.0 (Sec. I.2). Absolute graded-stability values, and diferences in their magnitudes across domains, should therefore be interpreted within a particular estimator and probe rather than as measurement-invariant quantities. Probe choice produces a related distinction. The SVM probe broadly reproduces the main findings obtained with sAwMIL, whereas the Mass Mean probe yields weaker domain separation, larger departures from probabilistic coherence, and less consistent behavioral associations (Sec. I.3). This weaker replication is not surprising because Mass Mean is not designed to recover calibrated trivalent probability distributions, which graded stability explicitly requires. These analyses suggest that a reproducible relational signal persists across operationalizations, even though its absolute numerical realization depends on the conditional estimator and probe. Faithfully recovering that signal therefore depends on measurement choices that preserve the probabilistic properties required by the construct.

The behavioral challenge experiment tests whether graded stability distinguishes resistance to repeated conversational challenge among beliefs with similar individual belief probabilities. Among beliefs matched on individual belief probability, lower-stability beliefs exhibit greater mean movement under repeated challenge in 83.3% of instruction tuned model–domain settings. The consistency of this direction across models is notable, but the individual efects are heterogeneous and often imprecisely estimated: most 95% bootstrap intervals include zero (Sec. G.2). We consequently view the behavioral experiment as evidence that graded stability captures diferences in resilience beyond individual belief probability, rather than as evidence for a universal mapping from γ to conversational behavior. More broadly, resistance to revision is not always desirable. Instability can indicate unwanted fragility, but revision may be appropriate when new information exposes an error or creates a legitimate conflict with existing commitments. Conversely, an incorrect judgment that resists all counterevidence would be highly stable without being reliable. Graded stability is therefore best understood as a descriptive property of an LLM belief system rather than a score to optimize. Distinguishing unwarranted fragility from appropriate responsiveness will require manipulating the relevance, reliability, and evidential force of the information introduced to the model.

Our results also illustrate the value and limitations of applying formal probabilistic theories to LLM representations. LLMs are autoregressive sequence models, not agents explicitly trained to maintain a globally coherent probability system over propositions, and we find correspondingly little exact CCK coherence. However, the measured individual belief and conditional probability distributions generally require only modest adjustments to reach the nearest coherent system. Formal coherence therefore does not have to be treated as an all-or-nothing claim about whether an LLM literally instantiates a particular epistemic theory. It can instead provide a reference structure against which empirical representations are measured. More generally, the calibrated probe probabilities used throughout should be understood as representation-based estimates of epistemic support rather than direct observations of a uniquely defined latent credence distribution of the LLM. From this perspective, approximate coherence makes the formal framework useful, while departures from coherence become objects of study rather than reasons to abandon it. The same principle applies to the definition of graded stability itself. Our formulation weights each admissible conditioning proposition equally and adopts Cooper–Cantwell–Kleene semantics for trivalent conditionals However, alternative weighting schemes or conditional semantics would define related but distinct notions of stability. Understanding which of these notions best predicts particular forms of model behavior is an important direction for future work.

Finally, our experiments deliberately use controlled factual domains containing explicit True, False, and Neither cases. This structure is essential to the present study as it provides comparatively well-defined trivalent supervision, permits systematic construction of conditional training examples, and supplies propositional objects that map cleanly onto the formal account of belief and conditioning. The same control limits ecological scope. Accordingly, diferences in γ across these domains reflect both the target propositions and the composition of the benchmark-specific non-disbelief sets over which stability is evaluated, rather than an intrinsic stability of the semantic domain alone. Rich natural-language claims may contain multiple propositions, presuppositions, context-sensitive content, or ambiguous conditioning relations, and real conversational contexts need not resemble the within-domain non-disbelief sets studied here. Extending graded stability to richer benchmarks and cross-domain belief systems will therefore require more than simply applying the same pipeline to longer text; it will require defining what the relevant propositions and conditioning relations are.

<table><tr><td>Official Name</td><td>Short Name</td><td># Layers</td><td># Parameters</td><td>Release Date</td><td>Source</td><td>Citation</td></tr><tr><td>Gemma-7b-it</td><td>gemma-7b</td><td>28</td><td>8.54 B</td><td>Feb 21, 2024</td><td>Google</td><td>[34]</td></tr><tr><td>Gemma-2-9b-it</td><td>gemma-2-9b</td><td>42</td><td>9.24 B</td><td>Jun 27, 2024</td><td>Google</td><td>[34]</td></tr><tr><td>Gemma-2-27b-it</td><td>gemma-2-27b</td><td>46</td><td>27.23 B</td><td>Jun 27, 2024</td><td>Google</td><td>[34]</td></tr><tr><td>Llama-3.2-3b-Instruct</td><td>11ama-3.2-3b</td><td>28</td><td>3.21 B</td><td>Sep 25, 2024</td><td>Meta</td><td>[35]</td></tr><tr><td>Llama-3.1-8b-Instruct</td><td>11ama-3.1-8b</td><td>32</td><td>8.03 B</td><td>Jul 23, 2024</td><td>Meta</td><td>[35]</td></tr><tr><td>Llama-3.1-70b-Instruct</td><td>1lama-3.1-70b</td><td>80</td><td>70.55 B</td><td>Jul 23, 2024</td><td>Meta</td><td>[35]</td></tr><tr><td>Mistral-7b-Instruct-v0.3</td><td>mistral-7b</td><td>32</td><td>7.25 B</td><td>May 22, 2024</td><td>Mistral AI</td><td>[36]</td></tr><tr><td>Mistral-Nemo-Instruct-2407</td><td>mistral-12b</td><td>40</td><td>12.25 B</td><td>Jul 18, 2024</td><td>Mistral AI</td><td>[37]</td></tr><tr><td>Mistral-Small-3.1-24B-Instruct-2503</td><td>mistral-3.1-24b</td><td>40</td><td>23.57 B</td><td>Mar 17, 2025</td><td>Mistral AI</td><td>[38]</td></tr><tr><td>Qwen2.5-7B-Instruct</td><td>qwen-2.5-7b</td><td>28</td><td>7.62 B</td><td>Sep 19, 2024</td><td>Alibaba Cloud</td><td>[39, 40]</td></tr><tr><td>Qwen2.5-14B-Instruct</td><td>qwen-2.5-14b</td><td>48</td><td>14.80 B</td><td>Sep 19, 2024</td><td>Alibaba Cloud</td><td>[39, 40]</td></tr><tr><td>Qwen2.5-72B-Instruct</td><td>qwen-2.5-72b</td><td>80</td><td>72.70 B</td><td>Sep 19, 2024</td><td>Alibaba Cloud</td><td>[39, 40]</td></tr></table>

Table 1. LLMs used in graded stability experiments. We list the oficial names of the LLMs according to the HuggingFace repository [41], where they are publicly available. We further specify the shortened name used throughout the paper, the number of layers, parameter count, release date, source organization, and oficial citation.

What remains consistent across these methodological and formal choices is the distinction between the strength of an individual belief and its stability within a broader system of beliefs. Evaluations of factual reliability typically ask whether a model’s answer is correct, how strongly it is supported, or whether confidence tracks accuracy. Our results show that relationships among beliefs provide complementary information. Propositions with similar individual support can difer systematically in stability, diferent models share substantial structure in which propositions are unusually stable or fragile, and, under our primary operationalization, these diferences are associated with downstream behavioral resilience. Graded belief stability therefore provides a relational view of factual reliability that complements existing measures of accuracy and uncertainty. Characterizing this structure may help explain no only what an LLM appears to believe, but how those beliefs are situated within a broader epistemic system and which are most susceptible to change as their informational context shifts.

## 4 Methods

We first describe the LLMs, datasets, and probing framework used throughout our experiments. We then detail how we estimate individual statement probabilities, construct empirical belief and non-disbelief sets, and measure conditional probabilities. Finally, we describe the analyses used to validate graded stability, characterize its variation across domains and models, and evaluate its relationship with behavioral resilience under conversational challenge. Al code and datasets associated with the experiments can be found at https://github.com/samanthadies/graded stability.

## 4.1 Models and data

## 4.1.1 Large language models

We evaluate 12 open-weight, instruction-tuned LLMs from four model families: Gemma, Llama, Mistral, and Qwen. For Gemma, we consider 7B, 9B, and 27B. For Llama, we use 3B, 8B, and 70B. For Mistral, we use 7B, 12B, and 24B. Finally, for Qwen we use 7B, 14B, and 72B. We report the exact model versions and the number of decoder layers in Table 1. Matched base models are listed in Section C. All models are loaded in evaluation mode with CUDA and BF16. We render statements using the models’ native chat templates as single user messages without an assistant response.

## 4.1.2 Datasets

Our datasets, introduced in [20], include three domains: City Locations, Medical Indications, and Word Definitions (Tab. 2). Each domain contains True, False, and Neither statements. The Neither statements pair synthetic entities generated using character n-gram Markov-chain models fit to the corresponding real entity sets and validated to reduce the likelihood that they correspond to existing entities. They therefore provide controlled cases for which an LLM should ideally suspend belief, making the datasets well suited to our trivalent belief framework. These synthetic cases instantiate a controlled form of epistemic indeterminacy associated with unfamiliar content. However, they do not exhaust the ways in which an LLM may suspend judgment in natural-language settings. Further dataset construction and validation details are provided in Section B.1.

<table><tr><td>Dataset</td><td>True</td><td>False</td><td>Synthetic</td><td>Examples</td></tr><tr><td>City</td><td>A: 1392</td><td>A: 1358</td><td>A:876</td><td>T. The city of Surat is located in India. F. The city of Palembang is located in the Dominican Republic.</td></tr><tr><td>Locations</td><td>N:1376</td><td>N:1374</td><td>N:876</td><td>S. The city of Norminsk is located in Jamoates.</td></tr><tr><td></td><td>A: 1439</td><td></td><td></td><td>T. Pentobarbital is indicated for the treatment of insomnia.</td></tr><tr><td>Medical Indications</td><td>N:1522</td><td>A: 1523 N:1419</td><td>A:478 N: 522</td><td>F. Vancomycin is not indicated for the treatment of lower respiratory tract infections.</td></tr><tr><td></td><td></td><td></td><td></td><td>S. Alumil is indicated for the treatment of reticers. T. Hoagy is a synonym of an Italian sandwich.</td></tr><tr><td>Word</td><td>A: 1234</td><td>A: 1277</td><td>A:1747</td><td>F. Decalogue is an astronomer.</td></tr><tr><td>Definitions</td><td>N:1235</td><td>N:1254</td><td>N:1753</td><td>S. Dostab is a scencer.</td></tr></table>

Table 2. Summary of datasets and statement types. Number of afirmative (A) and negated (N) statements across the three domains, along with examples. Each dataset includes True (T), False (F), and Synthetic (S) statements. Synthetic statements serve as Neither statements constructed to minimize prior LLM exposure. An earlier version of this table was introduced in [20].

We use the fixed training, calibration, and test partitions introduced with the original datasets [20] (Sec. B.2), containing approximately 55%, 20%, and 25% of statements, respectively, with entities kept exclusive across splits. The training split is used to fit each probe, while the calibration split is used to map raw probe scores to calibrated probability distributions for individual statements, $\pi _ { \mathcal { M } } ( s _ { i } )$ , and Direct Conditional statement pairs, $\pi _ { \mathcal { M } } ^ { \mathrm { d i r e c t } } ( s _ { i } , s _ { j } )$ according to the procedure in Section D.1. The fitted probes and calibration mappings are then applied to test statements and statement pairs.

The statements in the train and calibration sets are reused to train and calibrate the individual-statement probe and the Direct Conditional probe. The test set statements form the basis of our belief and non-disbelief sets $B _ { { \mathcal M } }$ and $\mathcal { X } _ { \mathcal { M } }$

## 4.1.3 Probes

Each statement $s _ { i }$ is fed into a model M, resulting in an activation $z _ { i } ^ { ( \ell ) }$ at each layer ℓ. We train each probe over the activations corresponding to the training statements, using the ground-truth True, False, and Neither labels as training targets. The probes produce raw score vectors $\mathbf { r } ( s _ { i } )$ over the K classes. Because these raw scores cannot be directly interpreted as probabilities, we fit a multinomial logistic-regression calibration model on the calibration split that maps each raw score vector $\mathbf { r } ( s _ { i } )$ to a calibrated class-probability distribution $\pi _ { \mathcal { M } } ( s _ { i } )$ over the K classes (Sec. D.1). Complete training specifications and hyperparameters for sAwMIL are reported in Section D.2, and its layer-selection procedure and selected layers are reported in Section D.3. Corresponding details for the secondary SVM and Mass Mean probes are provided in Section I.3. Throughout, we treat these calibrated probe probabilities as representation-based estimates of the model’s epistemic support over the supervised trivalent states. Calibration makes the probe outputs probabilistically interpretable with respect to those states, but does not by itself establish that they are identical to a uniquely defined latent credence distribution of the LLM.

Primary probe: sAwMIL Our main probe is the sparse-aware multiple-instance learning (sAwMIL) probe [20]. sAwMIL is a max-margin probe that is trained over a bag of token-level representations for each statement, rather than a single statement-level vector. It was also designed specifically to handle more than two classes. For K classes, sAwMIL trains K separate one-versus-all heads which are combined into a multiclass classifier using a softmax function.

Secondary probes: SVM and Mass Mean We use two additional probes, an SVM and the Mass Mean probe [17], to test whether our stability results are qualitatively robust to the choice of probe. The corresponding analyses are reported in Sec. I.3. Like sAwMIL, the SVM is a max-margin classifier that learns K one-versus-all heads. However, sAwMIL is a multi-instance learning probe, while the SVM operates only on the activation corresponding to a given statement’s final token. The Mass Mean probe is also a single-instance probe that operates on the activation corresponding to a given statement’s final token. It was initially designed as a binary probe. However, we extend it to a multiclass probe so that it can be used in our three-class, trivalent probability setting. For each class $j \in \{ 0 , \ldots , K - 1 \}$ , the Mass Mean probe takes the mean representation of the training examples in class $j , \mu _ { j }$ , the mean representation of the examples in all other classes, $\mu { \to } j$ , and computes the diference vector $\Delta \mu _ { j } = \mu _ { j } - \mu _ { \lnot j }$ . It then normalizes $\Delta \mu _ { j }$ to the unit norm and places the decision boundary halfway between $\mu _ { j }$ and $\mu _ { \neg j }$ . Statements are scored based on their signed projection along the $\Delta \mu _ { j }$ vector. This again results in K one-versus-all heads. Outputs from both SVM and Mass Mean undergo the same calibration step used for sAwMIL.

## 4.2 Estimating probabilities for individual statements and constructing empirical belief sets

For each model M, dataset, and probe, we train a three-class probe independently at each layer ℓ using the training split and fit its probability-calibration mapping using the calibration split. We then evaluate the calibrated probe at each layer on the fixed calibration partition $\mathcal { D } _ { \mathrm { c a l } }$ using multiclass log loss,

$$
\mathcal { L } _ { \mathrm { l o g } } ^ { ( \ell ) } = - \frac { 1 } { \vert \mathcal { D } _ { \mathrm { c a l } } \vert } \sum _ { ( s _ { i } , y _ { i } ) \in \mathcal { D } _ { \mathrm { c a l } } } \log \pi _ { y _ { i } } ^ { ( \ell ) } ( s _ { i } ) ,\tag{16}
$$

and select

$$
\ell ^ { \star } = \arg \operatorname* { m i n } _ { \ell } \mathcal { L } _ { \log } ^ { ( \ell ) } .\tag{17}
$$

Section D.3 reports the full layer-selection procedure and the resulting selected layers for sAwMIL. Corresponding results for the secondary probes are reported in Section I.3.

After selecting $\ell ^ { \star }$ , we retain the fitted and calibrated probe from that layer and use its outputs on the test statements throughout the downstream analysis rather than retraining the probe after layer selection. For each statement, we record both $\pi _ { \boldsymbol { \mathcal { M } } } ( s _ { i } ) = [ \pi _ { T } ( s _ { i } ) , \pi _ { F } ( s _ { i } ) , \pi _ { N } ( s _ { i } ) ]$ and the predicted state $\hat { y } _ { i } = \arg \operatorname* { m a x } _ { c \in \{ T , F , N \} } \pi _ { c } ( s _ { i } )$ These predictions define the empirical belief and non-disbelief sets used in all subsequent analyses.

For each test statement, we map its predicted class $\hat { y } _ { i }$ to a belief state, where ${ \hat { y } } _ { i } = T$ is a belief, ${ \hat { y } } _ { i } = F$ is a disbelief, and ${ \hat { y } } _ { i } = N$ is a suspended belief. For clarity, we restate the empirical belief and non-disbelief sets defined in Equation (5):

$$
\mathcal { B } _ { \mathcal { M } } = \{ s _ { i } : \hat { y } _ { i } = T \} , \qquad \mathcal { X } _ { \mathcal { M } } = \{ s _ { i } : \hat { y } _ { i } \in \{ T , N \} \} .
$$

Crucially, this is based on the probe’s prediction, not the statement’s ground-truth label, meaning that a False or Neither statement could be classified as a model belief. Thus, $B _ { \mathcal { M } } \subseteq \mathcal { X } _ { \mathcal { M } }$ . Section E.1 reports the resulting belief and non-disbelief set sizes, empirical thresholds, and conditional-pair coverage for each LLM–dataset combination. From these belief and non-disbelief sets, we construct $\vert \mathcal { X } _ { \mathcal { M } } \vert - 1$ statement pairs for each $P \in B _ { \mathcal { M } }$ of the form $( P , x )$ These statement pairs are the target of our conditional probability estimators described in Section 4.3 and ultimately define graded stability as described in Section 4.4.

For each LLM, probe, and dataset combination, we use the empirical belief threshold from Equation (6), restated for clarity:

$$
t _ { \mathcal { M } } = \operatorname* { m i n } _ { s _ { i } \in { \mathcal { B } _ { \mathcal { M } } } } \pi _ { T } ( s _ { i } ) .
$$

This belief threshold is derived after categorical beliefs have been identified and corresponds to the smallest True-class probability among statements classified by the probe as True. This guarantees that the probability $\pi _ { T } ( P _ { i } )$ that statement $P _ { i }$ is true is at least $t _ { \mathcal { M } }$ for all of model M’s beliefs $P _ { i } \in B _ { \mathcal { M } }$ . The strict persistence criterion $> t _ { \mathcal { M } }$ is separate from this categorical membership rule: a conditional estimate exactly equal to $t _ { \mathcal { M } }$ is counted as nonpersistent, while membership in $_ { B _ { \mathcal { M } } }$ remains determined by the trivalent probe prediction. Moreover, conditional persistence is determined by True-state support relative to $t _ { \mathcal { M } }$ rather than by the conditional probe’s argmax state, so a conditional can satisfy the persistence criterion even when True is not its most probable class.

## 4.3 Estimating conditional probability

For each belief $P \in B _ { \mathcal { M } }$ and non-disbelief $x \in \mathcal { X } _ { \mathcal { M } } \backslash \{ P \}$ , we aim to estimate $\operatorname* { P r } _ { \mathcal { M } } ( P \mid x )$ so that we can compute $\gamma _ { \mathcal { M } } ( P )$ , the proportion of conditioning propositions x for which belief in $P$ persists $( \mathrm { i . e . , P r } _ { \mathcal { M } } ( P \mid x ) > t _ { \mathcal { M } } )$ . Our primary operationalization, Direct Conditional, represents $P \mid x$ directly as a conditional statement and probes for its veracity. Our secondary approach, Joint-to-Conditional, represents x and $P$ jointly, probes their full $3 \times 3$ joint states, then derives the conditional mathematically. All main-text analyses use the Direct Conditional estimator. We use Joint-to-Conditional as a secondary estimator to test robustness to the conditional-probability operationalization (Sec. E.2, I.2).

Both approaches use the same general probing framework as the estimation of probabilities for individual statements described in Section 4.2, including the data splits and probability-calibration procedure. However, we construct new conditional or joint training and calibration examples from the existing train and calibration sets, use new training targets for the combined statements, and score LLM, probe, and dataset-specific $( P , x )$ pairs. The Direct

<table><tr><td>y ↑ sn</td><td>TNFTNFNNN</td></tr><tr><td>si</td><td>TNFTNFTNF</td></tr><tr><td>s</td><td>TTTNNNFFE</td></tr></table>

Table 3. Cooper–Cantwell conditional truth table. We report whether the conditional $s _ { i } \to s _ { j } \equiv s _ { j } \mid s _ { i }$ is True (T), False (F), or Neither (N) based on the truth values of $s _ { i }$ and $s _ { j }$ according to the Cooper–Cantwell conditional [28, 31]. When the antecedent $s _ { i }$ is True or Neither, the conditional takes the truth value of the consequent $s _ { j }$

Conditional and Joint-to-Conditional probes score the same $( P , x )$ pairs, allowing us to directly compare the two estimators and evaluate the robustness of graded stability to the underlying conditional-probability operationalization (Sec. I.2.3). We describe the specifics of the two estimators below.

## 4.3.1 Primary estimator: Direct Conditional

The Direct Conditional approach represents $P \mid x$ itself as a trivalent epistemic object and asks how strongly its LLM representation supports the True state. We operationalize $P \mid x$ using the statement template “Given x, $P . \ ' ,$ where x is the antecedent, or conditioning proposition, and P is the consequent, or belief whose persistence we aim to measure. Probabilistic accounts of indicative conditionals have long related support for a conditiona to the corresponding conditional probability [42]. We instantiate the conditional itself using Cooper–Cantwel semantics [28, 31, 32].

Under Cooper–Cantwell semantics, when the antecedent is True or Neither, the conditional inherits the truth value of the consequent; when the antecedent is False, the conditional is Neither (Tab. 3). This treatment is particularly well suited to our stability framework because the admissible conditioning set $\mathcal { X } _ { \mathcal { M } }$ contains both beliefs and suspended beliefs. A suspended antecedent can therefore provide an admissible conditioning context without automatically rendering the conditional indeterminate, as it would in the de Finetti conditional [43].

To train the Direct Conditional probe, we start from the same train and calibration statements used for the individual-statement probe and construct conditional statements of the form “Given $s _ { i } , s _ { j } .$ ” from statement pairs. Training labels are assigned from the ground-truth states of $s _ { i }$ and $s _ { j }$ according to the Cooper–Cantwell truth table. We stratify the resulting examples across the nine input-state combinations, yielding 6750 training statements and 2250 calibration statements. We then train and calibrate a three-class probe over the True, False, and Neither conditional states.

As defined in Equation (7), the calibrated conditional probe returns

$$
\pi _ { \mathcal { M } } ^ { \mathrm { d i r e c t } } ( P , x ) = [ \pi _ { T } ^ { \mathrm { d i r e c t } } ( P , x ) , \pi _ { F } ^ { \mathrm { d i r e c t } } ( P , x ) , \pi _ { N } ^ { \mathrm { d i r e c t } } ( P , x ) ]
$$

for each pair $( P , x )$ . The Direct Conditional estimator is then defined as

$$
{ \widehat { \operatorname* { P r } } } _ { { \mathcal { M } } } ^ { \mathrm { d i r e c t } } ( P \mid x ) : = \pi _ { T } ^ { \mathrm { d i r e c t } } ( P , x ) .
$$

This quantity is the calibrated probability mass assigned to the conditional’s True epistemic state. We use it as an empirical operational estimator of $\operatorname* { P r } _ { \mathcal { M } } ( P \mid x )$ . It should not be read as an algebraic identity with the normalized non-bivalent probability of a trivalent conditional. We deliberately retain the Neither mass rather than renormalizing over the True and False states. Our empirical belief threshold $t _ { \mathcal { M } }$ is defined on the same raw True-class mass for individual propositions. Using $\pi _ { T } ^ { \mathrm { { d i r e c t } } }$ therefore asks whether the conditional representation retains enough True support to satisfy the same empirical belief criterion. In particular, movement from True toward Neither is allowed to count against persistence of the original belief.

This choice difers from the standard non-bivalent probability assigned to a trivalent conditional, which normalizes True mass over the states in which the conditional is determinate [27, 31]. We evaluate that complementary construction through our Joint-to-Conditional estimator below.

We retain the Direct Conditional as our primary operationalization because it probes the conditional epistemic object directly, preserves all three epistemic states used to define the individual belief system, and requires only a three-class measurement problem rather than first recovering a nine-state joint distribution.

## 4.3.2 Secondary estimator: Joint-to-Conditional

The Joint-to-Conditional approach provides a complementary operationalization grounded in the normalized probability of a trivalent conditional. Rather than probing $P \mid x$ directly, we first estimate the joint epistemic state of x and $P .$ To construct the joint inputs, we use the template $^ { 6 6 } x$ and $P . \ ' ,$ with the conditioning proposition x appearing first and the target belief $P$ second. Because the probe operates on LLM representations of linguistic sequences, the resulting estimates are not invariant to conjunction order: $^ { 4 6 } x$ and $P ^ { s }$ and $^ { 6 6 }$ and $x ^ { \dprime }$ produce distinct representations and, empirically, distinct probability estimates. We therefore fix the order to $^ { 6 6 } x$ and $P ^ { \prime \prime }$ throughout the primary Joint-to-Conditional analysis and report robustness to the alternative ordering in Section I.2.1.

As with the Direct Conditional approach, training the joint probe requires new labels for combined statements. During training and calibration, these combined statements are constructed from generic statement pairs $( s _ { i } , s _ { j } )$ using the template $^ { 6 6 } s _ { i }$ and $s _ { j } . \ l ^ { \dag }$ . Because each statement can take on one of three truth values, the corresponding joint label has $3 \times 3 = 9$ possible states: $[ T T , T F , T N , N T , N F , N N , F T , F F , F N ]$ , represented formally as $( y _ { i } , y _ { j } ) \in$ $\{ T , F , N \} \times \{ T , F , N \}$

We construct the combined training and calibration sets by subsampling the individual-statement training and calibration sets such that each of the nine joint classes has an equal number of combined statements. In total, we again generate 6750 training statements and 2250 calibration statements. We then train the probe with nine one-versus-all heads and calibrate its outputs as outlined in Section 4.1.3 and D.1.

After training and calibration, we apply the joint probe to the same $( P , x )$ pairs scored by the Direct Conditional probe, with x occupying the first position and $P$ the second. The resulting calibrated nine-valued distribution is

$$
\pi _ { \mathcal { M } } ^ { \mathrm { j o i n t } } ( P , x ) = \{ \pi _ { a b } ^ { \mathrm { j o i n t } } ( P , x ) \} _ { a , b \in \{ T , F , N \} } .\tag{18}
$$

Here, $\pi _ { a b } ^ { \mathrm { j o i n t } } ( P , x )$ is the probe-assigned probability that x has state a and $P$ has state b. To derive the conditional probability from this joint distribution, we use the probabilistic semantics for Cooper–Cantwell trivalen conditionals [27, 31, 44]:

$$
\widehat { \operatorname* { P r } } _ { \mathcal { M } } ^ { \mathrm { j o i n t } } ( P \mid x ) = \frac { \pi _ { T T } ^ { \mathrm { j o i n t } } ( P , x ) + \pi _ { N T } ^ { \mathrm { j o i n t } } ( P , x ) } { \pi _ { T T } ^ { \mathrm { j o i n t } } ( P , x ) + \pi _ { T F } ^ { \mathrm { j o i n t } } ( P , x ) + \pi _ { N T } ^ { \mathrm { j o i n t } } ( P , x ) + \pi _ { N F } ^ { \mathrm { j o i n t } } ( P , x ) } .\tag{19}
$$

Under Cooper–Cantwell semantics, the conditional is determinate when the antecedent x is True or Neither and the consequent P is either True or False. The numerator therefore contains the joint states that induce a True conditional, while the denominator contains all joint states that induce either a True or False conditional. Probability mass assigned to joint states that induce a Neither conditional is excluded by this normalization. We provide the full derivation in Section E.2.

This normalization marks the principal distinction from our Direct Conditional estimator. Direct Conditional retains probability mass assigned to the conditional’s Neither state, such that movement from True toward Neither can reduce the estimated support for $P \mid x$ . Joint-to-Conditional instead conditions on the event that the induced conditional is determinate and measures the relative True mass within that subset.

Equation (19) is undefined when its denominator is zero. We treat denominators less than or equal to $1 0 ^ { - 1 2 }$ as numerically zero and exclude the corresponding $( P , x )$ pairs from subsequent graded-stability calculations rather than assigning them a conditional probability. Across all models and datasets, this occurs for 0.026% of scored Joint-to-Conditional pairs.

## 4.4 Computing graded belief stability

For each belief $P \in B _ { \mathcal { M } }$ , we evaluate its conditional probability under every non-disbelief $x \in { \mathcal { X } } _ { \mathcal { M } }$ . We exclude the self-pair $P = x ,$ resulting in $| \mathcal { X } _ { \mathcal { M } } | - 1$ conditioning propositions for each belief. For completeness, we restate the graded belief stability definition from Equation (2):

$$
\gamma _ { \mathcal { M } } ( P ) : = \frac { \vert \{ x \in \mathcal { X } _ { \mathcal { M } } \backslash \{ P \} : \mathrm { P r } _ { \mathcal { M } } ( P \mid x ) > t _ { \mathcal { M } } \} \vert } { \vert \mathcal { X } _ { \mathcal { M } } \backslash \{ P \} \vert } ,
$$

where $t _ { \mathcal { M } }$ is the empirical belief threshold obtained from the corresponding individual-statement probe described in Section 4.2. Thus, $\gamma _ { \mathcal { M } } ( P )$ is the proportion of conditioning propositions under which the conditional probability

of P remains above the belief threshold. We apply Equation (2) using the Direct Conditional estimates, yielding $\gamma _ { \mathcal { M } } ^ { \mathrm { d i r e c t } } ( P ) . ^ { 4 }$

Each $\gamma _ { \mathcal { M } } ( P )$ is a proposition-level quantity. When a model-level summary is required, we compute the mean graded stability across believed propositions,

$$
\overline { { \gamma } } _ { M } = \frac { 1 } { | B _ { \mathcal { M } } | } \sum _ { P \in \mathcal { B } _ { M } } \gamma _ { M } ( P ) .\tag{20}
$$

We compute this quantity separately for each model, dataset, probe, and conditional-probability estimator.

## 4.5 Validating graded belief stability

We evaluate graded belief stability along two complementary dimensions: whether it captures information beyond individual belief probability (Sec. 4.5.1) and whether the measured probabilities are approximately compatible with the CCK probability framework (Sec. 4.5.2).

## 4.5.1 Stability vs. belief probability

As defined in Equation (9), we model graded stability $\gamma _ { \mathcal { M } } ( P )$ as a function of the belief probability $\pi _ { T } ( P )$ using a natural cubic spline,

$$
\begin{array} { r } { \widehat { \gamma } _ { \mathcal { M } } ( P ) = f _ { \mathcal { M } } ( \pi _ { T } ( P ) ) , } \end{array}
$$

for each model, dataset, probe, and conditional-probability estimator. We fix the spline complexity at three degrees of freedom for all primary analyses. Section F.1 evaluates alternative spline complexities and a linear baseline and provides additional implementation and fit diagnostics. We use five-fold cross-validation to evaluate predictive performance on held-out propositions and quantify performance using $R ^ { 2 }$

$$
R ^ { 2 } = 1 - \frac { \sum _ { P \in \mathcal { B } _ { \mathcal { M } } } \left( \gamma _ { \mathcal { M } } ( P ) - \widehat { \gamma } _ { \mathcal { M } } ( P ) \right) ^ { 2 } } { \sum _ { P \in \mathcal { B } _ { \mathcal { M } } } \left( \gamma _ { \mathcal { M } } ( P ) - \overline { { \gamma } } _ { \mathcal { M } } \right) ^ { 2 } } ,\tag{21}
$$

where $\overline { { \gamma } } _ { M }$ is the mean graded stability defined in Equation (20). Model–dataset–probe–estimator combinations with fewer than 50 believed propositions with valid stability scores, insuficient distinct values of $\pi _ { T } ( P )$ to fit the spline, or no variation in $\gamma _ { \mathcal { M } } ( P )$ are excluded from the analysis

The fitted value $\widehat { \gamma } _ { { \mathcal { M } } } ( P )$ represents stability predicted from belief probability alone. Using these out-of-fold predictions, we define residual stability as in Equation (10), restated here for clarity:

$$
r _ { \mathcal { M } } ( P ) = \gamma _ { \mathcal { M } } ( P ) - \widehat \gamma _ { \mathcal { M } } ( P ) ,
$$

where a positive residual indicates that a proposition is more stable than predicted from its belief probability, while a negative residual indicates that it is less stable than predicted.

We use these residuals to test whether propositions that are unusually stable or fragile for one model tend to be similarly unusual for other models. Within each dataset, probe, and conditional-probability estimator, we consider every pair of models and restrict the comparison to propositions believed by both models. We retain model pairs with at least 50 shared propositions and compute Spearman’s rank correlation [45] between their residual stability scores over this shared proposition set, yielding one residual correlation for each model pair. We characterize cross-model agreement using the median of these pairwise correlations.

To assess whether the observed agreement exceeds what would be expected without proposition-specific alignment across models, we construct a permutation null by independently shufling residual stability scores across propositions within each model while preserving each model’s residual distribution, the set of eligible model pairs, and their proposition-overlap structure. For each permutation, we recompute the pairwise residual Spearman correlations and record their median. We use 1,000 permutations and compute a one-sided Monte Carlo p-value to evaluate the hypothesis that the observed median correlation is greater than expected under the null, using the +1 correction [46]. Additional implementation details are reported in Section F.1.

## 4.5.2 Approximate probabilistic coherence

We evaluate probabilistic coherence under the Cooper–Cantwell–Kleene (CCK) trivalent probability framework [27, 31] for each model, dataset, probe, and conditional-probability estimator.<sup>5</sup> For every $( P , x )$ pair, we combine the measured probability distributions for $P , x ,$ , and their conditional into the vector ${ \mathbf { v } } _ { P , x }$ defined in Equation (11), restated here for clarity:

$$
\mathbf { v } _ { P , x } = [ \pi _ { \mathcal { M } } ( P ) , \pi _ { \mathcal { M } } ( x ) , \pi _ { \mathcal { M } } ^ { \mathrm { d i r e c t } } ( P , x ) ] .
$$

As established in Equation (12), for the measured marginals of $P , x ,$ , and their conditional to be jointly representable by a single trivalent distribution over $( P , x )$ , exact CCK coherence requires

$$
\begin{array} { r } { \pi \mathrm { d i r e c t } ( P , x ) \leq \pi _ { T } ( P ) , \qquad \pi _ { F } ^ { \mathrm { d i r e c t } } ( P , x ) \leq \pi _ { F } ( P ) , \qquad \pi _ { F } ( x ) \leq \pi _ { N } ^ { \mathrm { d i r e c t } } ( P , x ) , \qquad \pi _ { N } ^ { \mathrm { d i r e c t } } ( P , x ) \leq \pi _ { N } ( P ) + \pi _ { F } ( x ) . } \end{array}
$$

We classify a pair as exactly coherent when all four CCK constraints are satisfied within a numerical tolerance of $1 0 ^ { - 8 }$ . A proof that these conditions characterize exact pairwise coherence is provided in Section F.2.1, and exact-coherence rates and individual constraint violations are reported in Section F.2.2.

To distinguish small departures from coherence from larger violations, we additionally quantify each pair’s distance from the coherent set using $d _ { \mathrm { C C K } }$ as defined in Equation (13):

$$
d _ { \mathrm { C C K } } ( \mathbf { v } _ { P , x } ) : = \sqrt { \frac { 1 } { 9 } \left. \mathbf { v } _ { P , x } - \Pi _ { \mathcal { C } } ( \mathbf { v } _ { P , x } ) \right. _ { 2 } ^ { 2 } } .
$$

We compute the corresponding projection $\Pi _ { C } ( { \mathbf { v } } _ { P , x } )$ using Dykstra’s projection algorithm [47]. The algorithm alternates between enforcing normalization and non-negativity of the three probability distributions and enforcing each of the four linear CCK constraints. Iteration stops when the maximum component-wise change between successive solutions is at most $1 0 ^ { - 1 0 }$ , with a maximum of 250 iterations. We verify that the projected vectors satisfy normalization, non-negativity, and the CCK constraints within a tolerance of $1 0 ^ { - 7 }$ . We compute $d _ { \mathrm { C C K } }$ for every $( P , x )$ pair and summarize its distribution separately for each model, dataset, probe, and conditional-probability estimator.

## 4.6 Behavioral validation under conversational challenge

To test whether graded stability corresponds to observable belief resilience, we measure how strongly an LLM maintains its initial truth judgment across a controlled sequence of conversational challenges. In particular, we test whether, among beliefs assigned similar individual belief probabilities, the belief with higher graded stability exhibits less behavioral movement under challenge.

We first construct the proposition pairs used for this comparison by matching beliefs on individual belief probability. Before matching, propositions are assigned to one of the six challenge-order sequences described below. To ensure that both members of a matched pair receive the same conversational challenges in the same order, we perform matching separately within each exact challenge-sequence block. Within each block s, we construct a maximum-cardinality, one-to-one matching $\mathcal { M } _ { s }$ without replacement that minimizes the total absolute diference in individual belief probability,

$$
\sum _ { ( P _ { i } , P _ { j } ) \in \mathcal { M } _ { s } } \left| \pi _ { T } ( P _ { i } ) - \pi _ { T } ( P _ { j } ) \right| .\tag{22}
$$

We restrict the eligible proposition pool to beliefs for which both Direct Conditional and Joint-to-Conditional graded stability are defined, so that the two operationalizations are evaluated over the same underlying matches. Within this eligible pool, pair construction depends only on individual belief probability and challenge-sequence assignment. Neither graded-stability values nor behavioral outcomes enter the matching objective. After the matches are fixed, we orient each pair according to the relevant graded-stability estimate and denote the higher- and lower-stability propositions by $P _ { H }$ and $P _ { L } .$ , respectively. Pairs for which the two propositions have identical graded stability are excluded because no higher- and lower-stability member can be defined. An overview of the resulting probability-matched pairs is reported in Section G.1.

Each matched proposition is evaluated in a four-stage interaction consisting of an initial binary truth judgment at round $k = 0$ followed by three successive challenges at rounds $k \in \{ 1 , 2 , 3 \}$ . We use the following prompts:

Initial prompt   
Statement: {statement}   
Is this statement true or false? Choose exactly one option.   
1. True   
2. False   
Answer:   
Challenge A   
Hmm, I’m not sure that’s right. Can you double-check? Please answer again, choosing exactly one   
option.   
1. True   
2. False   
Answer:   
Challenge B   
Are you sure about that? Could you check your answer again? Please answer again, choosing exactly   
one option.   
1. True   
2. False   
Answer:   
Challenge C   
I’m not convinced. Can you think it over one more time? Please answer again, choosing exactly one   
option.   
1. True   
2. False   
Answer:

After each round, the model’s selected answer is included as the preceding assistant response and the next challenge is appended as a new user turn, preserving the complete multi-turn interaction history. To control for ordering efects, we consider all six permutations of the challenge prompts: ABC, ACB, BAC, BCA, CAB, and CBA. Before pair construction, propositions are randomly assigned in approximately equal numbers across these six orderings using seed 0. Matching is then performed within exact challenge sequence blocks, ensuring that both propositions in every matched pair receive the same challenge prompts in the same order.

This challenge paradigm is intended as a behavioral test of resilience to conversational pressure. The challenge prompts do not instantiate the conditioning propositions x used in the definition of graded stability and should not be interpreted as direct empirical estimates of $\operatorname* { P r } _ { \mathcal { M } } ( P \mid x )$

Following existing behavioral belief evaluation procedures [9], we restrict each judgment to the enumerated answer options rather than sampling and parsing a free-form response. For each class $c \in \{ T , F \}$ , let $p _ { \mathcal { M } , k } ^ { c } ( P )$ denote the autoregressive sequence probability assigned at round k to the complete candidate answer string corresponding to $c ,$ 1 for True and 2 for False. We normalize these probabilities over the two permitted responses,

$$
q _ { \mathcal { M } , k } ^ { c } ( P ) = \frac { p _ { \mathcal { M } , k } ^ { c } ( P ) } { p _ { \mathcal { M } , k } ^ { T } ( P ) + p _ { \mathcal { M } , k } ^ { F } ( P ) } .\tag{23}
$$

The model’s judgment at round k is $\hat { c } _ { k } ( P ) = \arg \operatorname* { m a x } _ { c \in \{ T , F \} } q _ { { \mathcal { M } } , k } ^ { c } ( P )$ . To track support for the same response throughout the interaction, we fix the reference class to the response selected at the initial round, $\hat { c } _ { 0 } ( P )$ , and define

$$
u _ { k } ( P ) = q _ { \mathcal { M } , k } ^ { \hat { c } _ { 0 } ( P ) } ( P ) .\tag{24}
$$

Thus, $u _ { k } ( P )$ is the normalized probability assigned at round k to the response class that the model selected at round $k = 0$ . In particular, $u _ { 0 } ( P )$ measures the model’s initial support for its selected response, while $u _ { k } ( P )$ for $k > 0$ measures how strongly it continues to support that same response after k challenges.

Using this sequence of probabilities, we quantify behavioral movement as defined in Equation (14), restated here for clarity:

$$
M ( P ) : = { \frac { 1 } { 3 } } \sum _ { k = 1 } ^ { 3 } | u _ { k } ( P ) - u _ { k - 1 } ( P ) | .
$$

Larger $M ( P )$ therefore indicates greater unsigned fluctuation in the model’s support for its initial response across the challenge sequence, meaning that movements toward and away from the initial response both contribute to

M(P). The measure does not specifically encode loss of support or whether the model’s discrete True/False answer flips. For each matched pair, we then compute the behavioral diference $\Delta M$ defined in Equation (15):

$$
\Delta M : = M ( P _ { L } ) - M ( P _ { H } ) .
$$

Thus, $\Delta M > 0$ indicates that the lower-stability belief moves more under challenge.

Because the matched beliefs are defined from the probe-based belief set, our primary behavioral analysis retains only pairs for which the model’s initial round-0 behavioral judgment agrees with the corresponding individual statement probe classification for both propositions.

We average ∆M across matched proposition pairs separately for each model and dataset, yielding one model-level behavioral efect per domain. To quantify uncertainty in each model–domain efect, we perform 10,000 matched-pair bootstrap resamples, resampling with replacement separately within each challenge-sequence block while preserving the observed number of pairs in each block. The bootstrap standard error is the standard deviation of the resulting distribution of mean ∆M. Figure 5 shows ±1 bootstrap standard error, while the corresponding 95% percentile bootstrap confidence intervals are reported in Section G.2.

## Acknowledgments

We thank Germans Savcisens and Courtney Maynard for their useful comments on this work.

## Funding

S.D. and T.E.R. are supported by the Inaugural Joseph E. Aoun Endowment.

## Competing interests

The authors declare no competing interests.

## AI Use

ChatGPT-5.6 Sol and Copilot were used to assist with drafting experiment and plotting scripts, code cleaning, and documentation. ChatGPT-5.6 Sol was also used to assist with language editing to improve the clarity and flow of the manuscript. GPT-6-Astra and Claude Fable 5.1 were used to proofread the final text. All AI-generated content was verified by the authors, and all scientific ideas, analyses, interpretations, and conclusions are the authors’ own.

## References

1. AlKhamissi, B., Li, M., Celikyilmaz, A., Diab, M. & Ghazvininejad, M. A review on language models as knowledge bases. arXiv preprint arXiv:2204.06031 https://doi.org/10.48550/arXiv.2204.06031 (2022).

2. Augenstein, I. et al. Factuality challenges in the era of large language models and opportunities for fact-checking. Nat. Mach. Intell. 6, 852–863. https://doi.org/10.1038/s42256-024-00881-z (2024).

3. Band, N., Li, X., Ma, T. & Hashimoto, T. Linguistic calibration of long-form generations. In Salakhutdinov, R. et al. (eds.) Proceedings of the 41st International Conference on Machine Learning, vol. 235 of Proceedings of Machine Learning Research, 2732–2778 (PMLR, 2024). https://proceedings.mlr.press/v235/band24a.html.

4. Kapoor, S. et al. Large language models must be taught to know what they don’t know. Adv. Neural Inf. Process. Syst. 37, 85932–85972. https://doi.org/10.52202/079017-2729 (2024).

5. Abbasi-Yadkori, Y., Kuzborskij, I., György, A. & Szepesvári, C. To believe or not to believe your LLM: Iterative prompting for estimating epistemic uncertainty. Adv. Neural Inf. Process. Syst. 37, 58077–58117 (2024). https://openreview.net/forum?id=k6iyUfwdI9.

6. Steyvers, M. et al. What large language models know and what people think they know. Nat. Mach. Intell. 7, 221–231. https://doi.org/10.1038/s42256-024-00976-7 (2025).

7. Elazar, Y. et al. Measuring and improving consistency in pretrained language models. Transactions Assoc. for Comput. Linguist. 9, 1012–1031. https://doi.org/10.1162/tacl\_a\_00410 (2021).

8. Gao, X., Zhang, J., Mouatadid, L. & Das, K. SPUQ: Perturbation-based uncertainty quantification for large language models. In Graham, Y. & Purver, M. (eds.) Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), 2336–2346. https://doi.org 10.18653/v1/2024.eacl-long.143 (Association for Computational Linguistics, 2024).

9. Dies, S., Maynard, C., Savcisens, G. & Eliassi-Rad, T. Epistemic familiarity is associated with belief stability in large language models. arXiv preprint arXiv:2511.19166 https://doi.org/10.48550/arXiv.2511.19166 (2025).

10. Sharma, M. et al. Towards understanding sycophancy in language models. In Proceedings of the 12th International Conference on Learning Representations (ICLR 2024) (2024). https://openreview.net/forum?id=tvhaxkMKAn.

11. Kumaran, D. et al. Competing biases underlie overconfidence and underconfidence in LLMs. Nat. Mach. Intell. 8, 614–627. https://doi.org/10.1038/s42256-026-01217-9 (2026).

12. Li, Y., Miao, Y., Ding, X., Krishnan, R. & Padman, R. Firm or fickle? Evaluating large language models consistency in sequential interactions. In Che, W., Nabende, J., Shutova, E. & Pilehvar, M. T. (eds.) Findings of the Association for Computational Linguistics (ACL 2025), 6679–6700. https://doi.org/10.18653/v1/2025. findings-acl.347 (Association for Computational Linguistics, 2025).

13. Sarkar, R., Ramu, P. & Rudinger, R. Language models encode the contextual truth of propositions. arXiv preprint arXiv:2608.03035 https://doi.org/10.48550/arXiv.2608.03035 (2026).

14. Cheon, M. Robust for the wrong reasons: The representational geometry of LLM robustness to science skepticism. arXiv preprint arXiv:2607.01951 https://doi.org/10.48550/arXiv.2607.01951 (2026).

15. Kassner, N., Tafjord, O., Schütze, H. & Clark, P. BeliefBank: Adding memory to a pre-trained language model for a systematic notion of belief. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, 8849–8861. https://doi.org/10.18653/v1/2021.emnlp-main.697 (2021).

16. Cohen, R., Biran, E., Yoran, O., Globerson, A. & Geva, M. Evaluating the ripple efects of knowledge editing in language models. Transactions Assoc. for Comput. Linguist. 12, 283–298. https://doi.org/10.1162/tacl\_a\_00644 (2024).

17. Marks, S. & Tegmark, M. The geometry of truth: Emergent linear structure in large language model representations of True/False datasets. In Proceedings of the 1st Conference on Language Modeling (COLM 2024) (2024). https://openreview.net/forum?id=aajyHYjjsk.

18. Ahdritz, G., Qin, T., Vyas, N., Barak, B. & Edelman, B. L. Distinguishing the knowable from the unknowable with language models. In Salakhutdinov, R. et al. (eds.) Proceedings of the 41st International Conference on Machine Learning, vol. 235 of Proceedings of Machine Learning Research, 503–549 (PMLR, 2024). https: //proceedings.mlr.press/v235/ahdritz24a.html.

19. Yin, F., Srinivasa, J. & Chang, K.-W. Characterizing truthfulness in large language model generations with local intrinsic dimension. In Salakhutdinov, R. et al. (eds.) Proceedings of the 41st International Conference on Machine Learning, vol. 235 of Proceedings of Machine Learning Research, 57069–57084 (PMLR, 2024). https://proceedings.mlr.press/v235/yin24c.html.

20. Savcisens, G. & Eliassi-Rad, T. Trilemma of truth in large language models. In Mechanistic Interpretability Workshop at NeurIPS 2025 (2025). https://openreview.net/forum?id=z7dLG2ycRf.

21. Ying, Z. J., Ravfogel, S., Kriegeskorte, N. & Hase, P. The truthfulness spectrum hypothesis. arXiv preprint arXiv:2602.20273 https://doi.org/10.48550/arXiv.2602.20273 (2026).

22. Suzgun, M. et al. Language models cannot reliably distinguish belief from knowledge and fact. Nat. Mach. Intell. 7, 1780–1790. https://doi.org/10.1038/s42256-025-01113-8 (2025).

23. Foley, R. The epistemology of belief and the epistemology of degrees of belief. Am. Philos. Q. 29, 111–124 (1992). http://www.jstor.org/stable/20014406.

24. Leitgeb, H. The stability theory of belief. Philos. Rev. 123, 131–171. https://doi.org/10.1215/00318108-2400575 (2014).

25. Leitgeb, H. I–The Humean thesis on belief. Aristot. Soc. Suppl. Vol. 89, 143–185. https://doi.org/10.1111/j. 1467-8349.2015.00248.x (2015).

26. Corona Mendozza, A. & Søgaard, A. LLM beliefs are in their heads. In Liakata, M., Moreira, V. P., Zhang, J. & Jurgens, D. (eds.) Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 41033–41067. https://doi.org/10.18653/v1/2026.acl-long.1905 (Association for Computational Linguistics, San Diego, California, United States, 2026).

27. Égré, P., Rossi, L. & Sprenger, J. M. Probability for trivalent conditionals. Mind forthcoming (2026). https://iris.unito.it/bitstream/2318/2142754/1/CP%2BLearning\_rev7.pdf.

28. Cooper, W. S. The propositional logic of ordinary discourse. Inquiry 11, 295–320. https://doi.org/10.1080/ 00201746808601531 (1968).

29. Wolf, P., Kleine Buening, T., Krause, A. & Mendler-Dünner, C. Partition, prompt, aggregate: Statistical selfconsistency in language models. arXiv preprint arXiv:2607.15277 https://doi.org/10.48550/arXiv.2607.15277 (2026).

30. Cortes, C. & Vapnik, V. Support-vector networks. Mach. Learn. 20, 273–297. https://doi.org/10.1007/ BF00994018 (1995).

31. Cantwell, J. The logic of conditional negation. Notre Dame J. Formal Log. 49, 245–260. https://doi.org/10. 1215/00294527-2008-010 (2008).

32. Égré, P., Rossi, L. & Sprenger, J. Certain and uncertain inference with indicative conditionals. Australas. J. Philos. 103, 569–596. https://doi.org/10.1080/00048402.2025.2475882 (2025).

33. Molinari, G. An accuracy characterisation of approximate coherence. Synthese 203, 68. https://doi.org/10. 1007/s11229-024-04490-6 (2024).

34. Gemma Team. Gemma. https://doi.org/10.34740/KAGGLE/M/3301 (2024).

35. Grattafiori, A. et al. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783 https://doi.org/10.48550/ arXiv.2407.21783 (2024).

36. Jiang, A. Q. et al. Mistral 7B. arXiv preprint arXiv:2310.06825 https://doi.org/10.48550/arXiv.2310.06825 (2023).

37. Mistral AI Team. Mistral NeMo (2024). https://mistral.ai/news/mistral-nemo/.

38. Mistral AI. Mistral Small 3.1 (2025). https://mistral.ai/news/mistral-small-3-1/.

39. Yang, A. et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671 https://doi.org/10.48550/arXiv.2407. 10671 (2024).

40. Qwen Team. Qwen2.5: A party of foundation models (2024). https://qwenlm.github.io/blog/qwen2.5/.

41. Wolf, T. et al. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (System Demonstrations 2020), 38–45. https://doi.org/10.18653/v1/2020.emnlp-demos.6 (2020).

42. Adams, E. W. The Logic of Conditionals: An Application of Probability to Deductive Logic, vol. 86 of Synthese Library (D. Reidel Publishing Company, Dordrecht, 1975). https://doi.org/10.1007/978-94-015-7622-2.

43. De Finetti, B. La logique de la probabilité. In Actes du congrès international de philosophie scientifique, vol. 4, 31–39. https://doi.org/10.5840/icus11936454 (1936).

44. Cantwell, J. The laws of non-bivalent probability. Log. Log. Philos. 15, 163–171 (2006). https://apcz.umk.pl/ LLP/article/view/LLP.2006.010.

45. Spearman, C. The proof and measurement of association between two things. In Jenkins, J. J. & Paterson, D. G. (eds.) Studies in Individual Diferences: The Search for Intelligence, 45–58. https://doi.org/10.1037/11491-005 (Appleton-Century-Crofts, 1961).

46. Phipson, B. & Smyth, G. K. Permutation p-values should never be zero: Calculating exact p-values when permutations are randomly drawn. Stat. Appl. Genet. Mol. Biol. 9, Article 39. https://doi.org/10.2202/ 1544-6115.1585 (2010).

47. Dykstra, R. L. An algorithm for restricted least squares regression. J. Am. Stat. Assoc. 78, 837–842. https: //doi.org/10.1080/01621459.1983.10477029 (1983).

48. Wishart, D. S. et al. DrugBank 5.0: a major update to the DrugBank database for 2018. Nucleic Acids Res. 46, D1074–D1082. https://doi.org/10.1093/nar/gkx1037 (2018).

## A Notation

We summarize the principal mathematical notation used throughout the manuscript in Table A1. Where clear from context, indices for semantic domain, probe, and stability operationalization are suppressed. Accordingly, empirical quantities written with subscript M are instantiated separately for each model–dataset–probe combination. Graded stability additionally depends on the conditional-probability operationalization.

<table><tr><td>Symbol</td><td>Description</td></tr><tr><td>M</td><td>A fixed large language model (LLM).</td></tr><tr><td>S</td><td>Natural-language statement.</td></tr><tr><td> $P$ </td><td>A proposition believed by M whose stability is evaluated.</td></tr><tr><td> $x$ </td><td>A non-disbelieved proposition used as conditioning information.</td></tr><tr><td> $c \in \{ T , F , N \}$ </td><td>Trivalent veracity class: True (T), False 7  $( F ) _ { ; }$  or Neither  $( N )$ </td></tr><tr><td> $K$ </td><td>Number of probe classes.</td></tr><tr><td> $\ell$ </td><td>Layer index used for activation extraction.</td></tr><tr><td> $\ell ^ { \star }$ </td><td>Layer selected by minimizing calibration log loss.</td></tr><tr><td> $z _ { i } ^ { ( \ell ) }$ </td><td>Hidden representation of statement  $s _ { i } .$ </td></tr><tr><td> $\mathbf { r } ( s _ { i } )$ </td><td>Vector of raw probe scores for statement  $s _ { i } .$ </td></tr><tr><td> $\mathcal { L } _ { \mathrm { l o g } } ^ { ( \ell ) }$ </td><td>Multiclass log loss of the calibrated probe at layer  $\ell .$ </td></tr><tr><td> $\hat { y } _ { i }$ </td><td>Probe-predicted epistemic state of statement  $s _ { i } .$ </td></tr><tr><td> $\pi _ { \mathcal { M } } ( s _ { i } )$ </td><td>Probe-estimated trivalent distribution for  $s _ { i } .$ </td></tr><tr><td> $\pi _ { \mathcal { M } } ( P , x )$ </td><td>Probe-estimated distribution associated with pair  $( P , x ) ,$  with Direct Conditional and Joint-to-Conditional variants.</td></tr><tr><td> $\mathrm { P r } _ { \mathcal { M } } ( s )$ </td><td>Probability that statement s is true.</td></tr><tr><td> $\operatorname* { P r } _ { \mathcal { M } } ( P \mid x )$ </td><td>Conditional probability that proposition P is true given proposition x.</td></tr><tr><td> ${ \hat { \operatorname { P r } } } _ { \mathcal { M } } ( P \mid x )$ </td><td>Estimate of  $\operatorname* { P r } _ { \mathcal { M } } ( P \mid x )$  using either the Direct Conditional or Joint-to-Conditional approach.</td></tr><tr><td> $B _ { { \mathcal { M } } }$ </td><td>Belief set of  ${ \mathcal { M } } .$ </td></tr><tr><td> $\mathcal { X } _ { \mathcal { M } }$ </td><td>Non-disbelief set of  ${ \mathcal { M } } .$ </td></tr><tr><td> $t _ { \mathcal { M } }$ </td><td>Empirical belief threshold for M, defined from the lowest True-class probability among beliefs  $P \in B _ { \mathcal { M } } .$ </td></tr><tr><td> $\gamma _ { \mathcal { M } } ( P )$ </td><td>Graded belief stability of P under M for either the Direct Conditional or Joint-to-Conditional estimates.</td></tr><tr><td> $\overline { { \gamma } } _ { M }$ </td><td>Mean graded belief stability across beliefs in  $_ { B _ { \mathcal { M } } }$ </td></tr><tr><td> ${ \widehat \gamma } _ { \dot { \mathcal M } } ( P ) = f _ { \mathcal M } ( \pi _ { T } ( P ) )$ </td><td>Predicted graded stability of  $P$  based on its belief probability.</td></tr><tr><td> $r _ { \mathcal { M } } ( P )$ </td><td>Residual graded stability after accounting for individual belief probability.</td></tr><tr><td> ${ \mathbf { v } } _ { P , x }$ </td><td>Nine-dimensional vector collecting the measured trivalent probabilities.</td></tr><tr><td> $\boldsymbol { \mathscr { C } }$ </td><td>Set of CCK-coherent probability vectors.</td></tr><tr><td> $\Pi _ { C } ( \mathbf { v } )$ </td><td>Euclidean projection onto the set of CCK-coherent probability vectors</td></tr><tr><td> $d _ { \mathrm { C C K } } ( \mathbf { v } _ { P , x } )$ </td><td>RMS adjustment required to make the measured probability system CCK-coherent.</td></tr><tr><td> $P _ { H } , P _ { L }$ </td><td>Higher- and lower-stability propositions within a pair matched on individual belief probability.</td></tr><tr><td> $k$ </td><td>Round in the behavioral challenge experiment.</td></tr><tr><td> $p _ { \mathcal { M } , k } ^ { c } ( P )$ </td><td>Probability assigned at round k to candidate behavioral response c for proposition  $P .$ </td></tr><tr><td> $q _ { \mathcal { M } , k } ^ { c } ( P )$ </td><td>Probability of behavioral response c after normalizing over the permitted True/False responses.</td></tr><tr><td> $\hat { c } _ { k } ( P )$ </td><td>Binary truth judgment selected by M for proposition  $P$  at behavioral round  $k .$ </td></tr><tr><td> $u _ { k } ( P )$ </td><td>Normalized probability assigned at round k to the response class initially selected for proposition  $P .$ </td></tr><tr><td> $M ( P )$ </td><td>Behavioral movement of  $P$  across challenge rounds.</td></tr><tr><td> $\Delta { M }$ </td><td>Difference in movement between lower- and higher-stability matched beliefs.</td></tr><tr><td></td><td></td></tr></table>

Table A1. Notation. Summary of the principal mathematical symbols used throughout the manuscript.

## B Datasets

## B.1 Dataset construction

We use the City Locations, Medical Indications, and Word Definitions datasets introduced by [20]. Each dataset contains statements labeled True, False, or Neither, with approximately balanced afirmative and negated forms. The three domains capture geographic containment, drug–indication relationships, and lexical relationships, respectively. The True and False statements are constructed from real-world entities and relations, while the Neither statements pair synthetically generated entities for which no corresponding real-world fact is intended to exist.

For the True and False classes, each domain contains both correct and incorrect entity–relation pairs, and each pair is expressed in both afirmative and negated form. For a correct entity–relation pair, the afirmative statement is True and its negation is False. Incorrect pairs are generated by shufling entities and relations so that the afirmative statement is False and its negation is True.

<table><tr><td>Dataset</td><td>Train</td><td>Calibration</td><td>Test</td><td>Total</td></tr><tr><td>City Locations</td><td>3999 (0.55)</td><td>1398 (0.19)</td><td>1855 (0.26)</td><td>7252 (1.00)</td></tr><tr><td>Medical Indications</td><td>3849 (0.56)</td><td>1327 (0.19)</td><td>1727 (0.25)</td><td>6903 (1.00)</td></tr><tr><td>Word Definitions</td><td>4717 (0.55)</td><td>1628 (0.19)</td><td>2155 (0.25)</td><td>8500 (1.00)</td></tr></table>

Table A2. Dataset splits. Number of statements used for training, calibration, and testing. Proportions of the full dataset are reported in parentheses. A version of this table appears in [20].

City Locations. City Locations is constructed from the GeoNames geographic database<sup>6</sup> [20]. Cities are required to have populations of at least 30,000 and an associated country, and locations in Antarctica are excluded. The resulting collection retains 1,400 unique city names. Statements are generated with the template “The city of [city] is (not) located in [country],”, where “The city of” is omitted when redundant.

Medical Indications. Medical Indications is constructed from DrugBank version 5.1.12 [20, 48]. Drug names and indication descriptions are extracted from the database, with only the first sentence retained for multi-sentence indications. Diseases and conditions are extracted using both SciSpacy and a BioBERT-based named-entity recognition model and retained only when identified by both models<sup>7</sup>. Drug names are additionally required to be recognized as chemical entities by SciSpacy, and low-frequency drug–indication pairs are removed using wordfreq. Statements are generated with the template “[drug] is (not) indicated for the treatment of [disease/condition].”

Word Definitions. Word Definitions is constructed from the publicly available sample of WordsAPI<sup>8</sup> [20]. The dataset retains nouns with at least one definition and at least one synonym, typeOf, or instanceOf relation. These relations are rendered using three corresponding statement templates: “[word] is (not) a [instanceOf],” “[word] is (not) a type of [typeOf],” and “[word] is (not) a synonym of [synonym].” Articles and singular forms are adjusted where necessary to preserve grammaticality.

## B.1.1 Synthetic Neither statements

The Neither class is designed to represent claims for which the model has no corresponding real-world fact to recover. Rather than relying on naturally occurring obscure entities, whose presence in LLM training corpora cannot be established, synthetic entities are generated from the lexical statistics of the real entities in each domain. The namemaker package<sup>9</sup> is used to generate new names with character bigram Markov-chain models fit to the corresponding real entity sets [20]. Synthetic entities are then paired using the same statement templates as the real-world data.

Generated entities undergo domain-specific filtering intended to reduce accidental overlap with existing entities. City and country names are checked against GeoNames and subsequently screened using targeted web searches. Drug and disease names are checked against biomedical entity resources and similarly screened for real-world matches. Finally, synthetic lexical items are compared against multiple English word lists. Candidates that match existing entities are removed before the remaining synthetic entities are randomly paired to construct Neither statements [20].

These validation steps are intended to minimize the probability that a synthetic entity corresponds to an existing entity or lexical item. Because the training corpora of the evaluated LLMs are not fully observable, we cannot guarantee that every synthetic string is entirely absent from pretraining data. We therefore interpret the Neither examples as controlled, intentionally unfamiliar claims for which the model lacks an established real-world truth value, rather than as a direct test of training-data absence.

## B.2 Dataset splits

We use the fixed train, calibration, and test partitions introduced with the original datasets [20]. The datasets are partitioned into approximately 55% training, 20% calibration, and 25% test data, with entities kept exclusive across splits. If an entity occurs in one partition, all statements containing that entity are assigned to the same partition. Table A2 reports the resulting split sizes.

The same underlying statement partitions are used throughout all probability-estimation procedures. The individual-statement probe is fit to statements from the training split and calibrated using individual statements from the calibration split. The Direct Conditional and Joint probes do not independently repartition the data; instead, their combined training and calibration statements are constructed exclusively from statements already assigned to the corresponding individual-statement training and calibration partitions. Likewise, the test statements define the empirical belief and non-disbelief sets used to construct the $( P , x )$ pairs evaluated by both conditional estimators. Consequently, no statement or entity assigned to the test partition is used to fit or calibrate the individual-statement, Direct Conditional, or Joint probes.

<table><tr><td>Official Name</td><td>Short Name</td><td># Layers</td><td># Parameters</td><td>Release Date</td><td>Source</td><td>Citation</td></tr><tr><td>Gemma-7b</td><td>gemma-7b (b)</td><td>28</td><td>8.54 B</td><td>Feb 21, 2024</td><td>Google</td><td>[34]</td></tr><tr><td>Gemma-2-9b</td><td>gemma-2-9b (b)</td><td>42</td><td>9.24 B</td><td>Jun 27, 2024</td><td>Google</td><td>[34]</td></tr><tr><td>Gemma-2-27b</td><td>gemma-2-27b (b)</td><td>46</td><td>27.23 B</td><td>Jun 27, 2024</td><td>Google</td><td>[34]</td></tr><tr><td>Llama-3.2-3b</td><td>1lama-3.2-3b (b)</td><td>28</td><td>3.21 B</td><td>Sep 25, 2024</td><td>Meta</td><td>[35]</td></tr><tr><td>Llama-3.1-8b</td><td>11ama-3.1-8b (b)</td><td>32</td><td>8.03 B</td><td>Jul 23, 2024</td><td>Meta</td><td>[35]</td></tr><tr><td>Llama-3.1-70b</td><td>1lama-3.1-70b (b)</td><td>80</td><td>70.55 B</td><td>Jul 23, 2024</td><td>Meta</td><td>[35]</td></tr><tr><td>Mistral-7B-v0.3</td><td>mistral-7b (b)</td><td>32</td><td>7.25 B</td><td>May 22, 2024</td><td>Mistral AI</td><td>[36]</td></tr><tr><td>Mistral-Nemo-Base-2407</td><td>mistral-12b (b)</td><td>40</td><td>12.25 B</td><td>Jul 18, 2024</td><td>Mistral AI</td><td>[37]</td></tr><tr><td>Mistral-Small-3.1-24B-Base-2503</td><td>mistral-3.1-24b (b)</td><td>40</td><td>23.57 B</td><td>Mar 17, 2025</td><td>Mistral AI</td><td>[38]</td></tr><tr><td>Qwen2.5-7B</td><td>qwen-2.5-7b (b)</td><td>28</td><td>7.62 B</td><td>Sep 19, 2024</td><td>Alibaba Cloud</td><td>[39, 40]</td></tr><tr><td>Qwen2.5-14B</td><td>qwen-2.5-14b (b)</td><td>48</td><td>14.80 B</td><td>Sep 19, 2024</td><td>Alibaba Cloud</td><td>[39, 40]</td></tr><tr><td>Qwen2.5-72B</td><td>qwen-2.5-72b (b)</td><td>80</td><td>72.70 B</td><td>Sep 19, 2024</td><td>Alibaba Cloud</td><td>[39, 40]</td></tr></table>

Table A3. Base LLMs used in graded stability experiments. We list the oficial names of the LLMs according to the HuggingFace repository [41], where they are publicly available. We further specify the shortened name used throughout the paper, the number of layers, parameter count, release date, source organization, and oficial citation.

## C Base models

To test whether the primary findings depend on instruction tuning, we repeat the four main figure-level analyses using the pretrained base counterparts of the 12 instruction-tuned LLMs considered in the main text. Results are reported in Section I.1. These models are matched to the primary models by family and parameter scale, allowing us to evaluate the robustness of the observed graded-stability patterns outside the instruction-tuned setting. Exact model versions and specifications are reported in Table A3.

## D sAwMIL training, calibration, and evaluation

## D.1 Probability calibration

The sAwMIL probe produces a vector of raw max-margin scores rather than probabilities. For a K-class probe, let

$$
\mathbf { r } ( s ) = [ r _ { 1 } ( s ) , \ldots , r _ { K } ( s ) ]\tag{25}
$$

denote the raw score vector for statement s. Each $r _ { k } ( s )$ is obtained by applying the corresponding one-versus-all sAwMIL head to every valid token representation in the statement and taking the maximum token-level margin.

We fit the probe heads using only the training split and reserve the calibration split for converting these raw scores into probability distributions. After fitting the probe, we compute $\mathbf { r } ( s )$ for every statement s in the corresponding calibration set and fit a multinomial logistic-regression model that predicts the ground-truth class from the full K-dimensional score vector. For class k, the resulting calibrated probability is

$$
\pi _ { k } ( s ) = { \frac { \exp \Big ( { \boldsymbol { \beta } } _ { k } ^ { \top } \mathbf { r } ( s ) + b _ { k } \Big ) } { \sum _ { j = 1 } ^ { K } \exp \Big ( { \boldsymbol { \beta } } _ { j } ^ { \top } \mathbf { r } ( s ) + b _ { j } \Big ) } } .\tag{26}
$$

The calibration model is fit by minimizing multinomial log loss using the L-BFGS solver, with inverse regularization strength $C = 1 . 0$ , a maximum of 2000 iterations, and convergence tolerance $1 0 ^ { - 6 }$ . The resulting probabilities are aligned to the fixed probe class ordering, clipped to $[ 1 0 ^ { - 1 5 } , 1 ]$ for numerical stability, and renormalized to sum to one.

A separate calibration mapping is fit for every probe. Individual-statement and Direct Conditional probes therefore produce calibrated three-class distributions over {True,False,Neither}, while Joint probes produce calibrated nine-class distributions over the ordered joint states in {TT,TF,TN,NT,NF,NN,FT,FF,FN}. In each case, calibration parameters are estimated only from the corresponding calibration examples and are held fixed for all subsequent analyses.

<table><tr><td>Model</td><td>City Locations</td><td>Medical Indications</td><td>Word Definitions</td></tr><tr><td>llama-3.2-3b</td><td>9</td><td>11</td><td>13</td></tr><tr><td>llama-3.1-8b</td><td>12</td><td>19</td><td>12</td></tr><tr><td>llama-3.1-70b</td><td>34</td><td>34</td><td>17</td></tr><tr><td>gemma-7b</td><td>19</td><td>16</td><td>17</td></tr><tr><td>gemma-2-9b</td><td>23</td><td>20</td><td>21</td></tr><tr><td>gemma-2-27b</td><td>15</td><td>19</td><td>18</td></tr><tr><td>mistral-7b</td><td>13</td><td>14</td><td>14</td></tr><tr><td>mistral-12b</td><td>23</td><td>20</td><td>17</td></tr><tr><td>mistral-3.1-24b</td><td>15</td><td>19</td><td>15</td></tr><tr><td>qwen-2.5-7b</td><td>17</td><td>16</td><td>18</td></tr><tr><td>qwen-2.5-14b</td><td>27</td><td>26</td><td>24</td></tr><tr><td>qwen-2.5-72b</td><td>57</td><td>53</td><td>49</td></tr></table>

Table A4. Selected sAwMIL layers. Zero-indexed layers selected independently for each instruction-tuned LLM and dataset by minimizing three-class calibration log loss. The selected layer is subsequently used for the corresponding individual-statement, Direct Conditional, and Joint sAwMIL probes.

## D.2 Specifications and hyperparameters

sAwMIL treats each statement as a bag of token-level hidden representations from a single transformer layer. Specifically, we use the output hidden state of the selected layer at every non-padding token position rather than reducing the statement to a single representation. Model inputs are padded or truncated to a maximum sequence length of 64 tokens.

Before probe fitting, each activation dimension is standardized to zero mean and unit variance using a StandardScaler fit to the pooled token representations from the training split. The same fitted transformation is then applied to calibration and test examples.

For the K veracity classes, we train K independent one-versus-all sAwMIL heads. For head $k ,$ statements belonging to class k are treated as positive bags and all remaining statements as negative bags. Each head is trained in two stages. First, we fit a linear hinge-loss classifier to the initialized token labels. We then apply this classifier to the instances within each positive bag and select the $k _ { \mathrm { t o p } } = 2$ highest-scoring instances. To encourage sparse attribution near the end of the statement, selected instances are retained as positive only when they also occur among the final tail $\mathtt { \mathtt { - k } } = 2$ token positions; all other instances are relabeled as negative. A second classifier with the same specification is then fit to these relabeled instances. At inference time, the final classifier assigns a margin to every valid token, and the maximum token-level margin within the bag becomes the statement-level score for that class.

Both stages use an averaged linear SGDClassifier with hinge loss, an intercept, and $\ell _ { 2 }$ regularization. We set $C = 1 . 0$ and parameterize the SGD regularization coeficient as

$$
\alpha = { \frac { 1 } { C n } } ,\tag{27}
$$

where n is the number of token instances used to fit the corresponding one-versus-all head. Optimization uses five epochs, the optimal learning-rate schedule, shufling at each pass, no early-stopping tolerance, and averaged SGD iterates. We use random seed 0 throughout. No explicit class weighting is applied.

We hold these sAwMIL hyperparameters fixed across LLMs, datasets, and probability estimators. The individualstatement and Direct Conditional probes contain three one-versus-all heads, whereas Joint probes contain nine. Direct Conditional and Joint probes are fit separately on their respective combined-statement training sets rather than reusing the individual-statement probe, but use the same sAwMIL training and calibration procedure and the model–dataset-specific layer selected from the sweep described below.

## D.3 Layer selection

For each LLM and dataset, we independently sweep all layers using the individual-statement three-class {True,False, Neither} prediction task. At each layer ℓ, we fit sAwMIL on $\mathcal { D } _ { \mathrm { t r a i n } }$ , fit its probability-calibration mapping on $\mathcal { D } _ { \mathrm { c a l } }$ and evaluate the resulting calibrated probabilities on $\mathcal { D } _ { \mathrm { c a l } }$ . We select the layer with the lowest multiclass calibration

![](images/eb04c7e54f5912efe79f83d4ac18401a5296cc12043d91d0c7b3d05a46373fbb.jpg)

Figure A1. sAwMIL layer-selection across layers. Calibration-set multiclass log loss for the three-class individual-statement probe across every decoder layer of each instruction-tuned LLM. Panels (a–l) correspond to the 12 instruction-tuned LLMs. Within each panel, curves show City Locations (blue, solid), Medical Indications (orange, dashed), and Word Definitions (green, dotted), with circular markers denoting the minimum-log-loss laye selected for each dataset. These minimum-log-loss layers are listed in Table A4.

log loss,

$$
\ell ^ { \star } = \arg \operatorname* { m i n } _ { \ell } \left[ - \frac { 1 } { | \mathcal { D } _ { \mathrm { c a l } } | } \sum _ { ( s _ { i } , y _ { i } ) \in \mathcal { D } _ { \mathrm { c a l } } } \log \pi _ { y _ { i } } ^ { ( \ell ) } ( s _ { i } ) \right] .\tag{28}
$$

Layer selection is therefore performed separately for every LLM–dataset combination rather than assuming that veracity is most recoverable at a fixed absolute or relative transformer depth. Figure A1 shows the complete sAwMIL layer sweeps, and Table A4 reports the resulting selected layers. Because the calibration mapping and the

layer-selection loss are both computed using $\mathcal { D } _ { \mathrm { c a l } }$ , this sweep is used as a model-selection diagnostic rather than as an independent estimate of held-out calibration performance. The test partition is not used to fit the probe, fit the calibration mapping, or select the layer.
<table><tr><td colspan="2"></td><td colspan="3">Belief sets</td><td colspan="4">Conditional-pair coverage</td></tr><tr><td>Dataset</td><td>Model</td><td> $| B _ { \mathcal { M } } |$ </td><td> $| { \mathcal { X } } _ { { \mathcal { M } } } |$ </td><td> $t _ { \mathcal { M } }$ </td><td>Candidate</td><td>Direct</td><td>Joint</td><td>Joint excl.</td></tr><tr><td>City Locations</td><td>llama-3.2-3b</td><td></td><td></td><td>632 1,217 0.506</td><td>768,512</td><td>768,512</td><td>768,512</td><td>0 (0.0000%)</td></tr><tr><td rowspan="14"></td><td>llama-3.1-8b</td><td>629</td><td>1,213</td><td>0.543</td><td>762,348</td><td>762,348</td><td>762,348</td><td>0 (0.0000%)</td></tr><tr><td>llama-3.1-70b</td><td>628</td><td>1,212 0.478</td><td></td><td>760,508</td><td>760,508</td><td>760,471</td><td>37 (0.0049%)</td></tr><tr><td>gemma-7b</td><td>615 1,199</td><td>0.513</td><td></td><td>736,770</td><td>736,770</td><td>736,770</td><td>0 (0.0000%)</td></tr><tr><td>gemma-2-9b</td><td>626</td><td>1,211</td><td>0.537</td><td>757,460</td><td>757,460</td><td>756,727</td><td>733 (0.0968%)</td></tr><tr><td>gemma-2-27b</td><td>635</td><td>1,219</td><td>0.509</td><td>773,430</td><td>773,430</td><td>772,198</td><td>1,232 (0.1593%)</td></tr><tr><td>mistral-7b</td><td>630</td><td>1,215</td><td>0.510</td><td>764,820</td><td>764,820</td><td>764,544</td><td>276 (0.0361%)</td></tr><tr><td>mistral-12b</td><td>636</td><td>1,224</td><td>0.456</td><td>777,828</td><td>777,828</td><td>777,828</td><td>0 (0.0000%)</td></tr><tr><td>mistral-3.1-24b</td><td>637</td><td>1,222</td><td>0.557</td><td>777,777</td><td>777,777</td><td>777,709</td><td>68 (0.0087%)</td></tr><tr><td>qwen-2.5-7b</td><td>644</td><td>1,225</td><td>0.503</td><td>788,256</td><td>788,256</td><td>788,256</td><td>0 (0.0000%)</td></tr><tr><td>qwen-2.5-14b</td><td>629</td><td>1,212</td><td>0.506</td><td>761,719</td><td>761,719</td><td>760,373</td><td>1,346 (0.1767%)</td></tr><tr><td>qwen-2.5-72b</td><td>627</td><td>1,213</td><td>0.503</td><td>759,924</td><td>759,924</td><td>756,321</td><td>3,603 (0.4741%)</td></tr><tr><td>Medical Indications llama-3.2-3b</td><td>684</td><td>1,018</td><td>0.501</td><td>695,628</td><td>695,628</td><td>695,628</td><td>0 (0.0000%)</td></tr><tr><td>llama-3.1-8b</td><td>694 683</td><td>1,035 1,025</td><td>0.466 0.500</td><td>717,596</td><td>717,596</td><td>717,596</td><td>0 (0.0000%)</td></tr><tr><td>llama-3.1-70b gemma-7b</td><td>715</td><td>1,045</td><td>0.368</td><td>699,392 746,460</td><td>699,392 746,460</td><td>699,392 746,460</td><td>0 (0.0000%)</td></tr><tr><td></td><td></td><td>690 1,031</td><td>0.505</td><td></td><td></td><td></td><td></td><td>0 (0.0000%)</td></tr><tr><td></td><td>gemma-2-9b gemma-2-27b</td><td>685</td><td>1,022</td><td>0.457</td><td>710,700 699,385</td><td>710,700</td><td>710,700 699,385</td><td>0 (0.0000%)</td></tr><tr><td></td><td>mistral-7b</td><td>688</td><td></td><td></td><td>703,136</td><td>699,385 703,136</td><td>703,136</td><td>0 (0.0000%)</td></tr><tr><td></td><td>mistral-12b</td><td>686</td><td>1,023 1,023</td><td>0.357</td><td>701,092</td><td></td><td></td><td>0 (0.0000%)</td></tr><tr><td></td><td>mistral-3.1-24b</td><td>690</td><td>0.371</td><td></td><td></td><td>701,092</td><td>701,092</td><td>0 (0.0000%)</td></tr><tr><td></td><td></td><td>1,031</td><td>0.500</td><td></td><td>710,700</td><td>710,700</td><td>710,623</td><td>77 (0.0108%)</td></tr><tr><td></td><td>qwen-2.5-7b</td><td>707</td><td>1,053 0.435</td><td></td><td>743,764</td><td>743,764</td><td>743,760</td><td>4 (0.0005%)</td></tr><tr><td></td><td>qwen-2.5-14b</td><td>710 1,050</td><td>0.500</td><td></td><td>744,790</td><td>744,790</td><td>744,779</td><td>11 (0.0015%)</td></tr><tr><td></td><td>qwen-2.5-72b</td><td>685</td><td>1,027 0.500</td><td></td><td>702,810</td><td>702,810</td><td>702,379</td><td>431 (0.0613%)</td></tr><tr><td>Word Definitions</td><td>llama-3.2-3b</td><td>610</td><td>1,584 0.395</td><td></td><td>965,630</td><td>965,630</td><td>965,630</td><td>0 (0.0000%)</td></tr><tr><td></td><td>llama-3.1-8b</td><td>641</td><td>1,606</td><td>0.447</td><td>1,028,805</td><td>1,028,805</td><td>1,028,805</td><td>0 (0.0000%)</td></tr><tr><td></td><td>llama-3.1-70b</td><td>638</td><td>1,610 0.447</td><td></td><td>1,026,542</td><td>1,026,542</td><td>1,026,542</td><td>0 (0.0000%)</td></tr><tr><td></td><td>gemma-7b</td><td>638</td><td>1,623 0.398</td><td></td><td>1,034,836</td><td>1,034,836</td><td>1,034,836</td><td>0 (0.0000%)</td></tr><tr><td></td><td>gemma-2-9b</td><td>606</td><td>1,580 0.469</td><td></td><td>956,874</td><td>956,874</td><td>956,874</td><td>0 (0.0000%)</td></tr><tr><td></td><td>gemma-2-27b</td><td>618</td><td>1,584 0.441</td><td></td><td>978,294</td><td>978,294</td><td>978,294</td><td>0 (0.0000%)</td></tr><tr><td></td><td>mistral-7b</td><td>639</td><td>1,612 0.437</td><td></td><td>1,029,429</td><td>1,029,429</td><td>1,029,429</td><td>0 (0.0000%)</td></tr><tr><td></td><td>mistral-12b</td><td>617</td><td>1,589 0.377</td><td></td><td>979,796</td><td>979,796</td><td>979,796</td><td>0 (0.0000%)</td></tr><tr><td></td><td>mistral-3.1-24b</td><td>619</td><td>1,592 0.461</td><td></td><td>984,829</td><td>984,829</td><td>984,829</td><td>0 (0.0000%)</td></tr><tr><td></td><td>qwen-2.5-7b</td><td>627</td><td>1,616 0.409</td><td></td><td>1,012,605</td><td>1,012,605</td><td>1,012,605</td><td>0 (0.0000%)</td></tr><tr><td></td><td>qwen-2.5-14b</td><td>620</td><td>1,594 0.425</td><td></td><td>987,660</td><td>987,660</td><td>987,660</td><td>0 (0.0000%)</td></tr><tr><td></td><td>qwen-2.5-72b</td><td>617</td><td>1,593 0.469</td><td></td><td>982,264</td><td>982,264</td><td>982,264</td><td>0 (0.0000%)</td></tr></table>

Table A5. Belief-set and conditional-pair coverage using sAwMIL. For each LLM and domain, $| \boldsymbol { B } _ { \mathcal { M } } |$ and $| { \mathcal { X } } _ { { \mathcal { M } } } |$ denote the numbers of beliefs and non-disbeliefs, respectively, and $t _ { \mathcal { M } }$ is the empirical belief threshold. Candidate is the number $| B _ { \mathcal { M } } | ( | \mathcal { X } _ { \mathcal { M } } | - 1 )$ of constructed $( P , x )$ pairs after excluding self-pairs. Direct and Joint report the numbers of valid conditional-probability estimates. Joint excl. reports the number and percentage of candidate pairs excluded because the Joint-to-Conditional denominator is at most $1 0 ^ { - 1 2 }$

## E Construction and coverage of conditional probability estimates

## E.1 Belief sets and conditional-pair coverage

Table A5 reports the empirical belief sets, non-disbelief sets, belief thresholds, and conditional-pair coverage for the primary sAwMIL probe. For each model and dataset, the candidate set contains every pair $( P , x )$ with $P \in B _ { \mathcal { M } }$ and $x \in \mathcal { X } _ { \mathcal { M } } \backslash \{ P \}$ , yielding $| B _ { \mathcal { M } } | \cdot ( | \mathcal { X } _ { \mathcal { M } } | - 1 )$ total candidate pairs. We report the number of candidate pairs successfully assigned a conditional-probability estimate by the Direct Conditional and Joint-to-Conditional estimators, together with the number of Joint-to-Conditional pairs excluded due to an undefined denominator in Eq. 19.

The Direct Conditional estimator yields a valid estimate for every candidate pair across all 36 model–dataset combinations. Joint-to-Conditional coverage is also nearly complete: only 7,818 of 29,732,369 candidate pairs (0.026%) are excluded because of a numerically zero denominator. No Joint-to-Conditional pairs are excluded for Word Definitions, and the largest exclusion rate for any individual model–dataset combination is 0.4741%.

## E.2 Joint-to-Conditional derivation

We derive the Joint-to-Conditional estimator in Equation (19) from the nine-state joint distribution $\pi _ { \mathcal { M } } ^ { \mathrm { j o i n t } } ( P , x )$ Recall that $\pi _ { a b } ^ { \mathrm { j o i n t } } ( P , x )$ denotes the probability assigned to the joint state in which the antecedent x has state $a \in \{ T , F , N \}$ and the consequent P has state $b \in \{ T , F , N \}$

Under Cooper–Cantwell semantics, the conditional P | x takes the truth value of P when x is True or Neither, and is Neither when x is False. Collapsing the nine joint states according to this truth table therefore gives the induced trivalent conditional distribution

$$
\pi _ { T } ^ { \mathrm { j o i n t } , C } ( P , x ) = \pi _ { T T } ^ { \mathrm { j o i n t } } ( P , x ) + \pi _ { N T } ^ { \mathrm { j o i n t } } ( P , x ) ,\tag{29}
$$

$$
\pi _ { F } ^ { \mathrm { j o i n t } , C } ( P , x ) = \pi _ { T F } ^ { \mathrm { j o i n t } } ( P , x ) + \pi _ { N F } ^ { \mathrm { j o i n t } } ( P , x ) ,\tag{30}
$$

$$
\pi _ { N } ^ { \mathrm { j o i n t } , C } ( P , x ) = \pi _ { T N } ^ { \mathrm { j o i n t } } ( P , x ) + \pi _ { N N } ^ { \mathrm { j o i n t } } ( P , x ) + \pi _ { F T } ^ { \mathrm { j o i n t } } ( P , x ) + \pi _ { F F } ^ { \mathrm { j o i n t } } ( P , x ) + \pi _ { F N } ^ { \mathrm { j o i n t } } ( P , x ) .\tag{31}
$$

Here, the superscript C denotes the conditional distribution induced from the joint probe.

Cantwell’s non-bivalent probability rule assigns a sentence its probability of being True conditional on its having a determinate truth value [44]. Applying this normalization to the induced conditional distribution yields

$$
\begin{array} { r l } & { \widehat { \mathrm { P r } } _ { \mathcal { M } } ^ { \mathrm { j o i n t } } ( P \mid x ) = \frac { \pi _ { T } ^ { \mathrm { j o i n t } , C } ( P , x ) } { \pi _ { T } ^ { \mathrm { j o i n t } , C } ( P , x ) + \pi _ { F } ^ { \mathrm { j o i n t } , C } ( P , x ) } } \\ & { \qquad = \frac { \pi _ { T } ^ { \mathrm { j o i n t } } ( P , x ) + \pi _ { N T } ^ { \mathrm { j o i n t } } ( P , x ) } { \pi _ { T T } ^ { \mathrm { j o i n t } } ( P , x ) + \pi _ { T F } ^ { \mathrm { j o i n t } } ( P , x ) + \pi _ { N T } ^ { \mathrm { j o i n t } } ( P , x ) + \pi _ { N F } ^ { \mathrm { j o i n t } } ( P , x ) } , } \end{array}\tag{32}
$$

(33)

recovering Equation (19). The numerator contains exactly the joint states that induce a True conditional, while the denominator contains all states that induce either a True or False conditional. The estimator is therefore undefined when the conditional is assigned probability one of being Neither; as described in Section 4.3.2, we treat denominators less than or equal to $1 0 ^ { - 1 2 }$ as numerically zero and exclude the corresponding pairs.

This normalization also clarifies the distinction between the two conditional-probability estimators. The Direct Conditional estimator uses the raw True-state mass $\pi _ { T } ^ { \mathrm { d i r e c t } } ( P , x )$ , whereas Joint-to-Conditional normalizes the induced True mass over the True and False conditional states.

## F Supplementary validation of graded belief stability

## F.1 Robustness of the stability–belief-probability relationship

Our primary analysis models graded stability as a function of individual belief probability using a natural cubic spline with three degrees of freedom. To evaluate sensitivity to this choice, we repeat the analysis using a linear baseline and natural cubic splines with 2, 3, 4, and 5 degrees of freedom.

We use five-fold cross-validation to evaluate the cubic spline models. Propositions are randomly permuted using a deterministic seed derived from base seed 0 and divided into five approximately equal folds. The same fold assignment is reused across spline specifications. Within each fold, we construct a centered natural cubic spline basis from the training propositions with fixed boundaries at 0 and 1, estimate its coeficients by ordinary least squares, and apply the resulting basis to the held-out propositions. The linear baseline is fit analogously using an intercept and linear term in $\pi _ { T } ( P )$ . We apply the same inclusion criteria as in the primary analysis, requiring at least 50 valid proposition-level stability scores and suficient distinct values of $\pi _ { T } ( P )$ to fit the requested spline.

For the cross-model residual-agreement analysis in Section 4.5.1, we use the out-of-fold residuals from the primary $d f = 3$ specification and retain model pairs sharing at least 50 believed propositions. We generate 1,000 permutation samples by independently shufling residuals across proposition identities within each model while preserving model overlap, recompute the median pairwise Spearman correlation for each permutation, and calculate the one-sided Monte Carlo p-value using the standard +1 correction. Permutations use deterministic seeds derived from the same base seed 0.

![](images/7c359d7f06a3171465da940d5097d44968465b7b95948c5fcafda3314adfe6b0.jpg)

![](images/f7bd9d04b2e1ff58fdf1bd72e625a8acbf205e59b6bbfbf60910b5c765680d14.jpg)  
Direct Median Primary df = 3

![](images/877c02a3c9de66d3758add6e478b8f9fcb8173ed9ba4e3849496eb44646dc734.jpg)  
Figure A2. Robustness of the natural cubic spline. Violin plots show the distribution across the 12 instruction-tuned LLMs of held-out $R ^ { 2 }$ for predicting graded stability from individual belief probability using a linear baseline and natural cubic splines with 2–5 degrees of freedom for (a) City Locations, (b) Medical Indications, and (c) Word Definitions. Horizontal black lines denote the median across models, and the shaded column marks the primary specification with 3 degrees of freedom. Predictive performance is similar across the 3–5 degree-of-freedom specifications, indicating that the main conclusion is not sensitive to the precise spline flexibility.

Across all three domains, the primary $d f = 3$ specification performs similarly to the more flexible $d f = 4$ and $d f = 5$ alternatives (Fig. A2). The linear baseline and $d f = 2$ spline tend to yield lower held-out $R ^ { 2 }$ , particularly for City Locations, indicating that some nonlinearity is useful. Increasing flexibility beyond the primary specification, however, produces little systematic improvement. The conclusion that individual belief probability predicts a meaningful but incomplete fraction of graded stability therefore does not depend on the exact spline complexity.

## F.2 Additional probabilistic-coherence analyses

## F.2.1 Exact pairwise coherence conditions

We show that the four constraints in Equation (12) are both necessary and suficient for the three measured distributions associated with a single $( P , x )$ pair to admit a common trivalent joint representation. For compactness, we write

$$
p _ { b } = \pi _ { b } ( P ) , \qquad a _ { b } = \pi _ { b } ( x ) , \qquad c _ { b } = \pi _ { b } ^ { \mathrm { d i r e c t } } ( P , x ) , \qquad b \in \{ T , F , N \} .
$$

Let $J _ { a b }$ denote the probability assigned by a latent joint distribution to the state in which x has value a and P has value b, with the first coordinate corresponding to the antecedent. The row marginals of J must equal $\pi _ { \mathcal { M } } ( x )$ and its column marginals must equal $\pi _ { \mathcal { M } } ( P )$

Under the Cooper–Cantwell truth table, such a joint distribution induces the conditional distribution

$$
c _ { T } = J _ { T T } + J _ { N T } ,\tag{34}
$$

$$
c _ { F } = J _ { T F } + J _ { N F } ,\tag{35}
$$

$$
c _ { N } = J _ { T N } + J _ { N N } + J _ { F T } + J _ { F F } + J _ { F N } .\tag{36}
$$

Necessity. Suppose first that a representing joint distribution J exists. Combining its marginals with the induced conditional distribution gives

$$
p _ { T } - c _ { T } = J _ { F T } \geq 0 ,\tag{37}
$$

$$
p _ { F } - c _ { F } = J _ { F F } \geq 0 ,\tag{38}
$$

$$
c _ { N } - a _ { F } = J _ { T N } + J _ { N N } \ge 0 ,\tag{39}
$$

$$
p _ { N } + a _ { F } - c _ { N } = J _ { F N } \ge 0 .\tag{40}
$$

It follows that

$$
c _ { T } \leq p _ { T } , \qquad c _ { F } \leq p _ { F } , \qquad a _ { F } \leq c _ { N } , \qquad c _ { N } \leq p _ { N } + a _ { F } ,
$$

which are exactly the four constraints in Equation (12).

Suficiency. Conversely, suppose the four constraints hold. Define the vectors

$$
{ \bf r } = ( c _ { T } , c _ { F } , c _ { N } - a _ { F } ) ,\tag{41}
$$

$$
{ \bf f } = ( p _ { T } - c _ { T } , p _ { F } - c _ { F } , p _ { N } + a _ { F } - c _ { N } ) ,\tag{42}
$$

with coordinates ordered as $( T , F , N )$ . The four coherence constraints imply that every coordinate of r and f is nonnegative. Because p and c are each normalized probability distributions,

$$
\sum _ { b \in \{ T , F , N \} } r _ { b } = 1 - a _ { F } , \qquad \sum _ { b \in \{ T , F , N \} } f _ { b } = a _ { F } , \qquad r _ { b } + f _ { b } = p _ { b } .
$$

The vector f can therefore be used as the False-antecedent row of the latent joint distribution, while r supplies the total mass of the True- and Neither-antecedent rows.

When $a _ { F } < 1$ , define

$$
J _ { F b } = f _ { b } , \qquad J _ { T b } = \frac { a _ { T } } { 1 - a _ { F } } r _ { b } , \qquad J _ { N b } = \frac { a _ { N } } { 1 - a _ { F } } r _ { b } , \qquad b \in \{ T , F , N \} .\tag{43}
$$

All entries are nonnegative. Their row sums are $a _ { F } , ~ a _ { T }$ , and $a _ { N }$ , respectively, and because $a _ { T } + a _ { N } = 1 - a _ { F }$ , the column sum for each b is

$$
J _ { T b } + J _ { N b } + J _ { F b } = r _ { b } + f _ { b } = p _ { b } .
$$

Thus, J has the required marginals. Moreover, the induced conditional has True and False masses $r _ { T } = c _ { T }$ and $r _ { F } = c _ { F } .$ , while its Neither mass is

$$
a _ { F } + r _ { N } = a _ { F } + \left( c _ { N } - a _ { F } \right) = c _ { N } .
$$

Hence, J reproduces all three measured distributions.

The remaining case is $a _ { F } = 1$ . Normalization then implies $a _ { T } = a _ { N } = 0$ . Because $a _ { F } \leq c _ { N }$ and c is normalized, $c _ { N } = 1$ and $c _ { T } = c _ { F } = 0$ . Setting

$$
J _ { F b } = p _ { b } , \qquad J _ { T b } = J _ { N b } = 0
$$

for every $b \in \{ T , F , N \}$ therefore provides a valid joint distribution with the required marginals and an everywhere-Neither conditional.

The four inequalities in Equation (12) are therefore necessary and suficient for exact pairwise compatibility, provided that the three measured triples are themselves normalized nonnegative probability distributions.

Projection onto the coherent set. The pairwise coherent set C used to compute $d _ { \mathrm { C C K } }$ is the intersection of three probability simplexes with the four closed linear half-spaces in Equation (12). It is therefore a nonempty, compact, and convex subset of R<sup>9</sup>. Consequently, every measured vector ${ \mathbf v } _ { P , x }$ has a unique Euclidean projection $\Pi _ { C } ( { \mathbf { v } } _ { P , x } )$ onto this set, and Equation (13) is well defined.

Scope of the coherence analysis. These conditions characterize compatibility for a single $( P , x )$ pair. Testing them separately across pairs does not establish that all measured probabilities for an LLM can be generated by one global joint distribution over its entire belief system. Such simultaneous compatibility is a stronger global feasibility problem requiring a shared probability distribution over admissible trivalent valuations of all propositions. Our coherence analysis tests the pairwise condition only.

## F.2.2 Exact coherence

Table A6 reports exact CCK-coherence rates and individual constraint violations for the primary Direct Conditional estimates obtained using sAwMIL. As described in Section 4.5.2, a $( P , x )$ pair is classified as exactly coherent when all four CCK inequalities are satisfied within the numerical tolerance of $1 0 ^ { - 8 }$ . For each inequality, we additionally report how frequently it is violated and the mean signed excess among the pairs that violate it.

Exact coherence is rare for every model and domain (Tab. A6), with the largest observed rate equal to 0.206%. The two most frequently violated conditions are $\pi _ { F } ^ { \mathrm { d i r e c t } } ( P , x ) \leq \pi _ { F } ( P )$ and $\pi _ { N } ^ { \mathrm { d i r e c t } } ( P , x ) \leq \pi _ { N } ( P ) + \pi _ { F } ( x )$ , whereas violations of the remaining two inequalities occur substantially less often. Because exact coherence requires all four constraints to hold simultaneously, these frequent violations drive the near-zero exact-coherence rates reported in the main text and drive the distance-to-coherence analysis in Section 4.5.2.

One likely contributor to this pattern is the scale of the corresponding individual-statement probabilities. Because $P$ is a probe-defined belief, $\pi _ { F } ( P )$ and $\pi _ { N } ( P )$ are often small. Similarly, because $\mathcal { X } _ { \mathcal { M } }$ contains non-disbeliefs, its members x often have small $\pi _ { F } ( x )$ . Even modest conditional probability mass assigned to False or Neither can therefore violate the second or fourth constraint. The fourth constraint may additionally be sensitive to the Direct Conditional representation itself: in “Given $x , P , \ "$ the conditioning proposition precedes P in the autoregressive sequence, allowing the representation of the conditional to incorporate x and potentially shift probability mass toward Neither relative to P in isolation.

<table><tr><td>Dataset</td><td>Model</td><td>Exact</td><td> $\pi _ { T } ^ { \mathrm { d i r e c t } } ( P , x ) > \pi _ { T } ( P ) \pi _ { F } ^ { \mathrm { d i r e c t } } ( P , x ) > \pi _ { F } ( P ) \pi _ { F } ( x ) > \pi _ { N } ^ { \mathrm { d i r e c t } } ( P , x ) \pi _ { N } ^ { \mathrm { d i r e c t } } ( P , x ) > \pi _ { N } ( P ) + \pi _ { F } ( x )$ </td><td></td><td></td><td></td></tr><tr><td>City Locations</td><td>llama-3.2-3b</td><td>0.000%</td><td>2.7% (0.095)</td><td>90.4% (0.154)</td><td>3.4% (0.122)</td><td>96.6% (0.207)</td></tr><tr><td rowspan="20"></td><td>llama-3.1-8b</td><td>0.000%</td><td>5.2% (0.054)</td><td>88.0% (0.051)</td><td>5.8% (0.047)</td><td>94.2% (0.080)</td></tr><tr><td>llama-3.1-70b</td><td>&lt; 0.001%</td><td>7.9% (0.031)</td><td>80.5% (0.035)</td><td>11.4% (0.026)</td><td>88.5% (0.047)</td></tr><tr><td>gemma-7b</td><td>&lt; 0.001%</td><td>3.8% (0.107)</td><td>92.4% (0.159)</td><td>7.9% (0.091)</td><td>92.1% (0.102)</td></tr><tr><td>gemma-2-9b</td><td>0.000%</td><td>3.1% (0.050)</td><td>93.9% (0.087)</td><td>11.0% (0.036)</td><td>89.0% (0.046)</td></tr><tr><td>gemma-2-27b</td><td>0.000%</td><td>2.4% (0.067)</td><td>93.7% (0.104)</td><td>5.9% (0.073)</td><td>94.1% (0.109)</td></tr><tr><td>mistral-7b</td><td>0.000%</td><td>3.5% (0.097)</td><td>91.7% (0.077)</td><td>8.8% (0.053)</td><td>91.2% (0.093)</td></tr><tr><td>mistral-12b</td><td>0.004%</td><td>9.0% (0.055)</td><td>62.5% (0.082)</td><td>11.1% (0.075)</td><td>88.7% (0.112)</td></tr><tr><td>mistral-3.1-24b</td><td>0.000%</td><td>8.8% (0.027)</td><td>86.0% (0.074)</td><td>21.2% (0.022)</td><td>78.8% (0.068)</td></tr><tr><td>qwen-2.5-7b qwen-2.5-14b</td><td>&lt; 0.001%</td><td>4.6% (0.052)</td><td>88.0% (0.125)</td><td>16.6% (0.052)</td><td>83.4% (0.112)</td></tr><tr><td>qwen-2.5-72b</td><td>0.000% 0.000%</td><td>8.8% (0.032)</td><td>78.8% (0.066)</td><td>18.5% (0.037)</td><td>81.5% (0.077)</td></tr><tr><td>Medical Indications</td><td></td><td>8.4% (0.031)</td><td>74.4% (0.056)</td><td>15.0% (0.039)</td><td>85.0% (0.077)</td></tr><tr><td>llama-3.2-3b</td><td>0.000%</td><td>3.6% (0.082)</td><td>63.8% (0.153)</td><td>12.1% (0.138)</td><td>87.9% (0.276)</td></tr><tr><td>llama-3.1-8b</td><td>0.008%</td><td>5.2% (0.092)</td><td>59.7% (0.111)</td><td>13.2% (0.132)</td><td>86.7% (0.252)</td></tr><tr><td>llama-3.1-70b</td><td>0.000%</td><td>7.4% (0.088)</td><td>65.8% (0.111)</td><td>13.6% (0.127)</td><td>86.4% (0.179)</td></tr><tr><td>gemma-7b</td><td>0.097%</td><td>2.1% (0.095)</td><td>60.6% (0.152)</td><td>20.8% (0.132)</td><td>79.0% (0.250)</td></tr><tr><td rowspan="20">Word Definitions</td><td>gemma-2-9b</td><td>0.000%</td><td>8.4% (0.094)</td><td>72.1% (0.119)</td><td>21.8% (0.129)</td><td>78.1% (0.148)</td></tr><tr><td>gemma-2-27b</td><td>0.015%</td><td>9.4% (0.103)</td><td>68.9% (0.100)</td><td>20.6% (0.134)</td><td>79.3% (0.150)</td></tr><tr><td>mistral-7b</td><td>&lt; 0.001%</td><td>7.3% (0.085)</td><td>63.8% (0.090)</td><td>21.6% (0.134)</td><td>78.4% (0.183)</td></tr><tr><td>mistral-12b</td><td>0.200%</td><td>13.4% (0.099)</td><td>45.4% (0.107)</td><td>31.7% (0.132)</td><td>67.9% (0.184)</td></tr><tr><td>mistral-3.1-24b</td><td>0.015%</td><td>11.9% (0.096)</td><td>58.2% (0.054)</td><td>19.8% (0.134)</td><td>80.1% (0.128)</td></tr><tr><td>qwen-2.5-7b</td><td>0.035%</td><td>3.5% (0.087)</td><td>75.8% (0.184)</td><td>23.4% (0.132)</td><td>76.2% (0.204)</td></tr><tr><td>qwen-2.5-14b</td><td>0.000%</td><td>6.3% (0.088)</td><td>72.9% (0.125)</td><td>25.3% (0.127)</td><td>74.7% (0.178)</td></tr><tr><td>qwen-2.5-72b</td><td>0.000%</td><td>7.1% (0.086)</td><td>74.7% (0.079)</td><td>22.6% (0.114)</td><td>77.4% (0.144)</td></tr><tr><td>llama-3.2-3b</td><td>0.073%</td><td>6.6% (0.091)</td><td>64.2% (0.100)</td><td>5.7% (0.106)</td><td></td></tr><tr><td>llama-3.1-8b</td><td>0.020%</td><td>4.6% (0.086)</td><td>77.5% (0.103)</td><td>6.8% (0.125)</td><td>93.9% (0.223)</td></tr><tr><td>llama-3.1-70b</td><td>0.018%</td><td>6.5% (0.085)</td><td>73.7% (0.103)</td><td>4.7% (0.121)</td><td>93.0% (0.177) 95.1% (0.185)</td></tr><tr><td>gemma-7b</td><td>0.206%</td><td>2.7% (0.088)</td><td>64.1% (0.140)</td><td>8.1% (0.118)</td><td>91.4% (0.259)</td></tr><tr><td>gemma-2-9b</td><td>0.012%</td><td>8.2% (0.109)</td><td>82.3% (0.135)</td><td>10.1% (0.123)</td><td>88.8% (0.120)</td></tr><tr><td>gemma-2-27b</td><td>0.003%</td><td>3.6% (0.109)</td><td>87.2% (0.145)</td><td>5.1% (0.110)</td><td>94.7% (0.169)</td></tr><tr><td>mistral-7b</td><td>0.030%</td><td>4.7% (0.094)</td><td>72.5% (0.097)</td><td>6.4% (0.124)</td><td>93.2% (0.207)</td></tr><tr><td>mistral-12b</td><td>0.054%</td><td>2.7% (0.092)</td><td>73.1% (0.132)</td><td>5.6% (0.126)</td><td>94.3% (0.323)</td></tr><tr><td>mistral-3.1-24b</td><td>0.026%</td><td>7.0% (0.101)</td><td>76.0% (0.103)</td><td>7.8% (0.125)</td><td>91.8% (0.173)</td></tr><tr><td>qwen-2.5-7b</td><td>0.080%</td><td>7.7% (0.108)</td><td>76.2% (0.135)</td><td>10.9% (0.120)</td><td>87.6% (0.189)</td></tr><tr><td>qwen-2.5-14b</td><td>0.008%</td><td>6.2% (0.093)</td><td>79.7% (0.132)</td><td>9.0% (0.113)</td><td>90.7% (0.162)</td></tr><tr><td>qwen-2.5-72b</td><td>0.021%</td><td>3.8% (0.082)</td><td>82.0% (0.164)</td><td>8.2% (0.117)</td><td>91.6% (0.189)</td></tr></table>

Table A6. Exact CCK coherence and constraint violations for Direct Conditional estimates using sAwMIL. For each LLM and domain, we report the percentage of $( P , x )$ pairs satisfying all four CCK constraints simultaneously within tolerance $1 0 ^ { - 8 }$ . Here, 0.000% signifies exactly zero fully coherent pairs, while $< 0 . 0 0 1 \%$ means a small number of exactly coherent pairs exists. Each remaining column reports the percentage of pairs violating each constraint, with the mean excess across violating pairs shown in parentheses. Exact coherence is rare across all models and domains, with no model–domain combination exceeding 0.206%. Violations occur most frequently for the constraints on conditional False and Neither probability mass.

## G Additional behavioral results

## G.1 Matched-pairs analysis

Table A7 reports diagnostics for the probability-matched pairs used in our primary behavioral analysis. As described in Section 4.6, matching is performed separately within each challenge-sequence block so that both propositions in a pair receive the same conversational challenges in the same order. The reported sample additionally requires both propositions’ initial behavioral judgments to agree with their individual-statement probe classifications, as in the primary analysis. For each model, n denotes the number of pairs, while $| \Delta \pi _ { T } |$ and $| \Delta \gamma |$ quantify pairwise diferences in individual belief probability and graded stability, respectively.

Across all three domains, matching yields small diferences in individual belief probability while retaining substantially larger diferences in graded stability. Mean $| \Delta \pi _ { T } |$ is at most 0.00553 across the reported model–domain combinations, whereas mean $| \Delta \gamma |$ ranges from 0.066 to 0.300. Thus, the matched pairs are closely aligned in individual belief probability while preserving meaningful variation in graded stability.

<table><tr><td>Dataset</td><td>Model</td><td>n</td><td> $| \Delta \pi _ { T } |$ </td><td></td><td> $| \Delta \gamma |$ </td></tr><tr><td>City Locations</td><td>llama-3.2-3b</td><td>315</td><td>0.00435</td><td>(0.01055)</td><td>0.235 (0.173)</td></tr><tr><td></td><td>llama-3.1-8b</td><td>305</td><td>0.00181</td><td>(0.00556)</td><td>0.090 (0.122)</td></tr><tr><td></td><td>llama-3.1-70b</td><td>287</td><td>0.00483</td><td>(0.02893)</td><td>0.066 (0.116)</td></tr><tr><td></td><td>gemma-7b</td><td>304</td><td>0.00377</td><td>(0.01042)</td><td>0.175 (0.175)</td></tr><tr><td></td><td>gemma-2-9b</td><td>291</td><td>0.00370</td><td>(0.01686)</td><td>0.082 (0.141)</td></tr><tr><td></td><td>gemma-2-27b</td><td>316</td><td>0.00362</td><td>(0.01372)</td><td>0.133 (0.143)</td></tr><tr><td></td><td>mistral-7b</td><td>303</td><td>0.00456</td><td>(0.01924)</td><td>0.113 (0.157)</td></tr><tr><td></td><td>mistral-12b</td><td>278</td><td>0.00495</td><td>(0.01547)</td><td>0.146 (0.189)</td></tr><tr><td></td><td>mistral-3.1-24b</td><td>307</td><td>0.00362</td><td>(0.01821)</td><td>0.100 (0.151)</td></tr><tr><td></td><td>qwen-2.5-7b</td><td>312</td><td>0.00350</td><td>(0.01001)</td><td>0.136 (0.157)</td></tr><tr><td></td><td>qwen-2.5-14b</td><td>290</td><td>0.00314</td><td>(0.01289)</td><td>0.103 (0.159)</td></tr><tr><td></td><td>qwen-2.5-72b</td><td>276</td><td>0.00282</td><td>(0.01189)</td><td>0.096 (0.178)</td></tr><tr><td>Medical Indications</td><td>llama-3.2-3b</td><td>342</td><td>0.00442</td><td>(0.00530)</td><td>0.234 (0.185)</td></tr><tr><td></td><td>llama-3.1-8b</td><td>344</td><td>0.00389</td><td>(0.00518)</td><td>0.248 (0.204)</td></tr><tr><td></td><td>llama-3.1-70b</td><td>340</td><td>0.00404</td><td>(0.00763)</td><td>0.198 (0.181)</td></tr><tr><td></td><td>gemma-7b</td><td>356</td><td>0.00430</td><td>(0.00639)</td><td>0.266 (0.217)</td></tr><tr><td></td><td>gemma-2-9b</td><td>344</td><td>0.00381</td><td>(0.00535)</td><td>0.210 (0.206)</td></tr><tr><td></td><td>gemma-2-27b mistral-7b</td><td>338</td><td>0.00408</td><td>(0.00648)</td><td>0.182 (0.204)</td></tr><tr><td></td><td></td><td>329</td><td>0.00456</td><td>(0.00993)</td><td>0.179 (0.223)</td></tr><tr><td></td><td>mistral-12b</td><td>339</td><td>0.00515</td><td>(0.00856)</td><td>0.200 (0.204)</td></tr><tr><td></td><td>mistral-3.1-24b</td><td>337</td><td>0.00379</td><td>(0.00564)</td><td>0.138 (0.176)</td></tr><tr><td></td><td>qwen-2.5-7b</td><td>353</td><td>0.00441</td><td>(0.00579)</td><td>0.263 (0.206)</td></tr><tr><td></td><td>qwen-2.5-14b</td><td>354</td><td>0.00406</td><td>(0.00494)</td><td>0.232 (0.206)</td></tr><tr><td>Word Definitions</td><td>qwen-2.5-72b</td><td>337</td><td>0.00401</td><td>(0.00719)</td><td>0.155 (0.191)</td></tr><tr><td></td><td>llama-3.2-3b</td><td>303</td><td>0.00510</td><td>(0.00960)</td><td>0.221 (0.233)</td></tr><tr><td></td><td>llama-3.1-8b</td><td>316</td><td>0.00408</td><td>(0.00787)</td><td>0.169 (0.191)</td></tr><tr><td></td><td>llama-3.1-70b</td><td>316</td><td>0.00429</td><td>(0.00772)</td><td>0.199 (0.205)</td></tr><tr><td></td><td>gemma-7b</td><td>317</td><td>0.00538</td><td>(0.00766)</td><td>0.262 (0.230)</td></tr><tr><td></td><td>gemma-2-9b</td><td>300</td><td>0.00488</td><td>(0.01071)</td><td>0.179 (0.190)</td></tr><tr><td></td><td>gemma-2-27b</td><td>307</td><td>0.00460</td><td>(0.01048)</td><td>0.204 (0.200)</td></tr><tr><td></td><td>mistral-7b</td><td>315</td><td>0.00475</td><td>(0.00910)</td><td>0.201 (0.218)</td></tr><tr><td></td><td>mistral-12b</td><td>308</td><td>0.00553</td><td>(0.01139)</td><td>0.300 (0.254)</td></tr><tr><td></td><td>mistral-3.1-24b</td><td>306</td><td>0.00462</td><td>(0.00839)</td><td>0.201 (0.218)</td></tr><tr><td></td><td>qwen-2.5-7b</td><td>311</td><td>0.00447</td><td>(0.00834)</td><td>0.189 (0.199)</td></tr><tr><td></td><td>qwen-2.5-14b</td><td>308</td><td>0.00459</td><td>(0.00962)</td><td>0.169 (0.187)</td></tr><tr><td></td><td>qwen-2.5-72b</td><td>305</td><td>0.00474 (0.00925)</td><td></td><td>0.253 (0.226)</td></tr></table>

Table A7. Behavioral matching quality. For each LLM and domain, n is the number of matched proposition pairs, $| \Delta \pi _ { T } |$ is the absolute diference in belief probability between the two propositions in each pair, and $| \Delta \gamma |$ is their absolute diference in graded stability. Values for $| \Delta \pi _ { T } |$ and $| \Delta \gamma |$ report the mean with standard deviation in parentheses. Across domains and models, matching produces small diferences in individual belief probability while retaining substantially larger diferences in graded stability.

## G.2 Uncertainty in behavioral resilience efects

Figure 5 reports ±1 bootstrap standard error to visualize the precision of each model–domain efect. Table A8 reports the corresponding 95% percentile bootstrap confidence intervals for mean $\Delta M$ . Mean $\Delta M$ is positive in 83.3% of model–domain settings. The 95% confidence interval is entirely positive in 11 of 36 (30.6%) settings, entirely negative in 1 of 36 (2.8%), and includes zero in the remaining 24 of 36 (66.7%).

<table><tr><td>Dataset</td><td>Model</td><td>Mean ∆M</td><td>Bootstrap SE</td><td>95% CI</td></tr><tr><td>City Locations</td><td>llama-3.2-3b llama-3.1-8b</td><td>-0.0256</td><td>0.0118</td><td>[−0.0488, -0.0018]</td></tr><tr><td></td><td>llama-3.1-70b</td><td>0.0220 0.0001</td><td>0.0308</td><td>[-0.0402, 0.0815]</td></tr><tr><td></td><td></td><td></td><td>0.0049</td><td>[-0.0098, 0.0095]</td></tr><tr><td></td><td>gemma-7b</td><td>0.0101</td><td>0.0080</td><td>[-0.0058, 0.0256]</td></tr><tr><td></td><td>gemma-2-9b</td><td>0.0180</td><td>0.0077</td><td>[0.0030, 0.0329]</td></tr><tr><td></td><td>gemma-2-27b</td><td>0.0784</td><td>0.0144</td><td>[0.0497, 0.1069]</td></tr><tr><td></td><td>mistral-7b</td><td>0.0426</td><td>0.0147</td><td>[0.0156, 0.0730]</td></tr><tr><td></td><td>mistral-12b</td><td>0.0938</td><td>0.0179</td><td>[0.0582, 0.1289]</td></tr><tr><td></td><td>mistral-3.1-24b</td><td>-0.0012</td><td>0.0076</td><td>[-0.0160, 0.0137]</td></tr><tr><td></td><td>qwen-2.5-7b</td><td>0.0107</td><td>0.0095</td><td>[-0.0080, 0.0293]</td></tr><tr><td></td><td>qwen-2.5-14b</td><td>0.0089</td><td>0.0068</td><td>[-0.0040, 0.0227]</td></tr><tr><td>Medical Indications</td><td>qwen-2.5-72b</td><td>0.0107</td><td>0.0064</td><td>[-0.0015, 0.0238]</td></tr><tr><td></td><td>llama-3.2-3b</td><td>-0.0070</td><td>0.0153</td><td>[-0.0368, 0.0231]</td></tr><tr><td></td><td>llama-3.1-8b</td><td>0.0255</td><td>0.0307</td><td>[-0.0357, 0.0860]</td></tr><tr><td></td><td>llama-3.1-70b</td><td>0.0097</td><td>0.0115</td><td>[−0.0131, 0.0323]</td></tr><tr><td></td><td>gemma-7b gemma-2-9b</td><td>0.0764</td><td>0.0169</td><td>[0.0438, 0.1096]</td></tr><tr><td></td><td></td><td>0.0522</td><td>0.0131</td><td>[0.0265, 0.0778]</td></tr><tr><td></td><td>gemma-2-27b</td><td>0.0289</td><td>0.0167</td><td>[-0.0040, 0.0611]</td></tr><tr><td></td><td>mistral-7b</td><td>0.0256</td><td>0.0213</td><td>[-0.0160, 0.0680]</td></tr><tr><td></td><td>mistral-12b</td><td>0.0344</td><td>0.0173</td><td>[0.0003, 0.0687]</td></tr><tr><td></td><td>mistral-3.1-24b</td><td>0.0005</td><td>0.0076</td><td>[-0.0140, 0.0156]</td></tr><tr><td></td><td>qwen-2.5-7b</td><td>0.1025</td><td>0.0147</td><td>[0.0740, 0.1313]</td></tr><tr><td></td><td>qwen-2.5-14b</td><td>0.0081</td><td>0.0125</td><td>[-0.0162, 0.0328]</td></tr><tr><td></td><td>qwen-2.5-72b</td><td>-0.0058</td><td>0.0112</td><td>[−0.0273, 0.0160]</td></tr><tr><td>Word Definitions</td><td>llama-3.2-3b</td><td>0.0050</td><td>0.0252</td><td>[-0.0418, 0.0561]</td></tr><tr><td></td><td>llama-3.1-8b</td><td>0.0607</td><td>0.0558</td><td>[-0.0515, 0.1659]</td></tr><tr><td></td><td>llama-3.1-70b</td><td>0.0211</td><td>0.0154</td><td>[-0.0088, 0.0510]</td></tr><tr><td></td><td>gemma-7b</td><td>0.0408</td><td>0.0139</td><td>[0.0135, 0.0681]</td></tr><tr><td></td><td>gemma-2-9b</td><td>0.0346</td><td>0.0164</td><td>[0.0034, 0.0672]</td></tr><tr><td></td><td>gemma-2-27b</td><td>0.0145</td><td>0.0228</td><td>[−0.0302, 0.0584]</td></tr><tr><td></td><td>mistral-7b</td><td>0.0568</td><td>0.0281</td><td>[0.0009, 0.1102]</td></tr><tr><td></td><td>mistral-12b</td><td>0.0257</td><td>0.0373</td><td>[-0.0471, 0.0995]</td></tr><tr><td></td><td>mistral-3.1-24b</td><td>0.0206</td><td>0.0148</td><td>[-0.0088, 0.0491]</td></tr><tr><td></td><td>qwen-2.5-7b</td><td>0.0226</td><td>0.0171</td><td>[-0.0116, 0.0555]</td></tr><tr><td></td><td>qwen-2.5-14b</td><td>-0.0039</td><td>0.0114</td><td>[−0.0266, 0.0179]</td></tr><tr><td></td><td>qwen-2.5-72b</td><td>-0.0096</td><td>0.0191</td><td>[−0.0476, 0.0266]</td></tr></table>

Table A8. Uncertainty in behavioral resilience efects. We report the mean behavioral movement diference ∆M, bootstrap standard error, and 95% percentile bootstrap confidence interval using the sAwMIL probe and Direct Conditional graded stability. Uncertainty is estimated from 10,000 sequence-stratified matched-pair bootstrap resamples. Mean ∆M is positive in 83.3% of model–domain settings; the 95% confidence interval is entirely positive in 30.6% of settings, entirely negative in 2.8%, and includes zero in the remaining 66.7%.

Layers

Parameter count

![](images/474c7e076547e2a2ccb796f8f09e3e8b5a2da9ee98f81a3b43023eb337331df8.jpg)

![](images/1b592ccd1ef42df4c962dbc53c3c3c22c55d665617d2c18992b00eb200e24d8e.jpg)

(c)  
![](images/662999193a66b15520b6439ff2afe488cfd43848b43000e8638903fa80525989.jpg)

![](images/2bbf12719cea44139bed3c5c08f8be86b12e3751b56964433f3a3e0cec045939.jpg)

![](images/83a2c0167923adea5fabc32d88617816e251bef0aba5d8f9325cd4906e4c2017.jpg)

(f)  
![](images/cdaebcc657a819d94973b6e6b58d02a175c3d366a662480aa5d0a2f38d15d992.jpg)  
Figure A3. Association of model scale with graded stability. Bars show mean graded stability $\overline { { \gamma } } _ { \mathcal { M } }$ for the 12 instruction-tuned LLMs, ordered by nominal parameter count in (a) City Locations, (b) Medical Indications, and (c) Word Definitions, and by number of decoder layers in (d) City Locations, (e) Medical Indications, and (f) Word Definitions. Colors denote Llama (red), Gemma (blue), Mistral (yellow), and Qwen (green) model families. Error bars denote ±1 bootstrap standard error. Neither parameter count nor model depth shows a consistent relationship with graded stability.

## H Model scaling

The between-model variation in graded stability raises the possibility that $\gamma _ { \mathcal { M } } ( P )$ increases systematically with model scale. We examine this relationship using two measures of scale: parameter count and number of layers. For each domain, we compute the mean graded stability γ across each model’s beliefs and order the 12 instruction-tuned LLMs by nominal parameter count (Fig. A3(a–c)) and model depth (Fig. A3(d–f)).

We do not observe a consistent scaling relationship under either measure. Mean stability generally increases across some of the smaller models in City Locations and Medical Indications, but these trends are neither monotonic nor consistent across domains. Word Definitions shows little evidence of increasing stability with either parameter count or model depth. Moreover, models with the same nominal parameter count or the same number of layers can exhibit substantially diferent mean stability. Thus, although model scale may contribute to some of the within-family diferences observed in Section 2.4, neither parameter count nor model depth provides a systematic explanation for variation in graded belief stability across models.

## I Robustness checks

## I.1 Base-model robustness

To test whether our primary findings depend on instruction tuning, we repeat the main-text analyses using the pretrained base counterparts of the 12 instruction-tuned LLMs. The models are matched by family and parameter scale, with exact specifications reported in Section C.

## I.1.1 Replication of primary analyses in base models

The principal representation-based findings are qualitatively similar in the base models. Individual belief probability continues to explain a meaningful but incomplete fraction of graded stability (Fig. A4), and the residual stability remaining after accounting for belief probability exhibits positive cross-model structure. Median residual correlations are $\rho = 0 . 3 8$ for City Locations, $\rho = 0 . 3 1$ for Medical Indications, and $\rho = 0 . 2 6$ for Word Definitions.

![](images/5528af0ef1db378800cdf493c6511ebd48c0c90b764766d2a11fa88ca516a7e0.jpg)  
Figure A4. Relationship between individual belief probability and graded stability in base models. We display the held-out $R ^ { 2 }$ for predicting graded stability from individual belief probability in (a) City Locations, (b) Medical Indications, and (c) Word Definitions for the 12 matched pretrained base models from the Llama (red), Gemma (blue), Mistral (yellow), and Qwen (green) families. Panel (d) shows the distribution of pairwise Spearman correlations between models’ residual graded-stability scores after removing the fitted belief-probability relationship with black lines denoting median correlations. As with the instruction-tuned models, belief probability explains only part of graded stability and the remaining proposition-level structure is positively shared across models.

The approximate-coherence result also persists in the base models (Fig. A5). Most measured probability systems remain relatively close to the CCK-coherent set, with City Locations generally requiring the smallest adjustments and Medical Indications and Word Definitions showing larger distances.

Domain-level variation in graded stability is likewise broadly preserved (Fig. A6). City Locations remains concentrated near high stability for most base models, whereas Medical Indications and Word Definitions generally exhibit broader distributions and more mass at intermediate stability values. The base models typically show greater between-model heterogeneity than the instruction-tuned models, particularly for the Medical Indications and Word Definitions domains. There are, however, several notable exceptions to the overall pattern. In City Locations, for example, Llama-3.2-3B is substantially more stable than in the instruction-tuned setting, whereas Mistral-12B is markedly less stable.

The behavioral replication is weaker (Fig. A7). Unlike the instruction-tuned models, for which lower-stability beliefs exhibit greater mean movement in most model–domain settings, the base-model efects are generally smaller and cluster more closely around zero. We interpret this diference cautiously because the behavioral challenge paradigm is itself an instruction-following, multi-turn interaction. The pretrained base models have not undergone the instruction-following post-training of their matched instruction-tuned counterparts, and in our implementation the conversational history is therefore supplied to base models as an explicit plain-text transcript rather than through a model-specific chat template. The weaker base-model behavioral association may therefore reflect the mismatch between the challenge task and the models’ training rather than a failure of the underlying representational stability measure.

![](images/663fcb47a743f44b65663914266bf12e78c084060cba91eb33fcfc83060c0e81.jpg)

Figure A5. Distance from probabilistic coherence in base models. Violin plots show the distribution across (P,x) pairs of the RMS adjustment $d _ { \mathrm { C C K } }$ required to project the measured probability distributions for individua statements and conditionals onto the nearest CCK-coherent probability system for (a) City Locations, (b) Medical Indications, and (c) Word Definitions for Llama (red), Gemma (blue), Mistral (yellow), and Qwen (green). Horizontal black lines denote within-model medians. Base models remain comparatively close to the coherent set, although distances and between-model heterogeneity are slightly larger than in the instruction-tuned models.  
![](images/320021a082b4da195ce2f9906005cdd6bf5978f55492af7b26e35fa8622a6b2b.jpg)  
Figure A6. Variation in graded belief stability across domains in base models. Violin plots show the proposition-level distribution of Direct Conditional graded stability for each of the pretrained base models in (a) City Locations, (b) Medical Indications, and (c) Word Definitions for Llama (red), Gemma (blue), Mistral (yellow), and Qwen (green). Horizontal black lines denote within-model medians. City Locations is generally concentrated at higher stability values, while Medical Indications and Word Definitions show broader and more model-dependent distributions, reproducing the main domain-level pattern with greater heterogeneity.

## I.1.2 Association with instruction tuning

We next compare each base model directly with its instruction-tuned counterpart while holding belief identity fixed. For each matched model pair, we restrict the comparison to propositions believed by both models and define the average association with instruction tuning as

$$
\Delta _ { \mathrm { i n s t } } : = \frac { 1 } { | \mathcal { B } _ { \mathrm { b a s e } } \cap \mathcal { B } _ { \mathrm { i n s t } } | } \sum _ { P \in \mathcal { B } _ { \mathrm { b a s e } } \cap \mathcal { B } _ { \mathrm { i n s t } } } \left[ \gamma _ { \mathrm { i n s t } } ( P ) - \gamma _ { \mathrm { b a s e } } ( P ) \right] .\tag{44}
$$

Thus, $\Delta _ { \mathrm { i n s t } } > 0$ indicates that instruction tuning is associated with greater graded stability for the same set of shared beliefs. Across the 36 matched model–domain comparisons, 28 (77.8%) have $\Delta _ { \mathrm { i n s t } } > 0 ~ ( \mathrm { F i g . ~ A 8 } )$ . The efect is therefore positive more often than not, but its magnitude varies substantially across models and domains and several comparisons are negative. We interpret instruction tuning as exhibiting a positive directional tendency rather than a uniform stabilizing efect.

![](images/61719045ad78b0a43372db569564ef50b9e5eb533f1c57534429e6332ad6d1c6.jpg)  
Figure A7. Behavioral resilience among probability-matched beliefs in base models. Bars show the mean diference in behavioral movement ∆M between matched lower- and higher-stability beliefs for the pretrained base models in (a) City Locations, (b) Medical Indications, and (c) Word Definitions, with error bars denoting ±1 bootstrap standard error, for Llama (red), Gemma (blue), Mistral (yellow), and Qwen (green). Positive values indicate greater movement for the lower-stability belief. In contrast to the clearer directional association in instruction-tuned models, base-model efects are smaller and more heterogeneous, suggesting that the behaviora relationship is sensitive to instruction-following post-training.  
Efect of instruction tuning on graded stability

![](images/bf53adfa9b3b2ea468123e4f17372f9a97af8afcb0b000d738151f45d467b18c.jpg)  
Figure A8. Association between instruction tuning and graded belief stability. Bars show the mean matched diference $\Delta _ { \mathrm { i n s t } }$ in graded stability between each instruction-tuned model and its pretrained base counterpart over propositions believed by both models for (a) City Locations, (b) Medical Indications, and (c) Word Definitions for Llama (red), Gemma (blue), Mistral (yellow), and Qwen (green). Positive values indicate greater graded stability after instruction tuning, and error bars capture ±1 standard error. Instruction-tuned models have higher measured stability in 77.8% of matched comparisons, but the magnitude and direction of the efect remain heterogeneous.

## I.2 Conditional-estimator robustness

Our primary analyses use the Direct Conditional estimator, which estimates conditional support from representations of statements of the form “Given x, P.” To test whether our results depend on this operationalization of $\operatorname* { P r } _ { \mathcal { M } } ( P \mid x )$ we repeat the principal analyses using the Joint-to-Conditional estimator described in Section 4.3.2. While Direct Conditional probes the conditional statement itself, Joint-to-Conditional represents x and P jointly and derives the corresponding conditional probability from the resulting nine-state distribution. We first examine an additional representational choice introduced by the Joint estimator, the ordering of x and $P$ in the conjunction, before replicating the main analyses and directly comparing the two graded-stability estimates.

Order sensitivity of pair-level ${ \widehat { \mathsf { P r } } } ( P \mid x )$ City Locations

![](images/92774b7278a0264a291046e3736c0c9b163a60edee000fa28b8971910a392f6b.jpg)

![](images/3c620dff6b4f9311505ef60df387a0b98e5178924db0474264458e96d0406231.jpg)  
(c)  
Word Definitions

![](images/e903ebdeeb709c10ccfd75977e7bbe436d56eba945acd47faec2c60bb57dd549.jpg)

Order sensitivity of graded stability γ(P)  
![](images/d58ddab99bdf8e5df9fe4fc0768b95e91adf66d58f172c8feb024ad5d6f4083b.jpg)

(e)  
![](images/679b18e329262a11ecf3143415ef09bb136dcd7f3cc1e4248e91c8237b381857.jpg)

(f)  
![](images/9c41c70b6cb36f74aa5043af784e5b2c40f17409eb3071ef2a49fce56f3c8af5.jpg)  
Figure A9. Sensitivity of the Joint-to-Conditional estimator to conjunction ordering. We compare Joint estimates obtained using the templates $^ { 6 6 } x$ and $P ^ { * }$ and $^ { 6 6 } P$ and $x ' .$ . For each model, bars show the median absolute change in ${ \widehat { \operatorname { P r } } } _ { \mathcal { M } } ^ { \mathrm { j o i n t } } ( P \mid x )$ across evaluated $( P , x )$ combinations when the conjunction order is reversed, for (a) City Locations, (b) Medical Indications, and (c) Word Definitions for Llama (red), Gemma (blue), Mistral (yellow), and Qwen (green). We additionally display (d–f) the median absolute change in $\gamma _ { \mathcal { M } } ^ { \mathrm { j o i n t } } ( P )$ across target beliefs after graded stability is recomputed using the reversed-order estimates. Error bars denote ±1 bootstrap standard error. Reversing the conjunction order changes both the estimated conditional probabilities and the resulting graded-stability scores, revealing an order sensitivity specific to the Joint-to-Conditional operationalization.

## I.2.1 Sensitivity to conjunction ordering

The Joint-to-Conditional estimator requires a natural-language representation of the two statements being jointly probed. In our primary implementation, we use $^ { 6 6 } x$ and $P ^ { \ast }$ , placing the conditioning proposition x first and the target belief P second. The calibrated Joint probe produces the nine-state distribution described in Section 4.3.2, from which we derive ${ \widehat { \operatorname { P r } } } _ { \mathcal { M } } ^ { \mathrm { { l o u n t } } } ( P \mid x )$ . To test whether this estimate depends on the conjunction order, we independently train and calibrate a second Joint probe using the reversed template $^ { 6 6 } P$ and $x . ^ { \mathfrak { n } }$

Figure A9 shows that this choice is consequential. Reversing the conjunction changes the estimated conditional probabilities, with the size of the change varying substantially across models and domains (Fig. $\mathrm { A 9 } ( \mathbf { a } \mathrm { - } \mathbf { c } ) ,$ ). These diferences also afect the downstream graded stability measure (Fig. A9(d–f)). We treat this sensitivity as an additional measurement choice introduced by the Joint-to-Conditional estimator. An autoregressive LLM produces diferent hidden representations when the same two propositions appear in diferent token orders, and its training objective does not require the probe-relevant representations of $^ { 6 6 } x$ and $P ^ { \ast }$ and $^ { 6 6 } P$ and $x '$ to be invariant. As such, the observed diferences should not necessarily be interpreted as evidence about an underlying order-sensitive joint belief distribution. Rather, they show that the Joint operationalization inherits sensitivity to the linguistic representation used to elicit that distribution. We fix the order to $^ { 6 6 } x$ and $P ^ { \prime \prime }$ throughout the analyses below, and Direct Conditional remains our primary estimator because it does not introduce this additional conjunction-ordering choice.

![](images/6d5525f40a4d2abdb2c7ef6907bf40b73a15060fe7d01a092354d9af7ce8bf73.jpg)  
Figure A10. Relationship between individual belief probability and Joint-to-Conditional graded stability. We display the held-out $R ^ { 2 }$ for predicting Joint-to-Conditional graded stability from individual belief probability in (a) City Locations, (b) Medical Indications, and (c) Word Definitions for the 12 instruction-tuned LLMs for Llama (red), Gemma (blue), Mistral (yellow), and Qwen (green). Panel (d) shows the distribution of pairwise Spearman correlations between models’ residual graded-stability scores after removing the fitted belief-probability relationship, with black lines denoting median correlations. As with Direct Conditional, individua belief probability explains only part of graded stability, and residual stability remains positively correlated across most model pairs.

## I.2.2 Replication with Joint-to-Conditional

We repeat the main-text analyses using Joint-to-Conditional graded stability for the same 12 instruction-tuned LLMs (Figs. A10–A13). The principal representation-based findings are broadly preserved, while the absolute scale of graded stability and its behavioral association show greater sensitivity to the conditional estimator.

Individual belief probability remains an incomplete predictor of Joint-to-Conditional graded stability (Fig. A10). Held-out $R ^ { 2 }$ remains well below one across every model and domain. Cross-model residual correlations are also predominantly positive, although less uniformly than under Direct Conditional: 71.2% are positive for City Locations, 89.4% for Medical Indications, and 98.5% for Word Definitions. The corresponding median residual Spearman correlations are $\rho = 0 . 0 5 , \rho = 0 . 1 4$ , and $\rho = 0 . 2 1$ , respectively. Thus, the Joint estimator preserves the central result that graded stability contains systematic variation not captured by individual belief probability, while the strength of the shared residual structure is estimator-dependent.

Using the same CCK projection analysis as in Section 4.5.2, but substituting the Joint-derived conditional estimates, we find that the measured probability systems again lie relatively close to the coherent set (Fig. A11). Importantly, coherence is not guaranteed by the Joint construction: the individual-statement distributions for P and x are estimated independently rather than obtained as marginals of the probed joint distribution. City Locations is typically closest to the coherent set, while Medical Indications and Word Definitions show larger adjustments.

The clearest estimator-dependent efect appears in the absolute distribution of graded stability (Fig. A12). Joint-to-Conditional scores are strongly concentrated near $\gamma = 1$ across all three domains. City Locations remains highly stable, but Medical Indications and Word Definitions are also shifted substantially toward the upper end of the scale, reducing the domain separation observed under Direct Conditional. This diference is consistent with the construction of the estimators: Joint-to-Conditional renormalizes over determinate conditional states, whereas Direct Conditional allows Neither probability mass to reduce the estimated support for $P \mid x .$ . The pronounced domain ordering in the primary analysis should therefore be interpreted as partly dependent on the numerical operationalization of conditional support, even though Direct and Joint stability remain strongly related within models (Section I.2.3).

![](images/6bcc687f1de2c4891402027327cf01ff2b98a949b2e61e308373faf4c0d08a7a.jpg)

![](images/e3f5f052588aa3b55b9f69eeaecf7a9c575546a6b1866c7458b3a2810ff12d36.jpg)

![](images/83f4eb3a0cdaddc9fb06650f57c1e83da4ab94d0b0b5c2ecfbaadb5410e6ce83.jpg)

Figure A11. Distance of Joint-to-Conditional probability estimates from probabilistic coherence. Violin plots show the distribution across $( P , x )$ combinations of the RMS adjustment d<sub>CCK</sub> required to project measured probability distributions for individual statements and conditionals onto the nearest CCK-coherent probability system for (a) City Locations, (b) Medical Indications, and (c) Word Definitions for Llama (red), Gemma (blue), Mistral (yellow), and Qwen (green). Horizontal black lines denote within-model medians. As under Direct Conditional, the measured systems generally lie close to the CCK-coherent set, with City Locations showing the smallest distances for most models.  
![](images/fff78ef9b77869bf8fd3ccb741c5ca99aa027b3b052e8f5f3081a80c796fd302.jpg)

![](images/e6da7edb39d59bc31d53566f502f24d52161372b6ebe2e3847a5f375952cb431.jpg)

![](images/60f6bff692d8d8600da7304c43d839b992cfc7f45bc63bf65b690e72f37976cb.jpg)  
Figure A12. Variation in Joint-to-Conditional graded belief stability across domains. Violin plots show the proposition-level distribution of Joint-to-Conditional graded stability for each of the 12 instruction-tuned LLMs in (a) City Locations, (b) Medical Indications, and (c) Word Definitions for Llama (red), Gemma (blue), Mistral (yellow), and Qwen (green). Horizontal black lines denote within-model medians. Joint-to-Conditional stability is strongly concentrated near $\gamma = 1$ across all three domains, producing weaker domain separation than under Direct Conditional.

The behavioral association is weaker but remains more often positive than negative under Joint-to-Conditional (Fig. A13). Mean ∆M is positive in 26/36 (72.2%) model–domain settings: 10/12 for City Locations, 9/12 for Medical Indications, and 7/12 for Word Definitions. Thus, probability-matched lower-stability beliefs still tend to move more under conversational challenge, but the relationship is less consistent than under Direct Conditional.

## I.2.3 Direct–Joint agreement

Finally, we ask whether the two estimators preserve the relative ordering of beliefs even where their absolute stabilit values difer. For each model and domain, we compute the Spearman correlation between $\gamma _ { \mathcal { M } } ^ { \mathrm { d i r e c t } } ( P )$ and $\gamma _ { \mathcal { M } } ^ { \mathrm { j o i n t } } ( \dot { P } )$ over propositions with valid stability estimates under both approaches (Fig. A14).

The two operationalizations produce positively correlated stability rankings in every model–domain setting. Median Spearman correlations across models are $\rho = 0 . 7 8$ for City Locations, $\rho = 0 . 8 1$ for Medical Indications, and

![](images/02a749b7f75728c8f427d4d38b1e8d8eb6125aa51b569840403ed98f20a19c90.jpg)  
Figure A13. Behavioral resilience using Joint-to-Conditional graded stability among  
probability-matched beliefs. Bars show the mean diference in behavioral movement ∆M between matched lower- and higher-stability beliefs for the 12 instruction-tuned LLMs in (a) City Locations, (b) Medical Indications, and (c) Word Definitions, with error bars denoting ±1 bootstrap standard error, for Llama (red), Gemma (blue), Mistral (yellow), and Qwen (green). Positive values indicate greater movement for the lower-stability belief. Mea $\Delta M$ is positive in 26/36 (72.2%) model–domain settings, although the association is less consistent than under Direct Conditional.

Agreement between graded stability operationalizations  
![](images/41c38190a7a18076417f304a042a9f5ba35afe9ec6b363fb98cd90f1cf51ad99.jpg)  
Figure A14. Agreement between Direct Conditional and Joint-to-Conditional graded stability. Bars show the Spearman correlation between $\gamma _ { \mathcal { M } } ^ { \mathrm { d i r e c t } } ( P )$ and $\gamma _ { \mathcal { M } } ^ { \mathrm { j o i n t } } ( P )$ for each instruction-tuned LLM in (a) City Locations, (b) Medical Indications, and (c) Word Definitions for Llama (red), Gemma (blue), Mistral (yellow), and Qwen (green). Correlations are positive for every model and domain, indicating substantial agreement in the relative ranking of beliefs by stability despite diferences in the estimators’ absolute scales.

$\rho = 0 . 7 1$ for Word Definitions. Thus, despite their diferences in representation and absolute numerical scale, Direct Conditional and Joint-to-Conditional generally agree on which beliefs are relatively more or less stable within a model.

## I.3 Alternative-probe robustness

We repeat the principal analyses using two single-instance linear probes, an SVM [30] and Mass Mean [17], to evaluate whether the graded-stability results depend on the sAwMIL probe used in the main text. Both alternative probes operate on the hidden representation of the final non-padding token. For each probe, model, and dataset, we independently select the transformer layer that minimizes three-class calibration log loss using the procedure in

<table><tr><td></td><td>City Locations</td><td>Mass Mean</td><td></td><td>Medical Indications Mass Mean</td><td></td><td>Word Definitions</td></tr><tr><td>Model</td><td>SVM</td><td></td><td>SVM</td><td></td><td>SVM</td><td>Mass Mean</td></tr><tr><td>llama-3.2-3b</td><td>11</td><td>7</td><td>12</td><td>11 13</td><td>11 13</td><td>11 14</td></tr><tr><td>llama-3.1-8b llama-3.1-70b</td><td>25 76</td><td>17 36</td><td>14 65</td><td>37</td><td>29</td><td>30</td></tr><tr><td></td><td>17</td><td>19</td><td>17</td><td>17</td><td>16</td><td>16</td></tr><tr><td>gemma-7b gemma-2-9b</td><td>22</td><td>23</td><td>20</td><td>21</td><td>20</td><td>19</td></tr><tr><td>gemma-2-27b</td><td>22</td><td>22</td><td>21</td><td>43</td><td>20</td><td>22</td></tr><tr><td>mistral-7b</td><td>14</td><td>16</td><td>14</td><td>15</td><td>16</td><td>15</td></tr><tr><td>mistral-12b</td><td>17</td><td>39</td><td>16</td><td>20</td><td>15</td><td>19</td></tr><tr><td>mistral-3.1-24b</td><td>19</td><td>21</td><td>19</td><td>17</td><td>18</td><td>20</td></tr><tr><td>qwen-2.5-7b</td><td>20</td><td>19</td><td>22</td><td>18</td><td>18</td><td>17</td></tr><tr><td>qwen-2.5-14b</td><td>27</td><td>30</td><td>31</td><td>30</td><td>29</td><td>23</td></tr><tr><td>qwen-2.5-72b</td><td>55</td><td>63</td><td>56</td><td>60</td><td>56</td><td>57</td></tr></table>

Table A9. Selected layers for the alternative probes. Zero-indexed layers selected independently for the SVM and Mass Mean probes for each instruction-tuned LLM. The selected probe-specific layer is subsequently used for the corresponding individual belief probability and Direct Conditional analyses.

Section D.3. The resulting layers are reported in Table A9 and are subsequently used for both the individual-statement and Direct Conditional analyses.

For the SVM probe, we standardize each activation dimension to zero mean and unit variance using a StandardScaler fit only on the training representations. We then fit one LinearSVC per class in a one-versus-all construction, with $C = 1 . 0 , \ell _ { 2 }$ regularization, squared-hinge loss, a maximum of 10,000 iterations, convergence tolerance $1 0 ^ { - 4 }$ , no class weighting, and random seed 0.

For Mass Mean, we construct one class-versus-rest direction for each class $j \in \{ 0 , . . . , K - 1 \}$ . Let $\mu _ { j }$ denote the mean training representation for class $j$ and $\mu _ { \lnot j }$ the mean training representation over all remaining classes. We compute the diference vector

$$
\Delta \mu _ { j } = \mu _ { j } - \mu _ { \lnot j } ,\tag{45}
$$

and normalize $\Delta \pmb { \mu } _ { j }$ to the unit $\ell _ { 2 }$ norm before scoring. The decision boundary is placed halfway between $\mu _ { j }$ and $\mu _ { \lnot j } ,$ and statements are scored by their signed projection along the normalized $\Delta \pmb { \mu } _ { j }$ direction relative to this midpoint.

For both probes, the resulting class scores are converted to probability distributions using the multinomial logistic-regression calibration procedure described in Section D.1. The calibrator is fit only on the calibration split with $C = 1 . 0$ , the L-BFGS solver, a maximum of 5,000 iterations, and tolerance $1 0 ^ { - 6 }$ . Direct Conditional probes are trained separately from the corresponding individual-statement probes on the same conditional training and calibration examples used for sAwMIL, while retaining the probe-specific layer selected from the sweep.

Across these analyses, the SVM largely reproduces the principal representation-based results obtained with sAwMIL, whereas Mass Mean is less consistent across several downstream analyses. Both alternative probes nevertheless preserve the central result that graded stability is not fully explained by the fitted relationship with individual belief probability (Fig. A15). For the SVM, median cross-model residual correlations are $\rho = 0 . 2 5 , 0 . 2 0$ , and 0.19 for City Locations, Medical Indications, and Word Definitions, respectively. The corresponding Mass Mean medians are $\rho = 0 . 1 7 , 0 . 1 4$ , and 0.22. Thus, under both alternative probes, substantial proposition-level variation remains after accounting for individual belief probability, and models continue to show positively correlated residual stability.

The probabilistic-coherence results show a clearer diference between the two alternative probes (Fig. A16). The SVM yields distance-to-coherence distributions broadly similar to those obtained with sAwMIL. By contrast, Mass Mean generally requires larger adjustments to reach the CCK-coherent set, particularly for City Locations and Word Definitions.

The domain-level characterization shows the same pattern (Fig. A17). Under the SVM, City Locations remains concentrated near high graded stability, while Medical Indications and Word Definitions exhibit broader distributions extending further into intermediate and low stability values. This separation is substantially weaker under Mass Mean, with several City Locations models in particular receiving much lower graded-stability estimates than unde either sAwMIL or the SVM.

Finally, the behavioral association is more robust under the SVM than under Mass Mean (Fig. A18). SVM-based graded stability remains predominantly associated with greater behavioral resilience among probability-matched beliefs, although the efects are more heterogeneous than in the primary sAwMIL analysis. Under Mass Mean, the estimated efects are generally smaller and less consistent in direction.

![](images/91f73bdeafa6e4c02382b5d7f3af9d01de5f32ca7097e0060c0baf61c8373cfc.jpg)

![](images/0cc3dc99b8eb912c4f6fc394c06e7e84a11e618d8d44fc4e934dca64d49ceb51.jpg)  
Figure A15. Relationship between individual belief probability and graded stability under alternative probes. For the SVM, we show held-out $R ^ { 2 }$ for predicting graded stability from individual belief probability in (a) City Locations, (b) Medical Indications, and (c) Word Definitions, respectively, for Llama (red), Gemma (blue), Mistral (yellow), and Qwen (green). Panel (d) shows pairwise cross-model Spearman correlations between residual graded-stability scores. Panels (e–h) report the corresponding Mass Mean results. Black lines in (d) and (h) denote median residual correlations. Under both probes, individual belief probability leaves substantial proposition-level variation unexplained and residual stability remains largely positively shared across models.

The weaker Mass Mean results are consistent with known sensitivities of centroid-based truth directions. Mass Mean was introduced as a binary method that estimates a truth direction from the diference between the centroids of True and False representations [17]. Previous work applying this probe in a three-class setting found that it can become unstable when Neither examples are incorporated, since changes in the composition of the representation sets directly alter the estimated class centroids [9]. Our trivalent implementation similarly constructs three classversus-rest centroid directions for True, False, and Neither and calibrates the resulting scores into a shared multiclass probability distribution.

![](images/8fad22c9fbf05b7bfd4951676f53cc1931b595652b64749e377f546e9ef97e50.jpg)

![](images/b0a54d17690981cb144531befa5b8ae6238ab4cd8f4ca13cf17b43fafd7d7716.jpg)

![](images/6a1f805f5fbd21efdb89b3189f1a5c8ea2dd1bce666b8f49dea0683bd5180264.jpg)

![](images/5fb0fad49e8f149b1897843786c9340993b95051502f3d2219b9cd0ae4d93755.jpg)

(e)  
![](images/37db048a1362f63214ef97decd3ac2cd20024b1fcda9cc136bcb7841dd752911.jpg)  
(f)

![](images/3b0ac5e24000449c0a9d4258097f57403733c94841a0ca9ea4382db69008d387.jpg)  
Figure A16. Distance from probabilistic coherence under alternative probes. Violin plots show the RMS adjustment d<sub>CCK</sub> required to project measured probability distributions for individual statements and conditionals onto the nearest CCK-coherent system. Panels (a–c) show results for the SVM for Llama (red), Gemma (blue), Mistral (yellow), and Qwen (green), and panels (d–f) for Mass Mean, with columns corresponding to City Locations, Medical Indications, and Word Definitions. Horizontal black lines denote within-model medians. The SVM broadly reproduces the approximate-coherence pattern obtained with sAwMIL, whereas Mass Mean yields larger and more heterogeneous distances from the coherent set.

This sensitivity is especially important for graded stability because our downstream quantities depend on the absolute allocation of probability mass across all three classes, not merely on whether a binary truth direction ranks True statements above False statements. Mass Mean can therefore remain informative as a veracity probe while being less well suited to the trivalent probability estimation required here. We consequently view the strong SVM replication as evidence that the principal results are not specific to the multi-instance structure of sAwMIL, while the Mass Mean results identify a meaningful sensitivity to probes that are not naturally designed for multiclass probability estimation.

![](images/7790502fd1392e6eb30d94f0b7a412c4a11be99dd39d34bdb2632bbc7a86603d.jpg)

![](images/30bd03050b4bfffc537c748ae8285623bcb95724a4b05a4299c43ec0691b3f27.jpg)

![](images/4f81389fe51b7c2ca7c63296b35dae23f8ac8814191b007e76ecd0a94a792216.jpg)

![](images/e2849e3e8ea039ea5872841dd97e25138e3f62bc275a95c949340ced5f61fd23.jpg)

![](images/2b42661c9b01dfd0732332e39155f29aa92d370f2ab698bfa30fe263768a1696.jpg)

![](images/2ada982797e4ab946c9765bdf0496eb24d51ad54ad25284c40279ddfd0b9376d.jpg)  
Figure A17. Variation in graded belief stability under alternative probes. Violin plots show proposition-level Direct Conditional graded-stability distributions. Panels (a–c) show results for the SVM for Llama (red), Gemma (blue), Mistral (yellow), and Qwen (green) and panels (d–f) for Mass Mean, with columns corresponding to City Locations, Medical Indications, and Word Definitions. Horizontal black lines denote within-model medians. The SVM preserves the primary domain-level pattern, whereas Mass Mean produces more heterogeneous distributions and substantially weaker separation across domains.

![](images/1b70568baacac49c5cbf5b2e577d6d39ac22ad2510db49d755bbf6e2447c15d2.jpg)

![](images/15705d141c235db52e5186096efb8b3034a8f23c2886a6729008432a0488b909.jpg)

![](images/f4af2d4fb63a02f689be62746bc4103fb4c27dd9f0a2b6ac86179d372e89ff43.jpg)

![](images/d6e471feb9f403ba4b9c87753e311c4131fb47c65781608a0f57eacf5799bd44.jpg)

![](images/f305a87384331559f8f4481d543d274f059301b12c7545691228b2f10e6bc889.jpg)

![](images/bf8e9a61b0edd8e1f0c3a4aecb0df5465be17e1727f5cd57be7b24ce2b5078b1.jpg)  
Figure A18. Behavioral resilience under alternative probes. Bars show the mean behavioral movement diference ∆M between probability-matched lower- and higher-stability beliefs. Panels (a–c) show results for the SVM for Llama (red), Gemma (blue), Mistral (yellow), and Qwen (green) and panels (d–f) for Mass Mean, with columns corresponding to City Locations, Medical Indications, and Word Definitions. Error bars denote ±1 bootstrap standard error, and positive values indicate greater movement for the lower-stability belief. The directional association remains visible across many SVM model–domain combinations but is weaker and less consistent under Mass Mean.