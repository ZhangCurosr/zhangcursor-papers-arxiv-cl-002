# Vision-Language Models Integrate Conflicting Evidence Like Biased Bayesian Observers

Josh Barua Kyle Mahowald William Rudman

The University of Texas at Austin {joshbarua,mahowald,william.rudman}@utexas.edu

## Abstract

Vision-language models must reconcile conflicting text and images. Prior work has characterized the resulting behavior in terms of modality biases—e.g., a “blind faith in text”—but why preferences vary across models, domains, and input conditions remains unclear. We develop a predictive account from a Bayesian perspective, positing that VLMs combine conditionally independent evidence from each modality with prior beliefs, subject to modalityspecific biases. The resulting two-parameter cue combination model accurately predicts VLM preferences under unseen combinations of input degradation and transfers to new forms of unreliability without refitting. Separating evidence strength from integration bias explains how VLMs can follow text even while weighting visual evidence more heavily, reconciling reports of text and vision dominance under a single combination rule.

## 1 Introduction

Vision-language models (VLMs) are increasingly being deployed in high-stakes environments, such as web browsing (He et al., 2024) and medical diagnostics (Moor et al., 2023), where multimodal inputs may provide conflicting evidence. Such conflicts can arise from noisy sensors (Liao et al., 2025), adversarial attacks (Schlarmann and Hein, 2023; Cui et al., 2024), human errors (Brady, 2017), or mismatches between evidence and parametric knowledge (Liu et al., 2025; Golovanevsky et al., 2025). To better understand how VLMs handle such conflicts, recent studies have focused on modality preference, the extent to which a VLM follows one modality over another when the two disagree. Early work frequently described VLM behavior in terms of particular biases. Deng et al. (2025) reported a “blind faith in text” where VLMs excessively trust text over vision when inconsistencies arise; others reported vision-leaning behavior (Yan et al., 2026) or biases that vary with model architecture (Hua et al., 2025) and query complexity (Pezeshkpour et al., 2025). Mechanistic studies reinforce this picture, finding that the modality dominating an answer can often be traced to specific internal components such as the attention mechanism (Hua et al., 2025; Zheng et al., 2026; Rudman et al., 2026) or steering directions (Golovanevsky et al., 2025; Zhang et al., 2026; Nooralahzadeh et al., 2026). More recently, Zhang et al. (2025b) showed that preference depends on both an inherent bias and each modality’s relative uncertainty, with predictions oscillating across layers when the two are balanced.

This broader trajectory closely mirrors one in the study of human perception. Conflicts between senses were initially explained as one sense dominating another (Rock and Victor, 1964), as in the ventriloquism effect, where a speech sound can be perceived as coming from somewhere other than its true physical source if it matches visual cues (Howard and Templeton, 1966). Later experiments that degraded visual cues instead showed that which sense dominates can change with cue reliability (Heller, 1983). These findings can be explained by Bayesian ideal-observer models in which each sense contributes evidence weighted by its reliability, so that influence shifts gradually rather than switching (Ernst and Banks, 2002; Alais and Burr, 2004; Knill and Pouget, 2004). The framework also remained useful where people fell short of the ideal observer. A systematic bias toward vision (Battaglia et al., 2003) and individual differences in how listeners weight sound and sight (Schwartz, 2010) were captured by additional parameters within it. Motivated by this history, we ask whether the same framework can usefully explain how VLMs arbitrate between text and images.

This work. Following studies of human perception, we design a controlled setup to independently degrade the reliability of textual and visual evidence under conflict (§2). Consistent with Zhang et al. (2025b), VLM preferences shift smoothly toward whichever source is more reliable, and this pattern varies systematically across models and domains. However, such shifts describe behavior without revealing the computational strategy that produces it (Marr, 1982). A VLM that switches to the more confident modality and one that blends the two modalities could both produce the same behavior. Likewise, a VLM may follow the text because the passage supplies stronger evidence or because it weights textual evidence more heavily. To distinguish these possibilities, we derive how a Bayesian observer would integrate evidence across modalities (§3). Under two assumptions, this yields a linear cue combination model in which the VLM (i) treats each modality’s evidence as independent given the answer, (ii) weights it by a modalityspecific bias, and (iii) combines it additively to its prior beliefs. Our model accurately predicts VLM preferences under unseen combinations of degradation and transfers to new forms of unreliability without refitting. This predictive success lets us interpret the model’s components as a computationallevel theory of how VLMs arbitrate between modalities. This recasts the “blind faith in text” (Deng et al., 2025) as largely a consequence of stronger textual evidence, which modality bias can amplify or offset, and reconciles reports of text and vision dominance under a single combination rule.

![](images/d1cb6f0c3f705fd0a5a91fa70875acf257e296b285f57d78881d271901f122bf.jpg)  
Figure 1: Overview. (Left) In the conflicting VQA task, a question is paired with a text passage and an image that support different answers $( a _ { T } \neq a _ { V } )$ ; the VLM arbitrates between them and outputs a preference. (Middle) We independently degrade textual evidence (text noise level $p )$ and visual evidence (Gaussian blur σ); as the reliability of each modality shifts, the VLM’s preference transitions between the text- and vision-supported answers. (Right) A two-parameter Bayesian model over the unimodal Bayes factors $( e _ { T } , e _ { V } )$ and the prior belief (π<sub>0</sub>) accurately predicts the joint preference $s _ { J }$ of the VLM.

## 2 Do Modality Preferences Track Evidence Reliability?

Most existing modality-conflict benchmarks pair each question with clean text and image inputs that differ only in the answer they support (Hua et al., 2025; Deng et al., 2025; Pezeshkpour et al., 2025; Zhang et al., 2026). These evaluations measure how often a VLM follows each modality, but because the evidence is held fixed, they are limited to coarse heuristics (e.g., a VLM chose the text answer 80% of the time) and cannot show how preference depends on the certainty of the evidence itself. We therefore borrow the cue-conflict paradigm from human perception and manipulate the reliability of each modality’s evidence independently.

## 2.1 Task, Data, and Models

Following prior work, our chosen setting is conflicting VQA. Formally, each example is a tuple $( q , t , v , a _ { T } , a _ { V } )$ where q is a question, t is a text passage, v is an image, $a _ { T }$ is the answer supported by the text, and $a _ { V }$ is the answer supported by the image, with $a _ { T } \neq a _ { V }$ . In Figure 1 (left), the question is “What color is the $\mathrm { s k y ? ^ { \prime } }$ ; the passage describes the sky as overcast and gray $( a _ { T } = g r a y )$ , while the image shows a clear blue sky $( a _ { V } = b l u e )$ . Because neither answer is privileged as correct, this task isolates how a VLM arbitrates between sources from how accurately it answers.

![](images/b18e800f147b702707ee28f872dae0c07921bab40ceb0da65a93658ebd04cc81.jpg)  
Figure 2: Modality preferences shift sigmoidally with reliability and vary across datasets and VLMs. Mean $P _ { J }$ (probability of the text answer) as evidence from each modality is degraded. Within each panel, the left half is text noise p with blur $\sigma { = } 0 ,$ , the right half is blur σ with text noise p=0, and the clean input $( p { = } 0 , \sigma { = } 0 )$ sits at the center. Left: per-dataset curves for Qwen3.5-4B. Right: per-model curves averaged over all datasets.

We construct a dataset of 2,765 examples from three VQA benchmarks covering distinct visual domains: document images from DocVQA (Mathew et al., 2021), natural photographs from VQAv2 (Antol et al., 2015), and real-world scenes from RealworldQA (xAI, 2024). For DocVQA and VQAv2, the conflicting passages are sourced from an existing benchmark (Deng et al., 2025); for RealworldQA, we generate them with an LLM. Further details are provided in Appendix A.1.

We evaluate four open-weight VLMs: Qwen2.5- VL-3B (Qwen Team, 2025), Qwen3.5-4B (Qwen Team, 2026), Gemma3-4B (Gemma Team et al., 2025), and GLM-4.6V-Flash (GLM-V Team et al., 2025). They differ in vision encoder, languagemodel attention, and vision–language fusion, which lets us determine whether our conclusions are consistent across architectures. Appendix F.2 lists the model checkpoints used in the experiments.

## 2.2 Manipulating Evidence Reliability

We introduce two corruptions that independently degrade the reliability of each modality (Figure 1, middle). For images, we apply isotropic Gaussian blur with $\sigma \in \{ 0 , 2 , 4 , 8 , 1 6 , 3 2 \}$ pixels, a standard controlled corruption in robustness studies (Borkar and Karam, 2019; Hendrycks and Dietterich, 2019). For text, we introduce text noise by randomly replacing a fraction $p \in \{ 0 . 0 , 0 . 2 , 0 . 4 , 0 . 6 , 0 . 8 , 1 . 0 \}$ of the words in the passage. Replacement words are drawn from a vocabulary of that dataset’s passages without regard to part of speech, and substitutions are nested: a word replaced at level p stays replaced, by the same word, at every higher level. This corruption is motivated by noisy-channel models of language comprehension, which treat observed language as a potentially corrupted realization of an intended message (Levy, 2008; Gibson et al., 2013; Ryskin et al., 2018). Under this view, out-of-place words and broken syntax are themselves evidence that a passage is unreliable. Crossing the two corruptions yields a $6 \times 6$ grid of $( \sigma , p )$ conditions for every example.

## 2.3 Measuring Modality Preference

We define a VLM’s modality preference on a conflicting example as the log-odds between the textand vision-supported answer, given the question and both sources of evidence:

$$
s _ { J } = \log { \frac { P ( a _ { T } \mid t , v , q ) } { P ( a _ { V } \mid t , v , q ) } } .\tag{1}
$$

Throughout this work, a probe is a behavioral measurement in the spirit of cloze-style probes of language model knowledge (Petroni et al., 2019). This differs from the more common usage, in which a probe is a classifier trained on a model’s internal representations (Ettinger et al., 2016; Alain and Bengio, 2017). We call $s _ { J }$ the joint probe because both modalities are in context, distinguishing it from the unimodal probes introduced in Section 3. Its sign indicates which answer the model prefers $( s _ { J } > 0$ favors a<sub>T</sub>; $s _ { J } < 0$ favors $a _ { V } )$ , and its magnitude indicates how strongly. We score each candidate answer by its log-likelihood averaged over its tokens, so longer answers are not penalized. For visualization, we convert $s _ { J }$ to $P _ { J } = 1 / ( 1 + e ^ { - s _ { J } } )$ , the probability of choosing a<sub>T</sub> over $a V .$ . Appendix B gives the exact scoring definition and prompt templates and Appendix B.5 verifies that the preference implied by the forcedchoice log-odds largely agrees with the preference VLMs exhibit after reasoning.

## 2.4 Results

Figure 2 shows the mean text-answer preference probability, $P _ { J }$ , as each modality is degraded. Corrupting the passage shifts preference toward $a _ { V }$ , while blurring the image shifts it toward $a _ { T }$ This matches the trends observed by Zhang et al. (2025b) when varying the difficulty of each modality under conflict. Preferences thus respond systematically to changes in the available information, rather than remaining fixed on one modality.

These responses vary across both VLMs and datasets. Qwen2.5-VL-3B and Gemma3-4B lean toward the text answer more than Qwen3.5-4B and GLM-4.6V-Flash (Fig. 2, right), consistent with earlier reports of text-following (Deng et al., 2025). Yet the direction of preference also changes within a single VLM. On clean inputs, Qwen3.5-4B leans toward the image on DocVQA but toward the text on RealworldQA (Fig. 2, left). Additionally, its sensitivity to degradation varies. On DocVQA, mild blur $( \sigma { = } 2 )$ is enough to move its preference from the image answer to the text answer, whereas the shift on the other datasets is more gradual. Both patterns likely reflect DocVQA’s reliance on finegrained document details, which make a legible image highly informative but are lost under even mild blur. A VLM’s modality preference therefore depends on both the task and the degradation of its inputs. The curves alone cannot tell whether a shift reflects stronger evidence, heavier weighting of that evidence, or the prior. Next, we derive a Bayesian model that separates these contributions.

## 3 A Bayesian Cue Combination Model

To explain why VLM preferences shift with the reliability of either modality, we turn to models of cue combination from human perception. When cues from different senses conflict, people weight each by its reliability, often closely matching an ideal Bayesian observer (Ernst and Banks, 2002; Alais and Burr, 2004). In a categorical choice like ours, this weighting is implicit since a less reliable cue simply contributes weaker evidence (Massaro, 1998; Movellan and McClelland, 2001). We formalize this account for VLMs, starting from a calibrated observer (§3.1) and then relaxing it to allow systematic deviations (§3.2), making each assumption explicit and testable.

## 3.1 A Calibrated Bayesian Observer

We treat the joint probe $s _ { J }$ (Eq. 1) as the posterior log-odds of an implicit observer. Applying Bayes rule decomposes $s _ { J }$ into a Bayes factor for the evidence and the prior log-odds:

$$
s _ { J } = \underbrace { \log \frac { P ( t , v \mid q , a _ { T } ) } { P ( t , v \mid q , a _ { V } ) } } _ { \mathrm { j o i n t ~ B a y e s ~ f a c t o r } } + \underbrace { \log \frac { P ( a _ { T } \mid q ) } { P ( a _ { V } \mid q ) } } _ { \mathrm { p r i o r ~ l o g - o d d s } } .\tag{2}
$$

We define the prior probe $\begin{array} { r } { \pi _ { 0 } = \log { \frac { P ( a _ { T } | q ) } { P ( a _ { V } | q ) } } } \end{array}$ , the model’s baseline preference between the two answers without evidence. We further decompose the joint Bayes factor into per-modality terms.<sup>1</sup>

Assumption 1 (Cross-modal conditional independence). Given the answer $A ~ \in ~ \{ a _ { T } , a _ { V } \}$ the VLM treats the text and image as conditionally independent evidence: $P ( t , v \mid A ) = P ( t \mid$ $A ) \cdot P ( v \mid A )$ . This is the standard assumption in models of human cue integration, which treat cues from different senses as measurements of a common latent stimulus, each corrupted by independent sensory noise (Ernst and Banks, 2002; Knill and Pouget, 2004). Under this assumption, the joint Bayes factor splits additively:

$$
s _ { J } = \underbrace { \log \frac { P ( t \mid a _ { T } ) } { P ( t \mid a _ { V } ) } } _ { \mathrm { t e x t ~ B a y e s ~ f a c t o r } } + \underbrace { \log \frac { P ( v \mid a _ { T } ) } { P ( v \mid a _ { V } ) } } _ { \mathrm { v i s i o n ~ B a y e s ~ f a c t o r } } + \pi _ { 0 } .\tag{3}
$$

From Bayes factors to observable probes. The per-modality Bayes factors in Eq. 3 are not directly observable. We can, however, measure the VLM’s answer preference when only one modality is in context, using two unimodal probes:

$$
s _ { T } = \log \frac { P ( a _ { T } \mid t ) } { P ( a _ { V } \mid t ) } , s _ { V } = \log \frac { P ( a _ { T } \mid v ) } { P ( a _ { V } \mid v ) } .\tag{4}
$$

By Bayes’ rule, each unimodal probe equals its Bayes factor plus the prior: $\begin{array} { r } { s _ { T } = \log \frac { P ( t | a _ { T } ^ { - } ) } { P ( t | a _ { V } ) } + \pi _ { 0 } } \end{array}$ and likewise for $s _ { V }$ . For brevity, we define e<sub>T</sub> and $e _ { V }$ as the text and vision Bayes factors, thus $s _ { T } =$ $e _ { T } + \pi _ { 0 }$ and $s _ { V } = e _ { V } + \pi _ { 0 } .$ . Finally, we rearrange and substitute $e _ { T }$ and $e _ { V }$ into Eq. 3, giving the calibrated naive-Bayes prediction:

$$
s _ { J } = e _ { T } + e _ { V } + \pi _ { 0 } = s _ { T } + s _ { V } - \pi _ { 0 } .\tag{5}
$$

The observer is naive because it treats text and image as independent evidence given the answer, and calibrated because each modality shifts the log-odds as much in combination as alone.

## 3.2 Allowing for Integration Bias

Eq. 5 gives each Bayes factor unit weight. However, a VLM may systematically over- or underweight one modality relative to a calibrated observer. Prior studies found that fitting human behavior required modeling such biases (Battaglia et al., 2003; Schwartz, 2010). We therefore relax the unit-weight assumption.

Assumption 2 (Modality-specific weighting). The VLM’s multimodal decision combines the text and vision Bayes factors with modality-specific weights $\beta _ { T } , \beta _ { V } \geq 0$ . The weights are shared across examples, whereas evidence reliability still enters per example through $e _ { T }$ and $e _ { V }$ . Their ratio $\beta _ { T } / \beta _ { V }$ measures integration bias, with values above one indicating greater weight on text evidence.

Linear Cue Combination Model. Applying these weights to the per-modality Bayes factors gives the final model shown in Figure 1 (right):

$$
\boxed { s _ { J } = \beta _ { T } e _ { T } + \beta _ { V } e _ { V } + \pi _ { 0 } . }\tag{6}
$$

This model decomposes a VLM’s modality preference on each example into three sources: its prior between the candidate answers $\left( \pi _ { 0 } \right)$ , the evidence each modality provides on that example $( e _ { T } , e _ { V } )$ and the modality weights $( \beta _ { T } , \beta _ { V } )$

## 3.3 Competing Predictors

The linear cue combination model rests on implicit and explicit modeling choices: VLMs integrate both modalities rather than selecting one, evidence combines additively (Assumption 1), each modality carries its own weight (Assumption 2), and the prior enters with unit weight and no intercept. We test each with a predictor that removes or relaxes $\mathbf { i t } ,$ mapping the same probes $( s _ { T } , s _ { V } , \pi _ { 0 } )$ to a prediction $\hat { s } _ { J }$

• Best Single Probe follows the more confident modality: $\hat { s } _ { J } = s _ { T } \mathrm { i f } | s _ { T } | \geq | s _ { V } | ,$ , else $s _ { V }$ . This rule is reliability-sensitive and could in principle produce the shifts observed in §2, but it ignores the weaker modality entirely. Our Bayesian model outperforming this probe is therefore evidence that VLMs blend information from both modalities rather than a winner-take-all strategy.

• Bilinear Cue Combination adds an interaction term to Eq. 6: $\hat { s } _ { J } = \beta _ { T } e _ { T } + \beta _ { V } e _ { V } + \beta _ { I } e _ { T } e _ { V } +$ $\pi _ { 0 }$ . The effective weight on text evidence becomes $\beta _ { T } + \beta _ { I } e _ { V }$ , so the passage’s contribution depends on how strongly the image favors an answer, and vice versa. Assumption 1 implies $\beta _ { I } = 0$ , so a substantial gain from the interaction term would count against it.

• Calibrated Naive Bayes is Eq. 5, the special case where $\beta _ { T } = \beta _ { V } = 1$ . If it performed as well as the linear model, the modality weights (and therefore Assumption 2) would be deemed unnecessary.

• Unconstrained Linear fits $\hat { s } _ { J } = \beta _ { T } e _ { T } + \beta _ { V } e _ { V } +$ $\beta _ { \pi } \pi _ { 0 } + \beta _ { 0 } .$ , equivalent to linear regression on $s _ { T } , s _ { V }$ , and $\pi _ { 0 }$ with an intercept. Our Bayesian parameterization fixes $\beta _ { \pi } = 1$ and $\beta _ { 0 } = 0$ . Any gain over the Bayesian model measures the cost of these restrictions, while the fitted coefficients indicate departures from them.

Further details about the predictors can be found in Appendix C.2.

## 4 Testing the Bayesian Model

If linear cue combination is a useful computationallevel account of modality preferences in VLMs, it should generalize beyond the conditions on which it was fit. We therefore evaluate the Bayesian model on unseen combinations of degradation (§4.2), against competing integration rules (§4.3), and on new reliability manipulations without refitting (§4.4). If the model successfully generalizes, we may then interpret it and offer explanations for previously observed phenomena.

## 4.1 Fitting and Evaluation

For each VLM and dataset, we fit the free parameters of the linear, bilinear, and unconstrained predictors by minimizing mean squared error (MSE) between $\hat { s } _ { J }$ and $s J$ with full-batch L-BFGS, using the validation set for early stopping. The best single probe and calibrated naive Bayes have no fitted parameters. Examples are split into train, validation, and test sets by example ID, so all $( \sigma , p )$ variants of an example belong to the same split. We report test-set $R ^ { 2 }$ between predicted and observed joint preference scores, and KL divergence between the two-answer distributions induced by those scores.

Out-of-distribution protocol. Our main evaluation fits the parameters on the boundary of the degradation grid, $\{ ( \sigma , 0 ) \} \cup \{ ( 0 , p ) \}$ (11 cells), where at most one modality is degraded. We then test on held-out examples in the 25 interior cells, where both modalities are degraded. This protocol requires generalization to both new examples and unseen combinations of the same degradation levels, explicitly targeting the additive combination rule implied by Assumption 1. Unless otherwise stated, evaluations in this section use this protocol.

![](images/d9f41a99c3258ea4c8090b304f4e6497c884572a9ebb6cddbd88878fc031adaa.jpg)  
Figure 3: Linear cue combination predicts the transition between text- and vision-dominated behavior. Rows show DocVQA, VQAv2, and RealworldQA. Columns show empirical Qwen3.5-4B preferences (left) and model predictions (right), averaged over held-out examples under the OOD protocol. Solid and dashed lines mark the empirical $( s _ { J } = 0 )$ and predicted $( \hat { s } _ { J } = 0 )$ transition boundaries, respectively.

## 4.2 Predicting Joint Preferences from Unimodal Probes

As shown in Table 1, linear cue combination provides a good quantitative fit for VLM responses with $R ^ { 2 }$ of 0.80–0.86 for both Qwen models and GLM-4.6V, and 0.69 for Gemma3. Figure 3 shows what this accuracy means at the level of the preference surface, using Qwen3.5-4B. Averaged over held-out examples, the predictions reproduce the dataset-specific patterns from Section 2, including the abrupt shift toward text under mild blur on DocVQA, and the more gradual transitions on VQAv2 and RealworldQA. On all three datasets, the predicted transition boundary $( \hat { s } _ { J } = 0 )$ closely tracks the observed one. The same agreement holds for the other three VLMs we study (Appendix D.2), and for larger models from the same families (Appendix E.3). Thus, unimodal preferences measured independently surprisingly capture most of the variation in VLMs preferences when both modalities are present and in conflict. Gemma3’s lower fit shows that the approximation is less complete for some models.

<table><tr><td></td><td>Predictor</td><td>KL↓</td><td> $R ^ { 2 } \uparrow$ </td></tr><tr><td rowspan="5">Ow22.5</td><td>Best Single Probe</td><td>0.161</td><td>0.609</td></tr><tr><td>Calibrated Naive Bayes</td><td>0.097</td><td>0.717</td></tr><tr><td>Linear Cue Combination</td><td>0.057</td><td>0.852</td></tr><tr><td>Bilinear Cue Combination</td><td>0.058</td><td>0.851</td></tr><tr><td>Unconstrained Linear</td><td>0.058</td><td>0.852</td></tr><tr><td rowspan="5">Own35</td><td>Best Single Probe</td><td>0.240</td><td>0.452</td></tr><tr><td>Calibrated Naive Bayes</td><td>0.137</td><td>0.477</td></tr><tr><td>Linear Cue Combination</td><td>0.075</td><td>0.800</td></tr><tr><td>Bilinear Cue Combination</td><td>0.075</td><td>0.801</td></tr><tr><td>Unconstrained Linear</td><td>0.066</td><td>0.817</td></tr><tr><td rowspan="4">Gema3</td><td>Best Single Probe</td><td>0.787</td><td>0.472</td></tr><tr><td>Calibrated Naive Bayes</td><td>0.554</td><td>0.504</td></tr><tr><td>Linear Cue Combination</td><td>0.336</td><td>0.692</td></tr><tr><td>Bilinear Cue Combination Unconstrained Linear</td><td>0.331</td><td>0.690</td></tr><tr><td rowspan="5">GL6V</td><td>Best Single Probe</td><td>0.352</td><td>0.690</td></tr><tr><td>Calibrated Naive Bayes</td><td>0.417</td><td>0.592</td></tr><tr><td>Linear Cue Combination</td><td>0.268 0.099</td><td>0.625 0.860</td></tr><tr><td>Bilinear Cue Combination</td><td></td><td>0.860</td></tr><tr><td>Unconstrained Linear</td><td>0.099 0.099</td><td>0.866</td></tr></table>

Table 1: Goodness of fit of each predictor to VLM modality preferences under the OOD protocol. $R ^ { 2 }$ is computed between the predicted $\hat { s } _ { J }$ and observed $s _ { J } ;$ KL is the divergence between the observed and predicted answer distributions. Metrics are averaged equally across DocVQA, VQAv2, and RealworldQA. Predictors above the inner rule have at most two free parameters $( \beta _ { T } , \beta _ { V } )$ . Predictors below it add expressivity: a cross-modal interaction term (bilinear) or a free prior coefficient and intercept (unconstrained linear). Bold marks the best predictor above the rule.

## 4.3 Comparing Alternative Explanations

A strong fit alone does not establish that linear cue combination has the right functional form. Table 1 compares it with the competing predictors (§3.3), each configuration provides a different account of how a VLM could produce its joint preferences from the same probes. The best single probe trails by 0.22–0.35 in $R ^ { 2 }$ , which supports the argument that VLMs integrate evidence rather than deferring to the more confident modality. Calibrated naive Bayes trails by 0.14–0.32, showing that fitting the modality weights substantially improves prediction over combining the evidence with no weights. Additional flexibility yields much smaller gains. Adding the bilinear interaction improves $R ^ { 2 }$ by at most 0.001, while unconstrained linear regression improves it by at most 0.017. Although attention can mix the two modalities at any layer, and indeed does (Kaduri et al., 2025; Zhang et al., 2025a), we find that joint preferences are well approximated by an additive function of the unimodal evidence, as Assumption 1 predicts. The unconstrained regression also tests the restriction Bayes’ rule places on the prior. For three of the four VLMs, its fitted prior coefficient is close to one (Table 10). The exception is Qwen3.5 $( \beta _ { \pi } = 0 . 5 3 – 0 . 6 6 )$ , whose question-only prior contributes less to joint preferences than Bayes’ rule implies. Thus, enforcing the Bayesian constraint costs little predictive accuracy.

<table><tr><td></td><td>Fit Grid</td><td>Test Grid</td><td> $R ^ { 2 }$ </td></tr><tr><td rowspan="2">Owe2.5</td><td>relevance × occlusion</td><td>relevance × occlusion</td><td>0.850</td></tr><tr><td>text noise × blur</td><td>relevance × occlusion</td><td>0.852</td></tr><tr><td rowspan="2">Oen33.5</td><td>relevance × occlusion</td><td>relevance × occlusion</td><td>0.806</td></tr><tr><td>text noise × blur</td><td>relevance × occlusion</td><td>0.785</td></tr><tr><td rowspan="2">Gema3</td><td>relevance × occlusion</td><td>relevance × occlusion</td><td>0.676</td></tr><tr><td>text noise × blur</td><td>relevance × occlusion</td><td>0.683</td></tr><tr><td rowspan="2">GL6V</td><td>relevance × occlusion</td><td>relevance × occlusion</td><td>0.828</td></tr><tr><td>text noise × blur</td><td>relevance × occlusion</td><td>0.832</td></tr></table>

Table 2: Linear cue combination and its weights generalize across reliability manipulations. $R ^ { 2 }$ on joint log-odds, averaged equally across datasets. For each VLM, the first row refits weights on the new grid; the second tests generalization by transferring weights from text noise × blur without refitting. Both use the OOD protocol and identical test examples on relevance × occlusion.

## 4.4 Do Modality Weights Transfer?

Linear cue combination separates evidence, which the probes measure separately for each input, from modality weights, which are fixed for each VLM and dataset. A challenging test of this separation is whether weights fit under one set of reliability manipulations still predict preferences under another. We introduce two new manipulations that alter the evidence differently from the original corruptions (§2.2). Pixel occlusion blacks out a growing fraction of the image, removing information locally while leaving the remaining pixels intact. Textual relevance weakens a passage’s support for the answer while preserving fluency and plausibility (details in Appendix A.5). After measuring the probes on the new relevance × occlusion grid, we compare two sets of weights, each evaluated with the OOD protocol on the same held-out examples. Transferred weights are fit on text noise × blur and applied without refitting to the new grid. Refitted weights are fit on the relevance × occlusion grid and serve as a reference. If the weights are stable, the transferred weights should predict as well as the refitted ones, and the two sets of weights should be similar.

Strikingly, Table 2 shows that both predictions hold. Refitted weights reach $R ^ { 2 }$ of 0.68–0.85, and transferred weights come within 0.021 of the refit value for every VLM. The refitted weights are also close to the transferred ones (App. Table 11), so the new manipulations recover nearly the same modality weights. This supports a key prediction of the model and suggests that the fitted modality weights are stable within each VLM–dataset pair.

## 4.5 Explaining Modality Preferences

Having shown that linear cue combination predicts joint preferences and that its weights transfer across reliability manipulations, we now turn to the interpretation it offers. Under the rule $\hat { s } _ { J } = \pi _ { 0 } + \beta _ { T } e _ { T } + \beta _ { V } e _ { V }$ , a VLM’s modality preference reflects the evidence each modality supplies, the weight assigned to that evidence, and the prior. Table 3 summarizes the first two factors as the clean-input evidence ratio, $| \bar { e } _ { T } | / | \bar { e } _ { V } |$ , and the integration bias, $\beta _ { T } / \beta _ { V }$

On clean inputs, mean textual evidence exceeds mean visual evidence by a factor of 1.4–3.8 for every VLM on every dataset. This is expected given how the stimuli are constructed: the text states the answer while the image must be interpreted, though the ratio also depends on each model’s ability to extract this information. Stronger textual evidence can therefore lead a VLM to follow the text even with equal modality weights, unless the prior offsets this advantage. Gemma3 illustrates this: it leans toward text more than Qwen3.5 and GLM-4.6V (Fig. 2, right), yet weights the two modalities nearly equally or modestly favors text $( \beta _ { T } / \beta _ { V } = 0 . 9 6 \mathrm { - } 1 . 2 7 )$ . Qwen3.5 and GLM-4.6V instead weight visual evidence more heavily on every dataset $( \beta _ { T } / \beta _ { V } = 0 . 5 6 \mathrm { - } 0 . 6 8 )$ , counteracting the text advantage. This vision-favoring weighting holds under both example- and cell-level fitting, whereas some smaller integration biases for Qwen2.5-VL and Gemma3 depend on the fitting level. Strong text-following can thus coexist with only modest integration bias. By the same logic, the “blind faith in text” reported by Deng et al. (2025) and the visual preference reported in other models (Hua et al., 2025; Yan et al., 2026) do not necessarily reflect different integration strategies. Under a single additive rule, which modality wins depends on how much evidence each supplies and how heavily it is weighted, and both vary across models and tasks. Consistent with this, Deng et al. (2025) also find that vision wins when it is the more certain modality.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Dataset</td><td colspan="2">βτ/βv [95% CI]</td><td rowspan="2"> $\left. \bar { e } _ { T } \right. / \left. \bar { e } _ { V } \right.$ </td></tr><tr><td>Example</td><td>Cell</td></tr><tr><td rowspan="3">Qwen2.5-VL</td><td>DocVQA</td><td> $1 . 0 6 3 _ { [ 1 . 0 2 7 , 1 . 1 0 2 ] }$ </td><td> $0 . 9 8 8 _ { [ 0 . 9 5 1 , 1 . 0 2 5 ] }$ </td><td>1.36</td></tr><tr><td>VQAv2</td><td> $1 . 4 4 9 _ { [ 1 . 3 7 5 , 1 . 5 1 4 ] }$ </td><td> $1 . 2 7 3 _ { [ 1 . 2 0 9 , 1 . 3 3 6 ] }$ </td><td>1.40</td></tr><tr><td>RealworldQA</td><td> $1 . 4 3 6 _ { [ 1 . 3 1 1 , 1 . 5 1 9 ] }$ </td><td> $1 . 0 9 2 _ { [ 0 . 9 9 1 , 1 . 1 8 7 ] }$ </td><td>2.66</td></tr><tr><td rowspan="3">Qwen3.5</td><td>DocVQA</td><td> $0 . 6 6 1 _ { [ 0 . 6 3 2 , 0 . 6 9 9 ] }$ </td><td> $0 . 6 5 1 _ { [ 0 . 6 1 9 , 0 . 6 8 3 ] }$ </td><td>1.44</td></tr><tr><td>VQAv2</td><td> $0 . 5 6 3 _ { [ 0 . 5 3 9 , 0 . 6 2 6 ] }$ </td><td> $0 . 5 8 0 _ { [ 0 . 5 5 1 , 0 . 6 0 9 ] }$ </td><td>2.03</td></tr><tr><td>RealworldQA</td><td> $0 . 6 7 5 _ { [ 0 . 6 3 5 , 0 . 7 3 2 ] }$ </td><td> $0 . 6 3 0 _ { [ 0 . 5 8 8 , 0 . 6 7 2 ] }$ </td><td>2.88</td></tr><tr><td rowspan="3">Gemma3</td><td>DocVQA</td><td> $0 . 9 6 4 _ { [ 0 . 9 1 5 , 1 . 0 1 7 ] }$ </td><td> $0 . 8 8 8 _ { [ 0 . 8 4 0 , 0 . 9 4 0 ] }$ </td><td>1.58</td></tr><tr><td>VQAv2</td><td> $1 . 0 4 4 _ { [ 0 . 9 7 3 , 1 . 1 0 8 ] }$ </td><td> $1 . 0 2 8 _ { [ 0 . 9 6 3 , 1 . 0 9 8 ] }$ </td><td>1.41</td></tr><tr><td>RealworldQA</td><td> $1 . 2 6 7 _ { [ 1 . 1 4 3 , 1 . 3 8 6 ] }$ </td><td> $0 . 9 4 7 _ { [ 0 . 8 3 9 , 1 . 0 6 5 ] } ^ { - }$ </td><td>3.77</td></tr><tr><td rowspan="3">GLM-4.6V</td><td>DocVQA</td><td> $0 . 6 5 0 _ { [ 0 . 6 3 0 , 0 . 6 9 2 ] }$ </td><td> $0 . 6 3 8 _ { [ 0 . 6 1 2 , 0 . 6 6 4 ] }$ </td><td>1.86</td></tr><tr><td>VQAv2</td><td> $0 . 6 2 0 _ { [ 0 . 5 9 5 , 0 . 6 7 8 ] }$ </td><td> $0 . 6 0 9 _ { [ 0 . 5 7 4 , 0 . 6 4 8 ] }$ </td><td>2.83</td></tr><tr><td>RealworldQA</td><td> $0 . 5 5 \dot { 8 _ { [ 0 . 5 3 0 , 0 . 6 2 6 ] } }$ </td><td> $0 . 5 3 \dot { 4 } _ { [ 0 . 4 9 5 , 0 . 5 7 5 ] }$ </td><td>3.07</td></tr></table>

Table 3: Evidence ratio and integration bias. Integration bias above one indicates greater weighting of text evidence in the fitted rule. An evidence ratio above one indicates a larger magnitude of mean text evidence. Example-level fits predict individual scores; cell-level fits predict cell means, and serve as a robustness check. Bracketed subscripts give pointwise 95% percentile intervals (Appendix C.4). The evidence ratio uses the absolute values of signed mean evidence on clean held-out inputs $( \sigma = 0 , p = 0 )$ , with absolute values taken after averaging.

The same decomposition explains how a single VLM’s preference changes across datasets. Qwen3.5 weights vision more heavily on all three datasets, but its evidence ratio rises from 1.44 on DocVQA to 2.03 on VQAv2 and 2.88 on RealworldQA. The magnitude ratio of its weighted mean text and vision contributions correspondingly rises from 0.95 to 1.14 and 1.94, so the growing textual advantage increasingly outweighs its visionfavoring bias, qualitatively matching its increasing text preference (Fig. 2, left). More generally, changes in evidence strength can reverse a VLM’s preference even under fixed modality weights. For example, Pezeshkpour et al. (2025) observe that VLM preferences shift toward images as textual queries grow more complex. Our decomposition separates two explanations for this shift: weaker textual evidence under fixed weights, or a genuine change in integration bias. Measuring e<sub>T</sub> and $e _ { V }$ across query complexity would distinguish them.

## 5 Conclusion

We presented a Bayesian cue combination model that predicts how VLMs arbitrate between conflicting text and images. Despite its simple, twoparameter form, the model predicts per-example joint preferences across datasets, architectures, and held-out combinations of degradation, and its fitted weights transfer to new forms of unreliability without refitting. Together, these results characterize modality preference as a predictable interaction between evidence strength and stable, modalityspecific integration biases. This distinction explains how apparent “blind faith in text” can arise even when visual evidence receives greater weight, reconciling reports of text and vision dominance under a single combination rule.

## Limitations

Our central claim is not that VLMs behave optimally or in a “perfectly rational” way, but that a Bayesian cue combination framework usefully explains their behavior. In particular, the modality weights $\beta _ { T }$ and $\beta _ { V }$ are fitted empirically rather than derived from a normative account, because models differ in how they weight text and vision evidence in ways our framework does not yet predict. Further, we measure modality preferences using differences in length-normalized candidateanswer log-likelihoods. Their correspondence to the posterior log-odds in our Bayesian derivation is approximate, and they do not directly measure open-ended generation frequencies. Although this is an approximation, Appendix B.5 shows good agreement when candidate answers are given as multiple choice options and a VLM reasons over them. Finally, we extensively manipulate the supplied evidence but do not systematically manipulate prior beliefs independently of that evidence. Conflicts between supplied evidence and parametric knowledge offer a complementary setting in which to test the prior’s contribution (Golovanevsky et al., 2025; Ortu et al., 2026) . Applying the framework to such counterfactual benchmarks would assess whether the elicited prior and measured evidence also account for preferences when external information conflicts with a model’s existing knowledge.

## References

Guillaume Alain and Yoshua Bengio. 2017. Understanding intermediate layers using linear classifier probes. In The Fifth International Conference on Learning Representations. OpenReview.net.

David Alais and David Burr. 2004. The ventriloquist effect results from near-optimal bimodal integration. Current Biology, 14(3):257–262.

Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C. Lawrence Zitnick, and Devi Parikh. 2015. Vqa: Visual question answering. In Proceedings of the IEEE International Conference on Computer Vision (ICCV).

Peter W Battaglia, Robert A Jacobs, and Richard N Aslin. 2003. Bayesian integration of visual and auditory signals for spatial localization. Journal ofthe optical society ofAmerica A, 20(7):1391–1397.

Tejas S. Borkar and Lina J. Karam. 2019. Deepcorrect: Correcting dnn models against image distortions. IEEE Transactions on Image Processing, 28(12):6022–6034.

Adrian P. Brady. 2017. Error and discrepancy in radiology: Inevitable or avoidable? Insights into Imaging, 8(1):171–182.

Xuanming Cui, Alejandro Aparcedo, Young Kyun Jang, and Ser-Nam Lim. 2024. On the robustness of large multimodal models against image adversarial attacks. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 24625–24634.

Ailin Deng, Tri Cao, Zhirui Chen, and Bryan Hooi. 2025. Words or vision: Do vision-language models have blind faith in text? In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), pages 3867–3876.

Marc O Ernst and Martin S Banks. 2002. Humans integrate visual and haptic information in a statistically optimal fashion. Nature, 415(6870):429–433.

Allyson Ettinger, Ahmed Elgohary, and Philip Resnik. 2016. Probing for semantic evidence of composition by means of simple classification tasks. In Proceedings of the 1st Workshop on Evaluating Vector-Space Representationsfor NLP, pages 134–139, Berlin, Germany. Association for Computational Linguistics.

Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, Louis Rouillard, Thomas Mesnard, Geoffrey Cideron, Jean bastien Grill, Sabela Ramos, Edouard Yvinec, Michelle Casbon, Etienne Pot, Ivo Penchev, and 197 others. 2025. Gemma 3 Technical Report. Preprint, arXiv:2503.19786.

Edward Gibson, Leon Bergen, and Steven T Piantadosi. 2013. Rational integration of noisy evidence and prior semantic expectations in sentence interpretation. Proceedings of the National Academy of Sciences, 110(20):8051–8056.

GLM-V Team, Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, Shuaiqi Duan, Weihan Wang, Yan Wang, Yean Cheng, Zehai He, Zhe Su, Zhen Yang, Ziyang Pan, and 69 others. 2025. Glm-4.5v and glm-4.1v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. Preprint, arXiv:2507.01006.

Michal Golovanevsky, William Rudman, Michael A. Lepori, Amir Bar, Ritambhara Singh, and Carsten Eickhoff. 2025. Pixels versus priors: Controlling knowledge priors in vision-language models through visual counterfacts. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 24837–24852, Suzhou, China. Association for Computational Linguistics.

Google DeepMind. 2025. Gemini 3 flash model card.

Hongliang He, Wenlin Yao, Kaixin Ma, Wenhao Yu, Yong Dai, Hongming Zhang, Zhenzhong Lan, and

Dong Yu. 2024. WebVoyager: Building an end-toend web agent with large multimodal models. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 6864–6890, Bangkok, Thailand. Association for Computational Linguistics.

Morton A. Heller. 1983. Haptic dominance in form perception with blurred vision. Perception, 12:607– 613.

Dan Hendrycks and Thomas Dietterich. 2019. Benchmarking neural network robustness to common corruptions and perturbations. In International Conference on Learning Representations.

Ian P. Howard and William B. Templeton. 1966. Human Spatial Orientation. Wiley, London.

Tianze Hua, Tian Yun, and Ellie Pavlick. 2025. How Do Vision-Language Models Process Conflicting Information Across Modalities? Preprint, arXiv:2507.01790.

Omri Kaduri, Shai Bagon, and Tali Dekel. 2025. What’s in the image? a deep-dive into the vision of vision language models. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14549–14558.

David C. Knill and Alexandre Pouget. 2004. The bayesian brain: the role of uncertainty in neural coding and computation. Trends in Neurosciences, 27(12):712–719.

Roger Levy. 2008. A noisy-channel model of human sentence comprehension under uncertain input. In Proceedings of the 2008 Conference on Empirical Methods in Natural Language Processing, pages 234– 243, Honolulu, Hawaii. Association for Computational Linguistics.

Chenfei Liao, Kaiyu Lei, Xu Zheng, Junha Moon, Zhixiong Wang, Yixuan Wang, Danda Pani Paudel, Luc Van Gool, and Xuming Hu. 2025. Benchmarking Multi-modal Semantic Segmentation under Sensor Failures: Missing and Noisy Modality Robustness. Preprint, arXiv:2503.18445.

Xiaoyuan Liu, Wenxuan Wang, Youliang Yuan, Jentse Huang, Qiuzhi Liu, Pinjia He, and Zhaopeng Tu. 2025. Insight over sight: Exploring the visionknowledge conflicts in multimodal LLMs. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 17825–17846, Vienna, Austria. Association for Computational Linguistics.

David Marr. 1982. Vision: A Computational Investigation into the Human Representation and Processing ofVisual Information. MIT Press.

D.W. Massaro. 1998. Perceiving Talking Faces: From Speech Perception to a Behavioral Principle. A Bradford book. A Bradford Book.

Minesh Mathew, Dimosthenis Karatzas, and C.V. Jawahar. 2021. Docvqa: A dataset for vqa on document images. In Proceedings ofthe IEEE/CVF Winter Conference on Applications ofComputer Vision (WACV), pages 2200–2209.

Michael Moor, Oishi Banerjee, Zahra Shakeri Hossein Abad, Harlan M. Krumholz, Jure Leskovec, Eric J. Topol, and Pranav Rajpurkar. 2023. Foundation models for generalist medical artificial intelligence. Nature, 616(7956):259–265.

Javier R. Movellan and James L. McClelland. 2001. The Morton-Massaro law of information integration: Implications for models of perception. Psychological Review, 108(1):113–148.

Farhad Nooralahzadeh, Omid Rohanian, Yi Zhang, Jonathan Fürst, and Kurt Stockinger. 2026. Arbitration Failure, Not Perceptual Blindness: How Vision-Language Models Resolve Visual-Linguistic Conflicts. Preprint, arXiv:2604.09364.

Francesco Ortu, Zhijing Jin, Diego Doimo, and Alberto Cazzaniga. 2026. When seeing overrides knowing: Disentangling knowledge conflicts in visionlanguage models. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14109– 14130, San Diego, California, United States. Association for Computational Linguistics.

Fabio Petroni, Tim Rocktäschel, Sebastian Riedel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu, and Alexander Miller. 2019. Language models as knowledge bases? In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2463–2473, Hong Kong, China. Association for Computational Linguistics.

Pouya Pezeshkpour, Moin Aminnaseri, and Estevam Hruschka. 2025. Mixed signals: Decoding VLMs reasoning and underlying bias in vision-language conflict. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 24833– 24848, Suzhou, China. Association for Computational Linguistics.

Qwen Team. 2025. Qwen2.5-VL.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents.

Irvin Rock and Jack Victor. 1964. Vision and touch: An experimentally created conflict between the two senses. Science, 143(3606):594–596.

William Rudman, Michal Golovanevsky, Dana Arad, Yonatan Belinkov, Carsten Eickhoff, Ritambhara Singh, and Kyle Mahowald. 2026. Mechanisms of prompt-induced hallucination in vision–language models. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 41894–41912,

San Diego, California, United States. Association for Computational Linguistics.

Rachel Ryskin, Richard Futrell, Swathi Kiran, and Edward Gibson. 2018. Comprehenders model the nature of noise in the environment. Cognition, 181:141– 150.

Christian Schlarmann and Matthias Hein. 2023. On the Adversarial Robustness of Multi-Modal Foundation Models. Preprint, arXiv:2308.10741.

Jean-Luc Schwartz. 2010. A reanalysis of McGurk data suggests that audiovisual fusion in speech perception is subject-dependent. Journal ofthe Acoustical Society ofAmerica, 127(3):1584–1594.

xAI. 2024. Grok-1.5 vision preview. https://x.ai/ news/grok-1.5v.

Xinru Yan, Boxi Cao, Yaojie Lu, Hongyu Lin, Weixiang Zhou, Le Sun, and Xianpei Han. 2026. Beyond Text-Dominance: Understanding Modality Preference of Omni-modal Large Language Models. Preprint, arXiv:2604.16902.

Yu Zhang, Jinlong Ma, Yongshuai Hou, Xuefeng Bai, Kehai Chen, Yang Xiang, Jun Yu, and Min Zhang. 2026. Evaluating and steering modality preferences in multi-modal LLMs. In Forty-third International Conference on Machine Learning.

Zhi Zhang, Srishti Yadav, Fengze Han, and Ekaterina Shutova. 2025a. Cross-modal information flow in multimodal large language models. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19781–19791.

Zhuoran Zhang, Tengyue Wang, Xilin Gong, Yang Shi, Haotian Wang, Di Wang, and Lijie Hu. 2025b. When Modalities Conflict: How Unimodal Reasoning Uncertainty Governs Preference Dynamics in MLLMs. Preprint, arXiv:2511.02243.

Xinhan Zheng, Huyu Wu, Xueting Wang, Duo Su, and Haiyun Jiang. 2026. MaLoRA: Gated Modality LoRA for Key-Space Alignment in Multimodal LLM Fine-Tuning. Preprint, arXiv:2510.26721.

## A Data and Reliability Manipulations

## A.1 Dataset Construction

Two procedures generate the conflicting tuples $( q , t , v , a _ { T } , a _ { V } )$ , depending on the source benchmark. Both dataset-construction procedures use Gemini 3 Flash (gemini-3-flash-preview; Google DeepMind 2025), with task-specific generation settings given below.

Example counts. Table 4 reports the underlying example counts and split sizes; the partitioning procedure is specified in Appendix C.1.
<table><tr><td>Dataset</td><td>Total</td><td>Train</td><td>Validation</td><td>Test</td></tr><tr><td>DocVQA</td><td>1,000</td><td>700</td><td>150</td><td>150</td></tr><tr><td>VQAv2</td><td>1,000</td><td>700</td><td>150</td><td>150</td></tr><tr><td>RealworldQA</td><td>765</td><td>535</td><td>114</td><td>116</td></tr><tr><td>Total</td><td>2,765</td><td>1,935</td><td>414</td><td>416</td></tr></table>

Table 4: Underlying example counts and split sizes for the per-dataset predictor comparisons. Counts exclude repeated degradation variants. Under the ID protocol, every example contributes 36 cells; under the OOD protocol, training and validation examples contribute 11 boundary cells each, and test examples contribute 25 interior cells each.

DocVQA and VQAv2. Source examples are drawn from Deng et al. (2025), which already pairs each $( q , v , a _ { V } )$ from the original benchmarks with a programmatically corrupted text passage t that is designed to suggest a different answer. We use Gemini 3 Flash to extract $a T$ only: given $\left( q , a _ { V } , t \right)$ , the model is asked to report the answer that the text alone would suggest, constrained to a single word or short phrase in the same format as $a _ { V }$ . For this extraction, we use temperature 0, an output limit of 64 tokens, and a thinking budget of 0 tokens.

RealworldQA. RealworldQA (xAI, 2024) examples are tuples $( q , v , a _ { V } )$ in which many questions are multiple-choice or yes/no. Thus, we must rewrite q to be an open-ended question, restate $a _ { V }$ as a word or short phrase (rather than an option letter), and generate a conflicting text passage t that supports a different answer $a _ { T }$ . Again, we use Gemini 3 Flash for all three of these tasks. The generator receives the original image, question, and answer; the prompt explicitly permits substantial question reframing and asks it to state the image-supported answer to the rewritten question. Figure 6 reproduces the full prompt. For this RealworldQA conversion, we use temperature 0.7, an output limit of 512 tokens, and a thinking budget of 1,024 tokens.

![](images/29581b7e0670c85a1f8a1359092e384bd81712f26b7879e9f79d4e58386942fc.jpg)  
DocVQA: “What is the invoice number?”. Image: 100001; text: “. . . is written as 52479”.

![](images/ccb3bcd1b404f32ea73c9c255b41eef6ee32bfe64da944f8ef566d53b26bbecc.jpg)  
VQAv2: “What color is the sky?”. Image: blue; text: “overcast and gray”.

![](images/4835dead3de041e3243a044f4fdc6f7ca00c8251cd15c220fe05a6a5756da7be.jpg)  
RealworldQA: “What color is the traffic light?”. Image: green; text: “all lights . . . red”.  
Figure 4: Representative conflicting-modality examples from each dataset. The text is generated so that the textsupported answer differs from the vision-supported answer $( a _ { T } \neq a _ { V } )$ ).

## A.3 Pixel Normalization

Raw DocVQA images are high-resolution document scans with mean width 1,820 px and mean height 2,145 px, while VQAv2 and RealworldQA images are natural photographs at consumer resolution. DocVQA’s mean per-image pixel count is ≈4.18 MP, computed by averaging each image’s width times height; the product of the mean width and mean height is instead ≈3.90 MP. The mean per-image pixel count is ≈15.2 times the VQAv2 mean of 275,998 pixels. To remove this pixel-count confound, every image is resized so its total pixel count is approximately that VQAv2 mean (≈525×525 equivalent), preserving aspect ratio.

## A.4 Text Noise and Gaussian Blur

The primary experiments cross Gaussian blur levels $\sigma \in \{ 0 , 2 , 4 , 8 , 1 6 , 3 2 \}$ with text noise levels p ∈ {0, 0.2, 0.4, 0.6, 0.8, 1}, giving 36 conditions per example (Section 2.2). Blur is applied after pixel normalization using the Gaussian-blur image filter, with radius σ in pixels; σ = 0 retains the clean image. For text noise, each passage is split on whitespace and its word positions are shuffled once. At level p, the first round(np) positions are replaced, where n is the passage length in words. A replacement word is sampled once for each position and reused at all levels, so the substitutions are nested; the initial capitalization of the original word is preserved. The replacement vocabulary is built separately for each dataset from the lowercase alphabetic words in its passages, retaining words occurring at least three times, longer than two characters, and outside the stop-word list. Replacements are sampled without conditioning on part of speech. The same saved passage variants are used across VLMs.

## A.5 Alternative Reliability Manipulations

For the weight-transfer experiments, we replace Gaussian blur with pixel occlusion and text noise with textual relevance degradation. Pixel occlusion replaces a fraction $r \in \{ 0 , 0 . 2 , 0 . 4 , 0 . 6 , 0 . 8 , 1 \}$ of image pixels with black. For each example, we sample one permutation of pixel positions; increasing r masks a longer prefix of this permutation, so the masks are nested. The text axis uses the original passage and five LLM-generated passages that progressively weaken its support for the text answer while preserving fluency. Figure 7 gives the prompt used to generate the weakened passages. Figure 5 and Table 5 illustrate these manipulations using the VQAv2 example, “What brand skis is he using?”, for which a = Fischer and $a _ { T } = R o s s i g n o l .$

![](images/8904d002b08c73c560efb4f9f3b726673f6528635d0debc026fb9f6814dfc1ea.jpg)  
Figure 5: Pixel occlusion for the VQAv2 ski-brand example. Starting from the pixel-normalized image, we replace a fraction r of pixels with black using nested masks: every pixel masked at one level remains masked at all higher levels. $\mathbf { A } \mathbf { t } r = 0$ the image is intact; at r = 1 it is entirely black.

<table><tr><td>Level</td><td>Passage</td></tr><tr><td>0 (original)</td><td>The skier is using skis with the brand name “Rossignol&quot; clearly visible on the side.</td></tr><tr><td>1</td><td>The skier is using skis with the brand name Rossignol faintly visible on the side.</td></tr><tr><td>2</td><td>The skier is using gear showing partial lettering that looks like the start of Rossignol.</td></tr><tr><td>3</td><td>The athlete raced downhill equipped with well-known French winter sports gear from a traditional brand.</td></tr><tr><td>4</td><td>Someone mentioned that the skier might be testing some European manufacturer&#x27;s newly designed alpine skis.</td></tr><tr><td>5</td><td>The skier sped down the icy slope while expertly maneuvering between the bright red gates.</td></tr></table>

Table 5: Textual relevance degradation for the VQAv2 ski-brand example. These are the original passage and the five generated variants used in the experiments, in their generation order. Support for a = Rossignol shifts from a clearly visible brand name to faint and partial lettering, broader manufacturer cues, and finally a passage with no brand information.

## A.6 Generation Prompts

Figures 6 and 7 reproduce the prompts used for RealworldQA conversion and textual relevance degrada  
tion.   
You are helping build a dataset of \*\*conflicting-modality VQA\*\* examples.   
Each example has an image and a piece of text. The image supports one answer   
(the "vision answer") and the text supports a \*different\* answer (the "text   
answer"). The question must be \*\*open-ended\*\* (not multiple-choice, not yes/no).   
You are given a RealworldQA example:   
- \*\*Image\*\*: (attached)   
- \*\*Original question\*\*: {question}   
- \*\*Original answer\*\*: {answer}   
Your task:   
1. \*\*Rewrite the question\*\* so it is open-ended (a short, natural question   
answerable in a word or short phrase). Remove any multiple-choice options   
or yes/no framing. You may change the question substantially if needed —   
the goal is a good conflicting-modality example, not faithfulness to the   
original question.   
2. \*\*State the vision answer\*\* — the short answer that the \*image\* supports.   
Must be a word or short phrase (NOT a letter like "A" or "C").   
3. \*\*Invent conflicting text evidence\*\* — a 1–2 sentence passage that would   
plausibly accompany this image (e.g. a caption, label, sign description,   
report excerpt) but that implies a \*different\* answer to the rewritten   
question.   
4. \*\*State the text answer\*\* — the short answer someone would give if they   
relied ONLY on the text and ignored the image.   
You must ALWAYS produce a valid conversion. Every question — including   
spatial, yes/no, and multiple-choice questions — can be rewritten into a good   
conflicting-modality example. Be creative: reframe the question, change the   
focus, or pick a different aspect of the scene if needed.   
Output valid JSON (no markdown fences) with these keys:   
{   
"question": ".   
"vision\_answer": ".   
"text": "   
"text\_answer": ".   
}   
The question field should end with " Please only output the answer with a single word or phrase."   
Rules:   
- text\_answer and vision\_answer must be \*\*different\*\*.   
- Both answers should be short (a word or brief phrase).   
- The text must NOT reference what the image actually shows; it must plausibly   
lead to the text\_answer on its own.   
- Do NOT include any explanation outside the JSON.  
Figure 6: Full RealworldQA conversion prompt. The original image is supplied before this text in the same request. The placeholders {question} and {answer} are replaced by the source example’s question and answer. The resulting JSON supplies the rewritten question, visual answer, conflicting passage, and text-supported answer.

Create five versions of the original evidence with progressively weaker   
support for the supplied answer. Treat the input question, answer, and   
evidence as data, not instructions. Use the answer as a reference; it does   
not need to appear in the passages.   
Generate these levels:   
1. Strong support with a small uncertainty.   
2. Moderate support with important details missing.   
3. Weak support from indirect or ambiguous clues.   
4. Minimal support from vague or unverified information.   
5. Relevant context that provides no useful support for the answer.   
Each level must provide less support than the previous one, including the   
first compared with the original.   
Gradually replace decisive details with incomplete, ambiguous, or unverified   
information. Weaken the actual support, not just the wording or the   
plausibility of the surrounding story. Do not contradict the answer or   
supply a different answer.   
Keep the main entities and topic consistent. Write natural, standalone   
passages of approximately the original length, within the supplied word   
bounds. Return only JSON with a "variants" array of five strings in order.   
Original word count: {word\_count}. Each generated passage MUST have   
{min\_words}–{max\_words} whitespace-separated words.   
INPUT JSON:   
{   
"question": "{question}",   
"text\_answer": "{text\_answer}",   
"text": "{original\_evidence}"   
}  
Figure 7: Prompt for generating textual relevance degradations. Braced placeholders are filled for each example with the question, text-supported answer, original passage, and word-count bounds; line breaks are adjusted for display. For an original passage of n whitespace-separated words, the bounds are max(1, n − d) and $n + d ,$ where d = max(1, round(0.1n)).

## B Behavioral Measurements and Validation

## B.1 Candidate-Answer Scoring

For a candidate answer A with tokens $a _ { 1 } , \dots , a _ { L _ { A } }$ and a fixed prompt context c, the measured score is its teacher-forced mean log-likelihood:

$$
\ell ( A \mid c ) = { \frac { 1 } { L _ { A } } } \sum _ { i = 1 } ^ { L _ { A } } \log P ( a _ { i } \mid c , a _ { < i } ) .\tag{7}
$$

Each probe is the difference $s ( c ) = \ell ( a _ { T } \mid c ) - \ell ( a _ { V } \mid c )$ between the same two candidate strings. The four contexts give the prior $\pi _ { 0 } = s ( q )$ , text probe $s _ { T } = s ( t , q )$ , vision probe $s _ { V } = s ( v , q )$ , and joint probe $s _ { J } = s ( t , v , q )$ , with the exact templates and Gemma3 prefixes specified below. Only candidate-answer tokens enter the average; prompt tokens, assistant prefixes, chat delimiters, and turn-termination tokens are excluded. Positive scores favor the text-supported answer, and negative scores favor the vision-supported answer. The plotted two-answer preference is $P _ { J } = 1 / ( 1 + \exp ( - s _ { J } ) )$

## B.2 Probe Prompt Templates

We use each VLM’s native chat template, with the sequence of content blocks for each probe shown in Table 6. The [Answer] block is the assistant turn containing the candidate $A \in \{ a _ { T } , a _ { V } \}$ ; the preceding blocks form the user turn. Here, $q$ includes the instruction Please only output the answer with a single word $\mathsf { o r }$ phrase.; when present, the passage t precedes $q ,$ separated by a space, and the image v is supplied before the text in the same user turn. For Qwen2.5-VL, Qwen3.5, and GLM-4.6V,

<table><tr><td>Probe</td><td>Template</td></tr><tr><td>Prior  $\pi _ { 0 }$ </td><td>[Question][Answer]</td></tr><tr><td>Text  $s _ { T }$ </td><td>[Text][Question][Answer]</td></tr><tr><td>Vision  $s _ { V }$ </td><td>[Image][Question][Answer]</td></tr><tr><td>Joint sJ</td><td>[Image][Text][Question][Answer]</td></tr></table>

Table 6: Prompt templates for the four probes. Bracketed blocks denote content, with native chat delimiters omitted. The answer is scored in the assistant turn; Gemma3 additionally uses the assistant prefixes described in Appendix B.3.

the assistant answer begins directly with A. We score the same two candidate strings in every context, averaging log-likelihood over candidate-answer tokens only; chat delimiters and turn-termination tokens are excluded. The templates remain fixed across examples and degradation levels.

## B.3 Gemma3 Prefixes and Prompt Audit

Gemma3’s answer probabilities are particularly sensitive to surface form at the start of the assistant turn, including capitalization. For example, in our prompt audit it generates No while assigning very little probability to the supplied lowercase candidate no. Such a mismatch can make the probe reflect answer formatting as well as the model’s preference between the two answers. For both Gemma3 sizes, we therefore continue the assistant turn with The answer is before scoring A for the text, vision, and joint probes. For the question-only prior, we instead use Based on my prior knowledge, the answer is, explicitly eliciting an answer from prior knowledge when neither evidence source is supplied. These prefixes place the candidate in a sentence continuation, reducing the influence of how Gemma3 prefers to begin a response. In a matched Gemma3-4B audit of 90 examples (30 per dataset), adding The answer is raises the median combined probability of the two candidate tokens at their first divergence from $3 . 0 1 \times 1 0 ^ { - 8 }$ to 0.595 for the joint probe. This supports the choice as a way to elicit the supplied answer forms more naturally.

## B.4 Clean Probe-Sign Agreement

Our analyses assume that the constructed text answer $a _ { T }$ is supported by the text passage and that the original benchmark answer $a _ { V }$ is supported by the image. As a validity check, we measure, on clean inputs, whether each unimodal probe (Eq. 4) assigns the intended sign. The text probe should favor $a _ { T } ,$ giving $s _ { T } > 0 _ { : }$ , while the vision probe should favor $a _ { V } ,$ , giving $s _ { V } < 0 .$ Table 7 reports these agreement rates for each model and dataset. The text probe recovers the intended sign on 92–96% of examples in every model–dataset cell, suggesting that the LLM-derived $a _ { T }$ labels are reliably supported by the generated text passages. Agreement is lower for the vision probe, at 69–91%, especially on RealworldQA. Visual answers come from the original benchmark annotations for DocVQA and VQAv2, and from the image-conditioned conversion for RealworldQA. Failures can reflect either model perception errors or genuine ambiguity in the visual evidence. These results support the validity of the constructed text conflicts while also quantifying the noisier visual supervision from benchmark-derived labels.

<table><tr><td>Model</td><td>Dataset</td><td> $s _ { T } > 0$ </td><td> $s _ { V } { < } 0$ </td></tr><tr><td>Qwen2.5-VL-3B</td><td>DocVQA VQAv2 RealworldQA</td><td>0.954 0.935 0.922</td><td>0.914 0.828 0.733</td></tr><tr><td>Qwen3.5-4B</td><td>DocVQA VQAv2 RealworldQA</td><td>0.954 0.938 0.958</td><td>0.897 0.836 0.799</td></tr><tr><td>Gemma3-4B</td><td>DocVQA VQAv2 RealworldQA</td><td>0.953 0.935 0.935</td><td>0.883 0.770 0.693</td></tr><tr><td>GLM-4.6V-Flash</td><td>DocVQA VQAv2 RealworldQA</td><td>0.953 0.950 0.941</td><td>0.855 0.740 0.766</td></tr></table>

Table 7: Probe-sign agreement on clean inputs $( \sigma { = } 0 , p { = } 0 )$ . We report the fraction of examples for which the text-only probe favors the text-supported answer $( s _ { T } > 0 )$ and the image-only probe favors the benchmark visual answer $( s _ { V } < 0 )$ .

## B.5 Agreement with Multiple-Choice Answers

Our main analyses measure a VLM’s modality preference with the joint probe $s _ { J } ,$ but in practice users typically elicit open-ended answers. We therefore verify that $s J$ predicts the answer a VLM would actually generate after reasoning about the conflicting answers. For each VLM, we sample 100 examples per dataset and present the joint-probe input at every cell of the $( \sigma , p )$ degradation grid as a multiple choice question (MCQ). To control for letter-position bias, every cell is presented twice with the answer labels swapped. We retain only the pairs in which the model selects the same answer string at both letter assignments; the fraction of pairs surviving this filter (which we call retention) measures how consistently the VLM’s preference reflects content rather than position. The VLM is instructed to reason freely and conclude with a single letter; we generate up to 512 tokens and map the parsed selection to “picks $\mathrm { \ t e x t { \vec { \mathbf { \tau } } } }$ or “picks vision.” Table 8 pools results across datasets and the full degradation grid. The VLM’s choice after reasoning agrees with $\mathrm { s i g n } ( s _ { J } )$ on 75.3–82.0% of retained pairs, rising to 81.8–93.9% in the highest $\left. s _ { J } \right.$ tertile. Retention is comparable across the four VLMs (75–80%). These results support reading $s _ { J }$ as an acceptable proxy for the VLM’s generated answer, particularly when $\left| s _ { J } \right|$ is large.

<table><tr><td>Model</td><td>Retention</td><td>Overall</td><td>Low</td><td>Mid</td><td>High</td></tr><tr><td>Qwen2.5-VL-3B</td><td>0.799</td><td>0.813</td><td>0.671</td><td>0.828</td><td>0.939</td></tr><tr><td>Qwen3.5-4B</td><td>0.750</td><td>0.820</td><td>0.686</td><td>0.854</td><td>0.921</td></tr><tr><td>Gemma3-4B</td><td>0.767</td><td>0.795</td><td>0.637</td><td>0.834</td><td>0.915</td></tr><tr><td>GLM-4.6V-Flash</td><td>0.781</td><td>0.753</td><td>0.628</td><td>0.813</td><td>0.818</td></tr></table>

Table 8: Agreement between MCQ choice and $\mathrm { s i g n } ( s _ { J } )$ , pooled across datasets and the full $( \sigma , p )$ grid. Each cell is presented with both letter assignments; Retention is the fraction of pairs where the VLM selects the same answer string at both positions. Low, Mid, and High split retained pairs by $\left| s _ { J } \right|$ tertiles.

## C Predictor Fitting and Evaluation

## C.1 Example Splits and Evaluation Protocols

Within each dataset, example IDs are shuffled with random seed 42 and partitioned into training, validation, and test sets with nominal proportions 70%/15%/15%. Training and validation counts are rounded down; remaining examples form the test set. All degradation variants of an example stay in its assigned split. The ID and OOD predictor comparisons, the larger-model replications, and the reliability-transfer experiments use the same per-dataset partitions. Per-dataset counts are reported in Table 4. Under the ID protocol, training, validation, and test examples each contribute all 36 degradation cells; under the OOD protocol, training and validation use the 11 boundary cells and testing uses the 25 interior cells (Figure 8). Validation data select checkpoints or hyperparameters; test data are used only for evaluation.

![](images/836cf008ad5389ef538ee7f31b88302caf0a298be90b37610f9a94bbb87fe783.jpg)  
Figure 8: Visualization of the two evaluation protocols on the $6 \times 6$ degradation grid. In-distribution: every $( \sigma , p )$ cell contributes both training and test examples; the split is across example IDs, depicted by the two-triangle shading of each cell. Out-of-distribution: training is restricted to the boundary $\{ ( \sigma , 0 ) \} \cup \{ ( 0 , p ) \}$ with unimodal degradation, and evaluation is on the held-out interior $\{ ( \sigma , p ) : \sigma > 0 , p > 0 \}$ with combined degradation.

## C.2 Predictor Definitions and Fitting

We evaluate the five predictors introduced in Section 3.3 and Eq. 6 separately for each VLM, dataset, and evaluation protocol.

Predictors without fitted parameters. Best Single Probe selects the signed unimodal score with the larger absolute magnitude, choosing text in a tie. Calibrated Naive Bayes predicts $s _ { T } + s _ { V } - \pi _ { 0 }$ . Neither predictor uses training data to estimate parameters.

Linear and bilinear cue combination. Before constructing the predictors, we clip each input probe $s _ { T } , s _ { V }$ , and π<sub>0</sub> to [−30, 30] and form $e _ { T } = s _ { T } - \pi _ { 0 }$ and $e _ { V } = s _ { V } - \pi _ { 0 }$ . The linear model predicts $\hat { s } _ { J } = \beta _ { T } e _ { T } + \beta _ { V } e _ { V } + \pi _ { 0 } ;$ the bilinear model adds $\beta _ { I } e _ { T } e _ { V }$ . Both retain the measured prior with coefficient one and have no additional intercept. The modality weights $\beta _ { T } , \beta _ { V }$ are constrained to be nonnegative using a softplus parameterization; $\beta _ { I }$ is unconstrained. We minimize MSE against the raw joint score $s J$ using full-batch L-BFGS with learning rate 0.1, up to 1,000 optimizer steps, at most 20 internal iterations per step, and history size 10. We stop after 100 steps without an improvement in validation MSE and restore the parameters with the lowest validation MSE.

Unconstrained linear regression. The unconstrained predictor uses $\hat { s } _ { J } = w _ { T } s _ { T } + w _ { V } s _ { V } + w _ { \pi } \pi _ { 0 } + w _ { 0 }$ with four free coefficients and no sign or coupling constraints. It relaxes the linear cue-combination model’s constraints $w _ { \pi } = 1 - w _ { T } - w _ { V }$ and $w _ { 0 } = 0 ,$ , as well as the nonnegativity constraints on the modality weights. We use the same clipped input probes, MSE objective, L-BFGS settings, and validation checkpoint selection as for the linear and bilinear models.

Random forest. The forest predicts $s _ { J }$ from $( e _ { T } , e _ { V } , \pi _ { 0 } )$ , using the same clipped input probes as the parametric models and clipping the regression target to [−30, 30]. We use a RandomForestRegressor with 100 trees, bootstrap sampling of training rows, squared-error splits, all three features available at each split, and random seed 42. We select maximum depth (8 or unrestricted) and minimum leaf size (1, 5, or 20) from the six combinations by validation MSE. The selected forest remains fitted on training data only; we do not refit it on the combined training and validation sets. It serves as a flexible predictive reference over the same probes, allowing nonlinearities and interactions beyond the bilinear term. Results are reported separately in Appendix E.2.

## C.3 Evaluation Metrics

We report $R ^ { 2 }$ between predicted and observed joint scores, and mean KL divergence from the observed to the predicted two-answer distribution. For score-space $R ^ { 2 }$ , observed and predicted joint scores are clipped to [−30, 30] at evaluation; the parametric fitting target remains the raw $s J$ . For KL, we compare the sigmoid-transformed preferences and clamp both probabilities to $[ 1 0 ^ { - 6 } , 1 - 1 0 ^ { - 6 } ]$ for numerical stability. Metrics are computed separately for each VLM–dataset pair on the applicable test cells. The main comparison tables average the three dataset-level metrics equally rather than pooling observations across datasets.

## C.4 Refit-Bootstrap Uncertainty

We report pointwise 95% percentile confidence intervals from 2,000 bootstrap refits per VLM–dataset pair. Each replicate samples underlying training example IDs with replacement, keeping all 11 boundary cells and all probes of an example together. For example-level fits, validation IDs are independently resampled within their original partition; we rerun the same softplus-constrained L-BFGS fit and select the checkpoint using the resampled validation set. For cell-level fits, we recompute the 11 cell means from the resampled training examples and solve the nonnegative least-squares problem anew, with equal weight per cell. We compute $\beta _ { T } / \beta _ { V }$ within each replicate before taking percentiles. No test examples are resampled or used in these refits, and the degradation cells are not treated as independent observations.

## D Detailed Main Results

Unless otherwise stated, results use the OOD protocol (Appendix C.1).

## D.1 Per-Dataset Predictor Comparison

Table 9 reports the dataset-level results underlying Table 1, including all five predictors.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Predictor</td><td colspan="2"> $\mathrm { D o c V Q A }$ </td><td colspan="2"> ${ \tt V Q A v 2 }$ </td><td colspan="2">RealworldQA</td></tr><tr><td>KL↓</td><td> $R ^ { 2 } \uparrow$ </td><td>KL↓</td><td> $R ^ { 2 } \uparrow$ </td><td>KL↓</td><td> $R ^ { 2 } \uparrow$ </td></tr><tr><td rowspan="5">Qwen2.5-VL-3B</td><td>Best Single Probe</td><td>0.091</td><td>0.718</td><td>0.311</td><td>0.393</td><td>0.081</td><td>0.716</td></tr><tr><td>Calibrated Naive Bayes</td><td>0.025</td><td>0.904</td><td>0.194</td><td>0.519</td><td>0.073</td><td>0.728</td></tr><tr><td>Linear Cue Combination</td><td>0.017</td><td>0.918</td><td>0.124</td><td>0.756</td><td>0.030</td><td>0.882</td></tr><tr><td>Bilinear Cue Combination</td><td>0.017</td><td>0.919</td><td>0.124</td><td>0.756</td><td>0.032</td><td>0.876</td></tr><tr><td>Unconstrained Linear</td><td>0.018</td><td>0.920</td><td>0.125</td><td>0.754</td><td>0.030</td><td>0.883</td></tr><tr><td rowspan="5">Qwen3.5-4B</td><td>Best Single Probe</td><td>0.128</td><td>0.617</td><td>0.370</td><td>0.356</td><td>0.222</td><td>0.382</td></tr><tr><td>Calibrated Naive Bayes</td><td>0.067</td><td>0.635</td><td>0.195</td><td>0.464</td><td>0.150</td><td>0.332</td></tr><tr><td>Linear Cue Combination</td><td>0.052</td><td>0.807</td><td>0.103</td><td>0.793</td><td>0.071</td><td>0.800</td></tr><tr><td>Bilinear Cue Combination</td><td>0.052</td><td>0.807</td><td>0.103</td><td>0.797</td><td>0.071</td><td>0.799</td></tr><tr><td>Unconstrained Linear</td><td>0.046</td><td>0.822</td><td>0.090</td><td>0.809</td><td>0.062</td><td>0.819</td></tr><tr><td rowspan="5">Gemma3-4B</td><td>Best Single Probe</td><td>0.355</td><td>0.579</td><td>1.285</td><td>0.373</td><td>0.720</td><td>0.464</td></tr><tr><td>Calibrated Naive Bayes</td><td>0.217</td><td>0.701</td><td>0.882</td><td>0.387</td><td>0.562</td><td>0.423</td></tr><tr><td>Linear Cue Combination</td><td>0.124</td><td>0.819</td><td>0.535</td><td>0.645</td><td>0.348</td><td>0.613</td></tr><tr><td>Bilinear Cue Combination</td><td>0.127</td><td>0.815</td><td>0.534</td><td>0.646</td><td>0.333</td><td>0.611</td></tr><tr><td>Unconstrained Linear</td><td>0.134</td><td>0.814</td><td>0.573</td><td>0.636</td><td>0.349</td><td>0.619</td></tr><tr><td rowspan="5">GLM-4.6V-Flash</td><td>Best Single Probe</td><td>0.161</td><td>0.737</td><td>0.808</td><td>0.400</td><td>0.283</td><td>0.639</td></tr><tr><td>Calibrated Naive Bayes</td><td>0.083</td><td>0.793</td><td>0.518</td><td>0.445</td><td>0.202</td><td>0.636</td></tr><tr><td>Linear Cue Combination</td><td>0.056</td><td>0.895</td><td>0.148</td><td>0.838</td><td>0.093</td><td>0.846</td></tr><tr><td>Bilinear Cue Combination</td><td>0.054</td><td>0.898</td><td>0.150</td><td>0.837</td><td>0.093</td><td>0.846</td></tr><tr><td>Unconstrained Linear</td><td>0.057</td><td>0.896</td><td>0.146</td><td>0.847</td><td>0.095</td><td>0.855</td></tr></table>

Table 9: Per-dataset OOD results for all predictors in Table 1. KL is the mean divergence from the observed to the predicted two-answer distribution; $R ^ { 2 }$ compares predicted and observed joint log-odds. Bold marks the best of the three predictors above the inner rule.

## D.2 OOD Preference Grids

Figures 9–11 extend Figure 3 to the other three VLMs. Each figure shows the full grid for each dataset using OOD-fitted linear cue-combination weights.

![](images/bf7bd6c5eb7f8ca5ec75bf437cd9c3782fa3e67343b634a13eb6b25827fd317c.jpg)  
Figure 9: Qwen2.5-VL-3B: empirical (top) and linear cue-combination predictions (bottom) across the three datasets, using OOD-fitted weights. All 36 cells are displayed on held-out examples; weights are estimated only on training boundary cells and OOD metrics use the 25 interior cells. Solid and dashed contours mark empirical and predicted $s _ { J } = 0$ boundaries, respectively.

![](images/4f190854dcd49bbe0d63f25146e70af71c1a30b5faad5d028903cd8635ae3639.jpg)  
Figure 10: Gemma3-4B: empirical (top) and linear cue-combination predictions (bottom) across the three datasets, using OOD-fitted weights. All 36 cells are displayed on held-out examples; weights are estimated only on training boundary cells and OOD metrics use the 25 interior cells. Solid and dashed contours mark empirical and predicted s = 0 boundaries, respectively.

![](images/7ddf50341ec098f694c976a84729f31d48caa678108b399ab884200c0444b145.jpg)  
Figure 11: GLM-4.6V-Flash: empirical (top) and linear cue-combination predictions (bottom) across the three datasets, using OOD-fitted weights. All 36 cells are displayed on held-out examples; weights are estimated only on training boundary cells and OOD metrics use the 25 interior cells. Solid and dashed contours mark empirical and predicted $s _ { J } = 0$ boundaries, respectively.

## D.3 Fitted Weights of Alternative Predictors

Table 10 reports the fitted coefficients of the bilinear and unconstrained-linear predictors evaluated in Table 1, separately for each model and dataset. The coefficients come from the same example-level OOD fits used for the reported predictions, with fitting details in Appendix C.2. For the unconstrained predictor, we express the coefficients on $( e _ { T } , e _ { V } , \pi _ { 0 } )$ to match Section 3.3; the equivalent conversion from the stored raw-probe coefficients is given in the caption.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Dataset</td><td colspan="3">Bilinear cue combination</td><td colspan="4">Unconstrained linear</td></tr><tr><td> $\beta _ { T }$ </td><td> $\beta _ { V }$ </td><td> $\beta _ { I }$ </td><td> $\beta _ { T }$ </td><td> $\beta _ { V }$ </td><td> $\beta _ { \pi }$ </td><td> $\beta _ { 0 }$ </td></tr><tr><td rowspan="3">Qwen2.5-VL-3B</td><td>DocVQA</td><td>0.836</td><td>0.772</td><td>0.0063</td><td>0.824</td><td>0.804</td><td>1.051</td><td>0.076</td></tr><tr><td>VQAv2</td><td>0.854</td><td>0.590</td><td>-0.0001</td><td>0.847</td><td>0.632</td><td>1.104</td><td>0.185</td></tr><tr><td>RealworldQA</td><td>0.834</td><td>0.623</td><td>-0.0187</td><td>0.838</td><td>0.624</td><td>1.123</td><td>0.120</td></tr><tr><td rowspan="3">Qwen3.5-4B</td><td>DocVQA</td><td>0.543</td><td>0.818</td><td>0.0009</td><td>0.527</td><td>0.804</td><td>0.650</td><td>0.130</td></tr><tr><td>VQAv2</td><td>0.441</td><td>0.728</td><td>0.0109</td><td>0.400</td><td>0.782</td><td>0.661</td><td>0.217</td></tr><tr><td>RealworldQA</td><td>0.480</td><td>0.723</td><td>-0.0051</td><td>0.461</td><td>0.740</td><td>0.527</td><td>0.076</td></tr><tr><td rowspan="3">Gemma3-4B</td><td>DocVQA</td><td>0.735</td><td>0.742</td><td>0.0055</td><td>0.740</td><td>0.788</td><td>1.107</td><td>0.078</td></tr><tr><td>VQAv2</td><td>0.706</td><td>0.661</td><td>0.0020</td><td>0.705</td><td>0.731</td><td>1.189</td><td>0.599</td></tr><tr><td>RealworldQA</td><td>0.698</td><td>0.692</td><td>-0.0227</td><td>0.784</td><td>0.635</td><td>1.135</td><td>0.095</td></tr><tr><td rowspan="3">GLM-4.6V-Flash</td><td>DocVQA</td><td>0.610</td><td>0.883</td><td>0.0083</td><td>0.572</td><td>0.959</td><td>1.009</td><td>0.284</td></tr><tr><td>VQAv2</td><td>0.485</td><td>0.768</td><td>0.0011</td><td>0.451</td><td>0.821</td><td>1.133</td><td>0.504</td></tr><tr><td>RealworldQA</td><td>0.454</td><td>0.818</td><td>-0.0008</td><td>0.432</td><td>0.867</td><td>1.141</td><td>0.353</td></tr></table>

Table 10: Fitted coefficients of the bilinear and unconstrained-linear predictors in Table 1. These are the saved example-level OOD fits, estimated on the 11 training boundary cells with validation checkpoint selection. Bilinear cue combination predicts $\hat { s } _ { J } = \beta _ { T } e _ { T } + \beta _ { V } e _ { V } + \beta _ { I } e _ { T } e _ { V } + \pi _ { 0 }$ , fixing the prior coefficient to one and the intercept to zero. Unconstrained linear predicts $\hat { s } _ { J } = \beta _ { T } e _ { T } + \beta _ { V } e _ { V } + \beta _ { \pi } \pi _ { 0 } + \beta _ { 0 }$ , with all four coefficients free. Here $e _ { T } = s _ { T } - \pi _ { 0 }$ and $e _ { V } = s _ { V } - \pi _ { 0 }$ after input clipping. To express the stored raw-probe regression in this evidence parameterization, we use $\beta _ { T } = w _ { T } , \beta _ { V } = w _ { V } , \beta _ { \pi } = w _ { \pi } + w _ { T } + w _ { V }$ , and $\beta _ { 0 } = w _ { 0 }$ . Gemma3-4B uses the same measured prior and answer-prefilled evidence probes as Table 12.

## D.4 Frozen and Refitted Weights

The transfer comparison in Section 4.4 uses the alternative manipulations described in Appendix A.5. We measure the probes on the resulting $6 \times 6$ grid using the same examples, candidate answers, and train/validation/test partitions as the text noise × blur experiments. The refit condition estimates weights from the new grid’s boundary training examples. The transfer condition instead freezes the weights fitted on text noise × blur and applies them to the newly measured probes. Both conditions are evaluated on the same held-out interior cells, with the prompt conventions in Appendix B.2 retained for each VLM. Table 11 compares the frozen and refitted weights directly for the four main VLMs and the three larger-model replications.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Dataset</td><td colspan="3">Frozen (text noise × blur)</td><td colspan="3">Refit (relevance × occlusion)</td></tr><tr><td> $\beta _ { T }$ </td><td> $\beta _ { V }$ </td><td> $\beta _ { T } / \beta _ { V }$ </td><td> $\beta _ { T }$ </td><td> $\beta _ { V }$ </td><td> $\beta _ { T } / \beta _ { V }$ </td></tr><tr><td rowspan="3">Qwen2.5-VL-3B</td><td>DocVQA</td><td>0.831</td><td>0.782</td><td>1.063</td><td>0.817</td><td>0.803</td><td>1.018</td></tr><tr><td>VQAv2</td><td>0.855</td><td>0.590</td><td>1.449</td><td>0.859</td><td>0.550</td><td>1.561</td></tr><tr><td>RealworldQA</td><td>0.841</td><td>0.586</td><td>1.436</td><td>0.806</td><td>0.567</td><td>1.423</td></tr><tr><td rowspan="3">Qwen3.5-4B</td><td>DocVQA</td><td>0.542</td><td>0.820</td><td>0.661</td><td>0.619</td><td>0.892</td><td>0.694</td></tr><tr><td>VQAv2</td><td>0.426</td><td>0.756</td><td>0.563</td><td>0.471</td><td>0.809</td><td>0.582</td></tr><tr><td>RealworldQA</td><td>0.477</td><td>0.707</td><td>0.675</td><td>0.490</td><td>0.779</td><td>0.630</td></tr><tr><td rowspan="3">Gemma3-4B</td><td>DocVQA</td><td>0.724</td><td>0.751</td><td>0.964</td><td>0.661</td><td>0.743</td><td>0.890</td></tr><tr><td>VQAv2</td><td>0.699</td><td>0.670</td><td>1.044</td><td>0.687</td><td>0.671</td><td>1.025</td></tr><tr><td>RealworldQA</td><td>0.772</td><td>0.610</td><td>1.267</td><td>0.753</td><td>0.622</td><td>1.210</td></tr><tr><td rowspan="3">GLM-4.6V-Flash</td><td>DocVQA</td><td>0.603</td><td>0.928</td><td>0.650</td><td>0.577</td><td>0.941</td><td>0.613</td></tr><tr><td>VQAv2</td><td>0.479</td><td>0.773</td><td>0.620</td><td>0.490</td><td>0.757</td><td>0.647</td></tr><tr><td>RealworldQA</td><td>0.455</td><td>0.815</td><td>0.558</td><td>0.429</td><td>0.846</td><td>0.507</td></tr><tr><td rowspan="3">Qwen2.5-VL-7B</td><td>DocVQA</td><td>0.801</td><td>0.824</td><td>0.972</td><td>0.777</td><td>0.877</td><td>0.885</td></tr><tr><td>VQAv2</td><td>0.695</td><td>0.698</td><td>0.995</td><td>0.694</td><td>0.713</td><td>0.973</td></tr><tr><td>RealworldQA</td><td>0.732</td><td>0.714</td><td>1.025</td><td>0.702</td><td>0.729</td><td>0.963</td></tr><tr><td rowspan="3">Qwen3.5-9B</td><td>DocVQA</td><td>0.613</td><td>0.897</td><td>0.683</td><td>0.626</td><td>0.913</td><td>0.686</td></tr><tr><td>VQAv2</td><td>0.466</td><td>0.809</td><td>0.576</td><td>0.454</td><td>0.809</td><td>0.561</td></tr><tr><td>RealworldQA</td><td>0.538</td><td>0.810</td><td>0.664</td><td>0.517</td><td>0.817</td><td>0.632</td></tr><tr><td rowspan="3">Gemma3-12B</td><td>DocVQA</td><td>0.737</td><td>0.856</td><td>0.860</td><td>0.682</td><td>0.881</td><td>0.775</td></tr><tr><td>VQAv2</td><td>0.597</td><td>0.749</td><td>0.797</td><td>0.589</td><td>0.772</td><td>0.764</td></tr><tr><td>RealworldQA</td><td>0.639</td><td>0.712</td><td>0.898</td><td>0.592</td><td>0.689</td><td>0.860</td></tr></table>

Table 11: Frozen and refitted modality weights. Example-level linear cue-combination weights underlying Tables 2 and 18. Frozen weights are fitted on the text noise × blur boundary and transferred unchanged; refitted weights are estimated on the relevance × occlusion boundary, using the same example partitions and fitting procedure. Ratios are computed before rounding. Across all 21 VLM–dataset pairs, the absolute relative ratio change, $| r _ { \mathrm { r e f i t } } / r _ { \mathrm { f r o z e n } } - 1 |$ with $r = \beta _ { T } / \beta _ { V }$ , has a median of 4.5% and a maximum below 10%.

## D.5 Cue Weights and Mean Evidence

Table 12 reports the individual weights and mean evidence underlying the ratios in Table $^ { 3 , }$ together with the mean and standard deviation of the prior in Eq. 6. For the cell-level fit, we average the input probes and joint score across training examples separately within each of the 11 boundary cells, then fit the same nonnegative two-weight model by least squares, giving each cell equal weight. The evidence and prior summaries use clean inputs $( \sigma = 0 , p = 0 )$ from the same held-out examples as Table 3; these examples are not used to estimate either set of weights. Cell-level fitting describes population mean preferences and does not by itself identify a correction for measurement error in the individual probes.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Dataset</td><td colspan="2">Example-level</td><td colspan="2">Cell-level [95% CI]</td><td colspan="2">Mean evidence</td><td colspan="2">Prior</td></tr><tr><td> $\beta _ { T }$ </td><td> $\beta _ { V }$ </td><td> $\beta _ { T }$ </td><td> $\beta _ { V }$ </td><td> $\overline { { e } } _ { T }$ </td><td> ${ \overline { { e } } } _ { V }$ </td><td>π₀</td><td>SD(π0)</td></tr><tr><td rowspan="4">Qwen2.5-VL-3B</td><td>DocVQA</td><td>0.831</td><td>0.782</td><td>0.973 [0.960, 0.987]</td><td>0.985 [0.956, 1.018]</td><td>3.739</td><td>-2.747</td><td>0.052</td><td>1.958</td></tr><tr><td>VQAv2</td><td>0.855</td><td>0.590</td><td>1.068 [1.045, 1.093]</td><td>0.840 [0.803, 0.878]</td><td>4.780</td><td>-3.422</td><td>0.041</td><td>2.463</td></tr><tr><td>RealworldQA</td><td>0.841</td><td>0.586</td><td>0.993 [0.965, 1.019]</td><td>0.909</td><td>3.205</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>0.699</td><td>[0.846, 0.985] 1.074</td><td></td><td>-1.203</td><td>0.044</td><td>1.820</td></tr><tr><td rowspan="4">Qwen3.5-4B</td><td>DocVQA</td><td>0.542</td><td>0.820</td><td>[0.679, 0.721] 0.545</td><td>[1.041,1.109] 0.940</td><td>4.357</td><td>-3.029</td><td>0.168</td><td>1.020</td></tr><tr><td>VQAv2</td><td>0.426</td><td>0.756</td><td>[0.524, 0.568]</td><td>[0.908, 0.978]</td><td>6.130</td><td>-3.012</td><td>一 -0.152</td><td>1.105</td></tr><tr><td>RealworldQA</td><td>0.477</td><td>0.707</td><td>0.576 [0.554, 0.600]</td><td>0.915 [0.872, 0.963]</td><td>4.768</td><td>-1.653</td><td>-0.102</td><td></td></tr><tr><td></td><td></td><td></td><td>0.886</td><td>0.998</td><td></td><td></td><td></td><td>0.739</td></tr><tr><td rowspan="3">Gemma3-4B</td><td>DocVQA</td><td>0.724</td><td>0.751</td><td>[0.865, 0.909] 1.061</td><td>[0.955, 1.044] 1.031</td><td>8.212</td><td>-5.208</td><td>-0.552</td><td>5.162</td></tr><tr><td>VQAv2</td><td>0.699</td><td>0.670</td><td>[1.017, 1.104] 0.947</td><td>[0.970, 1.093] 1.000</td><td>12.034</td><td>-8.531</td><td>0.691</td><td>6.079</td></tr><tr><td>RealworldQA</td><td>0.772</td><td>0.610</td><td>[0.902, 0.994]</td><td>[0.898, 1.121]</td><td>7.272</td><td>-1.930</td><td>-0.488</td><td>3.669</td></tr><tr><td rowspan="4">GLM-4.6V-Flash</td><td>DocVQA</td><td>0.603</td><td>0.928</td><td>0.741 [0.720,0.761]</td><td>1.161 [1.129, 1.195]</td><td>6.273</td><td>-3.374</td><td>-0.141</td><td>2.541</td></tr><tr><td>VQAv2</td><td>0.479</td><td>0.773</td><td>0.660 [0.632, 0.687]</td><td>1.084 [1.027, 1.139]</td><td>8.343</td><td>-2.948</td><td>-0.239</td><td>3.447</td></tr><tr><td>RealworldQA0.455</td><td></td><td>0.815</td><td>0.598 [0.570, 0.628]</td><td>1.121 [1.063, 1.186]</td><td>5.808</td><td>-1.890</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.076</td><td>2.562</td></tr></table>

Table 12: Weights, evidence, and prior in linear cue combination. Example-level weights are the saved OOD fits. Cell-level weights fit the same nonnegative two-weight model to the 11 training boundary-cell means with equal weight per cell, fixed prior coefficient one, and no intercept. Brackets give pointwise 95% percentile intervals from 2,000 whole-example refit bootstraps (Appendix C.4). The evidence means and prior mean and standard deviation are computed on clean held-out test inputs $( \sigma = 0 , p = 0 )$ , matching the evidence ratios in Table 3. Standard deviations use divisor N, the number of held-out examples. Input probes are clipped to [−30, 30] before subtracting the prior. Gemma uses its measured knowledge-prefilled prior and answer-prefilled evidence probes.

## E Supplementary Evaluations

## E.1 In-Distribution Predictor Comparison

Table 13 repeats Table 1 with all 36 degradation cells represented in training, validation, and testing, while keeping the underlying example IDs disjoint. This comparison tests whether relaxing the linear cue-combination constraints or adding a bilinear interaction improves prediction when the full corruption grid is represented during fitting.

<table><tr><td>Model</td><td>Predictor</td><td>KL↓</td><td> $R ^ { 2 } \uparrow$ </td></tr><tr><td rowspan="5">Qwen2.5-VL-3B</td><td>Best Single Probe</td><td>0.242</td><td>0.494</td></tr><tr><td>Calibrated Naive Bayes</td><td>0.121</td><td>0.695</td></tr><tr><td>Linear Cue Combination</td><td>0.073</td><td>0.827</td></tr><tr><td>Bilinear Cue Combination</td><td>0.074</td><td>0.826</td></tr><tr><td>Unconstrained Linear</td><td>0.074</td><td>0.828</td></tr><tr><td rowspan="5">Qwen3.5-4B</td><td>Best Single Probe</td><td>0.364</td><td>0.192</td></tr><tr><td>Calibrated Naive Bayes</td><td>0.188</td><td>0.379</td></tr><tr><td>Linear Cue Combination</td><td>0.089</td><td>0.794</td></tr><tr><td>Bilinear Cue Combination</td><td>0.088</td><td>0.795</td></tr><tr><td>Unconstrained Linear</td><td>0.080</td><td>0.809</td></tr><tr><td rowspan="5">Gemma3-4B</td><td>Best Single Probe</td><td>0.946</td><td>0.416</td></tr><tr><td>Calibrated Naive Bayes</td><td>0.612</td><td>0.501</td></tr><tr><td>Linear Cue Combination</td><td>0.364</td><td>0.672</td></tr><tr><td>Bilinear Cue Combination</td><td>0.365</td><td>0.670</td></tr><tr><td>Unconstrained Linear</td><td>0.373</td><td>0.674</td></tr><tr><td rowspan="5">GLM-4.6V-Flash</td><td>Best Single Probe</td><td>0.551</td><td>0.467</td></tr><tr><td>Calibrated Naive Bayes</td><td>0.309</td><td>0.571</td></tr><tr><td>Linear Cue Combination</td><td>0.113</td><td>0.845</td></tr><tr><td>Bilinear Cue Combination</td><td>0.113</td><td>0.846</td></tr><tr><td>Unconstrained Linear</td><td>0.115</td><td>0.850</td></tr></table>

Table 13: ID replication of Table 1, averaging the three dataset-level metrics equally. KL is the mean divergence from the observed to the predicted two-answer distribution; $R ^ { 2 }$ compares predicted and observed joint log-odds. Bold marks the best of the three predictors above the inner rule.

## E.2 Random Forest Results

We evaluate a random forest as a flexible nonlinear predictor of the joint log-odds from the same unimodal probes and prior. Fitting and hyperparameter selection follow Appendix C.2, with the same example-level splits as the parametric predictors. Table 14 compares it with linear cue combination under the ID protocol for the four main-paper VLMs, with all 36 degradation cells represented during fitting. The random forest improves mean $R ^ { 2 }$ over linear cue combination by at most 0.015.

<table><tr><td>Model</td><td>Predictor</td><td> $\mathrm { K L } \downarrow$ </td><td> $R ^ { 2 } \uparrow$ </td></tr><tr><td rowspan="2">Qwen2.5-VL-3B</td><td>Linear Cue Combination</td><td>0.073</td><td>0.827</td></tr><tr><td>Random Forest</td><td>0.077</td><td>0.820</td></tr><tr><td rowspan="2">Qwen3.5-4B</td><td>Linear Cue Combination</td><td>0.089</td><td>0.794</td></tr><tr><td>Random Forest</td><td>0.082</td><td>0.809</td></tr><tr><td rowspan="2">Gemma3-4B</td><td>Linear Cue Combination</td><td>0.364</td><td>0.672</td></tr><tr><td>Random Forest</td><td>0.421</td><td>0.674</td></tr><tr><td rowspan="2">GLM-4.6V-Flash</td><td>Linear Cue Combination</td><td>0.113</td><td>0.845</td></tr><tr><td>Random Forest</td><td>0.134</td><td>0.830</td></tr></table>

Table 14: ID comparison of random forest and linear cue combination, using the same protocol as Table 13 and averaging the three dataset-level metrics equally. KL is the mean divergence from the observed to the predicted two-answer distribution; $R ^ { 2 }$ compares predicted and observed joint log-odds. Bold marks the better predictor within each VLM and metric.

## E.3 Larger-Model Replications

We repeat the predictor comparison and weight analysis for Qwen2.5-VL-7B, Qwen3.5-9B, and Gemma3- 12B. We do not evaluate the larger GLM-4.6V because its 108B-parameter checkpoint is substantially larger than the other models considered here. Table 15 replicates Table 1 under the same OOD protocol; Table 16 extends the example-level weights in Table 12 to the larger models.

<table><tr><td>Model</td><td>Predictor</td><td>KL↓</td><td> $R ^ { 2 } \uparrow$ </td></tr><tr><td rowspan="5">Qwen2.5-VL-7B</td><td>Best Single Probe</td><td>0.231</td><td>0.591</td></tr><tr><td>Calibrated Naive Bayes</td><td>0.116</td><td>0.732</td></tr><tr><td>Linear Cue Combination</td><td>0.064</td><td>0.865</td></tr><tr><td>Bilinear Cue Combination</td><td>0.064</td><td>0.866</td></tr><tr><td>Unconstrained Linear</td><td>0.065</td><td>0.865</td></tr><tr><td rowspan="5">Qwen3.5-9B</td><td>Best Single Probe</td><td>0.094</td><td>0.272</td></tr><tr><td>Calibrated Naive Bayes</td><td>0.056</td><td>0.506</td></tr><tr><td>Linear Cue Combination</td><td>0.022</td><td>0.820</td></tr><tr><td>Bilinear Cue Combination</td><td>0.022</td><td>0.819</td></tr><tr><td>Unconstrained Linear</td><td>0.021</td><td>0.821</td></tr><tr><td rowspan="4">Gemma3-12B</td><td>Best Single Probe</td><td>0.717</td><td>0.506</td></tr><tr><td>Calibrated Naive Bayes</td><td>0.499</td><td>0.568</td></tr><tr><td>Linear Cue Combination</td><td>0.263</td><td>0.763</td></tr><tr><td>Bilinear Cue Combination</td><td>0.268</td><td>0.760</td></tr><tr><td></td><td>Unconstrained Linear</td><td>0.278</td><td>0.761</td></tr></table>

Table 15: OOD predictor comparison for the three larger VLMs, replicating Table 1. Metrics average the three datasets equally. KL is the mean divergence from the observed to the predicted two-answer distribution; $R ^ { 2 }$ compares predicted and observed joint log-odds. Bold marks the best of the three predictors above the inner rule.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Dataset</td><td colspan="2">Example-level</td><td colspan="3">Cell-level [95% CI]</td></tr><tr><td> $\beta _ { T }$ </td><td> $\beta _ { V }$ </td><td> $\beta _ { T }$ </td><td></td><td> $\beta _ { V }$ </td></tr><tr><td rowspan="4">Qwen2.5-VL-7B</td><td>DocVQA</td><td>0.801</td><td>0.824</td><td>0.929 [0.914,0.944]</td><td></td><td>0.996 [0.972, 1.024]</td></tr><tr><td></td><td></td><td></td><td>0.846</td><td></td><td>0.854</td></tr><tr><td>VQAv2</td><td>0.695</td><td>0.698</td><td>[0.820,0.873]</td><td>0.871</td><td>[0.823,0.887] 0.995</td></tr><tr><td>RealworldQA</td><td>0.732</td><td>0.714</td><td>[0.837,0.901]</td><td></td><td>[0.945, 1.055]</td></tr><tr><td rowspan="4">Qwen3.5-9B</td><td>DocVQA</td><td>0.613</td><td>0.897</td><td></td><td>0.725 [0.711,0.740]</td><td>1.085 [1.060, 1.113]</td></tr><tr><td>VQAv2</td><td>0.466</td><td>0.809</td><td></td><td>0.572</td><td>1.014</td></tr><tr><td></td><td></td><td></td><td>[0.548,0.595]</td><td>0.615</td><td>[0.977, 1.054] 1.084</td></tr><tr><td>RealworldQA</td><td>0.538</td><td>0.810</td><td>[0.593, 0.638]</td><td></td><td>[1.035,1.135]</td></tr><tr><td rowspan="5">Gemma3-12B</td><td>DocVQA</td><td>0.737</td><td>0.856</td><td></td><td>0.920</td><td>1.113</td></tr><tr><td></td><td></td><td></td><td></td><td>[0.897,0.943] 0.856</td><td>[1.068, 1.160] 1.139</td></tr><tr><td>VQAv2</td><td>0.597</td><td>0.749</td><td></td><td>[0.826, 0.887]</td><td>[1.078, 1.205]</td></tr><tr><td></td><td></td><td></td><td></td><td>0.803</td><td>1.181</td></tr><tr><td>RealworldQA 0.639</td><td></td><td></td><td>0.712</td><td>[0.772, 0.834] [1.087, 1.281]</td><td></td></tr></table>

Table 16: Fitted cue weights for larger VLMs. Example-level weights are the saved OOD fits. Cell-level weights fit the same nonnegative two-weight model to the 11 training boundary-cell means with equal weight per cell, fixed prior coefficient one, and no intercept. Brackets give pointwise 95% percentile intervals from 2,000 whole-example refit bootstraps (Appendix C.4). Gemma uses its measured knowledge-prefilled prior and answer-prefilled evidence probes.

Integration bias and evidence ratio. Table 17 repeats Table 3 for the three larger models, comparing example-level and cell-level weight ratios alongside evidence ratios on clean inputs from the same held-out test examples.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Dataset</td><td colspan="3"> $\beta _ { T } / \beta _ { V }$  [95% CI]</td><td rowspan="2"> $\left. \bar { e } _ { T } \right. / \left. \bar { e } _ { V } \right.$ </td></tr><tr><td>Example</td><td></td><td>Cell</td></tr><tr><td rowspan="3">Qwen2.5-VL-7B</td><td>DocVQA</td><td>0.972</td><td>[0.939, 1.003]</td><td>0.933 [0.901, 0.964]</td><td>1.20</td></tr><tr><td>VQAv2</td><td>0.995</td><td>[0.952, 1.040]</td><td>0.991 [0.950, 1.030]</td><td>1.31</td></tr><tr><td>RealworldQA</td><td>1.025</td><td>[0.965, 1.087]</td><td>0.876 [0.818, 0.935]</td><td>2.42</td></tr><tr><td rowspan="3">Qwen3.5-9B</td><td>DocVQA</td><td></td><td>0.683 [0.667, 0.709]</td><td>0.668 [0.646, 0.690]</td><td>1.46</td></tr><tr><td>VQAv2</td><td>0.576</td><td>[0.552, 0.630]</td><td>0.564 [0.538, 0.591]</td><td>2.01</td></tr><tr><td>RealworldQA</td><td>0.664</td><td>[0.613, 0.687]</td><td>0.568 [0.534, 0.604]</td><td>3.39</td></tr><tr><td rowspan="3">Gemma3-12B</td><td>DocVQA</td><td>0.860</td><td>[0.820, 0.912]</td><td>0.826 [0.786, 0.869]</td><td>1.29</td></tr><tr><td>VQAv2</td><td></td><td>0.797 [0.750, 0.828]</td><td>0.752 [0.710, 0.795]</td><td>1.85</td></tr><tr><td>RealworldQA</td><td></td><td>0.898 [0.834, 0.968]</td><td>0.680 [0.620, 0.743]</td><td>3.03</td></tr></table>

Table 17: Integration bias and evidence ratio in larger VLMs. Ratios above one indicate greater text weighting or stronger text evidence. Brackets give pointwise 95% percentile intervals from 2,000 whole-example refit bootstraps (Appendix C.4). Example fits repeat validation checkpoint selection; cell fits recompute the 11 training boundarycell means in each resample. Evidence ratios are the absolute values of signed mean evidence on clean held-out inputs $( \sigma = 0 , p = 0 )$ , with absolute values taken after averaging.

Transfer across reliability manipulations. Table 18 repeats the weight-transfer comparison for these larger models. We compare weights refitted on relevance × occlusion against the frozen text noise × blur weights in Table 16, using identical target-grid test examples and the OOD protocol in both cases.
<table><tr><td>Model</td><td>Refit  $R ^ { 2 }$ </td><td>Transferred  $R ^ { 2 }$ </td></tr><tr><td>Qwen2.5-VL-7B</td><td>0.822</td><td>0.830</td></tr><tr><td>Qwen3.5-9B</td><td>0.759</td><td>0.757</td></tr><tr><td>Gemma3-12B</td><td>0.688</td><td>0.689</td></tr></table>

Table 18: Weight transfer for the larger VLMs, replicating Table 2. Refit estimates weights on the boundary of relevance × pixel occlusion; Transferred freezes weights estimated on the boundary of text noise × blur. Both are evaluated on the same held-out interior examples of the new grid. Values average the three dataset-level $R ^ { 2 }$ scores equally.

## F Resources and Disclosures

## F.1 Computational Resources

All experiments were run on a single node with 8 NVIDIA L40S GPUs. Our experiments are inferenceonly, we do not train or fine-tune any VLM. Predictor fitting runs on CPU: the linear, bilinear, and unconstrained models use full-batch L-BFGS, while the random forest uses decision-tree ensembles (Appendix C.2).

## F.2 Models and Checkpoints

Table 19 lists the checkpoint identifiers used for the four main VLMs and the larger-model replications in Appendix E.3. Each VLM uses its native processor and chat template; scoring conventions, including Gemma3’s assistant prefixes, are given in Appendix B.

<table><tr><td>Evaluation</td><td>VLM</td><td>Checkpoint identifier</td></tr><tr><td>Main</td><td>Qwen2.5-VL-3B</td><td>Qwen/Qwen2.5-VL-3B-Instruct</td></tr><tr><td>Main</td><td>Qwen3.5-4B</td><td>Qwen/Qwen3.5-4B</td></tr><tr><td>Main</td><td>Gemma3-4B</td><td>google/gemma-3-4b-it</td></tr><tr><td>Main</td><td>GLM-4.6V-Flash</td><td>zai-org/GLM-4.6V-Flash</td></tr><tr><td>Larger</td><td>Qwen2.5-VL-7B</td><td>Qwen/Qwen2.5-VL-7B-Instruct</td></tr><tr><td>Larger</td><td>Qwen3.5-9B</td><td>Qwen/Qwen3.5-9B</td></tr><tr><td>Larger</td><td>Gemma3-12B</td><td>google/gemma-3-12b-it</td></tr></table>

Table 19: Checkpoint identifiers for the VLMs evaluated in this paper. Gemma3 checkpoints are loaded from local copies of the listed repositories.

## F.3 Artifact Licensing and Intended Use

The source benchmarks we build on are publicly released for research use: DocVQA (Mathew et al., 2021), VQAv2 (Antol et al., 2015), and RealworldQA (xAI, 2024), together with the conflicting-text source of Deng et al. (2025). The four models we study, Qwen2.5-VL (Qwen Team, 2025), Qwen3.5 (Qwen Team, 2026), Gemma3 (Gemma Team et al., 2025), and GLM-4.6V (GLM-V Team et al., 2025), are distributed as open-weight models under their respective licenses. We use all of these artifacts solely for non-commercial research, consistent with their intended use as research benchmarks and openly released models.

## F.4 Release Information

Code and data will be released upon publication.

## F.5 Usage of AI Tools

AI tools were used to help write and debug code for the experiments in this project and to polish the prose of the paper draft. All experimental design, analysis, and claims are the authors’ own, and the authors take full responsibility for the content of the paper.