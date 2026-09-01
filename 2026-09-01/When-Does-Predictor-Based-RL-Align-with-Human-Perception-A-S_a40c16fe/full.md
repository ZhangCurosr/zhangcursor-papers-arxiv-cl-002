# When Does Predictor-Based RL Align with Human Perception? A Study of Subjective Rewards in Codec-Based Speech Language Models

Joonyong Park, Jerry Li

Spellbrush

jyjoon97@gmail.com jerry@sizigistudios.com

## Abstract

Codec-based text-to-speech (TTS) models make language-model post-training applicable to speech generation, but it remains unclear when learned perceptual predictors can serve as reinforcement-learning rewards without losing alignment with human listeners. We study this question with Group Relative Policy Optimization (GRPO) using learned rewards for anime-like speaking style, naturalness, likability, and arousal. To prevent perceptual rewards from being optimized through transcript drift, we introduce a character-error-rate (CER) zone constraint and compare policy optimization with Best-of-N reranking under the same reward gate. Across single-reward runs, each reward primarily improves its own target metric, showing that subjective predictors are not interchangeable quality surrogates. Multi-rater A/B tests further show uneven human transfer, while a reward-gap analysis separates average transfer from within-axis calibration: signed reward gaps significantly predict listener choices in the pooled analysis, whereas residual CER gaps do not, but per-axis calibration remains heterogeneous. Best-of-8 is a strong human-level baseline and is not clearly worse than GRPO perceptually, suggesting that GRPO should be viewed as amortizing reward-selected behavior into the policy rather than uniformly outperforming reranking. These results support analyzing subjective speech rewards as predictor– axis–base tuples and provide practical diagnostics for selecting rewards before multi-reward speech post-training.

## 1 Introduction

Modern text-to-speech (TTS) systems increasingly generate speech through discrete acoustic tokens. A neural audio codec first converts a waveform into a sequence of discrete codes, and a speech language model then predicts these codes autoregressively from text and, in zero-shot settings, a short reference audio prompt. The generated token sequence is finally decoded back into a waveform. This codec-based formulation underlies recent systems (Wang et al., 2023; Borsos et al., 2023; Du et al., 2024; Ye et al., 2025b), and makes speech generation resemble conditional language modeling over acoustic tokens. As a result, posttraining methods developed for large language model (LLM), including reinforcement learning (RL) from automatic or learned rewards, can now be applied to TTS policies. Recent RL work for TTS has shown that such methods can improve automatically measurable properties of generated speech, including transcription fidelity, target-text likelihood, speaker similarity, duration control, and prosodic stability (Liu et al., 2025; Li et al., 2026; Zhong et al., 2025; Gao et al., 2025).

However, many useful speech-generation objectives are not verifiable in the same way as transcription accuracy. Naturalness, expressiveness, likability, affective intensity, and domain-specific speaking style are perceptual attributes: they are ultimately defined by listener judgments rather than by a symbolic target string. A natural approach is to approximate such listener-defined attributes with learned perceptual predictors, such as mean opinion score (MOS) or speech-quality predictors, and use their scalar outputs as training rewards (Saeki et al., 2022; Mittag et al., 2021; Chen et al., 2024). The difficulty is that a learned perceptual predictor is only a proxy (Ziegler et al., 2019; Gao et al., 2023). A speech policy can increase the predictor score by exploiting predictor blind spots, producing outputs that are high-scoring but unintelligible, unstable, artifact-laden, or not actually preferred by listeners. This risk is especially acute in speech because generated waveforms must jointly satisfy transcription fidelity, acoustic quality, speaker similarity, naturalness, and prosodic or stylistic appropriateness (Chen et al., 2025; Mittag et al., 2021). Existing TTS-RL systems therefore often rely on verifiable or automatically computed reward components, such as character error rate (CER), word error rate (WER), negative loglikelihood (NLL), speaker similarity, duration, entropy, and rule-based prosody rewards (Liu et al., 2025; Li et al., 2026; Zhong et al., 2025; Gao et al., 2025). These rewards are useful for stabilizing generation, but they do not answer when a learned subjective predictor can be optimized while remaining aligned with listener judgments.

This paper asks when predictor-based RL can move a codec speech language model along a subjective reward axis while preserving humanperceptual alignment. We study this question with group relative policy optimization (GRPO) using learned perceptual predictors as rewards. To prevent subjective rewards from being optimized through transcript drift, we use a CER-zone hard constraint: perceptual rewards are active only when the generated speech remains sufficiently intelligible, and outputs that violate the transcription constraint receive a fixed negative reward. We also compare GRPO with Best-of-N reranking under the same reward gate, separating inference-time reward selection from policy-level movement. Because different reward settings vary not only in perceptual target but also in predictor architecture, training data, score scale, and base-model distribution, we analyze each setting as a predictor–axis– base tuple rather than attributing success or failure to the subjective axis alone.

Our study combines three levels of evidence. First, we compare single-reward GRPO runs under the same CER-zone scaffold to determine whether each reward produces a targeted machine-level shift or merely a generic quality change. Second, we use multi-rater A/B tests to evaluate whether those machine-level shifts are perceived by listeners. Third, we analyze reward signal quality using diagnostics such as reward-gap calibration, baseoutput score spread, domain match, and withinzone signal strength. This design allows us to distinguish three phenomena that are often conflated: whether a subjective predictor can be optimized, whether the resulting policy shift remains humanaligned, and whether the reward is suitable for inclusion in future multi-reward post-training.

Our main contributions are:

• A controlled training-and-evaluation scaffold for subjective speech rewards. We instantiate GRPO for learned perceptual predictors under a CER-zone hard constraint, compare it with Best-of-N reranking to separate reward-based sample selection from policylevel movement, and evaluate both first-shot behavior and CER-retry behavior used for human evaluation.

• A multi-rater human study of predictor– axis–base tuples. We show that machinelevel reward gains transfer unevenly to listeners: some tuples yield strong human-aligned shifts, some yield only modest transfer, and some fail on average despite high-confidence successes.

• Diagnostics for reward signal quality. We evaluate reward-gap calibration, base-output spread, domain match, and within-zone signal strength as diagnostics for deciding which subjective rewards are suitable for RL or future multi-reward post-training.

Together, these results characterize not merely whether subjective predictors can be optimized, but when predictor-based RL remains aligned with human perception in codec-based speech language models. We release code, prompts, generated audio samples, and reward scores at https://github. com/sizigi/animeGRPO.<sup>1</sup>

## 2 Related Work

Codec-based speech language models. Codecbased speech language models formulate TTS as autoregressive generation over discrete acoustic tokens produced by neural audio codecs such as SoundStream and EnCodec (Zeghidour et al., 2022; Défossez et al., 2023). Systems such as VALL-E, AudioLM, CosyVoice, and Llasa then generate these tokens from text and optional acoustic context (Wang et al., 2023; Borsos et al., 2023; Du et al., 2024; Ye et al., 2025b,a).

RL and preference optimization for TTS. Group Relative Policy Optimization (GRPO) was introduced as a critic-free variant of Proximal Policy Optimization that estimates advantages from group scores (Shao et al., 2024). Recent work has begun to apply GRPO, preference optimization, or reward-based fine-tuning to TTS. GRPO for TTS has optimized automatic speech recognition (ASR)- derived rewards such as CER and NLL (Liu et al., 2025). DMOSpeech 2 applies GRPO to duration prediction with speaker similarity and WER-based rewards (Li et al., 2026). Multi-Reward GRPO combines intelligibility and speaker-similarity objectives with rule-based rewards for length, decoding stability, and prosody alignment (Zhong et al., 2025). DiffRO optimizes neural codec language models with differentiable reward prediction from speech tokens (Gao et al., 2025). SpeechAlign studies preference-based optimization for speech generation using direct preference optimization (DPO), proximal policy optimization (PPO), and Best-of-N selection (Zhang et al., 2024). These studies establish that reward-based post-training can improve TTS, but they primarily focus on verifiable, automatically computed, or task-specific reward components.

![](images/730acb18ba38c564aa139743a305aa650a14d48600d724e3887bdb9c0106cad8.jpg)  
Figure 1: Overview of constrained perceptual GRPO.

Learned perceptual predictors. A long line of speech evaluation work aims to predict subjective listener judgments automatically. Early neural MOS predictors such as MOSNet model human naturalness ratings for converted or synthesized speech (Lo et al., 2019). More recent non-intrusive quality predictors, including UTMOS and NISQA, estimate naturalness or multidimensional speech quality without reference audio (Saeki et al., 2022; Mittag et al., 2021). Such predictors turn perceptual judgments into scalar model outputs and can therefore serve as proxy objectives for generation or post-training (Chen et al., 2024). However, a predictor score is not equivalent to a human judgment: learned predictors can be miscalibrated, outof-domain, insensitive to relevant perceptual differences, or vulnerable to overoptimization. This motivates evaluating not only whether a perceptual score increases, but whether the increase corresponds to listener preference.

Reward overoptimization and calibration. Learned rewards are known to suffer from overoptimization: a policy can obtain high proxy reward while degrading the true human objective (Gao et al., 2023). Common mitigations include KL regularization, constrained optimization, improved reward modeling, and inference-time reranking (Ziegler et al., 2019; Achiam et al., 2017). In speech, this problem is compounded by the need to satisfy multiple coupled constraints, including intelligibility, speaker consistency, acoustic quality, and style. This makes calibration between machine reward differences and human-perceived differences particularly important. This concern applies both to policy optimization and to inferencetime Best-of-N selection, since both can overoptimize a proxy reward (Gao et al., 2023).

## 3 Constrained Perceptual GRPO 3.1 Problem setup

Given a text prompt x, the codec speech language model samples a discrete acoustic-token sequence $s = ( s _ { 1 } , . . . , s _ { T } ) \sim \pi _ { \theta } ( \cdot \mid x )$ . A codec decoder maps s to a waveform $w = \operatorname { D e c } ( s )$ . Perceptual predictors are applied to the decoded waveform, but for brevity we write $p ( s ) : = p ( \mathrm { D e c } ( s ) )$ . The goal is to improve a target perceptual attribute while keeping the generated speech intelligible and close to a frozen reference policy $\pi _ { \mathrm { r e f } }$

## 3.2 CER-zone reward template

Let $c ( s , x )$ be the character error rate (CER) between an automatic-speech-recognition (ASR) transcript of $w = \operatorname { D e c } ( s )$ and the input text x. For a learned perceptual predictor score $p ( s )$ , we define

$$
R ( s , x ) = \left\{ \begin{array} { l l } { \tilde { p } ( s ) + b , } & { c ( s , x ) \leq \tau _ { l } , } \\ { \tilde { p } ( s ) , } & { \tau _ { l } < c ( s , x ) \leq \tau _ { h } , } \\ { - \rho , } & { c ( s , x ) > \tau _ { h } . } \end{array} \right.\tag{1}
$$

The three cases correspond to CLEAN, FEASIBLE, and VIOLATE zones. Here $\tilde { p } ( s )$ is the normalized non-negative predictor score, b is the CLEAN-zone bonus, and $\rho$ is the fixed VIOLATE penalty. In all experiments, we set $\tau _ { l } = 0 . 1 0 , \tau _ { h } = 0 . 3 0 , b = 0 . 5$ and $\rho = 1 . 0$ . This hard-zone design prevents a high perceptual score from numerically compensating for a transcription failure. The predictor-specific definitions of $\tilde { p } ( s )$ are given in §4.2.

## 3.3 GRPO optimization and stopping

We use GRPO that estimates advantages by comparing multiple rollouts for the same prompt. For each prompt, we sample $K = 4$ rollouts and compute a scalar reward $r _ { i }$ by subtracting an adaptive in-reward KL penalty from $R ( s _ { i } , x )$ . We then normalize rewards within the rollout group as

$$
\hat { A } _ { i } = ( r _ { i } - \mu _ { r } ) / ( \sigma _ { r } + \epsilon ) .
$$

Implementation details, including the verl and vLLM rollout setup, are reported in Appendix A.

We do not select checkpoints by the raw perceptual predictor score alone. Instead, we select the checkpoint that maximizes constraint-aware validation reward while monitoring CER violations and KL drift. This is important because larger predictor scores or reward gaps can arise from proxy overoptimization, transcript drift, or predictor blind spots. Selected checkpoints and full hyperparameters are reported in Appendix A, and KL trajectories are reported in Appendix J.

## 4 Experimental Setup

## 4.1 Base speech model

We use Llasa as the codec-based TTS backbone and XCodec2 as the acoustic tokenizer and waveform decoder (Ye et al., 2025b,a). We used a public checkpoint<sup>2</sup>, a multilingual variant trained with Emilia and Multilingual LibriSpeech (MLS), which provide Japanese-containing in-the-wild multilingual speech and multilingual read-speech data, respectively (He et al., 2024; Pratap et al., 2020). Our GRPO implementation builds on verl (Sheng et al., 2025). All main runs update the full actor parameters, and the KL reference policy is a frozen copy of the same base checkpoint. The auxiliary English ANIMESCORE experiment is reported in Appendix H and is used only as supporting evidence for cross-language behavior.

## 4.2 Reward predictors

We use three learned perceptual predictors in the main experiments and one additional arousal predictor for a training-only diagnostic in Appendix I.

ANIMESCORE<sup>3</sup> is a pairwise-preference-trained anime-likeness predictor (Park and Li, 2026), using a WavLM-base encoder, temporal modeling, and a RankNet-style ranking head. Its raw output is a signed score, which we normalize as

$$
\tilde { p } ( s ) = \operatorname* { m a x } ( 0 , p ( s ) + 3 . 0 ) .
$$

UTMOS is used off-the-shelf from UTMOS22- strong<sup>4</sup> (Saeki et al., 2022). It predicts a MOSlike naturalness score on [1, 5], and we set $\tilde { p } ( s ) =$ $p ( s ) / 5$

LIKABILITY is a likability predictor trained for this study on CocoNut-Humoresque (Suda et al., 2024). It uses pretrained WavLM encoder<sup>5</sup> and its raw score is the expected class value on [1, 6]. We use the original CocoNut-Humoresque split and train with class-weighted cross-entropy rather than regression to preserve score spread for GRPO. For reward computation, we set

$$
{ \tilde { p } } ( s ) = { \frac { p ( s ) - 1 } { 5 } } .
$$

We additionally use an off-the-shelf MSP-Dim arousal predictor only for the training-dynamics negative case discussed in §7.3; details are in Appendix I.

For the CER gate, we transcribe generated waveforms with Whisper large-v3 (Radford et al., 2023) and compute CER against the canonical written prompt. No additional language-specific text normalization is applied. Additional reward-predictor training and implementation details are provided in Appendix B.

## 4.3 Training and evaluation prompts

GRPO training uses 900 Japanese Wikipediaderived prompts. All main reward-axis runs use the same prompt training set to isolate the reward predictor as the intended varying factor. Checkpoint selection uses a disjoint 100-prompt validation set.

The main evaluation set contains 50 held-out Japanese prompts and is disjoint from GRPO training, validation, and reward-model training data. It is partitioned into five groups: emotional, animestylized, neutral, long-form narrative, and linguistically challenging prompts. Auxiliary English prompts are independently sourced rather than translations of the Japanese prompts and are described in Appendix H.

## 4.4 Decoding and evaluation protocol

We distinguish two evaluation modes. First-shot evaluation uses a single stochastic generation with fixed seed and no filtering or regeneration. We use this setting to measure the model’s unfiltered behavior, including the raw violation rate, defined as the fraction of outputs with CER > 0.30.

CER-retry evaluation is used to prepare audio for human level evaluation tests. Its purpose is to reduce obvious transcript-failure confounds while avoiding asymmetric post-hoc filtering. For both base and GRPO systems, we first generate with fixed seed. If the output has CER > 0.30, we regenerate with retry seeds and select the first candidate with CER ≤ 0.30. If no candidate satisfies the threshold, we keep the lowest-CER candidate. This rule is applied symmetrically to both sides of each pair. All 50 prompts are retained, including residual CER violators after all retries, to avoid post-hoc filtering by a metric tied to the RL scaffold.

## 4.5 Evaluation metrics

We evaluate each reward-axis run with both machine-level and human-level evaluations.

Machine-level evaluation. For each system, we generate speech on the 50-prompt held-out test set and report changes from the base model. For the target perceptual axis, we report the corresponding predictor delta; we also score each generated sample with the other predictors to measure cross-axis side effects. To measure transcription fidelity, we compute character error rate (CER) between the input text and a Whisper transcript of the generated waveform, and report both mean and median CER. We also report the violation rate, defined as the fraction of generated samples with CER > 0.30. For first-shot evaluation, this violation rate is measured from a single seed generation. For CER-retry evaluation and human-evaluation audio, it is measured after applying the symmetric retry protocol described in §4.4.

We also compare GRPO with Best-of-N reranking. For Best-of-N, we sample N candidates from the base model, score each candidate with the same CER-gated reward used for GRPO, and select the highest-scoring candidate. This comparison tests whether reward-selected samples already exist in the base model’s support, while GRPO tests whether such selection behavior can be amortized into the policy.

<table><tr><td>Reward</td><td>∆CER</td><td>∆AS</td><td>∆Likab.</td><td>∆UTMOS</td></tr><tr><td>ANIMESCORE</td><td>-0.030</td><td>+1.353</td><td>+0.075</td><td>-0.037</td></tr><tr><td>LIKABILITY</td><td>-0.018</td><td>+0.029</td><td>+0.167</td><td>+0.090</td></tr><tr><td>UTMOS</td><td>-0.030</td><td>-0.072</td><td>+0.140</td><td>+0.485</td></tr></table>

Table 1: Cross-axis mean objective shifts between base and Zone-CER GRPO outputs. Values are changes from the corresponding base outputs.

Human-level evaluation. We conduct pairwise listening tests through Lancers<sup>6</sup>, a Japanese crowdsourcing platform. The main human study covers three reward axes: ANIMESCORE, UTMOS, and LIKABILITY. Each axis contains 50 paired items, each item receives 5 independent ratings, and each axis is rated by 10 distinct Japanese listeners with no listener overlap between axes. The auxiliary English ANIMESCORE study is reported in Appendix H.

Each pair is presented as A/B audio with randomized side assignment, and listeners are not told which side corresponds to base, GRPO, or Best-of-N. For ANIMESCORE, listeners choose the clip that sounds more anime-like, defined as voice-actor-like speaking style and performance. For UTMOS, listeners choose the clip with better naturalness and audio quality, defined as fewer artifacts, noise, and distortions. For LIKABILITY, listeners choose the clip that sounds more pleasing or comfortable in voice and speaking manner. A separate skip path exists for problematic clips, but those are excluded from win-rate calculations.

At the item level, a system wins an item if more than half of raters prefer that system. Human win rate (HWR) is the fraction of items for which the listener majority prefers the target system. Machine win rate (MWR) is the fraction of items for which the reward predictor prefers the target system. Agreement is the fraction of items for which the machine winner and the human-majority winner are identical. Unless otherwise stated, HWR, MWR, and agreement are item-level metrics: when individual listener votes are analyzed, we explicitly label the metric as Vote HWR. We report item-level majority results as the primary human-evaluation metric, with Wilson 95% confidence intervals in Appendix E.

## 5 Machine-Level Behavior and Baselines

## 5.1 Cross-axis specificity and machine-level baselines

We first examine whether subjective rewards induce targeted machine-level shifts, whether the CERzone scaffold improves the reward–intelligibility trade-off, and whether policy optimization provides benefits beyond inference-time reranking.

Table 1 evaluates each single-reward GRPO run on all reward axes. The largest positive shift appears on the optimized axis in every row, indicating that subjective rewards mainly induce axis-specific movement rather than a generic quality improvement. This diagonal pattern shows that no single subjective reward is a universal surrogate for all desired speech properties.

## 5.2 Policy optimization versus Best-of-N reranking

We then compare three ways of using the same reward signal: Best-of-N does not update the policy: it samples N candidates from the base model, scores them with the CER-gated reward, and serves the highest-scoring candidate. Targetonly and Zone-CER are policy-trained systems. Target-only optimizes the perceptual predictor without the CER-zone constraint, while Zone-CER uses the full reward in Eq. (1).

Table 2 reports method-specific machine behavior: Base, Target-only, and Zone-CER are evaluated from one policy sample, while Best-of-N selects the highest-scoring candidate among N base samples. The Best-of-N rows show that rewardselected samples already exist in the base model’s support. On ANIMESCORE, Zone-CER produces a larger target shift and lower median CER than Best-of-8, but Best-of-8 has fewer violations. On UTMOS, Best-of-8 and Zone-CER achieve almost identical target gains; Best-of-8 has lower median CER, while Zone-CER has fewer violations. On LIKABILITY, Zone-CER gives the largest target gain, but the gain is small and comes with higher median CER than Best-of-8, although with fewer violations. Thus, GRPO is not uniformly better than reranking; the value of policy optimization depends on the reward axis and deployment trade-off. The Target-only rows additionally isolate the role of the CER-zone constraint: removing the CER zone worsens median CER and violation rate, while Zone-CER preserves or improves the target gain relative to Target-only with the largest differences on ANIMESCORE and UTMOS.

<table><tr><td>Axis</td><td>Method</td><td></td><td>∆Target CER med. Viol.(%)</td><td></td></tr><tr><td rowspan="5">ANIMESCORE Base</td><td></td><td>0.00</td><td>0.058</td><td>24.0</td></tr><tr><td>Best-of-4</td><td>+0.83</td><td>0.064</td><td>10.0</td></tr><tr><td>Best-of-8</td><td>+1.21</td><td>0.070</td><td>10.0</td></tr><tr><td>Target-only</td><td>+1.08</td><td>0.087</td><td>24.0</td></tr><tr><td>Zone-CER</td><td>+1.35</td><td>0.054</td><td>16.0</td></tr><tr><td rowspan="5">UTMOS</td><td>Base</td><td>0.00</td><td>0.058</td><td>24.0</td></tr><tr><td>Best-of-4</td><td>+0.32</td><td>0.052</td><td>10.0</td></tr><tr><td>Best-of-8</td><td>+0.47</td><td>0.055</td><td>10.0</td></tr><tr><td>Target-only</td><td>+0.25</td><td>0.154</td><td>14.0</td></tr><tr><td>Zone-CER</td><td>+0.49</td><td>0.074</td><td>6.0</td></tr><tr><td rowspan="5">LIKABILITY</td><td>Base</td><td>0.00</td><td>0.058</td><td>24.0</td></tr><tr><td>Best-of-4</td><td>+0.12</td><td>0.046</td><td>10.0</td></tr><tr><td>Best-of-8</td><td>+0.14</td><td>0.043</td><td>10.0</td></tr><tr><td>Target-only</td><td>+0.15</td><td>0.119</td><td>16.0</td></tr><tr><td>Zone-CER</td><td>+0.17</td><td>0.098</td><td>6.0</td></tr></table>

Table 2: Machine-level comparison across reward axes. ∆Target denotes mean objective shifts depending on the axis (∆AS, ∆UTMOS, or ∆Likab.). Base, Target-only, and Zone-CER use first-shot sample; Best-of-N selects the highest-scoring candidate among N base samples.
<table><tr><td colspan="4">Axis HWR MWR Agree ∆Target</td></tr><tr><td colspan="4">GRPO vs Base</td></tr><tr><td>ANIMESCORE</td><td>80.0 88.0</td><td>88.0</td><td>+1.24 +0.46</td></tr><tr><td>UTMOS</td><td>62.0 74.0 36.0</td><td>80.0</td><td>+0.17</td></tr><tr><td>LIKABILITY</td><td>56.0</td><td>76.0</td><td></td></tr><tr><td colspan="4">GRPO vs Best-of-8</td></tr><tr><td>ANIMESCORE</td><td>52.0</td><td>50.0 74.0</td><td>+0.07</td></tr><tr><td>UTMOS</td><td>46.0 36.0</td><td>68.0</td><td>-0.01</td></tr><tr><td>LIKABILITY</td><td>48.0 32.0</td><td>62.0</td><td>+0.03</td></tr></table>

Table 3: Human alignment results for GRPO against Base and Best-of-8. All HWR, MWR, and Agree values are item-level percentages over 50 paired items using majority vote from 5 raters per item. ∆Target is computed as the mean GRPO target score minus the comparison system’s target score.

## 6 Human Evaluation

## 6.1 Axis-level transfer

Table 3 shows that predictor-level gains transfer unevenly to listeners. ANIMESCORE is the strongest positive case: the listener majority prefers the GRPO output on most items, and item-level machine–human agreement reaches 88%. UT-MOS shows high machine–human agreement, but only a modest aggregate human preference shift. LIKABILITY is the main negative average-transfer case under this predictor–axis–base tuple: humans prefer the base overall. However, the rewardgap analysis below shows that this average failure masks a calibrated high-confidence region. These results separate two notions of transfer. Axislevel transfer asks whether the optimized system is preferred on average, while reward-gap calibration asks whether score differences explain itemlevel listener choices. Under this distinction, ANI-MESCORE is the clearest average-transfer success, whereas LIKABILITY and UTMOS provide the clearest evidence of within-axis reward-gap calibration.

## 6.2 Comparison between GRPO and Best-of-N in human preference

Table 3 also compares GRPO with Best-of-8 on each reward axis. Across all three axes, human preference is near chance: HWR is 52.0 for ANI-MESCORE, 46.0 for UTMOS, and 48.0 for LIKA-BILITY. Thus, listeners do not clearly prefer GRPO over Best-of-8. GRPO should therefore be interpreted as policy-level movement that amortizes reward-selected behavior rather than as a perceptual improvement over a strong Best-of-8 baseline.

## 6.3 Reward-gap calibration

Average win rates can hide whether a reward difference is perceptually meaningful. We therefore test whether signed pairwise reward gaps predict listener choices beyond residual intelligibility differences. Specifically, we pool the three main Japanese axes and the auxiliary English ANI-MESCORE study, and fit a vote-level logistic regression predicting whether an individual listener chooses the GRPO side from the within-axis standardized signed reward gap $z _ { R }$ and the within-axis standardized CER gap $z _ { C }$ , with axis fixed effects and item-clustered robust standard errors. $z _ { R }$ is positive when the target reward favors GRPO, and $z _ { C }$ is positive when the GRPO side has lower CER.

The signed reward gap is a strong predictor of human choice: a one-standard-deviation increase in reward gap in favor of GRPO increases the odds of choosing GRPO by 1.93×. By contrast, the CER gap is not predictive, and a Wald test rejects equality of the two slopes. Thus, within the CERretry evaluation regime, GRPO-side preference is better explained by signed reward advantage than by residual CER advantage.

Per-axis checks in Appendix F reveal heterogeneity. LIKABILITY and UTMOS have significant reward-gap slopes, while the ANIMESCORE slopes are positive but not significant. Reward-gap calibration should therefore be read as a local confidence diagnostic within a predictor–axis–base tuple, not as the sole explanation of average transfer. Full bin

<table><tr><td>Variable</td><td>β(SE) OR</td><td>95% CI</td><td>p</td></tr><tr><td>Reward gap zR</td><td>+0.657 (0.172) 1.93</td><td>[1.38, 2.70]</td><td>&lt; .001</td></tr><tr><td>CER gap zC</td><td>-0.041 (0.186) 0.96</td><td>[0.67, 1.38]</td><td>.83</td></tr></table>

Table 4: Vote-level logistic regression predicting whether a listener chooses the GRPO side, using axis fixed effects and item-clustered robust standard errors. $z _ { R }$ is the within-axis standardized signed reward gap in favor of GRPO, and $z _ { C }$ is the within-axis standardized CER advantage of GRPO.
<table><tr><td>Axis</td><td>Mean</td><td>Std</td><td>Range</td></tr><tr><td>ANIMESCORE</td><td>-0.39</td><td>1.53</td><td>[−2.52, 4.20]</td></tr><tr><td>UTMOS</td><td>+3.08</td><td>1.01</td><td>[1.30, 4.23]</td></tr><tr><td>LIKABILITY</td><td>+4.200.44</td><td></td><td>[2.78, 4.58]</td></tr></table>

Table 5: Base predictor distributions on base model with test dataset.

## statistics are reported in Appendix G. 6.4 Robustness to residual CER violations

Although the CER-retry protocol reduces transcript failures, a small number of item pairs remain above the CER threshold after all retries. We retain all 50 items to avoid post-hoc filtering by a metric tied to the RL objective. Excluding residual violator pairs changes vote-level HWR by +1.9 pp for ANI-MESCORE, −2.1 pp for UTMOS, and −4.5 pp for LIKABILITY; the qualitative conclusions remain unchanged.

## 7 Discussion: Diagnostics for Predictor–Axis–Base Tuples

The human study shows that increasing a subjective predictor score is not sufficient for human-aligned transfer. This is consistent with the broader RLHF observation that learned rewards are useful but imperfect proxies, and that optimizing them can diverge from the intended human objective (Ziegler et al., 2019; Gao et al., 2023). In speech, this proxy gap is shaped not only by the reward model, but also by the perceptual axis and the base model distribution. We therefore analyze each setting as a predictor–axis–base tuple. The diagnostics below are evidence-supported heuristics, not causal explanations or universal rules for reward success.

## 7.1 Reward-gap calibration

The strongest diagnostic observed in our study is reward-gap calibration, but it should be distinguished from average transfer. Instead of asking only whether the optimized system is preferred on average, reward-gap calibration asks whether larger signed reward advantages correspond to listener choices within a given predictor–axis–base tuple. The regression analysis in §6.3 supports this view at the pooled level: standardized signed reward gaps predict GRPO-side listener preference, whereas residual CER gaps do not. However, the per-axis fits in Appendix F show heterogeneity: LIKABILITY and UTMOS have significant rewardgap slopes, while the ANIMESCORE slopes are positive but not significant. This distinction explains why ANIMESCORE can be the strongest averagetransfer case even though its within-axis rewardgap slope is not significant. The optimization appears to move the distribution in a perceptually salient direction, while the magnitude of item-level reward gaps is less informative once many items already favor GRPO.

This diagnostic is especially useful for LIKA-BILITY. Although LIKABILITY fails on average, its per-axis reward-gap slope is significant, indicating that the predictor is informative in highconfidence regions. The negative average result therefore should not be read as evidence that likability is intrinsically difficult to improve. Rather, under this predictor–axis–base tuple, few items reach a reward-gap regime that is perceptually reliable.

## 7.2 Base-output spread

A second screening signal is the predictor’s score spread on base-model outputs. If a predictor assigns nearly identical scores to naturally occurring base outputs, this may indicate limited resolution on the target base distribution. Table 5 is consistent with part of this pattern: ANIMESCORE has the widest spread and gives the strongest positive transfer, while LIKABILITY has narrower spread and fails on average. The VAD-AROUSAL failure is discussed separately in Appendix I.

## 7.3 Within-zone signal under constraints

The VAD-AROUSAL run illustrates a constraintspecific failure mode. A predictor can be meaningful as a standalone evaluator but still fail as an RL reward if its variation inside the feasible CER zone is too small relative to the constraint penalty. In our VAD-AROUSAL run, validation reward improved mainly by reducing CER violations, while validation arousal stayed within seed-level variation. Because no human A/B study was conducted, we treat this as a training-dynamics negative case rather than evidence about perceptual transfer.

This result is consistent with the constrained-RL view that hard requirements should be separated from optimizable preferences (Achiam et al., 2017). The CER-zone scaffold prevents high perceptual scores from compensating for severe transcript drift, but it also requires the perceptual predictor to provide enough within-zone signal to affect GRPO ranking. Appendix I reports the retained training evidence and logging limitations.

Auxiliary cross-domain and cross-base checks. Appendix H reports an auxiliary English ANI-MESCORE study. It suggests that cross-language use of a style reward can be informative, but the study uses the same Japanese-native listener pool and is not treated as primary evidence for crosslingual human transfer.

Practical screening heuristic. Before scaling RL with a subjective speech reward, our results suggest four checks: measure reward-gap calibration with a small A/B study when possible; measure predictor spread on base-model outputs; check domain coverage or validate cross-domain use with listeners; and inspect whether the predictor has enough within-zone variation relative to constraint penalties. These checks do not prove causality, but they can identify rewards likely to produce lowconfidence or non-transferable policy movement.

## Conclusion

We studied when RL from learned subjective predictors transfers from machine-score gains to human-perceptual gains in codec-based speech language models. Our results show that predictor gains alone are not sufficient: average transfer and within-axis reward-gap calibration can diverge, and both must be considered alongside predictor resolution on the target base distribution, domain validation, and sufficient within-zone signal under intelligibility constraints. AnimeScore gives a strong in-domain positive case, UTMOS shows high machine–human agreement with only modest average preference shift, Likability fails on average but aligns in high-confidence regions, and VAD-Arousal fails as a training-time reward under our constrained scaffold. Best-of-N reranking further shows that reward-selected samples often already exist in the base model’s support; GRPO should therefore be viewed as an attempt to amortize such selection into policy-level movement, not as a uniform perceptual improvement over reranking. These findings suggest practical screening heuristics for subjective speech rewards before full RL or multi-reward post-training.

## Limitations

Our experiments are designed to compare subjective reward behavior under controlled conditions, but they do not fully disentangle reward-model architecture, perceptual axis, and base-model distribution. The reward models differ in training data, score scale, target construct, and base-model coverage; therefore, our results should be read as diagnostics for predictor–axis–base tuples rather than causal claims that one perceptual axis is intrinsically easier or harder than another.

Our evaluation is also limited by practical compute and annotation budgets. We evaluate a single primary decoding configuration, a fixed set of reward axes, Best-of-8 as the main reranking baseline, and a limited number of human listeners per item. Larger studies could add more base models, broader listener populations, native English listeners for the auxiliary English condition, larger reranking budgets, fixed-KL comparisons, multiseed training, and direct within-prompt rolloutspread measurements. These extensions would strengthen the generality of the proposed diagnostics, especially for future multi-reward speech posttraining.

## Ethical Considerations

This work studies post-training methods for controllable synthetic speech. Such methods can support creative and accessibility-oriented applications, but they can also lower the cost of generating speech in a target style without consent. We therefore frame our study around evaluation and diagnostics rather than deployment, and we release only artifacts intended for research use: code, prompts, generated audio samples, and reward scores. We do not redistribute merged base-model weights, reward model weights and or training corpora whose licenses or copyright status do not permit redistribution.

The ANIMESCORE reward model targets a stylistic dimension rather than speaker identity, but style and identity can interact in downstream use. Released materials therefore include responsibleuse guidance and are not intended for impersonation, voice cloning, or unauthorized style imitation. More broadly, subjective speech rewards should be developed with attention to fairness, speaker consent, dataset provenance, and copyright compliance. Future work should further examine how style-control rewards behave across demographic groups, listener communities, and culturally specific notions of expressiveness or likability.

## Use of AI Assistance

The authors used AI assistants for language polishing, LaTeX editing assistance, and brainstorming presentation of results. All scientific claims, experiments, analyses, and final manuscript content were verified and revised by the authors.

## References

Joshua Achiam, David Held, Aviv Tamar, and Pieter Abbeel. 2017. Constrained policy optimization. In Proceedings ofthe 34th International Conference on Machine Learning, pages 22–31.

Zalán Borsos, Raphaël Marinier, Damien Vincent, Eugene Kharitonov, Olivier Pietquin, Matt Sharifi, Dominik Roblek, Olivier Teboul, David Grangier, Marco Tagliasacchi, and Neil Zeghidour. 2023. AudioLM: A language modeling approach to audio generation. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 31:2523–2533.

Jingyi Chen, Ju-Seung Byun, Micha Elsner, and Andrew Perrault. 2024. DLPO: Diffusion model loss-guided reinforcement learning for fine-tuning TTS diffusion models. arXiv preprint arXiv:2405.14632.

Sanyuan Chen, Shujie Liu, Long Zhou, Yanqing Liu, Xu Tan, Jinyu Li, Sheng Zhao, Yao Qian, and Furu Wei. 2025. VALL-E 2: Neural codec language models are human parity zero-shot text-to-speech synthesizers. In International Conference on Learning Representations.

Erica Cooper, Wen-Chin Huang, Tomoki Toda, and Junichi Yamagishi. 2022. Generalization ability of MOS prediction networks. In IEEE International Conference on Acoustics, Speech and Signal Processing, pages 8442–8446.

Alexandre Défossez, Jade Copet, Gabriel Synnaeve, and Yossi Adi. 2023. High fidelity neural audio compression. arXiv preprint arXiv:2210.13438.

Zhihao Du, Yujie Wang, Qian Chen, Han Yang, Zhifu Wang, Heng Lu, Lina Tan, Wen Wang, and Yong Zhang. 2024. CosyVoice: A scalable multilingual zero-shot text-to-speech synthesizer based on supervised semantic tokens. arXiv preprint arXiv:2407.05407.

Changfeng Gao, Zhihao Du, and Shiliang Zhang. 2025. DiffRO: Differentiable reward optimization for LLM based TTS system. arXiv preprint arXiv:2507.05911.

Leo Gao, John Schulman, and Jacob Hilton. 2023. Scaling laws for reward model overoptimization. In Proceedings of the 40th International Conference on Machine Learning.

Haorui He, Zengqiang Shang, Chaoren Wang, Xuyuan Li, Yicheng Gu, Hua Hua, Liwei Liu, Chen Yang, Jiaqi Li, Peiyang Shi, Yuancheng Wang, Kai Chen, Pengyuan Zhang, and Zhizheng Wu. 2024. Emilia: An extensive, multilingual, and diverse speech dataset for large-scale speech generation. arXiv preprint arXiv:2407.05361.

Yinghao Aaron Li, Xilin Jiang, Fei Tao, Cheng Niu, Kaifeng Xu, Juntong Song, and Nima Mesgarani. 2026. DMOSpeech 2: Reinforcement learning for duration prediction in metric-optimized speech synthesis. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 31814–31822.

Chang Liu, Ya-Jun Hu, Ying-Ying Gao, Shi-Lei Zhang, and Zhen-Hua Ling. 2025. Group relative policy optimization for text-to-speech with large language models. arXiv preprint arXiv:2509.18798.

Chen-Chou Lo, Szu-Wei Fu, Wen-Chin Huang, Xin Wang, Junichi Yamagishi, Yu Tsao, and Hsin-Min Wang. 2019. MOSNet: Deep learning-based objective assessment for voice conversion. In Proceedings ofInterspeech 2019, pages 1541–1545.

Gabriel Mittag, Babak Naderi, Assmaa Chehadi, and Sebastian Möller. 2021. NISQA: A deep CNN-selfattention model for multidimensional speech quality prediction with crowdsourced datasets. In Proceedings ofInterspeech 2021, pages 2127–2131.

Joonyong Park and Jerry Li. 2026. AnimeScore: A preference-based dataset and framework for evaluating anime-like speech style. arXiv preprint arXiv:2603.11482.

Vineel Pratap, Qiantong Xu, Anuroop Sriram, Gabriel Synnaeve, and Ronan Collobert. 2020. MLS: A largescale multilingual dataset for speech research. arXiv preprint arXiv:2012.03411.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. 2023. Robust speech recognition via large-scale weak supervision. In Proceedings of the 40th International Conference on Machine Learning.

Takaaki Saeki, Detai Xin, Wataru Nakata, Tomoki Koriyama, Shinnosuke Takamichi, and Hiroshi Saruwatari. 2022. UTMOS: UTokyo-SaruLab system for VoiceMOS challenge 2022. In Proceedings ofInterspeech 2022, pages 4521–4525.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, and 1 others. 2024. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. 2025. HybridFlow: A flexible and efficient RLHF framework. In Proceedings of the Twentieth European Conference on Computer Systems, pages 1279–1297. Association for Computing Machinery.

Hitoshi Suda, Aya Watanabe, and Shinnosuke Takamichi. 2024. Who finds this voice attractive? a large-scale experiment using in-the-wild data. arXiv preprint arXiv:2407.04270.

Johannes Wagner, Andreas Triantafyllopoulos, Hagen Wierstorf, Maximilian Schmitt, Felix Burkhardt, Florian Eyben, and Björn W. Schuller. 2023. Dawn of the transformer era in speech emotion recognition: Closing the valence gap. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(9):10745– 10759.

Chengyi Wang, Sanyuan Chen, Yu Wu, Ziqiang Zhang, Long Zhou, and 1 others. 2023. VALL-E: Neural codec language models are zero-shot text to speech synthesizers. arXiv preprint arXiv:2301.02111.

Zhen Ye, Peiwen Sun, Jiahe Lei, Hongzhan Lin, Xu Tan, and 1 others. 2025a. Codec does matter: Exploring the semantic shortcoming of codec for audio language model. In Proceedings ofthe AAAI Conference on Artificial Intelligence.

Zhen Ye, Xinfa Zhu, Chi-Min Chan, Xinsheng Wang, Xu Tan, and 1 others. 2025b. LLaSA: Scaling traintime and inference-time compute for LLaMA-based speech synthesis. arXiv preprint arXiv:2502.04128.

Neil Zeghidour, Alejandro Luebs, Ahmed Omran, Jan Skoglund, and Marco Tagliasacchi. 2022. SoundStream: An end-to-end neural audio codec. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 30:495–507.

Dong Zhang, Zhaowei Li, Shimin Li, Xin Zhang, Pengyu Wang, Yaqian Zhou, and Xipeng Qiu. 2024. SpeechAlign: Aligning speech generation to human preferences. In Advances in Neural Information Processing Systems, volume 37.

Yicheng Zhong, Peiji Yang, and Zhisheng Wang. 2025. Multi-reward GRPO for stable and prosodic singlecodebook TTS LLMs at scale. arXiv preprint arXiv:2511.21270.

Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B. Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. 2019. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593.

## A Training Hyperparameters and Checkpoint Selection

Table 6 reports the common GRPO hyperparameters shared by the reward-axis runs and the per-axis selected checkpoints. All training is done with verl (Sheng et al., 2025) on a single H100 80 GB GPU with vLLM-backed rollouts. The reward predictors run on a second H100 to isolate codec decoding and predictor inference from the rollout engine.

<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Framework / rollout</td><td>verl /vLLM (gpu_memory_utilization=0.6)</td></tr><tr><td>Rollouts per prompt K</td><td>4</td></tr><tr><td>Train batch size</td><td>16 (PPO mini-batch 16, micro-batch 4 per GPU)</td></tr><tr><td>Actor optimizer / LR</td><td>AdamW /5×10−7</td></tr><tr><td>PPO clip ratio</td><td>0.1</td></tr><tr><td>Entropy coefficient</td><td>0</td></tr><tr><td>Top-p / Temperature / Rep. penalty</td><td>0.85 / 1.0 / 1.05</td></tr><tr><td>Max prompt / response length</td><td>512 / 2048 codec tokens</td></tr><tr><td>Speech-only token mask</td><td>enabled</td></tr><tr><td>KL mode</td><td>in-reward (use_k1_loss=False)</td></tr><tr><td>Adaptive KL controller</td><td>init β = 0.05, target KL = 0.05, horizon = 2000</td></tr><tr><td>save_freq/test_freq</td><td>30 / 30 steps</td></tr><tr><td>Hardware</td><td>1 × H100 80 GB for actor + 1 × H100 80 GB for reward predictors</td></tr><tr><td>Wall time per step</td><td>~115 s with Whisper in reward, ~43 s without Whisper</td></tr><tr><td>JP validation set</td><td>Validation and evaluation sets n = 100</td></tr><tr><td>EN validation set</td><td>n = 100</td></tr><tr><td>Paper evaluation set</td><td></td></tr><tr><td></td><td>n = 50 per language</td></tr></table>

Table 6: GRPO training hyperparameters.

## B Reward Predictor Implementation

This appendix documents the reward predictors used in the paper: backbone, checkpoint, training data, output scale, and release plan. Each predictor receives generated XCodec2 tokens, decodes them to a 16 kHz mono waveform, and returns a scalar score.

ANIMESCORE. ANIMESCORE is a pairwisepreference-trained anime-likeness predictor (Park and Li, 2026). It uses a microsoft/wavlm-base SSL encoder with the CNN frozen and the transformer fine-tuned, a learned mixture over the last four hidden layers, a BiLSTM, and a two-layer MLP head. The model is trained with a RankNetstyle pairwise ranking loss on the AnimeScore preference corpus and follows the SSL-MOS finetuning recipe of Cooper et al. (2022). The output is an unbounded scalar; its empirical range on basemodel outputs is approximately [−3, +5]. We release the checkpoint with the artifact bundle where licensing permits.

UTMOS. UTMOS is used off-the-shelf from UTMOS22-strong (Saeki et al., 2022). No finetuning is performed by us. The output is a naturalness MOS-like score in [1, 5], and we divide it by 5 before applying the CER-zone reward.

LIKABILITY. LIKABILITY is trained for this study on the CocoNut-Humoresque likability dataset (Suda et al., 2024). The model uses microsoft/wavlm-base-plus with the CNN frozen and the transformer fine-tuned, mean pooling, a Linear(768,6) classification head, and a softmax expectation over six discrete classes. The output is the expected class value on [1, 6]. We use the original CocoNut-Humoresque split and train with inverse-frequency class-weighted crossentropy rather than regression to preserve score spread for GRPO. Training uses AdamW with learning rate 10<sup>−5</sup> and weight decay 0.01, linear warmup followed by cosine decay, Gaussian noise augmentation at SNR 30–50 dB, and random gain perturbation of ±3 dB. The checkpoint is selected by validation SRCC. On the Humoresque test set (n = 283), the model obtains SRCC 0.712, LCC 0.703, MSE 0.274, and prediction range 2.71. The classification formulation is intentional: regression counterparts mean-collapse, compressing the prediction range and producing a flatter reward landscape for GRPO. We release the checkpoint with the artifact bundle, subject to the Humoresque licensing terms.

VAD-AROUSAL. VAD-AROUSAL is the offthe-shelf MSP-Dim valence-arousal-dominance predictor (Wagner et al., 2023). It uses a wav2vec2-large-robust backbone with a regression head emitting valence, arousal, and dominance. We use only the arousal output and clip it to [0, 1]. No fine-tuning is performed by us.

Whisper / CER. For the CER gate, we use Whisper large-v3 (Radford et al., 2023). Language detection is automatic. CER is computed against the canonical written prompt, with no additional Japanese- or English-specific text normalization. Thus, minor whitespace and punctuation differences can propagate into CER; this conservative choice is part of the intelligibility signal controlled by the CER-zone reward.

<table><tr><td>Group</td><td>Description</td><td>n</td></tr><tr><td>G1</td><td>emotional expression</td><td>18</td></tr><tr><td>G2</td><td>anime-stylized text</td><td>14</td></tr><tr><td>G3</td><td>neutral conversational</td><td>10</td></tr><tr><td>G4</td><td>long-form narrative</td><td>3</td></tr><tr><td>G5</td><td>linguistically challenging</td><td>5</td></tr><tr><td>Total</td><td></td><td>50</td></tr></table>

Table 7: Composition of test dataset.

<table><tr><td>Axis</td><td>base resid. GRPO resid. union resid.</td></tr><tr><td>ANIMESCORE JP 4/50</td><td>3/50 4/50</td></tr><tr><td>ANIMESCORE EN 1/50</td><td>1/50 1/50</td></tr><tr><td>3/50</td><td>3/50 4/50</td></tr><tr><td>UTMOS LIKABILITY 6/50</td><td>3/50 7/50</td></tr></table>

Table 8: Residual CER > 0.30 after the CER-retry pass. The union residual count is the number of item pairs where either side remains above the CER threshold.

Reward functions. The token-to-reward mapping is implemented separately for each axis: ANIMESCORE+CER, UTMOS+CER, LIKABIL-ITY+CER, VAD-AROUSAL+CER, and AS-only.

## C Evaluation Set

Table 7 summarizes the held-out prompt set used for machine evaluation and human listening tests.

Test dataset is held out from reward-model training, GRPO training, and checkpoint-selection validation.

Residual violations after retry. Table 8 reports how often either side remains above $\tau _ { h } = 0 . 3 0$ after the CER-retry pass. The union column gives the number of item pairs excluded in the clean-only robustness check. Here the target side is the GRPO output in the GRPO-vs-base comparison.

Vote-level robustness. Table 9 reports vote-level Human-WR and agreement before and after excluding item pairs where either side remains above $\tau _ { h }$ after retry. The “all” rows match the vote-level summaries in Appendix E.

Logistic check. The reward-gap regression in $\mathsf { A p - }$ pendix F further shows that the CER-gap slope is statistically indistinguishable from zero, while the reward-gap slope is significant. Together with Table 9, this indicates that residual CER differences are unlikely to be the primary driver of the main human-evaluation conclusions.

## D Human Recruitment

Collecting human-evaluation data. Humanevaluation votes are stored with randomized item identifiers and anonymized listener identifiers. We do not release platform-specific worker identifiers or personally identifying information. We collected basic self-reported metadata, including demographics, for quality control and aggregate analysis.

Participant recruitment and compensation. Listeners were recruited through Lancers, a Japanese crowdsourcing platform. Each listening session contained 25 A/B pairs, took approximately 10 minutes, and was compensated at 200 JPY per completed session. Before starting, workers were shown task instructions explaining the target comparison question, playback requirement, skip conditions, and research use of the responses. The human-evaluation interface included a pre-task instruction and consent page and a pairwise A/B listening page, shown in Figure 2.

## E Human Evaluation Statistics

This appendix provides confidence intervals and vote-level summaries for the human-evaluation results. Table 10 reports item-level Wilson intervals over the 50 majority-vote items for GRPO vs Base. Table 11 reports the corresponding item-level intervals for GRPO vs Best-of-8. Table 16 reports vote-level summaries with item-clustered bootstrap intervals. The English ANIMESCORE row is auxiliary and is discussed separately in Appendix H.

## F Reward-Gap Logistic Regression

Outcome variable. Let $y _ { i j } \in \{ 0 , 1 \}$ be listener i’s preference on item j. We set $y _ { i j } = 1$ if the listener chooses the target side, i.e., the GRPO output in the GRPO-vs-base comparisons. The signed reward gap $\Delta r = r _ { \mathrm { t a r g e t } } - r _ { \mathrm { b a s e } }$ indicates how strongly the target reward favors the GRPO side.

Predictors. z is the within-axis z-score of $\Delta r . \textit { \textbf { z } }$ is the within-axis z-score of ∆CER = $\mathrm { C E R } _ { \mathrm { b a s e } } - \mathrm { C E R } _ { \mathrm { t a r g e t } }$ , so positive $z _ { C }$ indicates that the target side is more intelligible.

Data. The regression uses 1000 vote-level observations from 200 item pairs across four axes: ANIMESCORE EN, ANIMESCORE JP, LIKABIL-ITY, and UTMOS. Each axis contains 50 item pairs with 5 listener votes per item. Rows produced only for visualization are not used in this regression.

<table><tr><td>Axis</td><td>excl. pairs HWR all</td><td></td><td>HWR clean</td><td></td><td></td><td>∆HWR Agree all Agree clean</td><td>∆Agr</td></tr><tr><td>ANIMESCORE JP</td><td>4</td><td>72.0</td><td></td><td>73.9 +1.9 pp</td><td>78.4</td><td></td><td> $8 0 . 9 \ \pm 2 . 5 \mathrm { p p }$ </td></tr><tr><td>ANIMESCORE EN</td><td>1</td><td>62.8</td><td></td><td>62.9 +0.1 pp</td><td>68.8</td><td></td><td> $6 9 . 4 \ \mathrm { \ t 0 . 6 p p }$ </td></tr><tr><td>UTMOS</td><td>4</td><td>56.4</td><td>54.3</td><td>−2.1 pp</td><td>72.0</td><td></td><td> $7 1 . 3 - 0 . 7 \mathrm { p p }$ </td></tr><tr><td>LIKABILITY</td><td>7</td><td>38.0</td><td></td><td>33.5 -4.5 pp</td><td>67.6</td><td></td><td>66.5 −1.1 pp</td></tr></table>

Table 9: Vote-level Human-WR and machine–human agreement before and after excluding residual CER-violator item pairs. Clean-only percentages are computed after removing all five votes for each excluded item pair. The clean-only analysis changes the exact percentages but does not change the qualitative conclusions: ANIMESCORE remains positive, UTMOS remains modestly positive, and LIKABILITY remains the negative average-transfer case.

![](images/c827bc5077a723955454d2cda0c39c3ed5e2a961f2a9ffc1552e20b4a3874b70.jpg)  
(a) Pre-task instruction and consent page.

![](images/c02ec4811c7cca6dea37159973af4251775efe9454ee9753824d7fa809b0601d.jpg)  
(b) Pairwise A/B listening interface.  
Figure 2: Screenshots of the human-evaluation interface. Before starting, participants read the task description, consent notice, and demographic questions. During evaluation, they listened to both clips in each pair and selected the side that better matched the target perceptual criterion.

Primary fit. We fit a logistic regression with axis fixed effects and item-clustered robust standard errors. Table 12 reports the primary pooled fit. The reference axis is ANIMESCORE EN.

Equality test and likelihood-ratio tests. A Wald test rejects equality of slopes: $\beta _ { R } - \beta _ { C } = + 0 . 6 9 8 ,$ $\mathrm { S E } ~ = ~ 0 . 2 6 0 , z ~ = ~ + 2 . 6 8$ , and $p = 7 . 4 \times 1 0 ^ { - 3 }$ Adding z to a model with axis fixed effects and z<sub>C</sub> improves fit by $\chi _ { 1 } ^ { 2 } = 1 5 . 8 2$ with $p = 7 \times 1 0 ^ { - 5 }$ Adding z<sub>C</sub> to a model with axis fixed effects and z<sub>R</sub> does not improve fit, with $\chi _ { 1 } ^ { 2 } = 0 . 0 6 \mathrm { a n d } p = 0 . 8 0$ Per-axis fits. Table 17 reports descriptive peraxis logistic fits.

Multicollinearity. The pooled Pearson correlation between z<sub>R</sub> and z<sub>C</sub> is r = +0.16. Within axis, the correlations are +0.16 for ANIMESCORE EN, +0.13 for ANIMESCORE JP, +0.40 for LIKABIL-ITY, and −0.05 for UTMOS. The two predictors are therefore weakly correlated, so the joint-model coefficients are interpretable as partial effects.

Sensitivity to EN inclusion. Restricting the regression to the three JP axes preserves the qualitative pattern that reward-gap slopes are positive. We report the four-axis pooled regression in the main text because it has the broadest coverage and because the EN row is part of the auxiliary evidence for cross-language reward transfer. The main conclusion remains that standardized reward gaps predict human preference more strongly than residual CER gaps.

## G Reward-Gap Binned Analysis

This appendix gives a descriptive reward-gap bin analysis underlying the main-text calibration discussion. The formal test is the signed-gap logistic regression in Appendix F; the binned analysis is intended only to visualize low- and high-gap regimes. Because the bins are based on absolute reward-gap magnitude, the agreement column is the more direct direction-invariant summary, while the GRPO preference column shows how often listeners choose the optimized output within each bin. Table 18 reports the resulting vote-level bins.

## H English AnimeScore Evaluation

We replicate the JP ANIMESCORE experiment with the same checkpoint on a 50-prompt English test dataset. The EN prompt set is not a translation of the JP prompts; it is independently authored English text with the same five-group structure. Both the EN training set and held-out EN test dataset are disjoint from the JP corpora.

<table><tr><td>Axis</td><td>HWR</td><td>HWR CI</td><td>MWR</td><td>MWR CI</td><td>Agree</td><td>Agree CI</td></tr><tr><td>ANIMESCORE JP</td><td>80.0</td><td>[67.0,88.8]</td><td>88.0</td><td>[76.2,94.4]</td><td>88.0</td><td>[76.2,94.4]</td></tr><tr><td>UTMOS</td><td>62.0</td><td>[48.2,74.1]</td><td>74.0</td><td>[60.4,84.1]</td><td>80.0</td><td>[67.0,88.8]</td></tr><tr><td>LIKABILITY</td><td>36.0</td><td>[24.1,49.9]</td><td>56.0</td><td>[42.3,68.8]</td><td>76.0</td><td>[62.6,85.7]</td></tr><tr><td>ANIMESCORE EN</td><td>70.0</td><td>[56.2,80.9]</td><td>66.0</td><td>[52.2,77.6]</td><td>72.0</td><td>[58.3,82.5]</td></tr></table>

Table 10: Item-level Wilson 95% confidence intervals over 50 items for GRPO vs Base. HWR, MWR, and agreement are item-level percentages computed from 5-rater majority votes.
<table><tr><td>Axis</td><td>HWR</td><td>HWR CI</td><td>MWR</td><td>MWR CI</td><td>Agree</td><td>Agree CI</td></tr><tr><td>ANIMESCORE</td><td>52.0</td><td>[38.5,65.2]</td><td>50.0</td><td>[36.6,63.4]</td><td>74.0</td><td>[60.4,84.1]</td></tr><tr><td>UTMOS</td><td>46.0</td><td>[33.0,59.6]</td><td>36.0</td><td>[24.1,49.9]</td><td>68.0</td><td>[54.2,79.2]</td></tr><tr><td>LIKABILITY</td><td>48.0</td><td>[34.8,61.5]</td><td>32.0</td><td>[20.8,45.8]</td><td></td><td>62.0 [48.2,74.1]</td></tr></table>

Table 11: Item-level Wilson 95% confidence intervals for GRPO vs Best-of-8 over 50 paired items. HWR, MWR, and agreement are item-level percentages computed from 5-rater majority votes.

<table><tr><td>Term</td><td> $\hat { \beta }$ </td><td>SE z</td><td>p</td></tr><tr><td>Intercept</td><td>+0.642</td><td>0.316  $+ 2 . 0 4$ </td><td>0.042</td></tr><tr><td>ANIMESCORE JP</td><td>+0.389</td><td>0.473  $+ 0 . 8 2$ </td><td>0.411</td></tr><tr><td>LIKABILITY</td><td>-1.147</td><td>0.432 -2.65</td><td>0.008</td></tr><tr><td>UTMOS</td><td>-0.273</td><td>0.434 -0.63</td><td>0.530</td></tr><tr><td>ZR</td><td>+0.657</td><td>0.172 +3.82</td><td> $\mathbf { 1 . 4 \times 1 0 ^ { - 4 } }$ </td></tr><tr><td>zC</td><td>-0.041</td><td>0.186 -0.22</td><td>0.827</td></tr></table>

Table 12: Logistic regression of human pair-preference on standardized reward gap and CER gap. $\operatorname { O R } ( z _ { R } )$ = 1.93 with 95% CI [1.38, 2.70], while $\mathrm { O R } ( z _ { C } ) = 0 . 9 6 $ with 95% CI [0.67, 1.38].
<table><tr><td>Metric</td><td>base</td><td>GRPO</td><td>∆</td></tr><tr><td>ANIMESCORE mean</td><td>-0.55</td><td>+0.22</td><td>+0.77</td></tr><tr><td>CER mean</td><td>0.044</td><td>0.034</td><td>-0.010</td></tr><tr><td>CER median</td><td>0.023</td><td>0.020</td><td>-0.003</td></tr><tr><td>UTMOS</td><td>3.10</td><td>3.18</td><td>+0.08</td></tr></table>

Table 13: ANIMESCORE EN base vs. zone-CER step 900 GRPO on English test dataset, $n = 5 0$

Objective results. On EN test dataset with the same CER-retry protocol, zone-CER GRPO at step 900 yields $\Delta \mathrm { A N I M E S C O R E } = + 0 . 7 7$ over base. Table 13 summarizes the corresponding objective scores.

Human results. On the same 50 EN items, item-level Human-WR is 70.0% with Wilson CI [56.2, 80.9]. Machine-WR is 66.0%, item-level machine–human agreement is 72.0%, vote-level Human-WR is 62.8%, and vote-level agreement is 68.8%.

Limitation. The EN study was rated by the same Japanese-native listener pool that produced the JP results. It therefore tests whether a JP-trained anime-likeness reward transfers to EN audio under the same listener community, not whether Englishnative listeners would judge the shift similarly. We treat the EN result as auxiliary evidence, not as primary evidence for cross-lingual human transfer.

<table><tr><td>Quantity Value</td></tr><tr><td>Total training steps 1671</td></tr><tr><td>Validation evaluations 60 0.050 / 0.082</td></tr><tr><td>Init β / final β [0.002, 0.122]</td></tr><tr><td>βrange Plateau  $\beta$  median 0.025</td></tr><tr><td>Plateau βIQR [0.014, 0.052]</td></tr><tr><td>Plateau reward-KL penalty median 0.041</td></tr><tr><td>Max reward-KL penalty 0.094</td></tr></table>

Table 14: Adaptive-KL statistics for the primary ANI-MESCORE JP zone-CER run. Plateau is defined as steps $\geq 3 0 0$
<table><tr><td>Step</td><td>val_AS</td><td>val_CER val_viol. val_reward</td><td></td></tr><tr><td>1050</td><td>-0.138 0.230</td><td>0.27</td><td>1.902</td></tr><tr><td>1380</td><td>+0.562 0.244</td><td>0.24</td><td>2.579</td></tr><tr><td>1400</td><td>+0.625 0.268</td><td>0.26</td><td>2.584</td></tr><tr><td>1710 +0.955</td><td>0.308</td><td>0.31</td><td>2.556</td></tr></table>

Table 15: Validation trajectory at key steps for the primary ANIMESCORE JP zone-CER run. Raw AnimeScore continues rising after step 1400, but constraintaware validation reward declines as violations increase.

## I VAD-Arousal Training-Only Negative Result

This appendix documents the training-only negative result for VAD-AROUSAL. No human A/B study was conducted for this axis, so the evidence concerns training dynamics rather than perceptual transfer.

Validation arousal stayed in the range 0.61–0.64 during early training, yielding a net change of only $\Delta \approx + 0 . 0 1 4$ . At the same time, validation reward improved from −0.34 to −0.05, while the validation violation rate dropped from 0.62 to 0.45. This suggests that training primarily improved reward by moving samples out of the VIOLATE shelf, rather than by increasing arousal within the feasible zone.

<table><tr><td>Axis</td><td>GRPO votes</td><td>Vote HWR</td><td>Vote Agree</td><td>Cluster-boot CI</td></tr><tr><td>ANIMESCORE JP</td><td>180/250</td><td>72.0</td><td>78.4</td><td>[64.0,79.6] / [72.4,84.4]</td></tr><tr><td>UTMOS</td><td>141/250</td><td>56.4</td><td>72.0</td><td>[47.2,65.2] / [64.8,78.8]</td></tr><tr><td>LIKABILITY</td><td>95/250</td><td>38.0</td><td>67.6</td><td>[29.2,47.2] / [58.8,76.0]</td></tr><tr><td>ANIMESCORE EN</td><td>157/250</td><td>62.8</td><td>68.8</td><td>[54.8,70.4] / [62.4,75.2]</td></tr></table>

Table 16: Vote-level summaries for GRPO vs Base with item-clustered bootstrap 95% confidence intervals. The two intervals in the last column correspond to Vote HWR and vote-level agreement.
<table><tr><td>Axis</td><td> $\hat { \beta } _ { R } \left( \mathrm { S E } , p \right)$ </td><td> $\hat { \beta } _ { C } \ ( \mathrm { S E } , p )$ </td><td>ORR</td></tr><tr><td>ANIMESCORE EN</td><td>+0.50 (0.37, 0.17)</td><td>-0.14 (0.30, 0.64)</td><td>1.65</td></tr><tr><td>ANIMESCORE JP</td><td>+0.17 (0.36, 0.63)</td><td>-1.91 (0.78, 0.014)</td><td>1.19</td></tr><tr><td>LIKABILITY</td><td>+1.09 (0.47, 0.020)</td><td>+0.48 (0.40, 0.23)</td><td>2.98</td></tr><tr><td>UTMOS</td><td>+1.15 (0.45, 0.012)+0.67 (0.35, 0.058)</td><td></td><td>3.15</td></tr></table>

Table 17: Per-axis logistic fits. $\hat { \beta } _ { R }$ is positive in every axis; LIKABILITY and UTMOS reach $p < 0 . 0 5$ individually.

Per-rollout reward decompositions were not retained, so this result is treated as a trainingdynamics diagnostic rather than a causal attribution of individual GRPO updates.

## J Adaptive-KL Statistics

Table 14 summarizes the adaptive-KL controller for the primary ANIMESCORE JP zone-CER run. Table 15 reports key validation points used for checkpoint selection.

Step selection. Between steps 1400 and 1710, raw validation AnimeScore rises from +0.625 to +0.955, but the validation violation rate also rises from 0.26 to 0.31 and constraint-aware validation reward decreases from 2.584 to 2.556. We therefore select step 1400.

<table><tr><td>Axis</td><td>|∆reward| bin n items n votes</td><td></td><td></td><td>Vote HWR</td><td>Vote Agree</td></tr><tr><td>ANIMESCORE JP</td><td>0-0.3</td><td>5</td><td>25</td><td>28.0</td><td>64.0</td></tr><tr><td>ANIMESCORE JP</td><td>0.3-1.0</td><td>8</td><td>40</td><td>62.5</td><td>62.5</td></tr><tr><td>ANIMESCORE JP</td><td>1.0–2.5</td><td>19</td><td>95</td><td>65.3</td><td>72.6</td></tr><tr><td>ANIMESCORE JP</td><td>2.5-∞</td><td>18</td><td>90</td><td>95.6</td><td>95.6</td></tr><tr><td>ANIMESCORE EN</td><td>0-0.3</td><td>17</td><td>85</td><td>48.2</td><td>49.4</td></tr><tr><td>ANIMESCORE EN 0.3-1.0</td><td></td><td>18</td><td>90</td><td>61.1</td><td>71.1</td></tr><tr><td>ANIMESCORE EN 1.0–2.5</td><td></td><td>11</td><td>55</td><td>74.5</td><td>83.6</td></tr><tr><td>ANIMESCORE EN</td><td>2.5-∞</td><td>4</td><td>20</td><td>100.0</td><td>100.0</td></tr><tr><td>UTMOS</td><td>0-0.3</td><td>17</td><td>85</td><td>45.9</td><td>52.9</td></tr><tr><td>UTMOS</td><td>0.3-1.0</td><td>24</td><td>120</td><td>48.3</td><td>75.8</td></tr><tr><td>UTMOS</td><td>1.0–2.5</td><td>9</td><td>45</td><td>97.8</td><td>97.8</td></tr><tr><td>LIKABILITY</td><td>0-0.3</td><td>40</td><td>200</td><td>24.5</td><td>61.5</td></tr><tr><td>LIKABILITY</td><td>0.3-1.0</td><td>8</td><td>40</td><td>90.0</td><td>90.0</td></tr><tr><td>LIKABILITY</td><td>1.0–2.5</td><td>2</td><td>10</td><td>100.0</td><td>100.0</td></tr></table>

Table 18: Descriptive vote-level reward-gap binned analysis. Items are grouped by absolute target-reward gap. Vote HWR is the fraction of individual votes choosing GRPO, and Vote Agree is the fraction of individual votes matching the machine-preferred side. This table complements the signed-gap regression in Appendix F. High-gap LIKABILITY bins contain few item pairs and should not be interpreted as standalone statistical evidence.