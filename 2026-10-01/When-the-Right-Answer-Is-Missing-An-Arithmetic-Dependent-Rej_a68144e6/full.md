# When the Right Answer Is Missing: An Arithmetic-Dependent Rejection Bottleneck in Jev

Jike Zhong<sup>1</sup>\* Ming Li<sup>2</sup>\* Yuxiang Lai<sup>3</sup>\* <sup>1</sup>University of Southern California <sup>2</sup>University of Florida <sup>3</sup>Emory University

## Abstract

Typed decision models such as Jev offer an efficient alternative to generative LLMs in decision-making workflows by selecting directly from predefined options. When candidate sets contain no valid answer, TypeSafe recommends including an "other" or “noneof-the-above” option to enable rejection. In this report, however, we identify an arithmeticdependent rejection bottleneck: Jev reliably selects correct numerical answers when available but frequently accepts incorrect alternatives when they are absent despite an explicit rejection option. On paired arithmetic problems, answer-present accuracy reaches 99%, while correct rejection falls to 7%. Moreover, this gap persists across numerical magnitudes, operation depths, contextual formulations, and rejection labels, and extends to scenarios such as time calculation and capacity rounding. Yet native Boolean verification achieves 99% exactmatch accuracy on the same answer-absent arithmetic cases, showing that categorical rejection can fail even when the model successfully verifies candidate correctness. Finally, we show that a simple decision threshold selected on sep arate development problems raises arithmetic rejection accuracy from 7% to 79% while retaining 97% answer-present accuracy, substantially mitigating the failure without retraining or additional inference.

## 1 Introduction

Typed decision models such as Jev (TypeSafe AI, 2026b) have emerged as an efficient alternative to generative large language models (LLMs) (OpenAI, 2023; Gemini Team, 2023; Anthropic, 2024) for decision-making workflows. Unlike generative LLMs that produce token sequences autoregressively, Jev accepts a description of the current state and predefined output criteria, then directly returns structured categorical, Boolean, or ordinal judgments (TypeSafe AI, 2026a,b). This makes it a natural component for workflows that require bounded decisions rather than generated explanations, such as in record, action selection and policy evaluation.

![](images/eb65521d0addad227dfac7e98483ebf5d0c0a344c4fb2a30ecb29ee4cd987567.jpg)

![](images/ebfce502d301ffafe0744a891b67885a74473306c4001bdf187d913a045c35e8.jpg)

![](images/775958fb569f4abbf44dbc6c5b6e8ef9bc646259e6a0e6f1d4ad6c5e16c0937b.jpg)  
Figure 1: Arithmetic-dependent rejection and threshold calibration. (a) Menu Choice achieves 99% answerpresent accuracy but only 7% correct rejection even on simple three-operand arithmetic (a + b − c), while Boolean verification reaches 99% answer-absent exact match. (b) Rejection collapses after just two operations. (c) NONE scores retain a useful rejection signal. (d) A development-selected threshold calibration raises held-out rejection to 79% while retaining 97% answerpresent accuracy, without additional inference.

However, predefined candidate sets need not contain a valid answer. For example, in payment reconciliation, a system matching a \$100 payment to retrieved invoices must reject all candidates if none has charges minus credits totaling \$100. Recognizing when to reject is therefore essential for reliable decisions, as emphasized by research on LLM abstention (Wen et al., 2025; Madhusudhan et al., 2025; Tomani et al., 2024; Kirichenko et al., 2025; Wang et al., 2025). To support rejection, TypeSafe recommends including an other or noneof-the-above option (TypeSafe AI, 2026a). This raises a basic question: if a model can select the correct answer, can it also recognize when every supplied answer is wrong?

We study this question through paired answerpresent and answer-absent menus with exact ground truth. Surprisingly, Jev achieves 99% answer-present accuracy on bare arithmetic, even simple two-operation arithmetic (a + b − c), yet correctly rejects only 7% of answer-absent menus despite an explicit “None" option (Figure 1a). This rejection gap extends beyond bare expressions to arithmetic-dependent scenarios, including inventory updates, purchase totals, and time calculations. Moreover, rejection already collapses with just two elementary operations, showing that the failure arises even under minimal computational demands (Figure 1b). Together, these results highlight why high selection accuracy alone is an incomplete measure of reliability.

To identify the source of this bottleneck, we examine both the required computation and the decision interface. Native Boolean verification correctly rejects all three candidates on 99% of answerabsent arithmetic cases, demonstrating that Jev can evaluate their numerical correctness. Similarly, providing the correct numerical result directly restores perfect Menu Choice rejection with the same candidates. We further introduce candidate-level T/F Choice, which preserves the Boolean verification questions but uses categorical outputs. Its substantially lower accuracy shows that individual verification alone does not resolve the failure; the decision interface remains consequential.

We therefore ask whether rejection can be improved within Menu Choice itself and find that its output scores retain a useful rejection signal even when the selected answer is incorrect (Figure 1c). Motivated by this, we propose a simple threshold calibration and show that a threshold selected on separate development problems raises arithmetic rejection from 7% to 79% while retaining 97% answer-present accuracy, without retraining or additional inference (Figure 1d).

## Contributions.

• We identify an arithmetic-dependent rejection bottleneck: Jev reliably selects valid answers but frequently fails to reject invalid candidate sets.

• We conduct extensive ablations, controlled comparisons, and error analyses to characterize the roles of computation, decision interface, and task formulation.

![](images/aa2bad180ae9b7d7c50804f4cd50964ef730deb7ef0dfeb1480b60e2d0e0dbb3.jpg)  
Oracle: 641; reject all three candidates  
Figure 2: Three formulations of the same decision. Menu Choice selects a candidate or NONE. Our T/F Choice ablation verifies each candidate through the categorical API; native Boolean changes only the question type. Candidate questions share one request. The oracle answer is shown for illustration and is not supplied in the arithmetic input.

• We propose a simple threshold calibration that substantially improves rejection while retaining high selection accuracy, without retraining or additional inference.

## 2 Related Work

Selection and rejection. Prior work examines least-incorrect answer selection (Wang et al., 2025) and domain-dependent degradation under noneof-the-above questions (Tam et al., 2025). More broadly, studies of LLM abstention document failures on unanswerable questions (Madhusudhan et al., 2025; Kirichenko et al., 2025; Wen et al., 2025) and explore uncertainty-based rejection (Tomani et al., 2024). We study this distinction in typed decision APIs, isolating an arithmeticdependent rejection bottleneck through matched interface comparisons and computation controls, then evaluating threshold calibration as a repair.

Jev interface behavior. TypeSafe documents mathematical limitations and Boolean/Choice inconsistencies (TypeSafe AI, 2026b). Sun and Xu (2026) study option-name/rubric conflicts, while Zhang et al. (2026) examine request configuration on ContractNLI. In contrast, we isolate exact numerical rejection under explicit false and NONE alternatives. Alternative-label controls connect this analysis to schema sensitivity without reducing the phenomenon to a particular identifier.

![](images/871aa0e62749c6fffad2027b1a7b2b5fdb48ddfc69e0857c6fb05851637a6247.jpg)

(b) Paired coverage gap (95% CI)  
![](images/d6cd6882fbc4ce78db6c75a500e21b743b8e2ce602326cacab6dad5d387fcae7.jpg)  
Figure 3: Strong selection conceals task-dependent rejection failures. (a) Accuracy on 100 paired problems per task; P/A denotes answer present/absent. T/F and Boolean require all three judgments correct. Time and counting pool 20 screening and 80 fresh cases. (b) Present-minus-absent gaps with 95% paired-bootstrap intervals. Menu rejection deteriorates most on arithmetic and purchase totals; lookup eliminates the gap.

## 3 Evaluation Framework

## 3.1 Exact Decisions with Incomplete Menus

Let $x _ { j }$ denote the facts and rule for case $j ,$ and let $y _ { j } = f ( x _ { j } )$ be the deterministically computed answer. Each case has a three-candidate menu $C _ { j c } = ( a _ { 1 } , a _ { 2 } , a _ { 3 } )$ with coverage $c \in \{ 0 , 1 \}$ . Candidate correctness is

$$
z _ { j c i } = { \bf 1 } [ a _ { i } = y _ { j } ] , \qquad \sum _ { i = 1 } ^ { 3 } z _ { j c i } = c .\tag{1}
$$

An answer-present menu has exactly one correct candidate; an answer-absent menu has none. The model must select the correct candidate or reject the menu. Rejection means that the supplied candidates are incorrect, not that the underlying problem is unanswerable.

## 3.2 Three Decision Formulations

Figure 2 illustrates our three formulations on one saved arithmetic case. They use two native API types, Choice and Boolean, with a third Candidate T/F Choice as an ablation constructed using the Choice type, rather than a third native primitive.

Menu Choice. A categorical question selects from {A, B, C, NONE}, where each option has an explicit correctness criterion. Following Type-Safe’s recommendation, NONE denotes that no valid candidate (TypeSafe AI, 2026a). We score the API-returned choice $\widehat { y }$ against the correct option y.

Candidate T/F Choice. To test whether individual verification repairs rejection, our diagnostic ablation asks whether each candidate is correct using categorical true/false options, batching all three questions in one request. We predict $\hat { z } _ { i } ^ { \bar { C } } = \mathbf { 1 } [ q _ { i } \geq 0 . 5 ]$ , where $q _ { i }$ is candidate i’s true probability. A question is considered correct only if all three labels are correct.

Native Boolean. To isolate the effect of decision type, we submit the same candidate questions with Boolean type and predict $\widehat { z } _ { i } ^ { B } = \mathbf { 1 } [ b _ { i } \geq 0 . 5 ]$ where $b _ { i }$ is the returned truth probability. Matched payloads differ only in the type field, verified programmatically. Similarly, all three labels must be correct.

## 3.3 Metrics and Statistical Unit

Case-level correctness. For candidate verification, let $S = \{ i : \widehat { z } _ { i } = 1 \}$ . A single positive selects that candidate; no positives rejects the menu; multiple positives count as incorrect without scorebased tie-breaking. Since each menu has at most one valid candidate, decision accuracy equals allcandidate exact match: $\begin{array} { r } { A _ { h , c } = \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \mathbf { \hat { 1 } } [ \widehat { \boldsymbol { z } } _ { j c } ^ { h } = } \end{array}$ $z _ { j c } ]$ for interface h and coverage condition c.

Paired comparisons. The coverage gap $G _ { h } =$ $A _ { h , 1 } - A _ { h , 0 }$ compares answer-present and answerabsent accuracy on paired problems. Interface and distance comparisons likewise preserve pairing. Named contrasts use 20,000 case-bootstrap resamples (seed 2026092901). AUC describes candidatescore discrimination; accuracy and uncertainty use the underlying problem as the statistical unit. Pilot results report explicit denominators.

![](images/64ef4d607d1598848c358aa21ee85c0cd934f2757ffe56c7cd6869f529d00187.jpg)

![](images/f0b78b73aa21cb4a6468ab68f477eee94c723e5671681be35f3945615ea2a587.jpg)

![](images/e886835405023aa041e940b5a70539f09dfd47d1d100fc0ae5994387fd14ba63.jpg)  
Figure 4: Rejection fails with simple arithmetic and depends on candidate distance. (a) Fifty paired cases per operation depth; depth zero supplies the result. (b) One hundred fresh two-operation cases per magnitude band. (c) Fifty paired inventory cases: larger distractor errors improve categorical rejection but not Boolean verification. Depth and distance vary within cases; magnitude uses independent cases.

## 4 Experiments and Results

## 4.1 Experimental Setup

Tasks and paired menus. Our main evaluation uses 100 problems each for two-operation arithmetic $( a + b - c )$ , numerical lookup, and purchase totals $( q p + s )$ . Arithmetic samples $a \in [ 3 0 0 , 9 9 9 ]$ and $b , c \in [ 1 0 , 9 9 ]$ ; purchase totals use $q \in [ 3 , 1 1 ]$ ], $p \in [ 1 1 , 5 9 ]$ , and $s \in [ 5 , 2 4 ]$ . Present menus use answer offsets $\{ - 2 , 0 , + 3 \}$ and absent menus use $\{ - 2 , + 1 , + 3 \}$ , preserving the facts and two distractors while shuffling candidate order within each coverage condition. All interfaces receive the same menus; arithmetic and lookup additionally share answers and candidate text. Contextual controls comprise 128 inventory problems and 64 problems each for refund policy and logical eligibility. Subsequent experiments vary computation depth, numerical magnitude, candidate distance, operator structure, task family, and rejection encoding.

Models and scoring. We evaluate Jev and record structured predictions and probabilities. Programmatic generators provide exact ground truth. Menu accuracy scores the selected option. Candidate verification requires all three truth labels to be correct. We additionally use Qwen3.5-9B (w/ Reasoning) as a generative LLM control on the same menus.

## 4.2 Main Results: The Rejection Bottleneck

Menu Choice achieves 99% answer-present accuracy on simple arithmetic but only 7% correct rejection, a 92-point paired gap. Purchase totals show the same pattern: 94% versus 24%, a 70-point gap (Figure 3). Thus, successful selection does not imply reliable rejection when the correct answer is missing.

Contextual inventory extends the effect beyond bare expressions, with 127/128 correct selections but only 14/128 correct rejections. In contrast, refund-policy and logical-eligibility controls achieve 64/64 under both coverage conditions across all tested interfaces. The bottleneck therefore depends on the task rather than arising whenever a rejection option is needed. These earlier contextual categorical-verification controls use neutral identifiers rather than literal T/F labels.

## 4.3 Isolating Computation and Interface Effects

Removing computation restores rejection. To isolate computation from numerical candidate matching, we supply the correct result directly while preserving the arithmetic answers and menus. All three interfaces achieve 100% accuracy under both coverage conditions. For Menu Choice, rejection rises from 7% to 100%: the same numerical candidates support reliable rejection when calculation is removed.

Individual verification does not close the interface gap. To separate decomposition from question type, we compare candidate-level T/F Choice with native Boolean, changing only the type field in matched payloads. On answer-absent arithmetic, exact-match accuracy rises from 27% to 99%; purchase totals show the same direction of improvement. Thus checking candidates individually does not resolve the failure: the same verification questions produce substantially different accuracy across interfaces. For example, given 674 + 27 − 60 = 641 and candidates (642, 639, 644), Menu Choice and T/F Choice both accept 639, whereas Boolean correctly rejects all three. The failure therefore persists under categorical verification even when native Boolean verification succeeds.

![](images/8ea439fb4678e5a7783157461fa7fbaac14e00ba792c06ebc07764015b577f48.jpg)  
Figure 5: A capable generative control. Qwen3.5-9B (w/ reasoning) on the same cases per family and coverage. P/A denotes present/absent. Both computational menus are solved perfectly. Token-limit failures account for 66 of 67 errors across menu and joint-verification responses, distinguishing output completion from valid incorrect decisions.

A capable generative model rejects the same menus. To test whether the candidate construction itself prevents rejection, we evaluate reasoningenabled Qwen3.5-9B (Qwen Team, 2026) using menu and joint-verification prompts on the identical transfer cases. Figure 5 shows that Qwen reaches 100% present and absent menu accuracy on both arithmetic and purchase totals. Lookup scores 99%/94%, with all seven errors caused by truncation. Joint-verification accuracy is 92%/79% for arithmetic, 89%/90% for lookup, and 92%/97% for purchase. Overall, 66 of 67 errors are truncations. The perfect computational-menu results establish that the same candidate sets permit reliable rejection.

## 4.4 Depth, Magnitude, and Operator Structure

Rejection collapses with minimal computation. We vary additive depth from zero (the result is supplied) to five operations on 50 paired base cases, preserving answers and candidates. Menu present/absent accuracy falls from 100%/66% at one operation to 98%/4% at two. Boolean retains 90%/92% at two operations and 84%/90% at four. At five, Boolean declines to 58%/66%, while Menu selection remains 90% and rejection is 2%. Categorical rejection thus deteriorates well before Boolean verification (Figure 4a).

<table><tr><td>Two-operation pattern</td><td>Present</td><td>Absent</td></tr><tr><td> $a \times b \times c$ </td><td>9/10</td><td>10/10</td></tr><tr><td> $a \times b + c$ </td><td>9/10</td><td>3/10</td></tr><tr><td> $a \times b - c$ </td><td>8/10</td><td>1/10</td></tr><tr><td> $a + b \times c$ </td><td>10/10</td><td>1/10</td></tr><tr><td> $a - b \times c$ </td><td>8/10</td><td>0/10</td></tr><tr><td> $a \div b \div c$ </td><td>10/10</td><td>8/10</td></tr><tr><td> $a \div b + c$ </td><td>6/10</td><td>1/10</td></tr><tr><td> $a \div b - c$ </td><td>5/10</td><td>2/10</td></tr><tr><td> $a + b \div c$ </td><td>9/10</td><td>3/10</td></tr><tr><td> $a \times b \div c$ </td><td>10/10</td><td>9/10</td></tr></table>

Table 1: Rejection varies with operator composition. Multiplication/division chains achieve higher rejection accuracy than expressions combining these operations with addition or subtraction.

Single-digit calculations already expose the gap. To isolate magnitude, we generate 100 fresh twooperation $a + b - c$ problems in each band: 1–9, 10–99, 100–999, 1,000–9,999, and 10,000–99,999. Operands, intermediate sums, and answers remain within the band; distractor offsets stay fixed. Selection is 100%, 100%, 99%, 99%, and 99%, respectively, whereas rejection is 56%, 8%, 14%, 11%, and 14%. Failure therefore occurs even within 1–9 and remains severe, without a monotonic magnitude trend, across larger bands (Figure 4b).

Distance affects interfaces differently. To distinguish candidate proximity from problem magnitude, we give 50 shared inventory cases absent candidates at absolute-error bands 1–5, 20–25, 90– 100, and 195–200. Menu rejection increases from 20% to 88%, and T/F exact rejection from 16% to 84%. In contrast, Boolean declines from 100% to 84% (Figure 4c). Increasing numerical separation helps categorical rejection but does not produce a common difficulty ordering across interfaces.

Mixed operators reproduce the rejection gap. To test beyond additive arithmetic, we evaluate 50 single multiplications, 50 exact divisions, and 100 two-operation problems (10 per pattern), using standard precedence and exact integer division. Two-operation cases retain the original operand ranges and positive answers. Single multiplication and division retain high rejection accuracy (98% and 96%), whereas combining these operators with addition or subtraction reproduces the selection– rejection gap (Table 1). The bottleneck thus extends beyond additive expressions while remaining sensitive to operator composition.

![](images/310ae1c19a643dd7e6e3a7915d229b6b19f9b7dd6dd176a36d12b995ef7f9383.jpg)  
Figure 6: Rejection gaps extend to other arithmeticdependent tasks. Menu Choice results across 15 pilot families, with 20 paired cases each. Time calculation, capacity rounding, and calendar arithmetic show selection–rejection gaps, while several symbolic tasks succeed under both conditions. Time and counting show their initial pilot results; the dotted divider separates the five additional numerical tasks.

## 4.5 Task Transfer and Rejection Encoding

The gap extends to other arithmetic-dependent scenarios. Beyond bare expressions, we observe selection–rejection gaps in contextual inventory, purchase totals, time calculations, and capacity rounding (Figure 6). Time calculations reproduce the gap on fresh cases despite perfect answerpresent accuracy, while Boolean verification or directly supplying the result largely restores rejection. Capacity rounding and calendar arithmetic also show rejection failures in the smaller pilot sets, whereas object counting exhibits a milder gap. We hypothesize that the underlying arithmetic in these scenarios contributes to the bottleneck, with its severity depending on the task and operator structure.

Broader controls delimit the affected tasks. Five additional numerical families receive 20 paired cases each; Figure 6 reports all 15 screens. State tracking, set intersection, rule chains, and scheduling succeed perfectly. Unit conversion scores 20/20 in both conditions, and fraction comparison scores 20/20 versus 19/20. In contrast, capacity rounding scores 18/20 versus 2/20; computing $\lceil 9 6 / 1 4 \rceil = 7$ is a representative example. Calendar arithmetic scores 12/20 versus 0/20. Spatial navigation and string transformation have weaker answer-present performance, while program execution and interval overlap perform better with the answer absent.

Alternative rejection labels preserve the failure. To isolate wording, we rerun the same 100 arithmetic cases with Other and None of the above, preserving rejection criteria, candidate order, and inputs. Their present/absent accuracies are 99%/9% and 99%/6%, compared with the saved NONE baseline of 99%/7%. Separately, moving NONE across four positions recovers only 5, 0, 2, and 0 of 40 selected inventory failures. These results highlight that neither label substitution nor position changes provide a general repair, motivating the score-based calibration studied next.

## 5 Error Analysis and Threshold Calibration

## 5.1 Useful Scores, Incorrect Acceptance

The preceding controls show that the decision interface affects rejection. We next ask whether the returned scores retain enough information to repair these errors without changing the interface. In Figure 7, arithmetic T/F Choice often assigns incorrect candidates truth scores above .5, although correct candidates generally receive higher scores. We quantify this separation using the area under the receiver operating characteristic curve (AUC): the probability that a randomly chosen correct candidate scores above a randomly chosen incorrect candidate, with half credit for ties. An AUC of .5 indicates chance-level ordering and 1 indicates perfect ordering. Arithmetic T/F achieves an AUC of .985 despite only 88/200 exact-match menus; purchase T/F shows the same separation between ranking and decisions, with an AUC of .942 and 62/200 exact matches.

This strong ordering suggests that threshold placement contributes to the errors. Under the default .5 threshold, arithmetic T/F accepts multiple candidates on 55/200 menus, compared with 3/200 for Boolean; purchase exhibits the same pattern. Selecting the highest-scoring candidate would hide these conflicting judgments and would still require a separate rejection rule when every candidate is wrong. Instead, the score separation motivates calibrating the acceptance threshold while retaining all-candidate exact-match evaluation.

![](images/4c3296eb45c03cdfcdaf10c16b5508dd71bb46f8ab79dc731a85634b5276e8f7.jpg)  
Figure 7: Score separation motivates threshold calibration. Empirical CDFs of Menu Choice rejection probabilities (left) and candidate truth scores for T/F Choice and Boolean (center/right). Answer-absent menus tend to receive higher rejection scores, while correct candidates generally receive higher truth scores than incorrect candidates. This separation motivates adjusting decision thresholds to reduce false acceptance. Gray dotted lines mark the default verification threshold; blue dashed lines mark development-selected thresholds.

Menu Choice offers a related opportunity through a different score. Answer-absent menus often assign more probability to NONE than answerpresent menus, even when NONE does not outrank the numerical alternatives. This motivates thresholding rejection probability directly. Thus candidate verification and menu selection call for different calibration rules, both using scores already returned by the model.

## 5.2 A Development-Selected Decision Rule

Menu rejection threshold. Let $p _ { \emptyset }$ be the probability of NONE and $p _ { i }$ the candidate probabilities. We replace categorical argmax with

$$
\widehat { d } _ { \tau } ( x , C ) = \left\{ \begin{array} { l l } { \emptyset , } & { p _ { \emptyset } \geq \tau , } \\ { \mathrm { a r g } \operatorname* { m a x } _ { i \in \{ 1 , 2 , 3 \} } p _ { i } , } & { p _ { \emptyset } < \tau . } \end{array} \right.\tag{2}
$$

This rule checks whether the menu should be rejected before selecting a candidate, allowing rejection without requiring NONE to have the largest probability.

Candidate acceptance threshold. For candidate verification, we instead threshold each T/F or Boolean truth score s : ${ \widehat { z } } _ { i } = \mathbf { 1 } [ s _ { i } \geq \tau ]$ and retain all-three exact-match scoring. A higher threshold targets the false acceptances identified above; one positive selects a candidate, no positives rejects the menu, and multiple positives remain an error.

Calibration protocol. To select an operating point that balances selection and rejection, we search $\tau \in \{ 0 , . 0 1 , . . . , 1 \}$ separately for each task/interface on 20 development problems, each with present and absent menus. The objective is mean case accuracy across the two balanced conditions; ties favor proximity to .5, then the smaller threshold. Arithmetic, lookup, and purchase use the earlier 20-case pilot for development and the disjoint 100-case transfer set for evaluation. Time and counting use their initial 20 for development and fresh 80 for evaluation. All calibration is performed offline on saved scores, and each selected threshold is applied unchanged to its evaluation set, requiring neither retraining nor additional inference.

## 5.3 Held-Out Recovery and Tradeoffs

Table 2 and Figure 8 show that these developmentselected thresholds recover rejection on held-out cases. For arithmetic menus, τ = .03 raises correct rejection from 7% to 79% while retaining 97% answer-present accuracy, compared with 99% originally. Balanced accuracy therefore rises from 53% to 88%: the rejection gains substantially exceed the selection losses.

Candidate-level calibration likewise converts the strong score ordering into more accurate decisions. With $\tau = . 7 4$ , arithmetic T/F exact match rises from 61%/27% to 86%/90% on present/absent menus. Boolean retains its default threshold of .5 and its 97%/99% accuracy. These results connect the score separation in Figure 7 to a practical correction of false acceptance.

![](images/ce9fe3d1c1f96c1d5a067752a067fb7f0a003dd712756552a99b29ce77fee88b.jpg)

![](images/6741a058f8f7a54314d45a9e8ff90b560f5d6ed98c78e5ac805e239a662121ef.jpg)  
Figure 8: Threshold adjustment recovers rejection on separate cases. (a) Arrows connect original menu decisions (open circles) to calibrated decisions (filled circles). Upward movement improves rejection; leftward movement loses selection accuracy. (b) Paired 95% bootstrap intervals quantify both effects. Arithmetic/purchase use 100 test cases; time/counting use 80. Thresholds are chosen only on the corresponding 20-case development sets.

<table><tr><td>Task</td><td>Interface</td><td>T</td><td>Present</td><td>Absent</td></tr><tr><td>Arithmetic</td><td>Menu T/F</td><td>.03 .74</td><td>99→97 61→86 97→97</td><td>7→79 27→90 99→99</td></tr><tr><td>Purchase</td><td>Boolean Menu T/F Boolean</td><td>.50 .03 .80 .90</td><td>94→86 45→73 73→87</td><td>24→79 17→82 76→97</td></tr><tr><td>Time</td><td>Menu T/F</td><td>.13 .66</td><td>80→80 61→59</td><td>49→72 49→65</td></tr><tr><td>Counting</td><td>Boolean Menu T/F Boolean</td><td>.50 .09 .79 .50</td><td>80→80 78→76 78→78 79→79</td><td>79→79 73→78 76→77 77→77</td></tr></table>

Table 2: Original → calibrated correct counts. Arithmetic and purchase denominators are 100 per coverage; time and counting denominators are 80. Lookup remains 100/100 throughout at τ = .5. No evaluation labels are used to select thresholds.

The same intervention also improves rejection beyond bare arithmetic. On fresh time cases, menu calibration increases rejection from 49/80 to 72/80 while preserving perfect selection. Purchase balanced accuracy improves from 59% to 82.5%, exchanging eight present-case successes for 55 absent-case rescues; counting exchanges two for five. Threshold calibration thus offers substantial recovery where the scores separate valid from invalid menus, with selection–rejection tradeoffs that vary by task.

## 6 Discussion and Conclusion

Computation and rejection interact. Strong selection and Boolean verification show that Jev can solve the underlying problems, while supplying computed answers restores Menu Choice rejection. However, matched T/F Choice does not recover Boolean accuracy, showing that decomposition alone is insufficient. Together, these controls link the bottleneck to computation and decision interface. The gap extends to contextual arithmetic, time calculation, and capacity rounding, while successful lookup, symbolic controls, and multiplication/division chains reveal its task dependence.

Implications for decision workflows. In payment reconciliation and similar workflows, retrieved candidates may all be invalid. An explicit rejection option therefore needs evaluation on paired answer-present and answer-absent cases. Boolean verification changes the interface, whereas threshold calibration improves decisions from existing scores without retraining or additional inference. Their effectiveness should be assessed through both rejection gains and selection losses; deterministic computation offers another remedy when structured inputs permit it.

Conclusion. In this report, we identify an arithmetic-dependent rejection bottleneck in Jev despite strong selection accuracy. Through paired coverage tests, matched interfaces, and lookup controls, we show that categorical rejection can fail even when candidate verification succeeds. Motivated by score-distribution analysis, our threshold calibration substantially improves held-out rejection without retraining or additional inference. These findings support evaluating selection and rejection jointly. Future work includes real-world retrieval settings, cross-task calibration, and training for reliable rejection.

## Limitations

We evaluate Jev through its hosted API without access to model weights or internal activations. Our experiments and interpretations therefore characterize observable behavior through controlled inputs and returned outputs, rather than directly identifying the internal mechanisms behind the rejection bottleneck.

## Ethics Statement

All experimental inputs are synthetic numerical or symbolic records without personal data. The study evaluates output reliability and does not execute financial, physical, or account actions. Its intended use is to improve validation of automated decisions and to make rejection errors visible alongside successful selection.

## References

Anthropic. 2024. Introducing the next generation of Claude.

Gemini Team. 2023. Gemini: A family of highly capable multimodal models. arXiv preprint arXiv:2312.11805.

Polina Kirichenko, Mark Ibrahim, Kamalika Chaudhuri, and Samuel J. Bell. 2025. AbstentionBench: Reasoning LLMs fail on unanswerable questions. arXiv preprint arXiv:2506.09038.

Nishanth Madhusudhan, Sathwik Tejaswi Madhusud han, Vikas Yadav, and Masoud Hashemi. 2025. Do LLMs know when to NOT answer? investigating abstention abilities of large language models. In Proceedings of the 31st International Conference on Computational Linguistics, pages 9329–9345.

OpenAI. 2023. GPT-4 technical report. arXiv preprint arXiv:2303.08774.

Qwen Team. 2026. Qwen3.5-9B model card. https: //huggingface.co/Qwen/Qwen3.5-9B. Accessed September 25, 2026.

Yu Sun and Junhao Xu. 2026. Type-safe is not errorfree: A constrained decision head follows the option name, not the rubric bound to it. arXiv preprint arXiv:2609.26758.

Zhi Rui Tam, Cheng-Kuang Wu, Chieh-Yen Lin, and Yun-Nung Chen. 2025. None of the above, less of the right: Parallel patterns between humans and LLMs on multi-choice questions answering. arXiv preprint arXiv:2503.01550.

Christian Tomani, Kamalika Chaudhuri, Ivan Evtimov, Daniel Cremers, and Mark Ibrahim. 2024. Uncertainty-based abstention in LLMs improves

safety and reduces hallucinations. arXiv preprint arXiv:2404.10960.

TypeSafe AI. 2026a. Choice: Primitives (questions). Accessed September 29, 2026.

TypeSafe AI. 2026b. Jev 1.13 jaggedness. https: //docs.typesafe.ai/model-jaggedness/jev-1. 13. Accessed September 25, 2026.

Haochun Wang, Sendong Zhao, Zewen Qiang, Nuwa Xi, Bing Qin, and Ting Liu. 2025. LLMs may perform MCQA by selecting the least incorrect option. In Proceedings ofthe 31st International Conference on Computational Linguistics, pages 5852–5862.

Bingbing Wen, Jihan Yao, Shangbin Feng, Chenjun Xu, Yulia Tsvetkov, Bill Howe, and Lucy Lu Wang. 2025. Know your limits: A survey of abstention in large language models. Transactions of the Association for Computational Linguistics, 13:529–556.

Fan Zhang, Yankai Chen, Zhuohan Xie, Yixi Zhou, Sijia Peng, Lei Fan, Xinhua Ji, Cunyuan Zheng, Huangyong Shan, Philip S. Yu, Xue Liu, Yu Chen, Preslav Nakov, and Songwei He. 2026. Same scores, different decisions: Evaluating JEV and language models for legal document understanding. arXiv preprint arXiv:2609.27678.