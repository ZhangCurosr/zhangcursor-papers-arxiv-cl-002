# WHEN CLIPPING REVERSES CORRECTION: FAILURE DYNAMICS OF POINTWISE FORWARD-KL ON-POLICY SELF-DISTILLATION

Di Huang<sup>1</sup>, Hao Li<sup>1</sup>, Yixin Chen<sup>1</sup>, Fuhai Li<sup>1,2∗</sup>

<sup>1</sup>Department of Computer Science, <sup>2</sup>Department of Pediatrics

Washington University in St. Louis

\*Correspondence: fuhai.li@wustl.edu

## ABSTRACT

On-policy self-distillation (OPSD) trains a student on its own generated responses using feedback from the same model conditioned on privileged information. On mathematical reasoning, the original OPSD study finds that stylistic tokens can dominate the training signal over math-related tokens, and that pointwise clipping of the forward KL objective stabilizes training. Pointwise clipping caps each vocabulary-wise forward KL term at a fixed threshold before summing over the vocabulary. Follow-up studies have adopted this clipping, but its effect on training has not been directly examined. In matched training runs differing only in whether clipping is applied, we observe that clipped runs produce substantially more repetitions that persist to the end of the response than their unclipped counterparts. We trace this failure to the clipped objective. We prove that the clipped objective can fail to correct the student toward the teacher and can instead push clipped and unclipped token probabilities away from its teacher. Our training runs agree with this analysis: inside repetitions, the clipped student places less probability than its teacher on leaving the repetition, and more on continuing it, whereas the unclipped runs stay close to their teachers.

## 1 INTRODUCTION

Outcome rewards give a reasoning model one scalar for thousands of tokens. On-policy distillation instead provides a per-token target from a teacher evaluated on the student’s own responses (Agarwal et al., 2024; Lu & Lab, 2025). Self-distillation uses the same network as both teacher and student, replacing a stronger teacher model with access to additional privilege information that is withheld from the student (Zhao et al., 2026a; Hubotter et al., 2026; Shenfeld et al., 2026). Self-distillation¨ provides dense supervision, but how the student learns from it depends on the training objective.

On-Policy Self-Distillation (OPSD) trains the student with a pointwise-clipped forward KL objective (Zhao et al., 2026a). At each response position, this objective clips each vocabulary-wise forward KL term at a fixed threshold before summing over the vocabulary. The authors introduced this pointwise clipping to keep stylistic tokens from dominating the math-related tokens in the training signal, and many subsequent methods retain it (Chen et al., 2026b; Li et al., 2026; Liu et al., 2026; Shrestha & Tessier, 2026).

Studies using this recipe report response length inflation (Yang et al., 2026), responses that fail to terminate within the generation budget (Chen et al., 2026b; Ichihara et al., 2026), redundant reasoning chain (Gu et al., 2026), and accuracy that declines after early gains (Zhao et al., 2026a; Chen et al., 2026b; Ichihara et al., 2026; Pan et al., 2026; Zhang et al., 2026). Proposed explanations include a frozen teacher that cannot adapt to the student (Chen et al., 2026b), shifts in the teacher’s style or reasoning pattern induced by the privileged context (Ichihara et al., 2026; Pan et al., 2026; Yang et al., 2026; Gu et al., 2026). No study reports that pointwise clipped objective contribute to the text degeneration, leaving this possibility open for further investigation.

Prior work offers a partial mathematical description of this clipped objective: a clipped term becomes constant and loses its direct gradient (Chen et al., 2026b), and the clipped sum is no longer a divergence (Chen et al., 2026b; Feng et al., 2026). These observations do not establish which student distributions the clipped objective favors, in particular whether minimizing it still brings a clipped token’s probability closer to the teacher’s.

We trained the pointwise-clipped forward KL OPSD recipe with a frozen reference-conditioned teacher and encountered a failure mode of text degeneration: generations ended in periodic tails that never terminated. When we removed only the clipping, nearly all of these loops disappeared. The teacher and its privileged context are identical in the two runs, so neither accounts for the difference. We therefore ask whether, and through what mechanism, this clipped objective contributed to this failure.

Our contributions therefore follow three questions: how the clipped objective changes the logit gradient, where its minimum lies, and what it does to training. The logit gradient (Section 4.1). We show that, relative to exact forward KL, the clipped objective reverses the direction in which gradient descent moves the logits, on every clipped token and on unclipped token whose student probability lies within a bounded range above the teacher’s. The minimizer (Section 4.2). We show that, at a response position with the unclipped and clipped token sets held fixed, the clipped objective is minimized only when the clipped tokens’ probability is reduced to zero rather than restored toward the teacher’s, and each unclipped token receives more probability than the teacher assigns it. A lower total probability on the clipped tokens always permits a lower objective value. Training runs (Sections 4.3). We train four matched pairs of runs that differ only in whether clipping is applied and vary whether the student and the teacher think. The clipped runs produce far more repetitions that continue to the end of the response than their unclipped runs. Inside repetitions, the clipped student places less probability than its teacher on leaving, and more on continuing, whereas the unclipped runs stay close to their teachers. These observations are consistent with the logit-gradient reversals we derived.

## 2 RELATED WORK

Pointwise-clipped forward KL. OPSD introduces the pointwise clipping with threshold τ (Zhao et al., 2026a) on each vocabulary-wise forward KL term. Later studies default threshold τ = 0.05 with training methods deviations(Tan & Hong, 2026b; Hou et al., 2026; Zhang et al., 2026; Liu et al., 2026; Wang et al., 2026; Chen et al., 2026a; Ichihara et al., 2026), at other values and/or with training methods deviations (Chen et al., 2026b; Ichihara et al., 2026; Shrestha & Tessier, 2026; Tan & Hong, 2026a; Liang et al., 2026), or without stating one (Li et al., 2026; Zhao et al., 2026b). Chen et al. (2026b) observe that a clipped term no longer supplies a gradient, so the cap stops the student from following large, mostly stylistic terms, and that a sum of capped signed terms is not a divergence. Feng et al. (2026) notes that the clipped objective equals exact forward KL wherever no term exceeds the threshold, so the two share their gradient and Hessian around the student that matches the teacher, but that elsewhere it can be negative.

Failures reported under the clipped objective. OPSD reports its best checkpoint, yet its comparison of objectives scores lower at 100 updates than at 50 (Zhao et al., 2026a). Chen et al. (2026b) finds that its OPSD baseline loses accuracy from 100 to 200 updates while truncation doubles, and also stating frozen teacher that cannot respond to what the student rejects as one of the limitations. Zhang et al. (2026) find that their three SmolLM3 students fall below the base model by 100 updates under multiple different configurations. Ichihara et al. (2026) finds that, a teacher with a math ques tion and a solution to a physics question leaves many generations repetitive and non-terminating, as compared to one with another math question’s solution instead, attribute this to the context without isolating which property is responsible, and leave deterioration under longer training unexplained. Gu et al. (2026) observe that “OPSD trajectories often recompute the same intermediate quantities or revise earlier steps without new information, leading to long and redundant reasoning chains,” and attribute this to a teacher conditioned on a single reference solution.

## 3 EXPERIMENTAL SETUP

Training design. Our goal is to reproduce the training design of OPSD, which we reimplement in the verl framework (Sheng et al., 2024). Each run starts from Qwen3-4B (Yang et al., 2025), with a trainable full-parameter student and a frozen teacher initialized from the same checkpoint. The teacher additionally receives a reference solution and provides its next-token distribution at every position of the student’s responses.

Table 1: Paired training families. All use Qwen3-4B, teacher top-128 support, $T = 1 . 1$ , 30 problems per update with one rollout per problem, learning rate of $1 \times 1 0 ^ { - 6 }$ , forward-KL distillation, and one trajectory per run. Clipped runs use $\tau = 0 . 0 5$ and their unclipped runs do not.
<table><tr><td>Family</td><td>Reference</td><td>Student/teacher thinking</td><td>Response cap</td></tr><tr><td>F</td><td>yes</td><td>off/on</td><td>18,432</td></tr><tr><td>M</td><td>yes</td><td>on/off</td><td>18,432</td></tr><tr><td>K</td><td>yes</td><td>off/off</td><td>8,192</td></tr><tr><td>L</td><td>yes</td><td>on/on</td><td>18,432</td></tr></table>

We train for 200 updates on the same 30k OpenThoughts (Guha et al., 2026) dataset used to train OPSD. We construct four matched clipped/unclipped pairs. Within each pair, the data, sampling setup, and model configuration are identical; only whether pointwise clipping is applied differs and we call the unclipped run the twin of the clipped run. Table 1 summarizes the run-specific configurations. The complete training hyperparameters and chat template are given in Appendix C. We refer to each pair by its letter in Table 1. F, M, K and L differ only in which side thinks. The letters index the order in which the configurations entered our run series, each added to isolate one variable from an earlier run.

Distillation objective. OPSD computes its loss over the full vocabulary. Our teacher runs as a separate inference service under SGLang (Zheng et al., 2024), and sending a full-vocabulary distribution for every response position is impractical, so our interface returns the teacher’s top-128 token log probabilities at each response position.

At response position t, let support $S _ { t }$ denote set of the teacher’s top-128 tokens, and let $a _ { t , i }$ and $b _ { t , : }$ be the teacher’s and the student’s logits for token i. Both distributions are formed by a softmax over $S _ { t }$ at temperature $T = 1 . 1$

$$
p _ { t , i } = \frac { \exp ( a _ { t , i } / T ) } { \sum _ { j \in S _ { t } } \exp ( a _ { t , j } / T ) } , \qquad q _ { t , i } = \frac { \exp ( b _ { t , i } / T ) } { \sum _ { j \in S _ { t } } \exp ( b _ { t , j } / T ) } , \qquad i \in S _ { t } ,\tag{1}
$$

so that the teacher distribution p and the student distribution q each sum to one on $p _ { t }$ $q _ { t }$ $S _ { t }$ . We then define the vocabulary-wise forward KL term of token i:

$$
d _ { t , i } = p _ { t , i } \log \frac { p _ { t , i } } { q _ { t , i } } .\tag{2}
$$

Here $d _ { t , i }$ is positive where the student assigns token i less probability than the teacher and negative where it assigns more. Exact forward KL sums $d _ { t , i }$ directly, whereas the pointwise-clipped forward KL of OPSD first replaces any $d _ { t , i }$ above τ by $\tau ,$ leaving smaller and negative $d _ { t , i }$ unchanged:

$$
D _ { \mathrm { F K L } , t } = D _ { \mathrm { K L } } ( p _ { t } \| q _ { t } ) = \sum _ { j \in S _ { t } } d _ { t , j } , \qquad D _ { \mathrm { c l i p } , t } = \sum _ { j \in S _ { t } } \operatorname* { m i n } ( d _ { t , j } , \tau ) , \qquad \tau = 0 . 0 5 .\tag{3}
$$

Despite the symbol, which follows OPSD’s notation, $D _ { \mathrm { c l i p } , t }$ is not a divergence, since it can be negative. The training loss is the mean of $D _ { \mathrm { c l i p } , t }$ in the clipped runs, or of $D _ { \mathrm { F K L } , t }$ in the unclipped runs, over all valid response positions in the update. We call the tokens in the teacher support $S _ { t }$ the candidate tokens at position t, to distinguish them from the token the student emits there, and a candidate token with $d _ { t , i } > \tau$ an over-threshold token. In a clipped run, the over-threshold tokens are the clipped tokens.

Trajectory analysis and evaluation. At every response position of every update, we record the teacher’s and the student’s log probabilities on the candidate tokens. We can therefore compute both the clipped objective and exact forward KL at the same response positions in both runs of each pair. Full capture details are provided in Appendix C.

We evaluate checkpoints saved every 25 training updates on AIME 2025 (Zhang & Math-AI, 2025) and AIME 2024 (Zhang & Math-AI, 2024). The main text reports AIME 2025 only, with two metrics: per-sample accuracy (avg@12) and the terminal loop rate (generation ended in periodic tails that never terminated). The complete AIME 2024 and AIME 2025 results, and evaluation configuration are given in Appendix D.

## 4 FAILURE DYNAMICS OF POINTWISE CLIPPING

## 4.1 POINTWISE CLIPPING REVERSES THE LOGIT GRADIENT

For an over-threshold candidate token, the exact forward KL and the pointwise-clipped forward KL produce gradients on that token’s logit with opposite signs. Exact KL gives a negative gradient, whereas the clipped objective gives a positive gradient. For a candidate token that is not overthreshold, the two gradients again have opposite signs exactly when the student’s probability exceeds the teacher’s but stays below the teacher’s probability divided by the teacher’s total probability on the tokens that are not over-threshold: exact KL then gives a positive gradient, whereas the clipped objective gives a negative one.

Fix one response position and suppress the index t. The teacher is frozen, so p and its support S do not depend on the student’s logits. We use $d _ { i } = p _ { i } \log ( p _ { i } / q _ { i } )$ from Equation 2. Let $z _ { i } = b _ { i } / T$ be the temperature-scaled student logit, so that $q _ { i } = \exp ( z _ { i } ) / \sum _ { j \in \mathcal { S } } \exp ( z _ { j } )$ . We call the derivative with respect to $z _ { i }$ as its logit gradient for candidate token i. Define

$$
A = \{ j \in \mathcal { S } : d _ { j } \leq \tau \} , \qquad C = \mathcal { S } \setminus A , \qquad P _ { A } = \sum _ { j \in A } p _ { j } .
$$

Here, A and $C$ are the active and clipped candidate token sets, and $P _ { A }$ is the teacher’s total probability mass on the active set. Equation 3 becomes

$$
D _ { \mathrm { c l i p } } = \sum _ { j \in A } p _ { j } \log \frac { p _ { j } } { q _ { j } } + | C | \tau .\tag{4}
$$

Proposition 1 (Logit gradient reversal on clipped candidate tokens). For any threshold $\tau > 0$ and any student distribution with $d _ { j } \neq \tau$ for every $j \in \mathcal S$ , the gradient of $D _ { \mathrm { c l i p } }$ with respect to $z _ { i }$ is

$$
{ \frac { \partial D _ { \mathrm { c l i p } } } { \partial z _ { i } } } = P _ { A } q _ { i } - p _ { i } { \bf 1 } \{ i \in A \} .\tag{5}
$$

Every clipped candidate token $i \in C$ satisfies

$$
\frac { \partial D _ { \mathrm { F K L } } } { \partial z _ { i } } = q _ { i } - p _ { i } < 0 , \qquad \frac { \partial D _ { \mathrm { c l i p } } } { \partial z _ { i } } = P _ { A } q _ { i } > 0 .\tag{6}
$$

Appendix A gives the proof. Both inequalities in Equation 6 are strict, and clipping is what makes their gradient signs differ. For a clipped candidate token $i \in C , d _ { i } > \tau > 0$ implies $p _ { i } > q _ { i }$ , so the exact KL logit gradient $q _ { i } - p _ { i }$ is negative. Pointwise clipping instead caps the term $d _ { j }$ of every $j \in C$ at the constant τ, which removes the gradient term $- p _ { i }$ in Equation 5. The remaining gradient $P _ { A } q _ { i }$ is positive because $q _ { i } > 0$ under softmax and $P _ { A } > 0$ . Since p and q are both normalized over the same support, there exists some $j$ with $q _ { j } \geq p _ { j }$ . Hence $d _ { j } \le 0 < \tau , \mathrm { s o } j \in { \cal A }$ , and therefore $P _ { A } \ge p _ { j } > 0 .$ , since Equation 1 gives $p _ { j } > 0$

Clipping also changes the logit gradient on the active candidate tokens when the student probability lies within a bounded range above the teacher’s. When the clipped set is nonempty, $\bar { P _ { A } } < 1$ , so $p _ { i } / P _ { A } > p _ { i }$ , for $i \in A$ . By equation 5, if the student’s probability in the active set satisfies

$$
p _ { i } < q _ { i } < p _ { i } / P _ { A } ,
$$

the exact KL gradient $q _ { i } - p _ { i }$ is positive while the clipped gradient $P _ { A } q _ { i } - p _ { i } < 0$ is negative.

## 4.2 CLIPPED OBJECTIVE MOVES PROBABILITY FROM CLIPPED TO ACTIVE TOKENS

Proposition 1 shows that, for a clipped candidate token, which already has less student than the teacher probability, the logit gradient of the clipped objective points toward lowering its logit, whereas that of exact KL points toward raising it. This reversal does not by itself determine whether the token’s probability is still restored toward the teacher’s, as under exact KL, because softmax probabilities depend on all logits. We therefore consider the clipped candidate tokens jointly and ask what happens at the minimizer of the clipped objective at a fixed response position: does an optimized student favor restoring the probability toward the teacher’s, or reduced even further?

At a fixed token position, for fixed nonempty active and clipped sets A and $C ,$ we consider the constrained problem:

$$
\begin{array} { l l } { \displaystyle \operatorname* { m i n i m i z e } \ } & { D _ { \mathrm { c l i p } } ( q ) = \displaystyle \sum _ { j \in \mathcal { S } } \operatorname* { m i n } ( d _ { j } , \tau ) } \\ { \mathrm { s u b j e c t ~ t o } } & { q _ { i } \geq 0 \ \mathrm { f o r ~ e v e r y } \ i \in \mathcal { S } , \quad \displaystyle \sum _ { j \in \mathcal { S } } q _ { j } = 1 , } \\ & { d _ { i } \leq \tau \ \mathrm { f o r ~ e v e r y } \ i \in A , \quad \quad d _ { i } > \tau \ \mathrm { f o r ~ e v e r y } \ i \in C . } \end{array}
$$

The feasible set of student distributions, denoted the region $R ,$ is thefixed clipping region of $( A , C )$ Unlike in Section 4.1, where $q$ is a softmax of finite logits and every $q _ { i } > 0$ , we allow $q _ { i } = 0$ so that a minimizer exists. For such a candidate token $i , d _ { i } = + \infty$ , the capped term is still $\tau ,$ , so it stays clipped. Everywhere in R, $D _ { \mathrm { c l i p } }$ is given by Equation 4 with these fixed sets.

Proposition 2 (Fixed-region minimizer of the clipped objective). The clipped objective has a unique minimizer over the fixed clipping region R,

$$
\begin{array} { r } { q _ { i } ^ { \star } = \displaystyle \int _ { 0 , } ^ { p _ { i } / P _ { A } , i \in { \cal A } , } \qquad D _ { \mathrm { c l i p } } ( q ^ { \star } ) = P _ { A } \log P _ { A } + | C | \tau . } \end{array}\tag{7}
$$

Let $\begin{array} { r } { Q _ { C } = \sum _ { j \in C } q _ { j } } \end{array}$ be the student’s total probability mass on the clipped set. For every value of $Q _ { C }$ attained in $R ,$ if its value is further added to the constraints without inconsistencies with the other constraints, the minimum of $D _ { \mathrm { c l i p } }$ over the distributions of $q \mathrm { ' s }$ in $R ( Q _ { C } )$ with that value is strictly increasing in $\mathit { Q } _ { \mathit { C } } \colon$

$$
D _ { \mathrm { c l i p } } ^ { \star } ( Q _ { C } ) = P _ { A } \log P _ { A } - P _ { A } \log ( 1 - Q _ { C } ) + | C | \tau , \qquad \frac { d D _ { \mathrm { c l i p } } ^ { \star } } { d Q _ { C } } = \frac { P _ { A } } { 1 - Q _ { C } } > 0 .\tag{8}
$$

Appendix B gives the proof. At the minimizer $q ^ { \star }$ of the clipped objective, the clipped candidate tokens’ total probability is not restored toward the teacher’s total $1 - \bar { P _ { A } }$ on these tokens but reduced further, to zero. All probability then lies on $A ,$ and the student’s probability on each active candidate token equals the teacher’s probability $p _ { i }$ multiplied by the same factor $1 / \dot { P _ { A } } > 1$ . Clipping therefore does more than limit the influence of dominant terms: inside a fixed clipping region, its minimizer assigns zero student probability to clipped candidate tokens, even though they already have less student probability than teacher probability, and gives each active candidate token more student probability than the teacher probability.

A softmax of finite logits, however, never attains $q ^ { \star }$ , because it gives every clipped token positive probability. Hence $Q _ { C } > 0$ , and $D _ { \mathrm { c l i p } } ( q ^ { \star } )$ is an unattained infimum. For such a student, Equation 8 characterizes the minimum attainable objective value for each $Q _ { C }$ . The lower the student’s total probability on the clipped candidate tokens, the closer the objective can come to this infimum, and restoring $Q _ { C }$ toward the teacher’s $1 - P _ { A }$ only moves this minimum attainable value farther from it.

## 4.3 CLIPPED UPDATES TURN STARTED REPETITION INTO PERSISTENT LOOPS

In matched runs that differ only in whether the objective is clipped, clipped training produces substantially more terminal loops than exact KL training (Figure 1; Table 3). We defined the repetitionrelated terminologies in Table 2 and use them throughout the section. The preceding two sections give a candidate cause: the clipped objective reverses the logit gradient of exact KL on every clipped token and on active tokens within a bounded range above the teacher’s probability. Within a fixed clipping region, its minimizer sets the clipped tokens’ probability to zero and gives each active token more probability than the teacher, instead of correcting the student toward the teacher. If, in training, the teacher’s exit tokens are clipped and the copy token lies in this range, the clipped objective would push the student to leave a repetition less often and to continue it more often than the teacher, so repetitions and terminal loops would be more than in the exact KL twin, although the size of this effect may differ across families. Both results describe one response position with a given clipped set, whereas in training the clipped set changes across sampled responses and positions, so whether the student’s probabilities move this way must be measured. We compare student and teacher exit mass at the first copy, which every repetition passes through, and the copy token’s logit gradient under both objectives before maturity. These comparisons can support the logit gradient reversal as the path from clipping to loops but cannot establish that path.

![](images/b1e92befb330af8e6ea53a2db14f6517dd43af758f564c6af1d34ec2b67bace4.jpg)  
Figure 1: Loop outcome over training, one trajectory per run. Each point is the number of the 30 rollouts sampled at that update that end in a terminal loop. Clipped runs in red, unclipped twins in blue. Shaded windows are three phases: early (updates 1–25, gray), ignition (F 51–75, M 76–100, K 151–175, L 101–125, orange), and late (updates 176–200, pink).

Table 2: Definitions used to characterize token repetition and analyze its continuation. The upper block defines periodic segments, repetitions, mature repetitions, and terminal loops. The lower block defines the first copy, copy token, and exit mass.
<table><tr><td>Term</td><td>Definition</td></tr><tr><td>Periodic segment</td><td>A maximal contiguous sequence  $x _ { s } , \ldots , x _ { e } ,$  where s and e are its start and end positions, with period l and length  $L = e - s + 1 > \ell . \ L$  satisfies  $x _ { t } ~ = ~ x _ { t - \ell }$  for  $s + \ell \leq t \leq e .$ </td></tr><tr><td>Repetition</td><td>A periodic segment with at least two complete period  $\lfloor L / l \rfloor \geq 2 .$ </td></tr><tr><td>Mature repetition</td><td>A repetition  $\lfloor { Z } / l \rfloor \ge 3$  with  $L \geq 3 0$  tokens. Mature repetition that reaches the last response token.</td></tr><tr><td>Terminal loop Copy token</td><td>At a position  $t \geq s + \ell$  while the periodic continuation remains, the token  $c _ { t } = x _ { t - \ell }$ </td></tr><tr><td></td><td>that continues the period. Its teacher and student probabilities are  $p _ { t , \mathrm { c o p y } }$  and qt,copy, respectively. Copy token belong to the second completed occurrence of the period, at positions</td></tr><tr><td>First copy</td><td> $s + \ell , \dots , s + 2 \ell - 1$ </td></tr><tr><td>Exit mass</td><td>At position t, the probability assigned to tokens other than the copy token:  $1 - p _ { t , \mathrm { c o p y } }$  for the teacher and  $1 - q _ { t , \mathrm { c o p y } }$  for the student.</td></tr></table>

Outcome. We define three 25-update phases: an early phase at the start of training, an ignition window at each clipped run’s loop onset, and a late phase at the end of training. In the early phase, terminal loops are nearly absent from every run. The exact KL runs remain there for the whole run, never exceeding 2 of 30 rollouts at any update. The clipped runs do not. Each run has a familyspecific onset after which terminal loops recur, and over the full run it produces 25 to 450 times as many of them as its twin. The result is shown in Figure 1. The thinking-mode configuration sets both the scale and the ignition time.

Trend. Table 3 follows repetitions $\left( N \geq 2 \right)$ through three phases. In the early phase the two runs of every pair are indistinguishable except run L. They generate repetition and mature repetition at the same rate. From ignition on, two amplifications hold. First, a repetition grows into a mature repetition $( N \geq 3$ and $L \ge 3 0 )$ more often under clipping than in the twin. The count N(mat. | $N \geq 2 )$ is 8.6 to over 400 times the twin’s, and the rate P(mat. | $N \geq 2 )$ 4.5 to over 150 times. Second, far more repetitions end as terminal loops: 26 to 381 per phase in the clipped runs, against at most two in any twin.

Table 3: Progression of repetition by training phase (period 2–50). Rows: $n ( N \geq 2 )$ , the number of repetitions; N(mat. | $\bar { N } \geq 2 )$ , the mature repetitions among repetitions; $\smash { P ( \mathrm { m a t . } \mid N \geq 2 ) }$ the proportion of repetitions that mature; N(term. | mat.), the terminal loops among mature repetition. Each cell reads early → ignition → late, the three phases of Figure 1, 750 rollouts each.
<table><tr><td>Quantity</td><td>Arm</td><td>F</td><td>M</td><td>K</td><td>L</td></tr><tr><td rowspan="2"> $n ( N \geq 2 )$ </td><td>clipped</td><td>11954954→2814</td><td>3852→2666→1304</td><td>1163→3757→4024</td><td>5952→5704→6374</td></tr><tr><td>twin</td><td>1237→2395→3154</td><td>3288→975→1496</td><td>1130→1042→1043</td><td>3991→2642→3121</td></tr><tr><td rowspan="2"> $N ( \mathrm { m a t . } \mid N \geq 2 )$ </td><td>clipped</td><td>1→790→749</td><td>6409→69</td><td>4→170→176</td><td>16→195→328</td></tr><tr><td>twin</td><td>5→3→7</td><td>16→1→8</td><td>64→8</td><td>7→20→9</td></tr><tr><td rowspan="2">P(mat. | N ≥ 2)</td><td>clipped</td><td>.001→.159→.266</td><td>.002→.153→.053</td><td>.003→.045→.044</td><td>.003→.034→.051</td></tr><tr><td>twin</td><td>.004→.001→.002</td><td>.005→.001→.005</td><td>.005→.004→.008</td><td>.002→.008→.003</td></tr><tr><td rowspan="2">N(term. | mat.)</td><td>clipped</td><td>0→307→381</td><td>1→206→26</td><td>0→109→155</td><td>1→30→35</td></tr><tr><td>twin</td><td>0→0→1</td><td>6→0→1</td><td>0→1→2</td><td>0→1→0</td></tr></table>

Family-specific loop patterns. With student thinking off and teacher thinking on (F), the student writes answer-mode derivations that the teacher scores where its own thinking block would begin. The student chains the repeated $\ " \Rightarrow \ "$ where the teacher prefers \text. This two-token cycle supplies most mature repetitions at ignition, and by the late phase all of them are terminal loops. With both modes off (K), student and teacher share the answer register, and most of its terminal loops are answer-closing formatting such as nested “\boxed{”. The teacher prefers a content command such as $\backslash \mathtt { t e x t  o r } \backslash \mathtt { f r a c }$ . With both modes on (L), most mature repetitions lie inside the thinking block and repeat an equation or a reasoning frame such as “Let me compute:”, where the teacher’s preferred alternative varies, most often a space, an opening parenthesis, or a different word such as “take”. Most of its mature repetitions exit rather than become terminal. With student thinking on and teacher thinking off (M), the teacher’s prompt contains a closed, empty thinking block, so the teacher scores the student’s thinking as answer text, and both runs stop emitting the closing tag. At ignition, M’s terminal loops repeat an answer-closing mark such as a check mark inside the unclosed thinking block, where the teacher prefers the end-of-response token <|im end|>.

Exit mass at the first copy. We dive deep into the training signal in each runs and how clipped objective applied to it. We examine whether clipping suppresses exit mass at the first copy of a repetition, terms defined in Table 2. Let $\mathcal { F } _ { u }$ denotes the window containing first-copy positions of all detected repetitions with periods $2 \leq \ell \leq 5 0$ across responses generated at update u. For each nonoverlapping five-update block B, we compute the student/teacher exit-mass ratio by summing each distribution’s exit mass over these positions:

$$
R _ { B } = \frac { \sum _ { u \in B } \sum _ { t \in \mathcal { F } _ { u } } ( 1 - q _ { u , t , \mathrm { c o p y } } ) } { \sum _ { u \in B } \sum _ { t \in \mathcal { F } _ { u } } ( 1 - p _ { u , t , \mathrm { c o p y } } ) } .
$$

We then measure the clipped ratio of the teacher’s exit mass that lies on clipped tokens. For the unclipped twins, we report the corresponding fraction on over-threshold tokens (Figure 2a). Pooled over the early phase, the student/teacher exit-mass ratio is .79 to 1.04. Under exact KL, the ratio is close to 1 across the whole run, even though 33 to 45 percent of the teacher’s exit mass still lies on over-threshold tokens. Under clipping, however, the exit mass ratio falls to .41 to .70 in the late phase, while 71 to 84 percent of the teacher’s exit mass is clipped. Thus, at the first copy, the clipped student places less probability than its teacher on leaving the repetition, and most of the teacher’s exit mass lies on the tokens whose logit gradient Proposition 1 reverses.

Copy token probability before maturity. We next examine the positions a repetition passes through before it becomes mature: how often the student places more probability than the teacher on the copy token, and whether each objective corrects this excess.

For each 25-update phase $\phi ,$ let $\mathcal { G } _ { \phi }$ denote the window of copied positions after the first copy and before maturity or an observed stop. We record the fraction at which the student assigns greater probability to the copy token than the teacher:

$$
R _ { \phi } = \frac { \sum _ { t \in \mathcal { G } _ { \phi } } \mathbf { 1 } \{ q _ { t , \mathrm { c o p y } } > p _ { t , \mathrm { c o p y } } \} } { | \mathcal { G } _ { \phi } | } ,
$$

(a)  
![](images/0e4de65a480e224796cf014c3e2ce0daceb987e519b326751d16e495bb7cb23f.jpg)

(b)  
![](images/f734fd278b63a5c0b0f8598aa07aad0a8a17266a2a371718ed1772493ea40ed3.jpg)  
(c)

![](images/a54f13e65638dfe849b5054e7c220d789d4a3158ba8203d384a02fc19f68a90c.jpg)  
clipped run (a) clipped ratio of teacher exit mass (b) lowered by the clipped objective unclipped twin (a) over-threshold ratio, twin (b) lowered by the exact kl (a) student = teacher (b) raised by the clipped objective (c) exact KL at the clipped run's positions

Figure 2: The copy token inside repetitions (period 2–50), with the student’s probabilities taken before the sampler’s truncation. (a) Student/teacher exit-mass ratio, one point per block of five updates $( 1 - 5 , . . . , 1 9 6 { - 2 0 0 } )$ . Lines without markers give the ratio of the teacher’s exit mass on clipped tokens, or on over-threshold tokens in the twins (right axis). (b) Copied positions between the first copy and maturity: fraction with $q _ { \mathrm { c o p y } } > p _ { \mathrm { c o p y } }$ . Dark bar: $q _ { \mathrm { c o p y } } < p _ { \mathrm { c o p y } } / P _ { A }$ , where the logit gradient of exact KL points toward lowering the copy logit and that of the clipped objective toward raising it. Light bar: both point toward lowering it. (c) The same copied positions, plus observed stopping positions before maturity: mean $- \partial D / \partial z _ { \mathrm { c o p y } }$ . Black and red: exact KL (hypothetical) and clipped KL, respectively, evaluated on the same positions and probabilities from the clipped run. Positive values point toward raising the copy logit. Phases as in Table 3.

where 1{·} is the indicator function and counts are pooled across repetitions and updates within the phase. The result is in Figure 2b. In the early phase, both runs of F, M, and L have $q _ { \mathrm { c o p y } } > p _ { \mathrm { c o p y } }$ at 29 to 45 percent of these positions. After ignition, this fraction falls to 4 to 6 percent in the exact KL runs but 32 to 77 percent in the clipped runs. In the clipped runs, $q _ { \mathrm { c o p y } }$ lies specifically between $p _ { \mathrm { c o p y } }$ and $p _ { \mathrm { c o p y } } / P _ { A }$ at 4 to 34 percent of positions. The dark part of each bar in Figure 2b shows the fraction of positions at which the copy token’s logit gradient is reversed: the clipped objective points toward raising the copy token logit where exact KL points toward lowering it.

We then measure the gradient of each objective $D \in \{ D _ { \mathrm { F K L } } , D _ { \mathrm { c l i p } } \}$ on the copy token. For each phase $\phi ,$ we record the mean ${ \mathrm { o f } } - \partial D / \partial z _ { \mathrm { c o p y } }$ over the window $\mathcal { G } _ { \phi }$ plus the observed stop positions, pooled across repetitions and updates. We report the negative of the gradient so that positive values point toward raising the copy token logit. We calculate $\partial D _ { \mathrm { F K L } } / \partial z _ { \mathrm { c o p y } } = q _ { \mathrm { c o p y } } - p _ { \mathrm { c o p y } }$ and $\partial D _ { \mathrm { c l i p } } / \partial z _ { \mathrm { c o p y } }$ , given by Equation 5. We evaluate both objectives on the positions and probabilities of the clipped run only, so that any difference between the two means comes from the objective alone. Figure 2c shows the result: the net direction of each objective’s logit gradient on the copy token over all positions of the window. In the early phase, both values are nearly zero in every family. In the ignition and late phases, the mean $\mathrm { o f } - { \partial D _ { \mathrm { F K L } } } / { \partial z _ { \mathrm { c o p y } } }$ is negative, pointing toward lowering the copy token logit, whereas that of $D _ { \mathrm { c l i p } }$ is slightly positive in every family, pointing toward raising it. At the positions that decide whether a repetition becomes mature, exact KL would thus correct the student’s excess probability on the copy token, and the clipped objective removes this correction.

![](images/1f527d5f79fa8a22578ffb5fcba0d5b5173a284fbf26771c4f74ca4e7df3d1c7.jpg)  
Figure 3: AIME 2025 across training. (a) Avg@12, the mean correct ratio over 12 samples for each of 30 problems at the checkpoint saved after each 25 updates. (b) The ratio of the 360 responses per checkpoint that end in a terminal loop. Solid curves with circles are the clipped runs, dashed curves with squares their unclipped twins. Each trained curve is one realized trajectory. The black arrow on each vertical axis marks the base model, the fixed starting checkpoint averaged over eight evaluation-sampling seeds.

## 5 HELD-OUT EVALUATION

We evaluate every checkpoint, saved at 25-update intervals, on the AIME 2025 problems. Figure 3 reports accuracy and terminal loop rate at saved checkpoints. It tests whether clipping increases looping and whether the additional loops accompany accuracy losses relative to the unclipped twin.

No run significantly outperforms the base model (Figure 3a). K show only minimal accuracy changes, whereas F, L, and M degrade substantially, and most clipped runs perform worse than their unclipped twins. M is the exception: both of its run collapse, and its twin reads zero only because it stops closing the thinking tag, so the strict scorer finds no answer.

Terminal loop rate separates the runs more sharply (Figure 3b). Every clipped run ends in a terminal loop far more often than the base model, while every unclipped twin stays near the base rate. Looped responses fail to reach a final answer and are therefore scored as incorrect. Accuracy losses in the clipped runs follow the same ordering as their loop rates: smallest in K, intermediate in F, and largest in L and M.

With 12 samples per problem, no clipped endpoint has lower pass@12 than its twin on either benchmark: every problem solved by the twin is also solved at least once by the clipped arm. We interpret this pattern as evidence that clipped training reduces per-sample reliability. However, the unclipped F and L twins remain below the fixed-base mean despite rarely looping. Thus, clipping-induced looping contributes to the observed degradation, but does not explain all accuracy loss relative to the base model.

Appendix D gives avg@12, terminal loop rate, and pass@12 at every checkpoint on AIME 2024 and AIME 2025 benchmarks.

## 6 CONCLUSION

Pointwise clipping was introduced to stabilize training against dominant stylistic tokens, but it also changes the direction in which forward kl corrects the student. In training this change appears as repetitions that continue to the end of the response, a loss of training stability. Because this failure develops over updates, studies that train with the pointwise-clipped objective could report accuracy and a text degeneration measure, across checkpoints, in addition to their best checkpoint. We hope that understanding these clipping dynamics helps make on-policy self-distillation stable enough for its gains to persist over longer training.

## REFERENCES

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from selfgenerated mistakes. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=3zKtaqxLhW.

Yunmeng Chen, Kunyu Wang, Peihan Li, Yi Wang, Shuyin Xia, Yi Liu, Xinyong Cheng, Dehui Wang, Xiangyong Zhai, Yanxing Liu, et al. Scope-opsd: Fisher-conditioned privileged subspaces for on-policy self-distillation. arXiv preprint arXiv:2609.12579, 2026a.

Yutong Chen, Guangfu Guo, Zhichao Xu, and Kunpeng Liu. Dualopsd: Adaptive privileged teachers for on-policy self-distillation. arXiv preprint arXiv:2608.26019, 2026b.

Yangyang Feng, Zhuoyan Feng, and Junlan Chen. Past: Privileged adaptation from complete student trajectories for on-policy self-distillation. arXiv preprint arXiv:2608.08726, 2026.

Siyi Gu, Jialin Chen, Sophia Zhou, Arman Cohan, and Rex Ying. Rethinking reward supervision: Rubric-conditioned self-distillation. arXiv preprint arXiv:2606.19327, 2026.

Etash Guha, Ryan Marten, Sedrick Keh, Negin Raoof, Georgios Smyrnis, Hritik Bansal, Marianna Nezhurina, Jean Mercat, Trung Vu, Zayne Sprague, et al. Openthoughts: Data recipes for reasoning models. In International Conference on Learning Representations, volume 2026, pp. 108059–108130, 2026.

ZhiYan Hou, Xinyu Tang, Hongyan An, Jianjin Zhang, Weizhen Wang, Yunyun Han, Gengsheng Li, Xiangzhao Hao, Haiyun Guo, Wenbin Hu, et al. Dash: Divergence-adaptive supervision horizons for on-policy self-distillation of reasoning models. arXiv preprint arXiv:2608.06243, 2026.

Jonas Hubotter, Frederike L¨ ubeck, Lejs Behric, Anton Baumann, Marco Bagatella, Daniel Marta,¨ Ido Hakimi, Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, et al. Reinforcement learn ing via self-distillation. arXiv preprint arXiv:2601.20802, 2026.

Yuki Ichihara, Naoto Iwase, Mohammad Atif Quamar, and Junpei Komiyama. Privileged solutions or context-induced teacher behavior? dissecting on-policy self-distillation. arXiv preprint arXiv:2608.09228, 2026.

Hynek Kydl´ıcek. Math-Verify: Math Verification Library. URL ˇ https://github.com/ huggingface/math-verify.

Yijiang Li, Bingyang Wang, Yijun Liang, Yunjie Tian, Di Fu, and Nuno Vasconcelos. On-policy self-distillation without any supervision. arXiv preprint arXiv:2608.06296, 2026.

Zihan Liang, Yufei Ma, Ben Chen, Zhipeng Qian, Xuxin Zhang, Huangyu Dai, and Lingtao Mao. Search-e1: Self-distillation drives self-evolution in search-augmented reasoning. arXiv preprint arXiv:2605.22511, 2026.

Xiaogeng Liu, Xinyan Wang, Yingzi Ma, Yechao Zhang, and Chaowei Xiao. When are teacher tokens reliable? position-weighted on-policy self-distillation for reasoning. arXiv preprint arXiv:2605.21606, 2026.

Kevin Lu and Thinking Machines Lab. On-policy distillation. Thinking Machines Lab: Connectionism, 2025. doi: 10.64434/tml.20251026. https://thinkingmachines.ai/blog/on-policy-distillation.

Leyi Pan, Shuchang Tao, Yunpeng Zhai, Lingzhe Zhang, Zhaoyang Liu, Bolin Ding, Aiwei Liu, and Lijie Wen. Rlcsd: Reinforcement learning with contrastive on-policy self-distillation. arXiv preprint arXiv:2606.11709, 2026.

Idan Shenfeld, Mehul Damani, Jonas Hubotter, and Pulkit Agrawal. Self-distillation enables con-¨ tinual learning. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/forum?id=qA6FgH0nnZ.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Samyak Shrestha and Alexander Tessier. Rethinking privileged information in on-policy selfdistillation. arXiv preprint arXiv:2608.18271, 2026.

Zhiquan Tan and Yinrong Hong. Paint: Partial-solution adaptive interpolated training for selfdistilled reasoners. arXiv preprint arXiv:2604.26573, 2026a.

Zhiquan Tan and Yinrong Hong. Self-supervised on-policy distillation for reasoning language models. arXiv preprint arXiv:2605.17497, 2026b.

Jiaxuan Wang, Xuan Ouyang, Zhiyu Chen, Yulan Hu, Zheng Pan, Xin Li, and Lan-Zhe Guo. Trace: Distilling where it matters via token-routed self on-policy alignment. arXiv preprint arXiv:2605.10194, 2026.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Yuxiao Yang, Xiaoyun Wang, and Weitong Zhang. Ogls-sd: On-policy self-distillation with outcome-guided logit steering for llm reasoning. arXiv preprint arXiv:2605.12400, 2026.

XiuYu Zhang, Wei Chow, Junfeng Fang, Zhenkai Liang, and Tat-Seng Chua. What does privileged information add to on-policy self-distillation? arXiv preprint arXiv:2609.20612, 2026.

Yifan Zhang and Team Math-AI. American invitational mathematics examination (aime) 2024, 2024.

Yifan Zhang and Team Math-AI. American invitational mathematics examination (aime) 2025, 2025.

Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-distilled reasoner: On-policy self-distillation for large language models. In Forty-third International Conference on Machine Learning, 2026a. URL https://openreview.net/ forum?id=Jpxfof0EaS.

Xuyang Zhao, Liting Zhang, Zichen Xu, Zhihu Wang, Xu Caiyue, Shiwan Zhao, and Qicheng Li. Is more privileged information better? From solution traces to problem-solving structure in selfdistilled reasoning. arXiv preprint arXiv:2608.01589, 2026b. URL https://arxiv.org/ abs/2608.01589.

Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody H Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E Gonzalez, et al. Sglang: Efficient execution of structured language model programs. Advances in neural information processing systems, 37:62557–62583, 2024.

## A PROOF OF PROPOSITION 1.

Fix a response position and suppress its index t. Hold the finite support S and the teacher distribution p fixed. By Equation 1, ${ \bf \nabla } p _ { j } , q _ { j } > 0$ for every $j \in \mathcal S$ , and both distributions sum to one on S. For the temperature-scaled student logits $z _ { j } = b _ { j } / \dot { T }$ , write

$$
Z = \sum _ { j \in \mathcal { S } } \exp ( z _ { j } ) , \qquad q _ { j } = \frac { \exp ( z _ { j } ) } { Z } .
$$

Consider any logit vector $z ^ { \circ }$ satisfying $d _ { j } \neq \tau$ for all $j \in { \mathcal { S } } ,$ , and let A and C be its active and clipped sets as defined in Section 4.1. Each $d _ { j } = p _ { j } \log ( p _ { j } / q _ { j } )$ is smooth in z, since $p _ { j }$ is fixed and $q _ { j }$ is a positive smooth function of z. Because S is finite and no $d _ { j }$ equals $\tau \mathrm { a t } z ^ { \circ } .$ , there is an open neighborhood of $z ^ { \circ }$ on which A and $C ,$ and hence $\begin{array} { r } { P _ { A } = \sum _ { j \in A } \overset { \vartriangle } { p _ { j } } } \end{array}$ , remain unchanged. On this neighborhood, Equation 4 gives

$$
D _ { \mathrm { c l i p } } = \sum _ { j \in A } d _ { j } + | C | \tau ,
$$

with A and $C$ fixed at their values at $z ^ { \circ }$ . This expression is smooth in $z ,$ so $D _ { \mathrm { c l i p } }$ is differentiable on the neighborhood and its gradient is obtained by differentiating the displayed expression.

For any $i , j \in S$ , the softmax identity log $q _ { j } = z _ { j } - \log Z$ gives

$$
d _ { j } = p _ { j } \log { p _ { j } } - p _ { j } z _ { j } + p _ { j } \log { Z } .
$$

Because p is fixed and

$$
{ \frac { \partial z _ { j } } { \partial z _ { i } } } = \mathbf { 1 } \{ j = i \} , \qquad { \frac { \partial \log Z } { \partial z _ { i } } } = { \frac { 1 } { Z } } { \frac { \partial Z } { \partial z _ { i } } } = { \frac { \exp ( z _ { i } ) } { Z } } = q _ { i } ,
$$

it follows that

$$
\frac { \partial d _ { j } } { \partial z _ { i } } = - p _ { j } { \bf 1 } \{ j = i \} + p _ { j } q _ { i } .\tag{9}
$$

Since $\begin{array} { r } { D _ { \mathrm { F K L } } = \sum _ { j \in \mathcal { S } } d _ { j } } \end{array}$ by Equation 3, summing Equation 9 over S gives

$$
\begin{array} { r } { \frac { \partial D _ { \mathrm { F K L } } } { \partial z _ { i } } = \displaystyle \sum _ { j \in \cal S } \left( - p _ { j } { \bf 1 } \{ j = i \} + p _ { j } q _ { i } \right) } \\ { = - p _ { i } + q _ { i } \displaystyle \sum _ { j \in \cal S } p _ { j } = q _ { i } - p _ { i } , } \end{array}
$$

at every logit vector, including points on the clipping boundaries. For the clipped objective, $| C | \tau$ has zero derivative on the neighborhood under consideration. Summing only over A therefore gives

$$
\begin{array} { l } { \displaystyle { \frac { \partial D _ { \mathrm { c l i p } } } { \partial z _ { i } } = \sum _ { j \in A } \bigl ( - p _ { j } { \bf 1 } \{ j = i \} + p _ { j } q _ { i } \bigr ) } } \\ { \displaystyle { } } \\ { \displaystyle { } } \\ { \displaystyle { } } \\ { { } = { - p _ { i } { \bf 1 } \{ i \in A \} + q _ { i } \sum _ { j \in A } p _ { j } } } \\ { \displaystyle { } } \\ { \displaystyle { } } \end{array}\tag{10}
$$

This establishes Equation 5.

Now let $i \in C .$ Then $d _ { i } > \tau > 0$ , and $p _ { i } > 0$ implies

$$
\log { \frac { p _ { i } } { q _ { i } } } > 0 , \qquad \mathrm { h e n c e } \qquad p _ { i } > q _ { i } .
$$

Consequently, $\partial D _ { \mathrm { F K L } } / \partial z _ { i } = q _ { i } - p _ { i } < 0$ . Since $i \not \in A$ , Equation 10 reduces to $\partial D _ { \mathrm { c l i p } } / \partial z _ { i } = P _ { A } q _ { i }$ To establish strict positivity, first note that $q _ { i } > 0$ . Moreover, $\begin{array} { r } { \sum _ { j \in \mathcal { S } } q _ { j } = \sum _ { j \in \mathcal { S } } \overset { \cdot } { p } _ { j } } \end{array}$ implies the existence of a token j with $q _ { j } \geq p _ { j }$ . For this token,

$$
d _ { j } = p _ { j } \log \frac { p _ { j } } { q _ { j } } \leq 0 < \tau ,
$$

so $j \in A$ and $P _ { A } \ge p _ { j } > 0$ . Therefore,

$$
\frac { \partial D _ { \mathrm { F K L } } } { \partial z _ { i } } = q _ { i } - p _ { i } < 0 , \qquad \frac { \partial D _ { \mathrm { c l i p } } } { \partial z _ { i } } = P _ { A } q _ { i } > 0 ,
$$

which proves Equation 6. Since $z ^ { \circ }$ was arbitrary away from the clipping boundaries, the result holds at every point specified in the proposition. □

A simpler example of updating dynamics To understand the direction of the updates on the logits as suggested by the previous proposition, the example below can give some intuitions. In the most general case, decreasing certain logits on index i does not automatically mean to decrease the corresponding probability $q _ { i } { \mathrm { - b u t } }$ in the case when other z values are fixed, this holds true. We can construct a continuous-time dynamical system to mimic the optimization process on the objective functions of the two model versions, defining the parameter updates as the negative gradient of the loss with respect to the logits $\begin{array} { r } { ( \frac { d z _ { i } } { d t } = - \nabla _ { z _ { i } } D ) } \end{array}$ . For the exact forward kl objective $D _ { F K L }$ , this logit update is defined continuously as:

![](images/ee1ab02892f9ba3774dfc38e7bc8851baec61f0b75aa09bed18d1900e7e48b7d.jpg)  
Figure 4: A simple example of updating direction of $\mathbf { q }$ under different losses(Forward KL v.s. Clipped Loss Updates.

$$
\frac { d z _ { i } } { d t } ( t ) = p _ { i } - q _ { i } ( t )
$$

For the clipped objective, the gradient space is instead split into a piecewise system governed by the local divergence threshold $d _ { j }$

$$
\frac { d z _ { i } } { d t } ( t ) = \left\{ \begin{array} { l l } { p _ { i } - P _ { A } q _ { i } ( t ) , } & { \mathrm { i f } \ d _ { j } \leq \tau } \\ { - P _ { A } q _ { i } ( t ) , } & { \mathrm { i f } \ d _ { j } > \tau } \end{array} \right.
$$

To map these raw logit updates $\textstyle { \frac { d z _ { i } } { d t _ { - } } }$ to the resulting probability trajectories $\frac { d q _ { i } } { d t }$ , we must account for the coupling effect of the softmax function, where the evolution of a single probability $q _ { i }$ depends on the updates to all logits in the vocabulary. However, to isolate the localized forces acting on a single prediction and visualize them within a simplified 2D direction field diagram, we introduce a specific framing assumption where all other logits $z _ { j } , j \neq i$ are temporarily held constant $\begin{array} { r } { ( \frac { d z _ { j } } { d t } = 0 ) } \end{array}$ . Under this isolated assumption, because the softmax function is strictly monotonically increasing with respect to its own logit, $\frac { d q _ { i } } { d t }$ acts as a positive monotonic scaling of $\textstyle { \frac { d z _ { i } } { d t } }$ . This strict sign preservation guarantees that the direction of the probability update exactly mirrors the direction of the logit update, providing the mathematical justification to map the negative gradients directly onto a 2D phase portrait to evaluate their critical points and structural stability. The direction field, null-cline, contour for the clipping region and the diagonal (for the correct alignment p=q) is shown in figure 4.

Applying this localized framework to the exact forward kl divergence (left) demonstrates that the gradient acts as a proportional restorative force across the entire probability space, driving the system toward a single, globally stable equilibrium along the critical boundary where $p _ { i } = q _ { i }$ (the diagonal). Because the directional derivative relies entirely on the unaltered residual, the vector field smoothly and pushes the predictions directly toward perfect calibration. This ensures the parameter space remains entirely free of dead zones, structural barriers, or competing attractors.

Conversely, the clipped objective (right) shatters the global stability of the forward kl. In the active set (unshaded area on bottom-right corner) where the divergence is low $( d _ { j } < \tau )$ , scaling the predicted term by $P _ { A }$ artificially shifts the fixed point away from true alignment, forcing the vectors toward a false attractor along the line $p _ { i } = P _ { A } q _ { i }$ (the blue line with slope that is less steep). When the local divergence exceeds the $\tau$ threshold, the system crosses into the clipped set (green shaded area on the top left), completely erasing the target probability $q _ { i }$ from the derivative calculation. Inside this boundary, the dynamic devolves into pure probability decay. Because the gradient equation yields a negative update $( - P _ { A } q _ { i } )$ , it relentlessly pushes the predictions until 0 is reached, structurally preventing the system from ever recovering true calibration when initialized or driven too far from the target. To sum up, the green shaded area along with the area between the red and blue lines on the right are where the arrows for the clipped objectives are in a different direction as compared with the ones for forward KL. One might observe that if you trace a phase portrait alone the arrows, then if you start with a point in either clipped or active set, the trajectory never goes into the other–this may give some intuition for boundary check for the next proposition as well.

## B PROOF OF PROPOSITION 2

Setup. Fix one response position. Let p be the teacher distribution on the finite candidate set S. It is a softmax output, so $p _ { i } > 0$ for every $i \in S$ , and $\textstyle \sum _ { i \in { \mathcal { S } } } p _ { i } = 1$ . Let $\tau > 0$ . Student distributions are the points of the closed simplex,

$$
q _ { i } \geq 0 \quad ( i \in \mathcal { S } ) , \qquad \sum _ { i \in \mathcal { S } } q _ { i } = 1 ,
$$

so probabilities equal to zero are allowed. We use the convention $p _ { j } \log ( p _ { j } / 0 ) = + \infty$ . The capped term of a zero-probability token is then min $\{ + \infty , \tau \} = \tau .$ , so

$$
D _ { \mathrm { c l i p } } ( q ) = \sum _ { j \in S } \operatorname* { m i n } \Bigl \{ p _ { j } \log \frac { p _ { j } } { q _ { j } } , \tau \Bigr \}
$$

is finite on the whole simplex.

Let A and C be fixed nonempty disjoint sets with $A \cup C = S$ . The fixed clipping region R is the set of student distributions q with

$$
p _ { i } \log \frac { p _ { i } } { q _ { i } } \leq \tau \quad ( i \in A ) , \qquad p _ { j } \log \frac { p _ { j } } { q _ { j } } > \tau \quad ( j \in C ) .
$$

Write $\textstyle P _ { A } = \sum _ { i \in A } p _ { i }$ and $\textstyle Q _ { C } = \sum _ { j \in C } q _ { j }$ . Since A and $C$ are nonempty and every $p _ { i } > 0 .$ , we have $0 < P _ { A } < 1$ . For $q \in R$ , every active term is at most $\tau ,$ so its minimum with τ is the term itself, and every clipped term exceeds $\tau ,$ , so its minimum with τ is τ . Hence, for $q \in R$ (Equation 4),

$$
D _ { \mathrm { c l i p } } ( q ) = \sum _ { i \in A } p _ { i } \log { \frac { p _ { i } } { q _ { i } } } + \sum _ { j \in C } \tau = F ( q _ { A } ) + | C | \tau , \qquad F ( q _ { A } ) = \sum _ { i \in A } p _ { i } \log { \frac { p _ { i } } { q _ { i } } } , \qquad q _ { A } = ( q _ { i } ) _ { i \in A } .
$$

This identity is valid on $R .$ Outside R it fails in general, because a term that exceeds τ is capped.

Claim. The distribution $q ^ { \star }$ with $q _ { i } ^ { \star } = p _ { i } / P _ { A }$ for $i \in A$ and $q _ { j } ^ { \star } = 0$ for $j \in C$ is the unique minimizer of $D _ { \mathrm { c l i p } }$ over R, and $D _ { \mathrm { c l i p } } ( q ^ { \star } ) = P _ { A }$ log $P _ { A } + | C | \tau$ (Equation 7).

Overview. The proof has two steps. Step 1 settles how much mass sits on the clipped set: any such mass can be moved onto an active token without leaving R, and doing so strictly lowers the loss, so only distributions with $Q _ { C } = 0$ can compete. Step 2 settles how that mass is split within the active set. Dropping the active-token clipping inequalities leaves a larger, smooth problem: minimize the strictly convex function $F$ over the positive allocations with $\textstyle \sum _ { i \in A } q _ { i } = 1$ . Its unique global minimizer is the Lagrange point $q _ { i } = p _ { i } / P _ { A }$ , which exceeds $p _ { i }$ because $\mathrm { \bar { \it P } } _ { A } < 1$ . The crucial implication is that the unique global minimizer of the larger, smooth problem satisfies the original clipping constraints. It is therefore also the unique minimizer of the original, restricted problem. The larger problem is only an auxiliary device: outside R the formula $F$ is not the clipped objective, and nothing is claimed there.

Membership facts (M1)–(M4). Steps 1 and 2 use four facts about when a token is active or clipped. These are the only places where the threshold enters the proof. Fix a token i and regard its term $p _ { i } \log ( p _ { i } / x )$ as a function of its student probability $x \ge 0$ , with value $+ \infty \operatorname { a t } x = 0$ . For $x > 0$

$$
{ \frac { d } { d x } } \Bigl [ p _ { i } \log { \frac { p _ { i } } { x } } \Bigr ] = { \frac { d } { d x } } \bigl [ p _ { i } \log p _ { i } - p _ { i } \log x \bigr ] = - { \frac { p _ { i } } { x } } < 0 ,
$$

so the term is strictly decreasing in x on $( 0 , \infty )$ , and at $x = p _ { i }$ it equals $p _ { i } \log 1 = 0$

(M1) Every active token has $q _ { i } > 0 . { \mathrm { ~ I f ~ } } q _ { i } = 0 .$ , the term is $+ \infty > \tau ,$ , so token i would be clipped, not active. Consequently $F$ is finite on R.

(M2) An active token that gains probability stays active. If $q _ { i } ^ { \prime } \ge q _ { i } > 0$ , then, because the term is decreasing, $p _ { i } \log ( p _ { i } / q _ { i } ^ { \prime } ) \leq p _ { i } \log ( p _ { i } / q _ { i } ) \leq \tau$

(M3) A clipped token that loses probability stays clipped, including at zero. Let $0 \leq q _ { i } ^ { \prime } \leq q _ { j }$ If $q _ { j } ^ { \prime } \ = \ 0$ , the term is $+ \infty > \tau$ . If $q _ { j } ^ { \prime } \ > \ 0$ , then $q _ { j } > 0$ as well and $p _ { j } \log ( p _ { j } / q _ { j } ^ { \prime } ) \geq$ $p _ { j } \log ( p _ { j } / q _ { j } ) > \tau .$

(M4) A token with $q _ { i } ~ \geq ~ p _ { i }$ is active for every $\tau \ > \ 0 .$ . Because the term is decreasing, $p _ { i } \log ( p _ { i } / q _ { i } ) \leq p _ { i } \log ( p _ { i } / p _ { i } ) = 0 < \tau$

Step 1: mass on the clipped set prevents optimality. Let $q \in R$ with $Q _ { C } > 0$ . Pick any active token $k \in A$ and move all clipped mass onto it:

$$
q _ { k } ^ { \prime } = q _ { k } + Q _ { C } , \qquad q _ { j } ^ { \prime } = 0 ( j \in C ) , \qquad q _ { i } ^ { \prime } = q _ { i } ( i \in A , i \ne k ) .
$$

Normalization. $\begin{array} { r } { \sum _ { i \in \cal S } q _ { i } ^ { \prime } = \sum _ { i \in \cal A , i \ne k } q _ { i } + ( q _ { k } + Q _ { C } ) + 0 = \sum _ { i \in \cal A } q _ { i } + Q _ { C } = 1 . } \end{array}$

Same region. Token k gains probability, so it stays active by (M2). Every clipped token drops to zero, so it stays clipped by (M3). The other active tokens are unchanged. Hence $q ^ { \prime } \in R$ , and the identity $D _ { \mathrm { c l i p } } \dot { = } F \dot { + } | C | \dot { \tau }$ applies to both q and $q ^ { \prime }$

Difference. The constants |C|τ cancel, and so do all active terms with $i \neq k$ . What remains is the k-th term:

$$
\begin{array} { r l } & { D _ { \mathrm { c l i p } } ( q ^ { \prime } ) - D _ { \mathrm { c l i p } } ( q ) = p _ { k } \log \displaystyle \frac { p _ { k } } { q _ { k } + Q _ { C } } - p _ { k } \log \displaystyle \frac { p _ { k } } { q _ { k } } } \\ & { \qquad = p _ { k } \left[ \log q _ { k } - \log ( q _ { k } + Q _ { C } ) \right] = p _ { k } \log \displaystyle \frac { q _ { k } } { q _ { k } + Q _ { C } } . } \end{array}
$$

Here $p _ { k }$ and $q _ { k }$ are the teacher and student probabilities of the single receiving token $k ,$ not the set masses. We have $p _ { k } > 0 , q _ { k } > 0 \mathrm { { b y } ( M 1 ) }$ , and $Q _ { C } > 0 , \mathrm { s o } 0 < \bar { q _ { k } } / ( q _ { k } + Q _ { \bar { C } } ) < 1$ , the logarithm is negative, and

$$
D _ { \mathrm { c l i p } } ( q ^ { \prime } ) < D _ { \mathrm { c l i p } } ( q ) .
$$

Hence every distribution in R with $Q _ { C } > 0$ has strictly larger loss than some distribution in R with $Q _ { C } = 0$

Step 2: optimal allocation on the active set. Throughout this step $p _ { i } > 0$ for every $i ,$ because the teacher is a softmax output, and $q _ { i } > 0$ for every $i \in A \ : \mathrm { b y \ ( M 1 ) }$ . Neither is an extra assumption. Consequently $F$ is finite and smooth wherever it is evaluated, and every Hessian entry $p _ { i } / q _ { i } ^ { 2 }$ below is strictly positive.

(a) The restricted problem. After Step 1 the competitors are the distributions $q \in R$ with $Q _ { C } = 0$ For such $q$ we have $q _ { j } = 0$ on $C , q _ { i } > 0$ on $A , \textstyle \sum _ { i \in A } ^ { \cdot } q _ { i } = 1$ , and $D _ { \mathrm { c l i p } } ( q ) = \bar { F } ( q _ { A } ) + | C | \tau$ . So we must minimize $F$ over

$$
K _ { \tau } = \Bigl \{ q _ { A } \in \mathbb { R } ^ { A } : q _ { i } > 0 , \sum _ { i \in A } q _ { i } = 1 , p _ { i } \log \frac { p _ { i } } { q _ { i } } \leq \tau \ \forall i \in A \Bigr \} .
$$

Conversely, every $q _ { A } \in K _ { \tau }$ , extended by zeros on $C ,$ is a distribution in R with $Q _ { C } = 0$

(b) The relaxed problem. Drop the threshold inequalities and let

$$
K = \Big \{ q _ { A } \in \mathbb { R } ^ { A } : q _ { i } > 0 , \sum _ { i \in A } q _ { i } = 1 \Big \} \ \supseteq \ K _ { \tau } .
$$

We minimize the same formula $F$ over $K$ . On $K \backslash K _ { \tau }$ the formula $F$ is no longer the clipped objective, because there some term exceeds τ and would be capped. The relaxed problem is an auxiliary smooth problem. It enters the argument only through the inclusion $K _ { \tau } \subseteq \bar { K }$ , and nothing is claimed about $D _ { \mathrm { c l i p } }$ outside $R .$

(c) Derivatives ofF. We differentiate on the positive orthant $\{ q _ { A } : q _ { i } > 0 \}$ , which is open, convex, and contains K. Write

$$
F ( q _ { A } ) = \sum _ { i \in A } \bigl [ p _ { i } \log p _ { i } - p _ { i } \log q _ { i } \bigr ] .
$$

Only the i-th summand depends on $q _ { i }$ , and $p _ { i }$ log $p _ { i }$ is a constant, so

$$
{ \frac { \partial F } { \partial q _ { i } } } = { \frac { \partial } { \partial q _ { i } } } \bigl [ p _ { i } \log p _ { i } - p _ { i } \log q _ { i } \bigr ] = 0 - p _ { i } \cdot { \frac { 1 } { q _ { i } } } = - { \frac { p _ { i } } { q _ { i } } } .
$$

For the second derivatives, differentiate $- p _ { i } / q _ { i } = - p _ { i } q _ { i } ^ { - 1 }$ with respect to $q _ { m } . \mathrm { H } m = i $

$$
{ \frac { \partial ^ { 2 } F } { \partial q _ { i } ^ { 2 } } } = { \frac { \partial } { \partial q _ { i } } } \left[ - p _ { i } q _ { i } ^ { - 1 } \right] = - p _ { i } \cdot \left( - 1 \right) q _ { i } ^ { - 2 } = { \frac { p _ { i } } { q _ { i } ^ { 2 } } } .
$$

If m $\neq i ,$ , the expression $- p _ { i } / q _ { i }$ does not depend on $q _ { m }$ , so $\partial ^ { 2 } F / \partial q _ { i } \partial q _ { m } = 0$ . The Hessian is therefore diagonal, $\nabla ^ { 2 } F ( q _ { A } ) = \operatorname { d i a g } \left( p _ { i } / q _ { i } ^ { 2 } \right) _ { i \in { \cal A } }$ , and for every vector $\mathbf { \chi } ) \in \mathbb { R } ^ { A }$

$$
v ^ { \top } \nabla ^ { 2 } F ( q _ { A } ) v = \sum _ { i \in A } \sum _ { m \in A } v _ { i } { \frac { \partial ^ { 2 } F } { \partial q _ { i } \partial q _ { m } } } v _ { m } = \sum _ { i \in A } { \frac { p _ { i } } { q _ { i } ^ { 2 } } } v _ { i } ^ { 2 } .
$$

Every coefficient $p _ { i } / q _ { i } ^ { 2 }$ is strictly positive, because $p _ { i } > 0$ and $q _ { i } > 0$ . If v $\neq 0 ,$ some $v _ { i } \neq 0 ,$ so the sum is strictly positive: the Hessian is positive definite at every point of the orthant.

(d) Strict convexity. A function $f$ on a convex set is strictly convex if $f ( \theta x + ( 1 - \theta ) y ) < \theta f ( x ) +$ $( 1 - \theta ) f ( y )$ for all $x \neq y$ in the set and all $\theta \in ( 0 , 1 )$ . We use two standard facts. (F1) A twice differentiable function on an open convex set whose Hessian is positive definite at every point is strictly convex. (F2) A differentiable strictly convex function lies strictly above each of its tangent planes: $f ( y ) > f ( x ) + \nabla f ( x ) ^ { \top } ( y - x )$ for all $x \neq y$ . By (c) and (F1), $F$ is strictly convex on the orthant, independently of normalization, and hence on its convex subset $K .$

(e) Lagrange point. Introduce a multiplier λ for the normalization constraint:

$$
\mathcal { L } ( q _ { A } , \lambda ) = F ( q _ { A } ) + \lambda \Big ( \sum _ { m \in A } q _ { m } - 1 \Big ) .
$$

Using (c) and $\begin{array} { r } { \frac { \partial } { \partial q _ { i } } \left( \sum _ { m \in A } q _ { m } - 1 \right) = 1 } \end{array}$

$$
{ \frac { \partial { \mathcal { L } } } { \partial q _ { i } } } = - { \frac { p _ { i } } { q _ { i } } } + \lambda = 0 \quad \Longrightarrow \quad \lambda = { \frac { p _ { i } } { q _ { i } } } \quad \Longrightarrow \quad q _ { i } = { \frac { p _ { i } } { \lambda } } \qquad { \mathrm { f o r ~ e v e r y ~ } } i \in A .
$$

So the ratio $p _ { i } / q _ { i }$ is the same for every active token. Substituting into the normalization constraint $\begin{array} { r } { \partial \mathcal { L } / \partial \lambda = \hat { \sum _ { i \in A } } q _ { i } - 1 = 0 , } \end{array}$

$$
1 = \sum _ { i \in A } { \frac { p _ { i } } { \lambda } } = { \frac { 1 } { \lambda } } \sum _ { i \in A } p _ { i } = { \frac { P _ { A } } { \lambda } } \quad \Longrightarrow \quad \lambda = P _ { A } \quad \Longrightarrow \quad q _ { i } ^ { \star } = { \frac { p _ { i } } { P _ { A } } } \qquad ( i \in A ) .
$$

This is the only stationary point, since the equations force $q _ { i } = p _ { i } / \lambda$ and then $\lambda = P _ { A }$ . It lies in $K \colon$ $q _ { i } ^ { \star } > 0$ and $\textstyle \sum _ { i \in A } q _ { i } ^ { \star } = { \dot { P } } _ { A } ^ { \star } / P _ { A } = 1$

(f) Global minimality in the relaxed problem. The gradient at the Lagrange point is

$$
\left. { \frac { \partial F } { \partial q _ { i } } } \right| _ { q _ { A } ^ { \star } } = - { \frac { p _ { i } } { q _ { i } ^ { \star } } } = - { \frac { p _ { i } } { p _ { i } / P _ { A } } } = - P _ { A } \quad { \mathrm { f o r ~ e v e r y ~ } } i \in A , \qquad { \mathrm { t h a t ~ i s } } , \qquad \nabla F ( q _ { A } ^ { \star } ) = - P _ { A } \mathbf { 1 } .
$$

Let $q _ { A } \in K$ with $q _ { A } \neq q _ { A } ^ { \star }$ . By (F2) with $x = q _ { A } ^ { \star }$ and $y = q _ { A }$

$$
F ( q _ { A } ) > F ( q _ { A } ^ { \star } ) + \nabla F ( q _ { A } ^ { \star } ) ^ { \top } ( q _ { A } - q _ { A } ^ { \star } ) ,
$$

and the last term vanishes because both allocations sum to one:

$$
\nabla F ( q _ { A } ^ { \star } ) ^ { \top } ( q _ { A } - q _ { A } ^ { \star } ) = \sum _ { i \in A } ( - P _ { A } ) ( q _ { i } - q _ { i } ^ { \star } ) = - P _ { A } \Big ( \sum _ { i \in A } q _ { i } - \sum _ { i \in A } q _ { i } ^ { \star } \Big ) = - P _ { A } ( 1 - 1 ) = 0 .
$$

Hence $F ( q _ { A } ) > F ( q _ { A } ^ { \star } )$ for every $q _ { A } \in K$ other than $q _ { A } ^ { \star }$ : the Lagrange point is the unique global minimizer of the relaxed problem.

(g) Membership of the candidate. Since $0 < P _ { A } < 1$ , we have $q _ { i } ^ { \star } = p _ { i } / P _ { A } > p _ { i }$ for every $i \in A$ and

$$
p _ { i } \log \frac { p _ { i } } { q _ { i } ^ { \star } } = p _ { i } \log \frac { p _ { i } } { p _ { i } / P _ { A } } = p _ { i } \log P _ { A } < 0 < \tau ,
$$

in agreement with (M4). Every active term lies strictly below the threshold, for every $\tau > 0$ , so the candidate cannot cross into the clipped set. Thus $q _ { A } ^ { \star } \in K _ { \tau }$ , and with $q _ { j } ^ { \star } = 0$ on C we get $q ^ { \star } \in R$

(h) Back to the restricted problem. Let $q _ { A } \in K _ { \tau }$ with $q _ { A } \neq q _ { A } ^ { \star }$ . Since $K _ { \tau } \subseteq K$ , part (f) gives ${ \dot { F } } ( q _ { A } ) > F ( q _ { A } ^ { \star } )$ , and $q _ { A } ^ { \star } \in K _ { \tau }$ by (g). So $q _ { A } ^ { \star }$ is also the unique global minimizer of the restricted problem: a minimizer over the larger set that lies in the smaller set is the minimizer over the smaller set. Its value is

$$
F ( q _ { A } ^ { \star } ) = \sum _ { i \in A } p _ { i } \log { \frac { p _ { i } } { p _ { i } / P _ { A } } } = \sum _ { i \in A } p _ { i } \log P _ { A } = \left( \sum _ { i \in A } p _ { i } \right) \log P _ { A } = P _ { A } \log P _ { A } .
$$

Conclusion. Let $q \in R$ with $q \neq q ^ { \star }$ . Case $Q _ { C } > 0 .$ . Step 1 gives $q ^ { \prime } \in R$ with $Q _ { C } = 0$ and $D _ { \mathrm { c l i p } } ( q ) > D _ { \mathrm { c l i p } } ( q ^ { \prime } )$ , and Step 2 gives $D _ { \mathrm { c l i p } } ( q ^ { \prime } ) = F ( q _ { A } ^ { \prime } ) + \mathsf { \bar { \Pi } } | C | \overset { \circ } { \tau } \geq \dot { F } ( q _ { A } ^ { \star } ) + | C | \tau = D _ { \mathrm { c l i p } } ( q ^ { \star } )$ So $\dot { D } _ { \mathrm { c l i p } } ( q ) > \dot { D } _ { \mathrm { c l i p } } ( q ^ { \star } )$ . Case $Q _ { C } = 0$ . Then $q _ { j } = 0 = q _ { i } ^ { \star }$ on $C ,$ so $q \neq q ^ { \star }$ means $q _ { A } \neq q _ { A } ^ { \star }$ , and Step 2 gives $D _ { \mathrm { c l i p } } ( q ) = F ( q _ { A } ) + | C | \tau > F ( q _ { A } ^ { \star } ) + | C | \tau = \bar { D } _ { \mathrm { c l i p } } ( q ^ { \star } )$

In both cases $D _ { \mathrm { c l i p } } ( q ) > D _ { \mathrm { c l i p } } ( q ^ { \star } )$ . Hence $q ^ { \star }$ is the unique minimizer of $D _ { \mathrm { c l i p } }$ over $R ,$ with $D _ { \mathrm { c l i p } } ( q ^ { \star } ) = P _ { A } \log \dot { P } _ { A } + | C | \tau$ . Existence is not assumed: the minimizer is exhibited. □

Fixed $Q _ { C }$ (Equation 8). Let c be a value of $Q _ { C }$ attained in R. For $q \in R$ with $Q _ { C } = c ,$ the loss is $F ( q _ { A } ) + | C | \tau$ with $q _ { i } > 0$ and $\textstyle \sum _ { i \in A } q _ { i } = 1 - c $ , and it does not depend on how c is split among the clipped tokens as long as it does not make any one of them active by going over the $d > \tau$ restriction. Repeat Step 2 with the constraint $\textstyle \sum _ { i \in A } { q _ { i } = 1 - c }$ . Stationarity again gives $q _ { i } = p _ { i } / \lambda ,$ and now

$$
1 - c = \sum _ { i \in A } { \frac { p _ { i } } { \lambda } } = { \frac { P _ { A } } { \lambda } } \quad \Longrightarrow \quad \lambda = { \frac { P _ { A } } { 1 - c } } \quad \Longrightarrow \quad q _ { i } ^ { \star } ( c ) = { \frac { 1 - c } { P _ { A } } } p _ { i } .
$$

The gradient there is $- p _ { i } / q _ { i } ^ { \star } ( c ) = - P _ { A } / ( 1 - c )$ in every coordinate, so for any other positive allocation with the same sum, $\begin{array} { r } { \nabla F ( q _ { A } ^ { \star } ( c ) ) ^ { \top } ( q _ { A } - q _ { A } ^ { \star } ( c ) ) = - \frac { P _ { A } } { 1 - c } \big ( ( 1 - c ) - ( 1 - c ) \big ) = 0 } \end{array}$ , and (F2) gives $F ( q _ { A } ) > F ( q _ { A } ^ { \star } ( c ) )$ : this is the unique global minimizer of the relaxed problem at level c.

Membership. Every clipped token has a term above $\tau > 0 .$ , so $\log ( p _ { j } / q _ { j } ) ~ > ~ 0$ and $q _ { j } ~ < ~ p _ { j }$ Summing over C gives $\begin{array} { r } { \dot { c } < \sum _ { j \in C } p _ { j } = 1 - P _ { A } } \end{array}$ , hence $( 1 - c ) / P _ { A } > \bar { 1 }$ and $q _ { i } ^ { \star } ( c ) > p _ { i }$ . By (M4) every active term is below $\tau ,$ , so the relaxed minimizer satisfies the threshold inequalities and is also the minimizer of the restricted problem at level c. Its loss is

$$
\begin{array} { l } { \displaystyle D _ { \mathrm { c l i p } } ^ { \star } ( c ) = \sum _ { i \in A } p _ { i } \log \frac { p _ { i } } { \left( 1 - c \right) p _ { i } / P _ { A } } + | C | \tau } \\ { = P _ { A } \log \frac { P _ { A } } { 1 - c } + | C | \tau = P _ { A } \log P _ { A } - P _ { A } \log ( 1 - c ) + | C | \tau , } \end{array}
$$

and

$$
\frac { d D _ { \mathrm { c l i p } } ^ { \star } } { d c } = - P _ { A } \cdot \frac { - 1 } { 1 - c } = \frac { P _ { A } } { 1 - c } > 0 .
$$

The level-c minimizer is unique in its active coordinates only.

Remarks. (i) F is not a KL divergence. Its arguments $p _ { A }$ and $q _ { A }$ are sub-probability vectors, and F can be negative: $F ( q _ { A } ^ { \star } ) = P _ { A }$ log $P _ { A } < 0$ . The proof uses only the strict convexity of $F .$

(ii) Scope. The statement concerns one fixed clipping region; no comparison across clipping patterns is claimed. A softmax over finite logits assigns every token positive probability, so such a student never equals $q ^ { \star } ;$ ; by the fixed- $\cdot Q _ { C }$ statement the optimal loss at level c decreases to $D _ { \mathrm { c l i p } } ( q ^ { \star } )$ as $c \downarrow 0 .$ so $q ^ { \star }$ is approached but not attained.

## C TRAINING AND EVALUATION CONFIGURATION, CAPTURE

Training configuration In all eight runs, the student and the teacher are initialized from the same Qwen3-4B checkpoint. Each clipped/unclipped pair receives the same per-update problems. Table 4 lists the detailed training configuration used in our experiment.

Table 4: Training Configuration.
<table><tr><td>Component</td><td>Executed value</td></tr><tr><td>Base checkpoint</td><td>Qwen/Qwen3-4B</td></tr><tr><td>Optimizer</td><td>torch.optim.AdamW; learning rate 10−6; β = (0.9,0.999); € = 10−8; weight decay 0.01</td></tr><tr><td>Schedule</td><td>Zero warmup. 10−⁶ constant.</td></tr><tr><td>Updates and checkpoints</td><td>200 updates; a checkpoint is saved every 25 updates</td></tr><tr><td>Seed</td><td>Training and data-loader seeds are 42</td></tr><tr><td>Global update</td><td>30 problems, one rollout each</td></tr><tr><td>Prompt length cap Training rollout</td><td>2,048 tokens; the response caps are given in Table 1 Temperature 1.1, top-p = 0.8, top-k = 20</td></tr><tr><td>sampler</td><td></td></tr><tr><td>Teacher sampler Loss</td><td>The teacher returns its top-128 token log probabilities at temperature 1.0 Forward KL on the teacher&#x27;s top-128 tokens, both distributions renormalized on</td></tr><tr><td></td><td>that support at temperature T = 1.1; clipped runs use τ = 0.05, unclipped runs apply no clip</td></tr><tr><td>Hardware</td><td>Four H200 GPUs per run: three for the student&#x27;s training and rollouts, one for the dedicated SGLang teacher</td></tr><tr><td>Software</td><td>Transformers 4.57.1; PyTorch 2.9.1+cu128; Ray 2.54.1; SGLang 0.5.9; CUDA toolkit 12.8.2</td></tr></table>

Message templates. We give the user messages and the chat template used to serialize the student and teacher prompts. The student’s user message is identical across all eight runs. Its serialized prompt differs only by the empty thinking block described below. Before chat serialization, the student’s single user message is:

Problem: {problem}   
Please reason step by step, and put your final answer within   
\boxed{}.

The teacher’s single user message is:

Problem: {problem} Here is a reference solution to this   
problem: === Reference Solution Begin === {answer} ===   
Reference Solution End === After reading the reference   
solution above, make sure you truly understand the reasoning   
behind each step - do not copy or paraphrase it. Now, using   
your own words and independent reasoning, derive the same   
final answer to the problem above. Please reason step by   
step, and put your final answer within \boxed{}.

The placeholder {answer} is filled with the full reference solution from the dataset, not the final answer alone. Qwen serializes the one-message prompt as

<|im start|>user   
{content}<|im end|>   
<|im start|>assistant

with no system or tool message. When thinking is disabled, the serialization additionally appends an empty <think> block followed by </think> and a blank line. The student’s sampled response token IDs are appended to the teacher prompt before teacher scoring.

Capture setting. At every response position of all 200 updates of all eight runs, the trainer stores the token the student emitted, the IDs of the teacher’s top-128 candidate tokens, the teacher’s log probabilities on those tokens, and the student’s log probabilities on the same tokens, gathered from its full-vocabulary log-softmax in the forward pass that computes the loss.

## D COMPLETE AIME2024 AND AIME2025 EVALUATION

Evaluation configuration. Checkpoints 25, 50, 75, 100, 125, 150, 175, and 200 are evaluated on AIME 2024 and AIME 2025. Each benchmark contains 30 problems.

Each saved checkpoint is evaluated with thinking enabled, drawing 12 samples per problem at temperature $. 6 , { \mathrm { t o p } } \cdot p = . 9 5 , { \mathrm { t o p } } - k = 2 0$ , and $\mathrm { m i n i m u m } \mathrm { - } p = 0 .$ , with a 38,912-token output budget. Scoring uses only the text after the last closing thinking tag: Math-Verify (Kydl´ıcek) compares it ˇ with the reference answer, and if that comparison fails, the last $\{ { \mathrm { b } } { \mathrm { o x } } { \mathrm { e d } } \{ \}$ expression is compared with the reference answer as a string or a number. We evaluate the same base-model checkpoint eight times for both benchmarks to measure variation from response sampling.

![](images/288b6a5d71d071cadcf11a1979b7f62bcc1811ecc4f8d65e2e5659aca97edcf4.jpg)

![](images/bcc4b8d24c8a297f625d9664bf2ecbdfd6974d61cdb13735f649cccba40079ce.jpg)  
Figure 5: AIME 2024 across training, in the layout of Figure 3. (a) Avg@12, the mean correct ratio over 12 samples for each of 30 problems at the checkpoint saved after each 25 updates. (b) The ratio of the 360 responses per checkpoint that end in a terminal periodic loop. Solid curves with circles are the clipped runs, dashed curves with squares their unclipped twins. Each trained curve is one realized trajectory. The black arrow on each vertical axis marks the base model, averaged over eight evaluation-sampling seeds. The values appear in Table 6.

Figure 5 visualize AIME 2024 evaluation result: (a) avg@12 and, (b) terminal loop rate.

Table 6 and 5 shows the numerical value of per checkpoint evaluated on AIME 2024 and AIME 2025, respectively: (a) avg@12, (b) terminal loop rate and (c) pass@12.

Table 5: Complete AIME 2025 evaluation at every checkpoint (columns: training update). (a) Avg@12, the mean correct ratio over 12 samples for each of 30 problems. (b) Terminal loop rate, the ratio of the 360 responses that end in a terminal periodic loop. (c) Pass@12, the ratio of the 30 problems solved by at least one of the 12 samples. Each trained row is one realized trajectory. The base row gives the mean and, in parentheses, the standard deviation over eight evaluation-sampling seeds of the fixed starting checkpoint.
<table><tr><td colspan="9">(a) avg@12</td></tr><tr><td>Run</td><td>25</td><td>50</td><td>75</td><td>100</td><td>125</td><td>150</td><td>175</td><td>200</td></tr><tr><td>F</td><td>.661</td><td>.653</td><td>.589</td><td>.550</td><td>.581</td><td>.525</td><td>.531</td><td>.544</td></tr><tr><td>F-no-clip</td><td>.622</td><td>.644</td><td>.578</td><td>.544</td><td>.619</td><td>.561</td><td>.575</td><td>.572</td></tr><tr><td>M</td><td>.592</td><td>.481</td><td>.378</td><td>.122</td><td>.050</td><td>.067</td><td>.031</td><td>.022</td></tr><tr><td>M-no-clip</td><td>.047</td><td>.000</td><td>.000</td><td>.000</td><td>.000</td><td>.000</td><td>.000</td><td>.000</td></tr><tr><td>K</td><td>.664</td><td>.653</td><td>.647</td><td>.619</td><td>.622</td><td>.550</td><td>.628</td><td>.600</td></tr><tr><td>K-no-clip</td><td>.647</td><td>.642</td><td>.633</td><td>.611</td><td>.644</td><td>.642</td><td>.608</td><td>.608</td></tr><tr><td>L</td><td>.636</td><td>.525</td><td>.533</td><td>.486</td><td>.469</td><td>.439</td><td>.469</td><td>.458</td></tr><tr><td>L-no-clip</td><td>.581</td><td>.531</td><td>.481</td><td>.508</td><td>.489</td><td>.506</td><td>.508</td><td>.517</td></tr><tr><td colspan="9">Base</td></tr><tr><td colspan="9">(b) terminal loop rate</td></tr><tr><td>Run</td><td>25</td><td>50</td><td>75</td><td>100</td><td>125</td><td>150</td><td>175</td><td>200</td></tr><tr><td>F</td><td>.008</td><td>.033</td><td>.128</td><td>.161</td><td>.153</td><td>.203</td><td>.194</td><td>.192</td></tr><tr><td>F-no-clip</td><td>.011</td><td>.003</td><td>.008</td><td>.019</td><td>.000</td><td>.006</td><td>.006</td><td>.008</td></tr><tr><td>M</td><td>.006</td><td>.036</td><td>.192</td><td>.494</td><td>.389</td><td>.422</td><td>.400</td><td>.400</td></tr><tr><td>M-no-clip</td><td>.050</td><td>.039</td><td>.036</td><td>.044</td><td>.031</td><td>.053</td><td>.042</td><td>.017</td></tr><tr><td>K</td><td>.014</td><td>.033</td><td>.036</td><td>.050</td><td>.058</td><td>.078</td><td>.086</td><td>.061</td></tr><tr><td>K-no-clip</td><td>.011</td><td>.011</td><td>.006</td><td>.006</td><td>.000</td><td>.003</td><td>.008</td><td>.011</td></tr><tr><td>L</td><td>.047</td><td>.144</td><td>.211</td><td>.231</td><td>.256</td><td>.328</td><td>.267</td><td>.303</td></tr><tr><td>L-no-clip</td><td>.003</td><td>.014</td><td>.019</td><td>.006</td><td>.011</td><td>.006</td><td>.022</td><td>.011</td></tr><tr><td colspan="9">Base</td></tr><tr><td colspan="9">(c) pass@ 12</td></tr><tr><td>Run</td><td>25</td><td>50</td><td>75</td><td>100</td><td>125</td><td>150</td><td>175</td><td>200</td></tr><tr><td>F</td><td>.800</td><td>.800</td><td>.867</td><td>.800</td><td>.833</td><td>.833</td><td>.867</td><td>.800</td></tr><tr><td>F-no-clip</td><td>.867</td><td>.800</td><td>.800</td><td>.733</td><td>.833</td><td>.833</td><td>.800</td><td>.800</td></tr><tr><td>M</td><td>.867</td><td>.700</td><td>.667</td><td>.367</td><td>.233</td><td>.267</td><td>.167</td><td>.133</td></tr><tr><td>M-no-clip</td><td>.133</td><td>.000</td><td>.000</td><td>.000</td><td>.000</td><td>.000</td><td>.000</td><td>.000</td></tr><tr><td>K</td><td>.833</td><td>.800</td><td>.833</td><td>.833</td><td>.867</td><td>.800</td><td>.833</td><td>.833</td></tr><tr><td>K-no-clip</td><td>.833</td><td>.800</td><td>.833</td><td>.767</td><td>.867</td><td>.833</td><td>.767</td><td>.800</td></tr><tr><td>L L-no-clip</td><td>.833 .767</td><td>.800</td><td>.767</td><td>.767</td><td>.833</td><td>.767</td><td>.767</td><td>.767</td></tr><tr><td></td><td></td><td>.767</td><td>.733</td><td>.767</td><td>.767</td><td>.767</td><td>.700</td><td>.767</td></tr><tr><td colspan="9">Base .817 (.018)</td></tr></table>

Table 6: Complete AIME 2024 evaluation at every checkpoint (columns: training update). (a) Avg@12, the mean correct ratio over 12 samples for each of 30 problems. (b) Terminal loop rate, the ratio of the 360 responses that end in a terminal periodic loop. (c) Pass@12, the ratio of the 30 problems solved by at least one of the 12 samples. Each trained row is one realized trajectory. The base row gives the mean and, in parentheses, the standard deviation over eight evaluation-sampling seeds of the fixed starting checkpoint.
<table><tr><td colspan="9">(a) avg@12</td></tr><tr><td>Run</td><td>25</td><td>50</td><td>75</td><td>100</td><td>125</td><td>150</td><td>175</td><td>200</td></tr><tr><td>F</td><td>.739</td><td>.756</td><td>.622</td><td>.617</td><td>.553</td><td>.539</td><td>.547</td><td>.533</td></tr><tr><td>F-no-clip</td><td>.728</td><td>.717</td><td>.694</td><td>.658</td><td>.672</td><td>.667</td><td>.683</td><td>.661</td></tr><tr><td>M</td><td>.672</td><td>.572</td><td>.447</td><td>.125</td><td>.069</td><td>.069</td><td>.036</td><td>.031</td></tr><tr><td>M-no-clip</td><td>.056</td><td>.000</td><td>.000</td><td>.000</td><td>.000</td><td>.000</td><td>.000</td><td>.000</td></tr><tr><td>K</td><td>.742</td><td>.733</td><td>.664</td><td>.681</td><td>.678</td><td>.700</td><td>.681</td><td>.700</td></tr><tr><td>K-no-clip</td><td>.703</td><td>.744</td><td>.722</td><td>.728</td><td>.742</td><td>.719</td><td>.728</td><td>.736</td></tr><tr><td>L</td><td>.700</td><td>.550</td><td>.467</td><td>.450</td><td>.408</td><td>.436</td><td>.433</td><td>.428</td></tr><tr><td>L-no-clip</td><td>.653</td><td>.611</td><td>.586</td><td>.586</td><td>.603</td><td>.614</td><td>.603</td><td>.614</td></tr><tr><td colspan="9">Base</td></tr><tr><td colspan="9">(b) terminal loop rate</td></tr><tr><td>Run</td><td>25</td><td>50</td><td>75</td><td>100</td><td>125</td><td>150</td><td>175</td><td>200</td></tr><tr><td>F</td><td>.011</td><td>.022</td><td>.169</td><td>.181</td><td>.236</td><td>.233</td><td>.247</td><td>.253</td></tr><tr><td>F-no-clip</td><td>.003</td><td>.000</td><td>.011</td><td>.008</td><td>.017</td><td>.011</td><td>.003</td><td>.000</td></tr><tr><td>M</td><td>.000</td><td>.028</td><td>.214</td><td>.481</td><td>.386</td><td>.475</td><td>.406</td><td>.444</td></tr><tr><td>M-no-clip</td><td>.039</td><td>.072</td><td>.069</td><td>.064</td><td>.072</td><td>.067</td><td>.067</td><td>.072</td></tr><tr><td>K</td><td>.019</td><td>.017</td><td>.061</td><td>.081</td><td>.064</td><td>.033</td><td>.058</td><td>.031</td></tr><tr><td>K-no-clip</td><td>.006</td><td>.003</td><td>.017</td><td>.014</td><td>.008</td><td>.014</td><td>.003</td><td>.025</td></tr><tr><td>L</td><td>.047</td><td>.197</td><td>.289</td><td>.353</td><td>.397</td><td>.369</td><td>.336</td><td>.369</td></tr><tr><td>L-no-clip</td><td>.000</td><td>.011</td><td>.017</td><td>.011</td><td>.019</td><td>.014</td><td>.000</td><td>.008</td></tr><tr><td colspan="9"></td></tr><tr><td colspan="9">(c) pass@12</td></tr><tr><td>Run</td><td>25</td><td>50</td><td>75</td><td>100</td><td>125</td><td>150</td><td>175</td><td>200</td></tr><tr><td>F</td><td>.867</td><td>.833</td><td>.900</td><td>.867</td><td>.900</td><td>.867</td><td>.867</td><td>.833</td></tr><tr><td>F-no-clip</td><td>.833</td><td>.833</td><td>.833</td><td>.867</td><td>.800</td><td>.833</td><td>.800</td><td>.833</td></tr><tr><td>M</td><td>.833</td><td>.800</td><td>.700</td><td>.467</td><td>.333</td><td>.333</td><td>.167</td><td>.100</td></tr><tr><td>M-no-clip</td><td>.233</td><td>.000</td><td>.000</td><td>.000</td><td>.000</td><td>.000</td><td>.000</td><td>.000</td></tr><tr><td>K</td><td>.833</td><td>.867</td><td>.833</td><td>.833</td><td>.867</td><td>.867</td><td>.833</td><td>.867</td></tr><tr><td>K-no-clip</td><td>.833 .867</td><td>.833</td><td>.867</td><td>.833</td><td>.867</td><td>.833</td><td>.833</td><td>.833</td></tr><tr><td>L L-no-clip</td><td>.833</td><td>.833 .833</td><td>.800 .800</td><td>.767</td><td>.767 .800</td><td>.800 .800</td><td>.833</td><td>.833</td></tr><tr><td>Base</td><td></td><td></td><td></td><td>.800</td><td>.838 (.033)</td><td></td><td>.800</td><td>.800</td></tr><tr><td colspan="9"></td></tr></table>