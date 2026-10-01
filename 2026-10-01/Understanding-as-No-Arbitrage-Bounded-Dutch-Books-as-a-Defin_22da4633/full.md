# Understanding as No-Arbitrage: Bounded Dutch Books as a Definition and Training Objective for Language Models

Daniel Dragonevskiy ispromashka@gmail.com

September 2026

## Abstract

Does a language model merely predict tokens, or does it understand what it says? The question stays undecidable until understanding is given some operational meaning. We propose one, borrowing from inferential-role semantics and the oldest device for keeping graded commitments honest: the Dutch book. A model understands a vocabulary, relative to an inference system and a degree (k, ε), if no trader from a bounded class T<sub>k</sub> can extract guaranteed profit above ε by betting against its elicited credences on logically related claims. That makes understanding measurable, down to a single word or relation.

Four results follow. A ladder theorem ties bounded trader classes to local relaxations of the marginal polytope: full coherence is NP-hard to check, so any bounded system can only understand by degrees, and the theorem predicts how that degree should generalize. A second theorem shows that the exact optimum of negative log-likelihood, on a corpus of mutually inconsistent but stylistically distinguishable authors, is incoherent across elicitation contexts with a guaranteed profit computable in closed form: the defect sits in the training objective, not the architecture, and scale does not remove it. A δ-generalization of Adams’ theorem shows coherent credences degrade additively along derivation chains, and that overconfidence about a derived claim becomes an arbitrage opportunity in its own right, unlike the silent error compounding of autoregressive sampling. Finally we specify a training scheme, Arbitr, in which a trader adversary supplies the missing cross-context term.

Five pre-registered experiments test this. E-A0 measures exploitability across models from 1.5B to 7B parameters: within-format credences are already close to coherent, but cross-format and negation books extract guaranteed profit up to 0.15 per unit stake, worst at 7B on facts it is unsure of. E-A1 trains against the exact adversary and cuts negation exploitability two orders of magnitude, transferring to an untrained, higher-level pattern in every seed, but half the runs collapse into an uninformative equilibrium, exactly as predicted for an objective missing its calibration term. E-A1b adds that term; every registered criterion then passes, at no cost to task accuracy. E-A2 replaces the hand-specified adversary with a learned one that discovers exploitable books on its own, and shows partial transfer to 7B. E-A3 tightens the attribution. An ablation without the trader shows that the cross-format gain comes from the adversarial term rather than the calibration anchor, while on negation the anchor alone accounts for a factor of about 2.4; the mechanism replicates on a second model family (Phi-3.5) and carries over to a negation template never seen in training, with a measurable residue of wording-specificity. It also exposes the approach’s sharpest limit: at 7B, near-zero measured incoherence coincides with near-maximal confidence on most items, so part of the apparent coherence at scale is confident degeneration rather than calibration. We are careful about what this does not show: factual hallucination is untouched, and coherence is necessary for knowledge, not suficient.

## 1 Introduction

Are large language models “stochastic parrots” that predict tokens without understanding, or does next-token prediction sufice for understanding on its own? The debate has run for years without a criterion either side could lose by (Bender and Koller, 2020; Mitchell and Krakauer, 2023; Søgaard, 2025). Probing work has weakened the strong “blind correlation” reading: nexttoken predictors pick up linearly decodable, causally efective world models (Li et al., 2023; Nanda et al., 2023; Gurnee and Tegmark, 2024). Yet the same models routinely violate elementary probability identities across logically related questions (Zhu and Grifiths, 2024; Andrews and Sarkar, 2026), and their multi-step reasoning degrades in ways their stated confidence never reflects (Dziri et al., 2023; Bachmann and Nagarajan, 2024).

We think the more useful question is not whether LLMs understand, but what measurable property would count as understanding, and what training objective would produce it. Inferential-role semantics ties the meaning of an expression to the inferences it licenses (Brandom, 1994); we operationalize that idea with the oldest formal tool for keeping graded commitments honest, the Dutch book (de Finetti, 1937). The resulting definition is behavioral, graded, word-level, and—the part that matters for training—adversarially trainable.

Definition (informal). Fix an inference system I and a family of elicitation contexts. A model understands a vocabulary to degree (k, ε) if no trader in a bounded class T (portfolios supported on at most k propositions instantiating rules of I, including cross-context “law of one price” books) achieves guaranteed profit exceeding ε against the model’s elicited credences.

Contributions. Exploitability-as-distance is classical (Schervish et al., 2002; De Bona and Finger, 2015), and we do not claim it. What we do claim:

1. A graded ladder (Thm. 5): T<sub>k</sub>-unexploitability turns out to equal k-local consistency, membership in a local relaxation of the marginal polytope. The polyhedral combinatorics here is standard (the local polytope of Wainwright and Jordan, 2008; Sherali–Adams relaxations Sherali and Adams, 1990; PSAT complexity Georgakopoulos et al., 1988; Pitowsky, 1991), and we are not claiming any of it as new. What is new, we think, is the dictionary between bounded traders and relaxation levels, and its consequence: since full coherence is NP-complete, a bounded agent can only understand by degrees, which turns “degree of understanding” into a complexity-theoretic quantity with a generalization prediction attached. Sum-of-squares pseudodistributions as “computationally bounded Bayesian beliefs” (Barak and Steurer, 2014) and bounded-trader logical induction (Garrabrant et al., 2016) anticipate the idea conceptually.

2. An objective-defect theorem (Thm. 7): on a mixture corpus of internally consistent, mutually inconsistent authors whose style correlates with identity, the exact NLL optimum turns out to be incoherent across contexts, with guaranteed arbitrage profit equal to the style–author correlation. Per context the optimum is coherent—a mixture of coherent credences is coherent for unconditional events, cf. opinion-pooling results (Madansky, 1964; Genest and Zidek, 1986)—so the defect is precisely located: NLL never scores one proposition across two contexts. No amount of data or scale removes this; only a cross-context term added to the objective can.

3. Priced additive degradation (Thm. 9), a δ-generalization of Adams’ theorem (Adams, 1975; Gilio, 2002): δ-coherent credences degrade at most additively along derivation chains, and overstating terminal confidence relative to premise prices is itself an arbitrage opportunity. That degradation is visible in the price is the useful part—autoregressive sampling compounds errors silently (Dziri et al., 2023; LeCun, 2023).

4. A training scheme (Arbitr, §5): a small trader network learns to construct profitable books over batches whose admissible worlds are known by construction, and the model is penalized by the realized profit. Coherentizing at any level cannot hurt proper-score accuracy (a small extension of Predd et al., 2009, Prop. 11), so the two loss terms do not fight each other at the optimum. Bid–ask spreads (lower/upper previsions, Walley, 1991) give “I have no basis to commit” an architectural home; calibration bettors (Foster and Vohra, 1998) punish unfounded precision and keep the spread from collapsing under gradient pressure. Per-word exploitability Inc gives individual lexical items their own understanding certificates.

We also ran a pre-registered measurement, E-A0, of T<sub>2</sub>/T<sub>3</sub>-exploitability across model scales (§6)—the cheapest way to find out whether the whole program is moot, i.e. whether deployed models were already close enough to coherent that none of this matters.

## 2 Related work

Incoherence measures. Measuring incoherence via normalized guaranteed Dutch-book loss goes back to Schervish et al. (2002, 2003). De Bona and Finger (2015) show that distance-based and Dutch-book-based inconsistency measures coincide and are LP-computable; Potyka (2014) studies p-norm minimal-violation measures; Stafel (2015, 2019) work out the philosophy. Our Prop. 3 restates this lineage compactly. Andrews and Sarkar (2026) compute LP Dutch books against LLMs, as a measurement only. Andrews (2026) proposes LP-computable penalties from representation theorems (de Finetti, Afriat, Echenique–Saito) as label-free evaluation metrics and candidate regularizers, with the bettor appearing only as an LP variable—no bounded trader classes, no learnable adversary, no training experiments. That is the gap this paper’s method and experiments fill.

Probability logic. Additive degradation under p-validity is Adams’ theorem (Adams, 1975); precise propagation of bounds in System P is due to Gilio (2002); probabilistic entailment as LP is Nilsson (1986); Hailperin (1996); PSAT complexity is Georgakopoulos et al. (1988); correlation polytopes and their local relaxations are Pitowsky (1991). Approximate coherence and its graded benefits: De Bona and Stafel (2017).

Bounded rationality via traders and pseudodistributions. Logical induction defines rational credence by unexploitability against polynomial-time traders (Garrabrant et al., 2016); sum-of-squares pseudodistributions formalize beliefs of computationally bounded Bayesians (Barak and Steurer, 2014). We borrow both intuitions and put them to work at finite, learnable scale.

LLM (in)coherence and consistency training. Systematic probability-identity violations in LLMs: Zhu and Grifiths (2024); non-Bayesian updating: Chen et al. (2026). Consistency has been enforced at decode time (Kassner et al., 2021; Mitchell et al., 2022) or by losses tied to fixed fact/rule sets (Calanzone et al., 2024); paraphrase-consistency losses: Elazar et al. (2021). RL-style “self-consistency” rewards agreement among samples, not probabilistic coherence over related propositions (Pres et al., 2026). Semantic entropy (Farquhar et al., 2024) is the special case of our law-of-one-price books restricted to paraphrase families. Betz and Richardson (2023) train coherent credences on synthetic corpora without an adversary. Negation-driven, objectivelevel failures: Asher and Bhar (2024). To put the kinship plainly: the training component here is consistency fine-tuning in the lineage of Elazar et al. (2021) and Calanzone et al. (2024), generalized to a market semantics. What is added is an adversary that searches for violations instead of enumerating them, settlement by admissible worlds instead of a fixed fact list, a graded trader-class ladder with its own generalization prediction, and a no-arbitrage reading that ties the loss back to a definition.

World models. The world-model program (LeCun, 2022) goes after the same slogan, “prediction is not understanding,” architecturally: replace token prediction with latent-state prediction and planning. Our results sit next to that, not against it—coherence of elicited credences is something any architecture serving as an epistemic interface needs, and Thm. 7 is about the objective given an architecture, not about the architecture itself. Nothing here bears on whether latent world models are needed for grounding or planning. But a JEPA-style system that answers questions still owes its users credences that don’t contradict themselves across phrasings.

Credal and adversarial learning. Credal deep learning trains classifiers to output probability intervals (Caprio et al., 2023; CreINNs, 2024; Credal Deep Ensembles, 2024); Credal LLMs (Manchingal et al., 2026) derive intervals post hoc from adapter ensembles. None provide avoiding-sure-loss guarantees or an anti-collapse mechanism; none train an LM end to end. Adversarial training with logical opponents exists in non-probabilistic form (Ehrenfeucht–Fraïssé games over graphs with a depth-k curriculum, Mannucci, 2025) and as LLM discriminators of reasoning soundness without stakes (Liu et al., 2025).

Hallucination lower bounds. Calibrated models forced to answer must hallucinate on missing mass (Kalai and Vempala, 2024); these bounds don’t care about representation or objective. Nothing below contradicts them—coherence is a diferent failure class (§8).

## 3 Setup

Let Φ be a finite set of natural-language propositions carrying a known logical structure (atoms, connectives, certified paraphrase/entailment relations from an inference system I). A world is an assignment $w \in \{ 0 , 1 \} ^ { \Phi }$ consistent with I; W denotes the set of worlds and $C : = \mathrm { c o n v } \{ w :$ $w \in W \}$ the coherent polytope (de Finetti): $q \in C$ if q extends to a probability measure over worlds.

A model M induces credences $q _ { M } ( \varphi ; c ) ~ \in ~ [ 0 , 1 ]$ through an elicitation family: contexts $c \in E ( \varphi )$ (question formats, languages, paraphrastic framings, irrelevant-context perturbations) mapped to the model’s normalized probability of an afirmative answer. Coherence of the assembled credence function requires both (a) membership in C and (b) law of one price (LOP): $q ( \varphi ; c ) = q ( \varphi ; c ^ { \prime } )$ for $c , c ^ { \prime } \in E ( \varphi )$ . LOP books are the cross-context instruments; semantic entropy (Farquhar et al., 2024) measures exactly LOP dispersion over paraphrase families.

A portfolio is $x \in \mathbb { R } ^ { \Phi } , \| x \| _ { 1 } \leq 1$ : buy $x _ { \varphi }$ shares of $\varphi$ at price $q ( \varphi )$ ; a share pays 1 if $\varphi$ holds. Guaranteed profit: $\begin{array} { r } { g ( x ) = \operatorname* { m i n } _ { w \in W } \sum _ { \varphi } x _ { \varphi } ( w ( \varphi ) - q ( \varphi ) ) } \end{array}$ ; exploitability In ${ \mathfrak { s } } ( q ) : = \operatorname* { m a x } _ { \| x \| _ { 1 } \leq 1 } g ( x )$

Definition 1 (Bounded trader classes). $\mathcal { T } _ { k }$ is the set of portfolios whose support lies in some $S \in S _ { k }$ , where $\boldsymbol { S _ { k } }$ collects supports of size $\leq k$ instantiating rules of I with $\leq k$ premises (negation and LOP pairs at $k { = } 2 ;$ conjunction, modus ponens, transitivity triples at $k { = } 3 ;$ longer chains and quantifier instances above). Inc<sub>k</sub>(q) is the corresponding restricted maximum.

Definition 2 (Graded understanding). M understands Φ to degree $( k , \varepsilon )$ , relative to I and the elicitation family, if ${ \mathrm { I n c } } _ { k } ( q _ { M } ) \leq \varepsilon$ . For a word v, the understanding certificate In $\mathbf { \dot { \omega } } _ { \mathbf { k } , v }$ restricts the maximum to books pivoting on v (supports whose propositions difer in the inferential role of v).

## 4 Results

Proposition 3 (Exploitability is distance; known). Inc $\begin{array} { r } { : ( q ) = \operatorname* { m i n } _ { \mu \in C } \| q - \mu \| _ { \infty } } \end{array}$ . In particular In $\mathsf { \Lambda } _ { : ( q ) } = 0 \mathrm { \Lambda } _ { i f f q \in C }$

Proof. For fixed x, min $_ { \neg \infty } \subset W \left. x , w - q \right. = \operatorname* { m i n } _ { \mu \in C } \left. x , \mu - q \right.$ (a linear function attains its minimum on a polytope at a vertex). Both feasible sets are convex and compact and the objective is bilinear, so Sion’s minimax theorem permits the exchange: Inc $\begin{array} { r } { \mathbf { \rho } : ( q ) = \operatorname* { m i n } _ { \mu \in C } \operatorname* { m a x } _ { \| \ b { x } \| _ { 1 } \leq 1 } \langle \ b { x } , \mu - } \end{array}$ $q \rangle = \operatorname* { m i n } _ { \mu \in C } \| \mu - q \| _ { \infty }$ □

Remark 4. This is a short route to a known equivalence between Dutch-book and distancebased incoherence measures (De Bona and Finger, 2015; Schervish et al., 2002; Potyka, 2014). We use it here only as a building block.

Theorem 5 (Ladder of understanding). (a) $\mathrm { I n c } _ { k } ( q ) = \mathrm { m a x } _ { \mathit { S } \in \mathcal { S } _ { k } }$ dist $\infty ( q | _ { S }$ , conv $( W | _ { S } ) )$ : bounded exploitability equals the largest violation of k-local coherence constraints, i.e. T<sub>k</sub>-unexploitability is membership in a local (Sherali–Adams-style) relaxation of the marginal polytope. (b) Inc $_ 2 \ \leq$ Inc<sub>3</sub> $\leq \cdots \leq$ Inc, with equality at $k = | \Phi |$ . (c) Verifying Inc $_ k ( q ) \leq \varepsilon$ for fixed k is polynomial; verifying In $\dot { \boldsymbol { { \mathbf { \mathit { z } } } } } ( \boldsymbol { q } ) = 0$ is NP-complete (Georgakopoulos et al., 1988; Pitowsky, 1991).

Proof. (a) Apply Prop. 3 to each admissible support $S ,$ noting conv $( W ) | _ { S } = \mathrm { c o n v } ( W | _ { S } )$ under projection, and that a portfolio confined to S interacts with worlds only through W|<sub>S</sub>. (b) Nesting of portfolio classes. (c) For fixed $k , \ | S _ { k } |$ is polynomial in |Φ| and each local LP has bounded size; global coherence is PSAT. □

Remark 6. Since full understanding is NP-hard, any bounded agent, human or machine, can only be coherent up to some level of the hierarchy. “Degrees of understandin $\mathrm { g } ^ { \mathrm { , } }$ is a rung, not a figure of speech. This gives a testable prediction (§6): a model trained against $\mathcal { T } _ { k }$ should generalize coherence to unseen instances of patterns at level $\leq k .$ , not at level k+1.

Theorem 7 (The exact NLL optimum is cross-context incoherent). Let the corpus be generated by two internally consistent authors: author 1 asserts φ, author 2 asserts ¬φ, with equal weights, and let style correlate with identity: contexts $c _ { 1 } , c _ { 2 }$ with $P ( c _ { 1 } \mid$ author $1 ) = P ( c _ { 2 } \ |$ author 2) = $( 1 + \rho ) / 2 , \rho > 0$ . The exact NLL optimum M<sup>∗</sup> (the true conditional distribution) elicits $q ( \varphi ; c _ { 1 } ) =$ $( 1 + \rho ) / 2$ and $q ( \varphi ; c _ { 2 } ) = ( 1 - \rho ) / 2$ , and the LOP book (sell $\varphi$ at $c _ { 1 }$ , buy at $c _ { 2 } )$ collects guaranteed profit $\rho$ regardless of the truth value of $\varphi :$ Inc $_ \mathrm { L O P } ( M ^ { * } ) \geq \rho$

Proof. $q ( \varphi ; c _ { i } ) = P ( \mathrm { a u t h o r } 1 \mid c _ { i } )$ by Bayes’ rule; the two share positions in $\varphi$ cancel in every world, leaving the price diference $\rho .$ □

Remark 8 (Locating the defect). The arithmetic here is elementary, Bayes’ rule and cancellation, and per context $M ^ { * }$ is coherent: a mixture of coherent credences is coherent for unconditional events (the coherent set is convex; pooling only breaks under conditionalization and across contexts, Madansky, 1964; Genest and Zidek, 1986). The real content is that no NLL optimum on such a corpus can be a faithful corpus simulator and an elicitation-invariant credence function at the same time.

One objection deserves an answer. $q ( \varphi ; c )$ is a conditional probability, and conditioning on informative context is not irrational—in the corpus, style really is evidence about the author. That’s true, and it’s the point. The law-of-one-price constraint only applies across elicitations the deployer has declared meaning-preserving. At deployment, the language or format of a question is the asker’s free choice, not a sample from the corpus’s author process, and it carries no evidence about $\varphi .$ . The simulator has no way to know this: the NLL objective never sees the elicitation family. So the corpus-rational conditional and the deployment-rational credence pull apart, and the gap is real in practice—a cross-format book collects its profit from the deployed system whatever the rationale behind its prices was (cf. the simulator-vs-agent distinction of Andreas, 2022). Read “defect,” then, as a mismatch between what the objective rewards (simulation) and what deployment needs (assistantship), not as an accusation of Bayesian error. Scale and data can’t close that gap; only a cross-context term in the objective can. Within-context violations seen in practice (Zhu and Grifiths, 2024) are a separate, capacity/representation story, with negation a proven mechanism (Asher and Bhar, 2024)—complementing Thm. 7 rather than competing with it.

Theorem 9 (Priced additive degradation; δ-Adams). Say q is δ-coherent w.r.t. I if for every rule instance $\begin{array} { r } { \Gamma \vdash \psi , q ( \psi ) \ge 1 - \sum _ { \gamma \in \Gamma } ( 1 - q ( \gamma ) ) - \delta } \end{array}$ . Let $\psi _ { 0 } , \ldots , \psi _ { n }$ be a derivation chain in which step i uses $\psi _ { i }$ and at most m side premises, each priced $\geq 1 - \epsilon$ . Then

$$
1 - q ( \psi _ { n } ) \leq \big ( 1 - q ( \psi _ { 0 } ) \big ) + n ( m \epsilon + \delta ) .
$$

Moreover, any assembled credence assigning ψ<sub>n</sub> a price exceeding the bound implied by the enforced constraints admits a book with guaranteed profit equal to the excess $( b y$ Thm. $5 ( a )$ applied to the chain’s supports).

Proof. Induction over steps using the δ-inequality; the second claim is the definition of local exploitability. □

Remark 10. At $\delta = 0$ this is the classical uncertainty-accumulation result (Suppes, 1966; Adams, 1975; Adams and Levine, 1975), and approximate propagation of uncertainty through inference rules is itself old news—Gilio’s coherence-based System P bounds (Gilio, 2000, 2002), Hailperin’s optimal LP bounds (Hailperin, 1996), the graded benefits of approximate coherence (De Bona and Stafel, 2017). We’re not claiming novelty for the additive skeleton or the $\varepsilon { - } \delta$ machinery. What the trader vocabulary adds is the pricing corollary: in autoregressive sampling a step error enters the conditioning context silently, and terminal confidence carries no trace of it (Dziri et al., 2023; Bachmann and Nagarajan, 2024). Under enforced coherence constraints, understated degradation becomes an arbitrage opportunity instead—an audit-LP catches it at inference, a trader adversary penalizes it during training. Combine this with a selective rule (assert only if $q \ge 1 - \alpha )$ and calibration on the assertion set, and asserted n-step conclusions come with a price-backed error bound.

Proposition 11 (Level-wise dominance; extension of Predd et al., 2009). Let $K \supseteq C$ be any closed convex set of a ladder level (a local relaxation), and let $q \notin K$ . The Euclidean projection $\pi _ { K } ( q )$ satisfies, for every world $w \in W \subseteq K , \| \pi _ { K } ( q ) - w \| ^ { 2 } \leq \| q - w \| ^ { 2 } - \| q - \pi _ { K } ( q ) \| ^ { 2 }$ coherentization at any level strictly improves Brier score in every possible world. Hence an arbitrage penalty does not trade of against proper-score accuracy at the optimum.

Proof. The obtuse-angle property of projections onto convex sets containing w.

Proposition 12 (Spreads; restatement of Walley, 1991). Interval credences $[ l ( \varphi ) , u ( \varphi ) ]$ avoid sure loss $i f f$ there exists $p \in C$ with $l \leq p \leq u$ pointwise. The natural extension—the LP bounds [min, max] p(ψ) over $\{ p \in C : l \leq p \leq u$ on quoted propositions}—is the canonical inference from partial commitments.

Remark 13. The spread gives “I have no basis to commit” an explicit home, with exact semantics: a wide interval with a nonempty core is honest ignorance; an empty core is incoherence, whatever the width. Inference becomes an amortized natural extension—the trained model approximates the LP bounds in a forward pass, and the exact LP audits it. Credal deep learning already trains interval outputs for classifiers (Caprio et al., 2023; CreINNs, 2024; Credal Deep Ensembles, 2024); what’s new here is doing it end-to-end for an LM, the avoiding-sure-loss criterion, and the anti-collapse game below.

Conjecture 14 (Equilibrium spread). Under a proper-scoring/coverage pressure (narrowing) opposed by logical traders and calibration bettors (Foster and Vohra, 1998) settling on a verifiable anchor set (widening where unfounded), equilibrium spread width on $\varphi$ converges to the epistemic uncertainty consistent with the anchor and $\varphi$ ’s logical relations to it. In the exchangeable toy case (Beta posterior over a settleable event family), the equilibrium width equals the credible-interval width.

This conjecture is the design’s real bet: it’s what should keep learned ignorance from collapsing under gradient pressure. Zeroing the spread without evidence creates exploitable calibration books—the loss goes up, not down.

## 5 The Arbitr objective

Batch construction. A generator instantiates rule patterns of I over templated content—fictional entities to isolate structure from knowledge, real-world content to probe what the deployed model actually believes—producing supports whose world sets are known by construction. Settlement of guaranteed profit is exact and cheap at training time this way; the NP-hardness of Thm. 5(c) only bites in free text, where the trained trader has to generalize and an exact audit-LP over the answer’s neighborhood does the certifying.

Players. The model quotes $q ( \varphi ; c )$ (or [l, u]) across the elicitation family. A small trader network sees the batch, its structure, and the quotes, outputs a portfolio in $\mathcal { T } _ { k }$ , and is trained to maximize realized guaranteed profit; the model minimizes $\mathcal { L } _ { \mathrm { t a s k } } + \lambda \mathbb { E } [ \mathrm { p r o f i t } ]$ , plus proper scoring on the anchor set. Training alternates GAN-style, with a curriculum climbing the ladder in k. Prop. 11 predicts no task regression at the optimum. The experiments below test that directly, and one of them (E-A1) shows it fails once the anchoring premise is dropped.

Certificates. At inference: (i) per-word $\operatorname { I n c } _ { k , v }$ maps (“the model does not understand larger : a transitivity book yields 0.31”); (ii) an audit-LP over the local neighborhood of an answer, run by a frozen tuple (weights × elicitation family × audit-LP × anchor × decision rule)—never by the model’s own trader; (iii) abstention when the spread exceeds threshold or a book is found, with Thm. 9 supplying the selective guarantee.

## 6 E-A0: measuring the target

Before training anything, the program must establish that deployed credences are exploitable enough to matter—the pre-registered kill threshold is median Inc < 0.05 across all models and formats, in which case the program closes.

Design (pre-registered). Five support patterns—negation, certified paraphrase (LOP), conjunction, transitivity (core), and modus ponens (exploratory: natural-language conditionals are pragmatically confounded w.r.t. material implication)—over two content classes (fictional entities; real-world comparatives and capitals), 600 supports, three elicitation formats (F1: Russian yes/no; F2: English true/false; F3: Russian numbered choice), each with and without an irrelevant-context distractor (Andrews and Sarkar, 2026). Credences are read from the firsttoken distribution over answer-variant tokens; supports with answer-mass coverage < 0.5 are excluded (fraction reported). Exploitability per support is the exact LP of Prop. 3; cross-format LOP books price one proposition across formats. Models: Qwen2.5-{1.5B, 3B, 7B}-Instruct and SmolLM3-3B (non-Qwen arm); 8×V100 node; 9,360 elicitations per model. The pre-registered primary metric was the median support-level Inc over pooled core patterns (no distractor), with kill threshold 0.05.

Pre-registered verdict. The kill criterion fired: primary medians came in at 0.024 [CI95 0.019, 0.032] for 1.5B, 0.000 for 3B, 0.0001 for 7B, and 0.019 for SmolLM3 (27% excluded by coverage). By the registered criterion, within-format, template-level incoherence is negligible for instruction-tuned models at 1.5B–7B. We treat this verdict as binding for the metric as registered; everything from here is post-hoc diagnosis.
<table><tr><td></td><td>Qwen2.5-1.5B</td><td>Qwen2.5-3B</td><td>Qwen2.5-7B</td><td>SmolLM3-3B</td></tr><tr><td>Primary median Inc (pre-reg.)</td><td>0.024</td><td>0.000</td><td>0.0001</td><td>0.019</td></tr><tr><td>Negation channel, F1 / F2 (median) Conditionals (MP, exploratory)</td><td>0.472 / 0.165</td><td>0.177 / 0.001</td><td>0.038 / 0.105</td><td>0.052 / 0.117</td></tr><tr><td>Cross-format book F1↔F2: mean profit</td><td>0.345 0.151</td><td>0.500 0.021</td><td>0.495 0.064</td><td>0.439</td></tr><tr><td>share of propositions with profit &gt; 0.25</td><td>25.0%</td><td>4.0%</td><td>13.3%</td><td>0.070</td></tr><tr><td></td><td></td><td></td><td></td><td>7.5%</td></tr></table>

Table 1: E-A0 (v2 run). Top: the pre-registered primary metric—the kill threshold of 0.05 fired for every model. Bottom: a post-hoc channel breakdown, restricted to the two readout-valid formats, no distractor. Cross-format profits are per unit stake on law-of-one-price books.

Post-hoc diagnosis. Three things survive scrutiny here. The aggregator, not the target, was the problem: pooled medians are dragged down by patterns—paraphrase, conjunction, transitivity—that a confidently degenerate responder satisfies for free, so exploitability concentrates in specific channels and tails rather than showing up in an average. The negation channel $( | q ( \varphi ) + q ( \lnot \varphi ) - 1 | / 2 )$ is large at 1.5B and still present at 7B (Table 1); the conditional channel is large everywhere, but confounded by conditional pragmatics as expected going in.

The dominant phenomenon turns out to be cross-elicitation incoherence, not within-format incoherence. Between the two individually valid formats, law-of-one-price books collect mean guaranteed profit of 0.15 at 1.5B and 0.064 at 7B, with 13–25% of propositions yielding profit above 0.25; the efect doesn’t shrink monotonically with scale (7B is worse than 3B). A model can be nearly coherent inside one format and still be incoherent across two phrasings of the same question. That’s exactly the failure Thm. 7 predicts for NLL-trained models—coherent per context, but assembled in a context-sensitive way—though matching a post-hoc pattern to a theorem is not the same as testing it.

Readout validity turned out to matter as much as anything we set out to measure. Three silent elicitation failures showed up during the campaign, and the coverage filter caught none of them on its own: a Cyrillic letter-option that collided with the first token of the Russian word for “true” (which inverted answers at high coverage, so the first run was discarded); a reasoningmode chat template that consumed the first answer token; and a numbered-choice format whose answers barely correlated with the other two $( r \approx 0 . 0 2 – 0 . 4 )$ and flipped under an irrelevant preamble. What caught all three was the cross-format books pricing them—which is itself a decent argument for auditing by book rather than by filter.

Relation-level certificates. The pre-registration promised word/relation-level exploitability maps as a byproduct, and Table 2 delivers them: negation and paraphrase books pivoting on each relation, valid formats only, no distractor. They pin down where the headline anomaly at 7B actually lives—the model is close to perfectly coherent on capitals and age order, but its median book profit on mountain-height and river-length comparatives is 0.43–0.48. Its crossformat instability sits precisely on the relations it doesn’t know for sure. In the vocabulary of Def. 2: it understands capital-of and fails to understand higher-than, and the certificate says by how much.
<table><tr><td>median Inc by relation</td><td>capital-of</td><td>higher-than</td><td>longer-than</td><td>older-than</td><td>ball-in-box</td></tr><tr><td>Qwen2.5-1.5B</td><td>0.027</td><td>0.249</td><td>0.081</td><td>0.075</td><td>0.091</td></tr><tr><td>Qwen2.5-3B</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.001</td></tr><tr><td>Qwen2.5-7B</td><td>0.000</td><td>0.428</td><td>0.482</td><td>0.000</td><td>0.007</td></tr><tr><td>SmolLM3-3B</td><td>0.002</td><td>0.115</td><td>0.038</td><td>0.062</td><td>0.045</td></tr></table>

Table 2: Relation-level understanding certificates from E-A0: median violation on negation/paraphrase books pivoting on each relation (F1/F2, no distractor, n ≈ 40–100 books per cell).

Consequence for the program. E-A0 kills the naive idea that within-format logical books alone give a rich training signal at 3B and up. It points the trader class toward cross-elicitation books and the negation channel instead, and it meant a fresh pre-registration was needed before any training experiment. The exploratory/confirmatory line stayed clean throughout: nothing from the post-hoc reading above went directly into a confirmatory claim. Each later experiment (E-A1, E-A1b, E-A2) registered its own endpoints and thresholds before running, and we report their verdicts by those registrations, failures included.

## 7 E-A1: training against the exact bounded adversary

Following on from E-A0, we ran a pre-registered, minimal training experiment. Qwen2.5-1.5B-Instruct, LoRA (r=16), fp32, 1000 steps, four seeds, two FLOP-matched arms fed identical batches. Arm A (control) trains only a task loss—cross-entropy toward the true yes/no token mass on real-world facts of known truth. Arm B adds λ=1 times the profit of the exact bestresponse $\mathcal { T } _ { 2 }$ trader: negation books $| p ( \varphi ) { + } p ( \neg \varphi ) { - } 1 | / 2$ within each format, and cross-format law-of-one-price books $| p _ { F 1 } ( \varphi ) { - } p _ { F 2 } ( \varphi ) | / 2$ . This is the analytically optimal adversary within the class; a learnable trader only earns its keep once books have to be discovered in free text. Evaluation runs on held-out content (deduplicated against training propositions by exact match), untrained level-3 patterns, held-out facts, and in-context transitive-inference questions.
<table><tr><td></td><td>base</td><td></td><td>A (task only) B (task + arbitrage)</td></tr><tr><td>Negation Inc, median (held-out)</td><td>0.237</td><td> $0 . 3 3 1 \pm 0 . 0 5 3$ </td><td> $\mathbf { 0 . 0 0 3 \pm 0 . 0 0 3 }$ </td></tr><tr><td>Cross-format LOP profit, mean</td><td>0.099</td><td> $0 . 0 9 4 \pm 0 . 0 3 6$ </td><td> $0 . 0 4 3 \pm 0 . 0 1 6$ </td></tr><tr><td>Transitivity Inc, mean (untrained, level 3)</td><td>0.004</td><td> $0 . 0 2 0 \pm 0 . 0 0 3$ </td><td> $\mathbf { 0 . 0 0 6 \pm 0 . 0 0 2 }$ </td></tr><tr><td>Conjunction Inc, mean (untrained, level 3)</td><td>0.070</td><td> $0 . 1 1 8 \pm 0 . 0 1 3$ </td><td> $0 . 0 6 9 \pm 0 . 0 2 6$ </td></tr><tr><td>Held-out fact accuracy</td><td>0.553</td><td> $0 . 8 6 2 \pm 0 . 0 1 9$ </td><td> $0 . 7 8 9 \pm 0 . 0 5 2$ </td></tr><tr><td>Transitive-chain QA accuracy</td><td>0.720</td><td> $0 . 9 5 0 \pm 0 . 0 2 8$ </td><td> $0 . 7 0 4 \pm 0 . 2 0 5$ </td></tr></table>

Table 3: E-A1 (4 seeds, mean ± std). P1 (negation, threshold ×3) succeeded at ×133 with separated intervals. P2 (LOP, threshold ×2) reached ×2.2 but the seed intervals overlap— formally a failure as registered, underpowered at four seeds. The task-parity guard failed on the mean (−7.3 pp).

Per-seed structure is the finding. Arm B is bimodal, and that’s the real result. Seed $s _ { 2 }$ is an existence proof of the equilibrium we were after: negation Inc at 0.006, LOP profit halved, fact accuracy 0.855 and chain accuracy 0.975—both at parity with the control—and transitivity Inc improved eightfold. Seeds $s _ { 0 }$ and $s _ { 3 }$ bought coherence a diferent way, at the cost of truthtracking: chain accuracy at chance. This is exactly the vacuity failure predicted in §8(3). Probing shows two collapse modes: one where every credence sits at 0.5 (coherent, but says nothing), and one with extreme, mutually consistent credences that have come unmoored from truth. The theory names the missing piece in advance: the calibration-bettor leg of Conjecture 14, which this minimal setup doesn’t have. So the guard failure here confirms that the design needs all three legs—it doesn’t refute Prop. 11, which assumes a proper-scoring anchor to begin with.

Ladder. Conjunction (level 3, untrained) doesn’t move beyond noise, which is what the ladder predicts. Transitivity does move, in all four B seeds, against the letter of that prediction. Two readings are possible here, and the base model settles between them. Relative to the trained control, B improves transitivity (0.020 → 0.006); but the control itself gets worse relative to the base model $( 0 . 0 0 4  0 . 0 2 0 )$ —task-only SFT induces incoherence on patterns it was never trained on. In E-A1 the B seeds straddle the base value (0.0025–0.0076 vs. 0.004), so the safe claim is prevention: arbitrage pressure on a relation’s level-2 books largely keeps SFT from degrading level-3 coherence over the same vocabulary. Whether it actually improves on the base is settled in E-A1b (Table 4), where the main arm does $_ \mathrm { g o }$ below the base value. Either way the efect tracks vocabulary, not pattern, and E-A1b’s two-sided registered test treats its replication as

confirmatory.

Takeaways. Two things stand out. Exploitability on the trained channel drops two orders of magnitude and carries over to held-out content—graded understanding, in the sense of Def. 2, can be trained into a model. And a one-legged objective reaches the good equilibrium in some seeds and collapses in others; the next section adds the missing leg and tests whether that fixes it. Nothing here makes the model smarter on its own—it makes credences agree with each other, and in the equilibrium that works, it does so at no measured cost to the task.

## 7.1 E-A1b: the three-legged objective

A second pre-registered experiment added the missing leg, and more seeds. A Brier anchor $\mu ( p - t ) ^ { 2 }$ on book propositions of known truth $( \mu { = } 0 . 5 ;$ the negated proposition anchored at $1 - t ,$ paraphrase pairs at the same $t ;$ fictional-content propositions left unanchored, where $p \approx 0 . 5$ is the honest answer), a λ grid (main arm $\lambda { = } 0 . 3 .$ 10 seeds; dose arm $\lambda { = } 1 . 0 ,$ 4 seeds; control, 10 seeds), and symmetric, pre-registered equilibrium selection: every 200 steps a checkpoint is scored on held-out validation books and facts by the fixed composite (1−acc)+ $\cdot \mathrm { I n c } _ { \mathrm { n e g } } + \mathrm { p r o f i t } _ { \mathrm { L O P } } .$ and the best one is kept.
<table><tr><td></td><td>base</td><td>A (control, 10 seeds)</td><td> $\mathrm { { B } _ { 0 . 3 } }$  (main, 10 seeds)</td><td> $\mathrm { B 1 . 0 ~ ( d o s e , ~ 4 ~ s e e d s ) }$ </td></tr><tr><td>Negation Inc, median</td><td>0.237</td><td> $0 . 2 8 2 \pm 0 . 0 9 7$ </td><td> $\mathbf { 0 . 0 0 1 5 \pm 0 . 0 0 2 8 }$ </td><td> $0 . 0 0 0 3 \pm 0 . 0 0 0 4$ </td></tr><tr><td>Cross-format LOP profit, mean</td><td>0.097</td><td> $0 . 1 1 6 \pm 0 . 0 3 0$ </td><td> $\mathbf { 0 . 0 4 1 \pm 0 . 0 1 2 }$ </td><td> $0 . 0 5 8 \pm 0 . 0 1 1$ </td></tr><tr><td>Transitivity Inc, mean (untrained)</td><td>0.0036</td><td> $0 . 0 2 0 \pm 0 . 0 0 5$ </td><td> $\mathbf { 0 . 0 0 1 5 \pm 0 . 0 0 1 4 }$ </td><td> $0 . 0 0 1 8 \pm 0 . 0 0 0 9$ </td></tr><tr><td>Held-out fact accuracy</td><td>0.553</td><td> $0 . 8 3 2 \pm 0 . 0 3 0$ </td><td> $\mathbf { 0 . 8 4 8 \pm 0 . 0 3 1 }$ </td><td> $0 . 8 5 0 \pm 0 . 0 2 4$ </td></tr><tr><td>Transitive-chain QA accuracy</td><td>0.730</td><td> $0 . 9 4 0 \pm 0 . 0 2 8$ </td><td> $0 . 9 1 4 \pm 0 . 0 9 3$ </td><td> $0 . 9 5 1 \pm 0 . 0 3 7$ </td></tr></table>

Table 4: E-A1b. Every pre-registered criterion passed. P1, negation reduction, hit ×193 against a threshold of ×3. P2, LOP reduction, hit $\times 2 . 8$ with separated seed intervals against a threshold of ×2—the criterion E-A1 missed on power alone. The task-parity guard passed, with $\mathrm { { B } _ { 0 . 3 } }$ actually above control. Stability was $8 / 1 0$ non-collapsed seeds against a bar of $8 / 1 0 ;$ of the two flagged seeds, one is a genuine degradation and one misses the chain-accuracy bar of 0.85 by a hair, at 0.840. Upward transfer to untrained transitivity books replicated in all ten seeds, now as a confirmatory test rather than an exploratory one—and measured against the base column, the transfer improves on the untrained model itself $( 0 . 0 0 3 6 \to 0 . 0 0 1 5 )$ , not just on the SFT-degraded control (0.020): task-only SFT induces level-3 incoherence, and arbitrage training both blocks and reverses it. The dose arm shows the anchor stabilizes λ=1 too (chain accuracy 0.951, versus collapse in E-A1 without the anchor).

The composite verdict registered ahead of time—P1 and P2 and the guard and stability— passed. With all three legs in place, the informative-coherent equilibrium is reachable reproducibly, at zero measured task cost (here, slightly negative), and the coherence it buys is a property of the trained words’ inferential roles: it transfers to unseen content and one rung up the ladder. Validation traces show the composite falling from 1.03 to 0.003 within 600 steps.

## 7.2 E-A2: scale transfer and a learnable adversary

A third pre-registration tested two things at once.

7B transfer, first. Qwen2.5-7B, fp16 mixed precision, four seeds per arm, same objective and data. The trained channel does transfer—negation exploitability drops from $0 . 1 1 8 \pm 0 . 1 0 0$ to 0.000, the registered criterion passes—but the LOP criterion just misses its $\times 2$ threshold, coming in at ×1.9 (the 7B target starts closer to the floor, $0 . 0 5 9  0 . 0 3 0 )$ , and both the task-parity guard and the stability criterion fail as registered: one of four seeds collapses on facts (0.513), one misses the stability bar by 0.011, and the remaining two are healthy (facts 0.789–0.829, chains near 1.0). We report these verdicts as registered and read them this way: the mechanism transfers, but the equilibrium hyperparameters $( \lambda , \mu ,$ learning rate) tuned at 1.5B are not scale-free, and stabilizing them takes re-tuning per scale (§7.3 revisits this with an exploratory retuning pass). One side observation: under the symmetric checkpoint-selection rule, the control arm picked step 0—the untrained base model—in 2 of 4 seeds. At 7B, task-only fine-tuning degrades validation coherence faster than it improves validation accuracy, the same pattern E-A1 found at 1.5B.

Second, a learnable adversary, still at 1.5B, four seeds. The exact best-response trader is replaced by a small MLP that allocates a unit budget across a pool of four genuine books and four decoys whose claimed relation is false $( \operatorname { s a y } , \ ^ { 6 } \varphi ^ { \prime } )$ paired with $^ { 6 6 } \mathrm { { \dot { 1 } t } }$ is not the case that $\psi ^ { , \dag }$ for some unrelated ψ). Decoys settle on the full square, so their guaranteed profit is identically zero, and validity is never revealed to the trader—it sees only prices and surface features like negation markers, token overlap, and format match. Every pre-registered criterion passed. The trader learned to put 75% of its budget on genuine books, up from about 50% at the start (seeds ranged 0.70–0.79), and the model it trained reached parity with the exact-adversary arm: negation at 0.000 against a bar of 0.01, LOP at 0.041 against 0.06, and the fact guard passing at 0.836. Chain accuracy was noisier under the GAN dynamic $( 0 . 7 3 5 \pm 0 . 1 0 2$ , not part of the registration)—adversarial training carries its own stability cost, which is a tuning question for later work. What this validates is the piece that free-text operation actually depends on: an adversary can find which books are worth pricing without anyone telling it.

Open after E-A2: spread-valued credences (Prop. 12, Conj. 14), books over genuinely free, uncurated text, scale-specific stabilization past 7B, and word-level certificate maps at scale.

## 7.3 E-A3: ablation, replication, and face-validity checks

A fourth pre-registration (2026-09-11) targeted the main open issues left after E-A2, which an internal simulated-referee pass over the earlier draft had flagged (see the limitations below): attribution of the E-A1b efect to the trader term specifically, a training-free baseline, an unseen-template discrimination test, a cross-family replication, and a face-validity column for Table eftab:ea0.

Ablation: is the efect the trader’s, or the anchor’s? A new arm, AN, drops the arbitrage term entirely and trains task loss plus the Brier anchor alone $( \lambda { = } 0 , \mu { = } 0 . 5$ , otherwise identical to $\mathrm { { B } _ { 0 . 3 } }$ in E-A1b: same data, seeds, and checkpoint selection). The two channels split cleanly. On the cross-format law-of-one-price axis—the paper’s central claim—the anchor alone does nothing: AN’s mean profit $( 0 . 1 3 7 \pm 0 . 0 2 5 )$ sits at parity with the untrained control $( 0 . 1 1 6 \pm 0 . 0 3 0 )$ , and only $\mathrm { { B } _ { 0 . 3 } }$ moves it $( 0 . 0 4 1 \pm 0 . 0 1 2$ , a further ×3.3 over AN with separated intervals). The trader is doing all the work here, as claimed. On negation, the anchor is not inert: it alone cuts the control’s median from 0.282 to 0.118 (×2.4), and the trader adds a further $\times 8 0$ on top of that (0.0015). The end-to-end ×193 reported in Table 4 is real as an $\mathrm { A - t o - B }$ comparison, but on this one channel a meaningful share of it belongs to the calibration anchor supervising $p ( \neg \varphi ) = 1 - t .$ not to the arbitrage term—because the anchor already constrains that identity for every anchored proposition. We now report the negation reduction decomposed rather than as a single multiplier, and read this as a partial, disclosed confirmation of the worry that the attribution to the trader was unestablished: real on law-of-one-price, overstated if left unstated on negation.

Training-free baseline. Decode-time projection onto the coherent polytope—the same LP that defines Inc—applied with no training at all to the base model’s raw credences on 206 heldout real-content books (412 propositions), trades unevenly: it raises negation-book accuracy by 8.5 points $( 0 . 5 2 7  0 . 6 1 2 )$ by reconciling $p ( \varphi )$ against $p ( \neg \varphi )$ , but costs 3.3 points on law-ofone-price books $( 0 . 5 0 0  0 . 4 6 8 )$ , where forcing two paraphrases to agree pulls a near-chance estimate further from the truth rather than toward it. Trained arbitrage $\mathrm { ( B _ { 0 . 3 } ) }$ both removes the incoherence and improves fact accuracy over control (0.848 vs. 0.832); the cheap decode-time alternative removes incoherence by construction but does not reliably buy accuracy. Training is not trivially dominated by the free baseline.

Table 1 face validity. The suspicion that the near-zero E-A0 medians at 3B and 7B reflect confident degeneration turns out to be correct, and worse at 7B than first suspected. Adding an informativeness column $( 1 - H ( p ) / H ( 0 . 5 )$ , so 0 is an honest coin flip and 1 is confident $0 / 1 )$ to the same per-pattern cells shows Qwen2.5-7B-Instruct sitting at informativeness $\approx 1 . 0$ almost everywhere—even the 10th percentile is 0.70–0.94 across patterns—simultaneously with the nearzero Inc reported in Table 1. The two move together with scale: 1.5B’s informativeness sits in a moderate 0.3–0.7 band (where low Inc is more plausibly honest coherent uncertainty), 3B and 7B are pinned near the ceiling on almost every cell. This means the paper’s monotonicity-with-scale reading needs a caveat it did not have: what looks like models getting more coherent with scale is at least partly models getting more confidently degenerate with scale, and the exploitability measure, as registered, cannot by itself tell the two apart. SmolLM3-3B, the one non-Qwen arm, does not follow the pattern—its real-content informativeness runs 0.34–0.48 points above its fictional-content informativeness on every core pattern (it is confident where it can know and unsure where it cannot, the opposite of confident degeneration)—so this looks like a property of a particular model family’s RLHF recipe rather than a universal scale artifact. We keep the E-A0 kill-criterion verdict as registered, but Table 1’s cross-model comparison should be read alongside this column, not instead of it.

Unseen-template discrimination. The original E-A1b adapters (arms A and $\mathrm { B _ { 0 . 3 } ) }$ no longer exist as saved weights—only their aggregate metrics survived the move to this machine—so the discrimination test could not be run against the trained trader arm directly; this is logged as a resource gap, not glossed over. It was run instead on the AN checkpoints against a Russian negation template never seen in any training book (“it is false that” in place of the trained “it is not the case that”). The result is informative anyway: AN’s median Inc on the unseen lexeme $( 0 . 1 2 8 { \pm } 0 . 0 9 2 )$ is statistically indistinguishable from its median Inc on the trained lexeme $( 0 . 1 1 8 \pm 0 . 0 7 8 )$ —transfer with essentially no loss. That is weak but real evidence against the surface-trick reading of C2: if the efect were tied to a specific string, swapping the string should have cost something, and it did not. A direct trader-arm version of this test (on $\mathrm { { B } _ { 0 . 3 } }$ once family F finishes, or on a re-trained Qwen replica) is the natural next step rather than a gap left open indefinitely.

Cross-family replication. Llama-3.2 and Gemma-2 sit behind a manual license-accept gate we had no credentials for on the compute we used; that decision was logged before training, not chosen after seeing results, and the next available candidate—Phi-3.5-mini-instruct (3.8B), ungated—became family F. Six seeds per arm (fewer than the ten used for the Qwen arms; a single guest GPU rather than the earlier cluster, declared as a resource constraint ahead of running), same data, same objective, same symmetric checkpoint selection as E-A1b. The mechanism replicates. Negation Inc collapses from a per-seed median of $0 . 0 8 6 \pm 0 . 0 5 8$ (control) to a value so small it rounds to 0.0000 at four decimals for every trained seed—by mean, a more conservative statistic, $0 . 2 2 1 \pm 0 . 0 1 3  0 . 0 7 6 \pm 0 . 0 1 6 , \mathrm { ~ a ~ } \times 2 . 9$ reduction with separated intervals. Cross-format law-of-one-price profit drops $0 . 1 2 5  0 . 0 6 1 \ ( \times 2 . 0 $ , at the registered threshold). The task-parity guard passes with the trained arm above control (0.822 vs. 0.776 fact accuracy), and $5 / 6$ seeds clear the stability bar. This is the second model family, on a diferent pretraining and instructiontuning pipeline than Qwen2.5, showing the same qualitative equilibrium: coherence gained at no cost to—here, a net gain in—task accuracy.

With family F’s checkpoints available, we also re-ran the unseen-template discrimination test of the previous paragraph directly on the trained trader arm, closing the gap left by the missing original Qwen weights. On the unseen lexeme, $\mathrm { B } _ { 0 . 3 , F }$ beats control by $\times 2 4 . 3 \ ( 0 . 1 2 6  0 . 0 0 5 2$ a stable ratio, not a near-zero-denominator artifact)—a direct answer to C2 rather than the indirect one available from the anchor-only arm. The honest complication: this transfer is not fully lossless the way the anchor-only case was. $\mathrm { B } _ { 0 . 3 , F } \mathrm { ^ { \prime } s }$ own Inc rises from an almost-perfect 0.0000076 on the trained lexeme to 0.0052 on the unseen one—small in absolute terms and still far below control, but a real relative jump that the anchor-only arm did not show $( 0 . 1 1 8  0 . 1 2 8$ statistically flat). The fair reading is not “purely inferential” and not “a syntactic trick”—it’s a trader efect that generalizes to a new negation lexeme well enough to remain the dominant factor, with a partial, measurable, honestly-reported residue of lexeme-specificity on top.

7B stability, revisited (exploratory). E-A2 left the 7B stability failure as an open problem, framed as hyperparameters not being scale-free. A small, explicitly exploratory grid— two configurations, two seeds each, no thresholds registered—tested whether that instability is a property of the scale or of the particular $( \lambda , \mu ) { = } ( 0 . 3 , 0 . 5 )$ pair copied over from 1.5B. Lowering λ to 0.15, and separately raising µ to 0.8, both eliminate the collapse entirely: all four new runs land in the healthy range (fact accuracy 0.76–0.80, chain accuracy 0.98–1.0), where E-A2’s original grid produced one outright collapse and one near-miss out of four. At n=2 per configuration this cannot support a quantitative claim, but it is enough to revise the qualitative one: the right reading is not “the equilibrium hyperparameters don’t scale,” but “the equilibrium hyperparameters need re-tuning per scale, and a small grid finds a stable point quickly once you look.” A properly powered version of this grid is on the roadmap rather than folded into this paper’s confirmatory claims.

## 8 What this does not deliver

(1) Factual hallucinations. Missing-mass lower bounds (Kalai and Vempala, 2024) don’t care about the objective; a coherent flat-earther is unexploitable by logical books. Knowledge only enters through the anchor. (2) Grounding. Settlement here is internal—logic plus verifiable anchors—not reference to the world; readers who take grounding to be part of what understanding means (Søgaard, 2025) should read our title’s “understanding” as the inferential piece only. (3) Vacuity. $q \equiv 0 . 5$ is coherent on its own; content comes from the task score, the calibration anchor, and coverage together, and E-A1 showed empirically what happens when one leg is missing. (4)

The decoder. Generation stays autoregressive; Arbitr changes what the probabilities mean and which ones are admissible, not how text gets sampled. (5) State of the art. We don’t claim it on open-domain tasks; what we do claim lives on four axes—exploitability reduction, reliable chain length, risk–coverage, and relation-level certificates.

Methodological limitations. A few limits are worth stating directly, not as a formality. Elicitation is fragile: credences come from the first answer-token, and three silent readout failures happened during the campaign, all caught by cross-format books rather than the coverage filter. A sturdier elicitation—averaging over sampled answers, or semantic clustering along the lines of Farquhar et al., 2024—is the right foundation for later rounds. Evaluation books are deduplicated from training by exact proposition text, but they come from the same generative template families; the untrained-pattern and cross-format results help here, but don’t settle the question, transfer to free text remains untested, and transfer to unseen template families rests on a single new negation template (§7.3). The results lean mainly on Qwen2.5, with SmolLM3 as a non-Qwen elicitation arm and Phi-3.5-mini-instruct as a second full training replication (§7.3), all LoRA-only from 1.5B to 7B; the 7B stability failure shows the hyperparameters don’t carry over across scale unchanged, though a small exploratory retune (§7.3) suggests the fix is re-tuning, not a scale ceiling. And several registered criteria were decided by seed-interval separation at n = 4–10; efects on task-side metrics sit within noise, and the headline reductions are specific to the trained channels. No external peer review has taken place: earlier drafts were stresstested with a simulated referee pass built from language-model reviewer personas, whose findings motivated E-A3, but that is a development aid and not a substitute for expert review. Likewise, “pre-registered” here means dated criteria files written before each run and kept in the project directory; they were not deposited with an external registry, so their dates rest on our own records, and they are released with this paper so that the thresholds can at least be checked against the reported verdicts.

## 9 Conclusion

Does the model understand? That turns into an experimental question once understanding means bounded no-arbitrage over inferential roles. The definition is graded, and it’s honest about complexity: NP-hardness forces a ladder rather than an all-or-nothing verdict. It locates part of the known incoherence in LLMs inside a provable property of the NLL optimum itself (Thm. 7), turns silent error compounding into priced degradation (Thm. 9), and comes with an adversary that searches for the violations worth punishing, rather than a fixed list of them like prior consistency losses.

The empirical quantity behaved the way the theory said it would. Deployed models turn out to be exploitable exactly where Thm. 7 says the defect should live, in cross-elicitation books. Training against the bounded adversary removes that exploitability on held-out content. The vacuous equilibrium the theory predicts shows up when the calibration leg is missing, and goes away when it’s restored. Pressure on a word’s level-2 inferential role leaks upward into its level-3 role, in all ten seeds we tried. And an adversary that’s never told which books are valid learns to find the exploitable ones by itself.

What’s left open is scale-specific stabilization—a 7B seed still collapses under hyperparameters tuned at 1.5B—along with genuinely free-text books and spread-valued credences. That’s the registered continuation.

## References

E. W. Adams. The Logic of Conditionals. Reidel, 1975.

E. W. Adams, H. P. Levine. On the uncertainties transmitted from premisses to conclusions in deductive inferences. Synthese 30, 1975.

J. Andreas. Language models as agent models. Findings of EMNLP, 2022.

I. Andrews. Revealed rationality: label-free evaluation and regularization from representation theorems. arXiv:2608.05015, 2026.

I. Andrews, S. Sarkar. Dutch books for language models. arXiv:2609.02797, 2026.

N. Asher, S. Bhar. Strong hallucinations from negation and how to fix them. arXiv:2402.10543, 2024.

G. Bachmann, V. Nagarajan. The pitfalls of next-token prediction. ICML, 2024.

B. Barak, D. Steurer. Sum-of-squares proofs and the quest toward optimal algorithms. ICM, 2014.

E. Bender, A. Koller. Climbing towards NLU. ACL, 2020.

G. Betz, K. Richardson. Probabilistic coherence, logical consistency, and Bayesian learning. PLoS ONE 18(2), 2023.

R. Brandom. Making It Explicit. Harvard UP, 1994.

D. Calanzone, S. Teso, A. Vergari. Logically consistent language models via neuro-symbolic integration. arXiv:2409.13724, 2024.

M. Caprio et al. Imprecise Bayesian neural networks. arXiv:2302.09656, 2023.

K. Wang et al. Credal deep ensembles. NeurIPS, 2024.

S. K. Manchingal, S. Nikolenko, F. Cuzzolin. Credal large language models for semantic commitment under uncertainty. arXiv:2608.23244, 2026.

K. Wang et al. CreINNs: credal-set interval neural networks. arXiv:2401.05043, 2024.

G. De Bona, M. Finger. Measuring inconsistency in probabilistic logic: rationality postulates and Dutch book interpretation. Artificial Intelligence 227, 2015.

G. De Bona, J. Stafel. Why be (approximately) coherent? Analysis 77(2), 2017.

B. de Finetti. La prévision: ses lois logiques, ses sources subjectives. Ann. Inst. H. Poincaré, 1937.

N. Dziri et al. Faith and fate: limits of transformers on compositionality. NeurIPS, 2023.

Y. Elazar et al. Measuring and improving consistency in pretrained language models. TACL 9, 2021.

S. Farquhar et al. Detecting hallucinations in LLMs using semantic entropy. Nature 630, 2024.

D. Foster, R. Vohra. Asymptotic calibration. Biometrika 85(2), 1998.

S. Garrabrant et al. Logical induction. arXiv:1609.03543, 2016.

C. Genest, J. Zidek. Combining probability distributions. Statistical Science 1(1), 1986.

G. Georgakopoulos, D. Kavvadias, C. Papadimitriou. Probabilistic satisfiability. J. Complexity 4, 1988.

A. Gilio. Precise propagation of upper and lower probability bounds in System P. NMR, 2000. arXiv:math/0003046.

A. Gilio. Probabilistic reasoning under coherence in System P. Ann. Math. AI 34, 2002.

W. Gurnee, M. Tegmark. Language models represent space and time. ICLR, 2024.

T. Hailperin. Sentential Probability Logic. Lehigh UP, 1996.

A. Kalai, S. Vempala. Calibrated language models must hallucinate. STOC, 2024.

N. Kassner et al. BeliefBank. EMNLP, 2021.

Y. LeCun. A path towards autonomous machine intelligence. OpenReview, 2022.

Y. LeCun. Do autoregressive LLMs dream? Public lectures, 2023.

K. Li et al. Emergent world representations (Othello-GPT). ICLR, 2023.

C. Chen, M. Jörke, A. Goliński, M. Fedzechkina, G. Sapiro, S. Williamson, N. Foti. LLMs are not (consistently) Bayesian: quantifying internal (in)consistencies of LLMs’ probabilistic beliefs. arXiv:2605.06915, 2026.

M. A. Mannucci. Logical GANs: adversarial learning through Ehrenfeucht–Fraïssé games. arXiv:2510.22824, 2025.

A. Madansky. Externally Bayesian groups. RAND RM-4141, 1964.

Q. Liu, L. Ye, W. Ma, Y.-C. Chou, A. Yuille. Generative adversarial reasoner: enhancing LLM reasoning with adversarial reinforcement learning. arXiv:2512.16917, 2025.

E. Mitchell et al. Enhancing self-consistency with ConCoRD. EMNLP, 2022.

M. Mitchell, D. Krakauer. The debate over understanding in AI’s LLMs. PNAS 120(13), 2023.

N. Nanda et al. Emergent linear representations in world models. BlackboxNLP, 2023.

N. Nilsson. Probabilistic logic. Artificial Intelligence 28, 1986.

I. Pitowsky. Correlation polytopes. Math. Programming 50, 1991.

N. Potyka. Linear programs for measuring inconsistency in probabilistic logics. KR, 2014.

J. Predd et al. Probabilistic coherence and proper scoring rules. IEEE Trans. Inf. Theory 55(10), 2009.

M. Schervish, T. Seidenfeld, J. Kadane. Measuring incoherence. Sankhy¯a A, 2002.

M. Schervish, T. Seidenfeld, J. Kadane. Measures of incoherence. Bayesian Statistics 7, 2003.

I. Pres, L. Ruis, M. Ghebreselassie, B. Z. Li, J. Andreas. Self-CTRL: self-consistency training with reinforcement learning. arXiv:2606.18327, 2026.

P. Suppes. Probabilistic inference and the concept of total evidence. In Aspects of Inductive Logic, North-Holland, 1966.

H. Sherali, W. Adams. A hierarchy of relaxations. SIAM J. Discrete Math. 3, 1990.

A. Søgaard. Do language models have semantics? ACL, 2025.

J. Stafel. Measuring the overall incoherence of credence functions. Synthese 192, 2015.

J. Stafel. Unsettled Thoughts. OUP, 2019.

M. Wainwright, M. Jordan. Graphical models, exponential families, and variational inference. FnT ML, 2008.

P. Walley. Statistical Reasoning with Imprecise Probabilities. Chapman & Hall, 1991.

J.-Q. Zhu, T. Grifiths. Incoherent probability judgments in LLMs. CogSci, 2024.