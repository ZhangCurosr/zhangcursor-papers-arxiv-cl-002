# When Does Selection Replace Extraction? A Pre-Registered Test of Agent Memory with a Typed Decision Model

Rishabh Sharma Independent Researcher rishabh.sharma1103@gmail.com

Rishika Lall Independent Researcher lallrishika@gmail.com

## Abstract

Does conversational memory need LLMextracted facts, or is selecting the right raw turns enough? Published results disagree. Extraction-based systems report gains from distilled facts. Recent studies find raw history with good ranking does as well, but disagree about whether ranking matters. We ran a preregistered study on held-out LoCoMo conversations and LongMemEval. At a tight budget on LoCoMo, raw turns selected by a single call to Jev, a typed decision model, are non-inferior to an LLM-extraction memory (one-sided 95% bound −3.0 points against a −5-point margin). Blind human grading narrows the margin but does not change the result. Raw turns cost 3,061× less to write, and the result holds with a second answer model. Within this study, reranking’s gain shrinks as the budget grows. It adds 17.4 points on LoCoMo and 9.1 on LongMemEval when three of 30 candidates are kept. At generous budgets it adds 1.5 and 1.1, and extraction systems are more accurate. This suggests why published results disagree. At matched context, Jev selects as accurately as an LLM reranker (non-inferiority bound −2.0) at a third of the latency, and more accurately than a multi-call graph traversal. Reranking lowers correct abstention. Plans, code and graded answers are released.

## 1 Introduction

Does conversational memory need LLM-extracted facts? The literature is split. Extraction-based systems report gains from distilling conversations before retrieval: mem0 (Chhikara et al., 2025) extracts facts from each message, the LongMemEval design study (Wu et al., 2025) finds that expanding index keys with extracted facts helps retrieval, and SeCom (Pan et al., 2025) segments sessions and compresses the segments before retrieval. Recent studies find that raw history, ranked well, does as well or better: SmartSearch (Derehag et al., 2026) retrieves from raw history with a deterministic pipeline and a learned ranking stage, and Fidelity Before Structure (An, 2026) finds verbatim chunks ahead of LLM-extracted artifacts in a controlled comparison. These two also disagree with each other: SmartSearch identifies ranking as the bottleneck, while Fidelity finds that reranking adds little.

We propose that the context budget, the number of retrieved items the answer model reads, accounts for part of the disagreement. When the budget keeps a few of many candidates, the choice of items decides the answer. Selection then matters, and a good selector over raw turns can stand in for extraction. When the budget is generous, similarity order already includes most of the evidence. Ranking then adds little, and extracted facts, which are more compact, are more accurate. We test the first half of this under a pre-registered plan, on conversations never used for development. Are raw turns with a single reranking call non-inferior to a strong extraction-based memory at a tight, matched budget? And how does the rerank’s value change as the budget grows? Non-inferiority means we test whether raw turns are at most 5 points worse, rather than whether the two systems differ at all.

The selector is Jev, TypeSafe’s typed decision model (Almeida, 2026; TypeSafe AI, 2026). It answers a fixed-option question with a probability in one short request. The extraction system is engram v2. engram is a memory system we built and described in an earlier preprint (Sharma, 2026c): an LLM extracts facts, and a typed decision model makes every later decision about them. engram v2 is the version used here; it extracts facts with gpt-4o-mini and types, relates and updates them with Jev. It was the most accurate system on our development conversation, and we chose it as the comparator because the test could fail against it. Choosing our own extraction system as the comparator gave us every reason to make it strong. engram v2’s read path is Turns + Jev’s read path over extracted facts instead of raw turns. H1 therefore holds the selector fixed and varies only what is stored: a controlled comparison of extraction and raw turns, in the spirit of Fidelity Before Structure’s.

Our contributions:

1. Within this study, reranking’s gain over similarity search shrinks as the budget grows: from +17.4 to +1.5 points on LoCoMo and from +9.1 to +1.1 on LongMemEval, as the answer model reads three and then twenty of 30 candidates. This suggests an explanation for why SmartSearch finds ranking decisive and Fidelity Before Structure finds it marginal, which we offer as an interpretation (§5.3, §6). Kang et al. (2026) give complementary evidence that memory design choices depend on the budget.

2. To our knowledge, the first pre-registered non-inferiority test in conversationalmemory evaluation, on held-out conversations, with a second answer model and blind human grading. It bounds what extraction adds at a tight budget. Under the worst grading we applied, extraction adds at most 4.7 points. Human grading puts the difference at −1.7 to −2.6 points (§5.1, §5.4). LazyMem (Yu et al., 2026) also prespecifies its automatic metric and uses a human audit as a sensitivity analysis; our study registers a plan, a primary test and a non-inferiority margin before any held-out run.

3. To our knowledge, the first answer-level, matched-context evaluation of a typed decision model as the selector. At matched context, Jev is non-inferior to a gpt-4o-mini listwise reranker (S4, lower bound −2.0) at about a third of the latency. It is also more accurate than Jev-Mem’s multi-call Jev graph traversal at matched context (S2) (§5.2, §5.7). This is consistent with MemReranker (Li et al., 2026a), where a small reranker matches gpt-4o-mini on key retrieval metrics.

4. Diagnostics. A per-category decomposition of where the selection ceiling comes from: shortlist misses (11.5% of questions) and rerank drops (9.2%), with temporal evidence dropped most at the rerank (§5.6). And a finding about evaluation: the LLM judge’s leniency interacts with answer length, so judge–human agreement differs by system (§5.4, §6). Judge leniency itself is documented elsewhere (Ren et al., 2026; Penfield Labs, 2026); the interaction with answer length is what we add.

What is not new: raw turns plus a reranker is a known pattern (Derehag et al., 2026; Wu et al., 2026), and engram v2 is a version of our earlier system (Sharma, 2026c). The contribution is the test, the within-study budget result and the typed selector, not a new architecture.

## 2 Related Work

Raw history against extraction. SmartSearch (Derehag et al., 2026) argues that neither LLM structuring at ingestion nor learned retrieval policies are necessary, and ranks raw history with a CrossEncoder and ColBERT fusion stage. Fidelity Before Structure (An, 2026) swaps only the stored representation inside one pipeline, with gpt-4o answering and a gpt-4o-mini judge giving binary grades. Verbatim chunks lead LLM-extracted artifacts by 15.9 points on LoCoMo (categories 1– 3, 699 questions). They lead by 22.0 points on LongMemEval-S (500 questions). In an external system anchor (their Appendix D), the official Mem0 package also trails verbatim chunks. With a gpt-4o-mini answerer it scores 36.6% against 47.9% (categories 1–3). With gpt-4o it scores 54.7% against 69.9% (1,540 questions, categories 1–4). Nano-Memory (Wu et al., 2026) answers from raw turns with retrieval and generation alone; EMem (Zhou and Han, 2025) builds a strong baseline from near-verbatim discourse units; Zeng et al. (2024) sweep chunks, triples, facts and summaries and find chunk-based and mixed stores strongest on LoCoMo; the LongMemEval design study (Wu et al., 2025) finds round-level storage best and factaugmented index keys helpful; and Letta reports 74.0% on LoCoMo for a gpt-4o-mini agent that stores conversation history in files, with no judge stated (Letta, 2025). Zero-Mem (Xiao et al., 2026) removes LLM calls from every memory operation: it keeps raw traces and retrieves with BM25, dense embeddings and an entity graph built by a non-generative NER model. It is fully deterministic, whereas our selector is a typed decision model, and we test selection against extraction under pre-registration; it reports F1, so its numbers are not comparable with ours. LazyMem (Yu et al., 2026) defers memory construction to query time:

a trained model retains and compresses only the query-relevant content of a broad retrieved pool. It rewrites text at read time, while our selector only chooses among raw turns. Our result agrees with this lineage at tight budgets and is smaller and more cautious than Fidelity’s gap, as expected for an extraction system that keeps source quotes. We extend it with a registered non-inferiority margin, held-out conversations, a budget analysis and percategory recall.

Budgets and compression. Kang et al. (2026) study a complementary budget-dependent decision: given identified evidence, whether to retain raw records or replace them with generated consolidations under a fixed answer-time budget. Consolidation helps when the budget is too small for the relevant raw evidence (up to 48 points on Long-MemEval at 32 tokens), and retention is preferable once it fits. We study end-to-end selection instead: whether query-time ranking of raw turns can substitute for write-time extraction when only a fraction of the retrieved candidates reaches the answer model. The two sets of results are consistent. Our tight budgets (129–265 tokens) already fit several raw turns, the regime where Kang et al. also find retention competitive. Their intervention acts after evidence has been identified, while ours tests whether selection itself removes the need for extraction. The extraction advantage we observe at generous budgets is partly our selector’s read-path ceiling (§5.3), and does not contradict their finding. EMBER (Li et al., 2026b) learns which verbatim evidence to retain under a fixed pre-query token budget, and a controlled comparison of memory substrates finds that none dominates across operating regimes (Huang et al., 2026).

Extraction systems. mem0 (Chhikara et al., 2025) extracts facts per message; we test mem0 OSS 2.1.0, and newer mem0 releases report higher, self-reported numbers (Mem0 Engineering Team, 2026). Graphiti/Zep (Rasmussen et al., 2025), A-MEM (Xu et al., 2025), MemGPT/Letta (Packer et al., 2023), EverMemOS (Hu et al., 2026) and Memora (Xia et al., 2026) structure memory with LLM calls at write time. Several recent systems make extraction cheaper: SimpleMem (Liu et al., 2026) compresses interactions into compact indexed memory units, LightMem (Fang et al., 2026) filters and groups content in stages and consolidates offline, and LeanMem (Liao et al., 2026) stores each kind of content as profile, event or sourcegrounded record memory.

Typed decisions in memory. Jev-Mem (Jiang et al., 2026a) was the first memory system built on Jev; it uses typed questions for typing, relations, routing, traversal and stopping over a multi-graph store. The AtMem–Jev article (Taghia, 2026) reports that Jev reranking raises ranking metrics. We measure a Jev reranker at the answer level, against an LLM reranker and against Jev-Mem at matched context.

Reranking in conversational memory. Smart-Search finds ranking to be the bottleneck; Fidelity finds reranking marginal. Training-Free Lexical– Dense Fusion (Lysenstøen, 2026) reports an offthe-shelf cross-encoder lowering Hit@1 on conversational queries, and ConvMemory v2 (Pan, 2026) reports gains from a cross-encoder fine-tuned for conversation. MemReranker (Li et al., 2026a), a small reasoning-aware reranker for agent memory, matches gpt-4o-mini on key retrieval metrics, consistent with our finding that a typed decision model selects as accurately as an LLM reranker (S4). EARM (Feng et al., 2026) treats LLM reranking as a per-query cost and amortizes it by reusing past relevance scores; the same cost argument motivates a selector that answers in one short request. Our budget analysis offers one way these findings fit together (§6).

Evaluation validity. Held-out conversation splits of LoCoMo already exist (Yan et al., 2025; Jiang et al., 2026b); our design adds preregistration and a non-inferiority margin. Same Ranking, Different Winner (Panthi and Abdelfattah, 2026) shows that retrieval credit depends on the stored form; we score shortlist recall on raw turns only. Fidelity reports judge–human agreement of $\kappa = 0 . 8 9 7$ on 100 questions, similar for short and long answers, with a judge instructed to be strict (their Appendix J.4); with mem0’s lenient LoCoMo judge, we found that agreement depended on the system’s answer style (§5.4, §6).

## 3 Systems

## 3.1 Setup and notation

A conversation is a sequence of turns x<sub>1</sub>, . . . , x<sub>T</sub>, each with its session date. A memory system has a write function W that builds a store, a read function that selects part of it for a question q, and an answer model L:

$$
\begin{array} { l l } { { M = W ( x _ { 1 } , \ldots , x _ { T } ) , } } & { { S ( q ) \subseteq M , } } \\ { { a = L \big ( q , \mathrm { r e n d e r } ( S ( q ) ) \big ) } } & { { } } \end{array}\tag{1}
$$

For Turns + Jev and Turns + cosine, M is the turns themselves, each stored with its date. For engram v2, M is a set of facts that an LLM extracted, each with a source quote and a validity window. For Jev-Mem, M is a graph whose nodes are turns and whose edges Jev types.

A typed question Q has a fixed option set $O _ { Q }$ Given a state s, Jev returns a probability for every option in one request, and a decision is the most probable option with that probability as its confidence:

$$
\begin{array} { r l } & { p ( o \mid s , Q ) , \quad o \in O _ { Q } , } \\ & { d ( s , Q ) = \arg \underset { o \in O _ { Q } } { \operatorname* { m a x } } p ( o \mid s , Q ) , } \\ & { \pi ( s , Q ) = \underset { o \in O _ { Q } } { \operatorname* { m a x } } p ( o \mid s , Q ) } \end{array}\tag{2}
$$

Jev does not generate text. It scores a closed set of options, so its output needs no parsing, and its confidence is a probability that can be thresholded.

Reading starts from a cosine shortlist of the n stored items closest to the question, with $n = 3 0$ and $e ( \cdot )$ the embedding:

$$
C ( q ) = \underset { n } { \mathrm { T o p } } \ \cos \bigl ( e ( q ) , e ( m ) \bigr ) , \quad m \in M\tag{3}
$$

Turns + Jev asks Jev one relevance question $Q _ { \mathrm { r e l } }$ about every shortlisted turn, in one request. It keeps the turns whose relevance $\rho$ exceeds $\tau = 0 . 5$ , in decreasing $\rho .$ The top $f = 1 0$ turns of the shortlist by cosine follow them (the cosine floor), and the answer model reads the first k:

$$
\begin{array} { r l } & { \rho ( m , q ) = p ( \mathrm { y e s } \mid m , q , Q _ { \mathrm { r e l } } ) , } \\ & { R ( q ) = \{ m \in C ( q ) : \rho ( m , q ) > \tau \} , } \\ & { S _ { k } ( q ) = \mathrm { f r s t } _ { k } \big ( R ( q ) \mathrm { b y } \rho , } \\ & { \mathrm { t h e n } \mathrm { T o p } _ { f } C ( q ) \setminus R ( q ) \mathrm { b y } \mathrm { c o s i n e } \big ) } \end{array}\tag{4}
$$

Turns + cosine reads the first k turns in cosine order. Below, the subscripts J, cos and E denote Turns + Jev, Turns + cosine and engram v2. Systems are compared at matched context. With $T _ { A } ( k )$ the mean rendered tokens per question of system A at $k ,$ and $k _ { B }$ the comparator’s own k (three), Turns + Jev runs at the k whose tokens are closest, ties going to the larger k:

$$
k ^ { * } = \arg \operatorname* { m i n } _ { k } \left| T _ { J } ( k ) - T _ { B } ( k _ { B } ) \right|\tag{5}
$$

The rerank’s gain over similarity search at the same k is

$$
\Delta ( k ) = \operatorname { A c c } _ { J } ( k ) - \operatorname { A c c } _ { \cos } ( k )\tag{6}
$$

The primary test H1 compares Turns + Jev with engram v2 question by question. Let $c _ { i } ^ { A }$ be one if system $A \ ' \mathrm { s }$ answer to question i is judged correct and zero otherwise, $d _ { i }$ the difference Turns + Jev minus engram v2, <sup>¯</sup>d its mean and s its standard deviation over the $N = 7 7 8$ questions. Turns + Jev is non-inferior if the one-sided 95% lower bound clears the margin δ = 5 points, with $z _ { \mathrm { 0 . 9 5 } } = 1 . 6 4 5 $

$$
\begin{array} { l } { { d _ { i } = c _ { i } ^ { J } - c _ { i } ^ { E } , } } \\ { { \bar { d } - z _ { 0 . 9 5 } \displaystyle \frac { s } { \sqrt { N } } > - \delta } } \end{array}\tag{7}
$$

The share of the gap between similarity search and extraction that the rerank closes, at the H1 budget, is

$$
G = { \frac { \operatorname { A c c } _ { J } - \operatorname { A c c } _ { \cos } } { \operatorname { A c c } _ { E } - \operatorname { A c c } _ { \cos } } }\tag{8}
$$

with each system at its matched k. The total cost per question adds the write cost of $r _ { w }$ turns, the turns written per question asked (4.0 on the benchmark), to the read and answer costs:

$$
C = r _ { w } c _ { \mathrm { w r i t e } } + c _ { \mathrm { r e a d } } + c _ { \mathrm { a n s w e r } }\tag{9}
$$

## 3.2 The systems

All systems use gpt-4o-mini to answer, textembedding-3-small to embed and jev-1.13.0 for every Jev decision. Figure 1 contrasts the write and read paths of Turns + Jev, engram v2 and Jev-Mem. We give the raw-turn systems descriptive names: Turns + Jev, Turns + cosine and Turns + LLM, registered as T0R, L0 and T0R-LLM in the plan. The post-hoc variant T0R-wide is Turns + Jev (wide).

Turns + Jev. The write path embeds each turn and stores it as “[date] speaker: text”, with no extraction and no LLM call. The read path is Eqs. (3)

![](images/da2845fb842b8330477166158fa215d4e2978cf47d8c13d85c1d91f6c237c702.jpg)  
Read time, per question: one answerer and one judge for every system  
Figure 1: Write path (per turn, top) and read path (per question, bottom) of Turns + Jev, Turns + cosine, engram v2 and Jev-Mem. Border colour says what does the work: code (blue), an LLM call (amber), a Jev typed decision (purple), a store (green), the answer model (red) and the judge (teal). The grid gives LLM calls and Jev requests per turn and Jev requests per question, from each system’s code (Jev-Mem: its default profile); Turns + Jev and Turns + cosine share a write path and differ only per question, where Turns + Jev makes one Jev request and Turns + cosine none. A design diagram; no measured data.

and (4); Jev’s relevance question asks whether each turn helps answer the question.

Turns + cosine. The same store, read in cosine order with no Jev call.

Turns + LLM. Turns + Jev’s store and shortlist, scored by a gpt-4o-mini listwise reranker instead of Jev.

Full context. Every turn of the conversation, rendered as Turns + Jev renders a line, in the answer prompt.

engram v2. engram (Sharma, 2026c), our earlier system. An LLM extracts facts from each message with mem0’s extraction prompt. Jev then answers typing questions and relation questions against up to ten candidate facts, and a belief policy closes superseded facts. The read path is Turns + Jev’s over facts instead of turns. v2 changes two things from the preprint’s version, both fixed on the development conversation before the v2-frozen tag. Extraction receives each message’s session date, so relative dates resolve to the conversation’s time. A same-attribute gate, one more Jev question per candidate, lets an update close a stored fact whose relation type differs. The v2 plan’s Deviations section (docs/V2\_PLAN.md §12) records both.

mem0 2.1.0. The default add() path: one LLM extraction call per message, with the session date as the observation date; reads are vector search.

Jev-Mem. Jev-Mem at commit 81574eb with its default profile and jev\_model pinned to jev-1.13.0, driven through its own API. Each turn is a node; each write makes two Jev requests (memory type, relations), and each read routes, traverses and stops with between two and sixteen Jev requests. Its returned turns are rendered as “[date] speaker: text” and answered with our prompt; its own prompts, best-of-three selection and judge are not used.

A worked example. Figure 2 traces one heldout question through Turns + Jev and engram v2 at the H1 budgets. It was chosen by a fixed rule, not for effect. The question must:

1. be an H1 question that the judge and the human grader both scored correct for Turns + Jev and wrong for engram v2;

2. be temporal (8 questions meet the first two conditions);

3. have an evidence turn that Jev’s rerank kept, not

the cosine floor;

4. have replayed contexts that match the recorded ones;

5. have the shortest Turns + Jev context among those left.

engram v2 extracted the evidence turn, but under the wrong speaker. Three other facts outranked it at k=3. Turns + Jev kept the verbatim turn with its date. Appendix J shows the opposite case, chosen by the same kind of rule: there the evidence turn never reached Turns + Jev’s shortlist, while engram v2’s extracted fact did.

## 4 Study Design

Pre-registration. The plan was deposited before any run on the data below (10.5281/zenodo.22970745, commit b3c5dc5, tag v3-frozen). An amendment, with the outcome paragraphs used in §5.1, followed (10.5281/zenodo.22977848, commit efae0b6, tag v3-amended) (Sharma, 2026a,b). It was deposited after Batch A, so the results of S1 and S2 were known when it added S7, the full LongMemEval run and the second answer model. It was deposited before any primary-test (H1) result was seen.

Data. LoCoMo (Maharana et al., 2024) numbers its question categories. We name them 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop and 5 adversarial, which matches the dataset’s counts over all ten conversations (282, 321, 96, 841 and 446 questions). The primary data are conv-44, conv-47, conv-48, conv-49 and conv-50, never run by any system before this study. They hold 3,122 turns and 778 scored questions, plus 209 adversarial ones. The scored questions are 140 multi-hop, 165 temporal, 50 open-domain and 423 single-hop. conv-30, conv-41, conv-42 and conv-43 (610 scored questions), held out in an earlier study, give an exploratory replication. Development used conv-26 only. LongMemEval\_S cleaned (Wu et al., 2025) provides a registered sample of 70 questions, with user turns only. The amendment added all 500 questions with user and assistant turns: 470 scored and 30 abstention.

Stack. Answers and judgments use gpt-4o-mini at temperature 0 with mem0’s LoCoMo answer and judge prompts; the judge returns CORRECT or WRONG. Tokens are counted with o200k\_base over the memory block the answer model sees.

Token matching. Every comparison between systems holds context fixed. The comparator runs at k=3, its natural setting, and Turns + Jev runs at the matched k of Eq. (5). A retrieval-only sweep over k from one to thirty gave the token counts. The sweep and the chosen k were saved before Turns + Jev answered at that k.

Tests. The primary test H1 is Eq. (7), with the judge’s labels. The margin is half the rerank’s measured effect on the development conversation. The seven secondary tests, under Holm correction at family-wise 0.05, are exact two-sided McNemar tests except S4, a non-inferiority test with the same margin: S1 Turns + Jev against Turns + cosine, S2 against Jev-Mem, S3 against mem0 and S4 against Turns + LLM on LoCoMo; S5 against mem0 and S6 against Turns + cosine on the LongMemEval sample; S7 against Turns + cosine on the full Long-MemEval set. The plan’s power analysis put the probability of passing H1 at 0.89 if the development difference held.

Checks. H1 and S1 were re-answered by Llama 3.3 70B Instruct via OpenRouter from the same contexts and judged by the same judge; a result is called model-robust only if it holds under both answer models. The first author graded every question on which the judge found exactly one of H1’s two answers correct, blind to system and judge label (§5.4, Appendix C). The prespecified judge decides the test, and the human audit is a sensitivity analysis, as in LazyMem (Yu et al., 2026). Shortlist recall measures where the LoCoMo evidence turns fall (§5.6, Appendix D).

Deviations. Every change after registration is dated in the plan and listed in Appendix B. Apart from the amendment above, none changed a test, the margin or the planned interpretation; the mem0 serving deviation adds a caveat to S3.

## 5 Results

## 5.1 Primary test (H1, registered)

The registered outcome paragraph, filled in (quoted with the plan’s names; T0R is Turns + Jev):

Pass. At matched context (265 tokens; engram v2 251), raw turns with a single rerank call were non-inferior to LLMextraction memory: difference −0.5 points, one-sided 95% lower bound −3.0, above the registered −5 margin. Whatever accuracy extraction adds at this budget is under 3.0 points, at 3,061× the write cost. The rerank closes 94% of the gap between similarity search and extraction. By category, T0R did not trail on multi-hop (74.3 vs 72.1, n=140) and trailed on open-domain (56.0 vs 60.0, n=50), contrary to what we registered for multi-hop and as we registered for opendomain.

Question: When did Calvin meet with the creative team for his new album?   
gold answer: 8 June, 2023 evidence: D8:1 conv-50, a held-out conversation

kept by Jev (P > 0.5) cosine floor kept by Jev, then cut at k not read

![](images/c830f4e0bb0422824eec147b12a8f3ef54e9c7480a49e39036e68789c491bc0d.jpg)  
Figure 2: One held-out question traced through both read paths, replayed offline from the frozen stores and the call cache (no API call; both contexts match the recorded token counts: Turns + Jev 292, engram v2 309). Each column lists the top four of the 30-item cosine shortlist and every item the answer model read, in cosine order, with Jev’s P(relevant): purple rows were kept by Jev, blue rows by the cosine floor, and the dashed row was kept by Jev but ranked below the cut at k=3. An illustration chosen by the rule in §3, not evidence. Appendix J shows a question where extraction wins, chosen by the same kind of rule.

Table 1: H1: Turns + Jev at k=6 (265 tokens per question) against engram v2 at k=3 (251 tokens), 778 scored questions of the five held-out conversations. The judge row is the registered test; the human rows replace the judge’s labels on the 141 graded discordant questions. Differences and bounds in points.
<table><tr><td>Grading</td><td>Turns + Jev</td><td>engram v2</td><td>Difference</td><td>One-sided 95% bound</td><td>Two-sided 95% CI</td><td>Non-inferior (margin -5)</td></tr><tr><td>Judge (registered)</td><td>77.0%</td><td>77.5%</td><td>-0.5</td><td>-3.0</td><td>[-3.5, +2.5]</td><td>yes</td></tr><tr><td>Human, strict</td><td>76.3%</td><td>78.0%</td><td>-1.7</td><td>-3.9</td><td>[-4.3, +0.9]</td><td>yes</td></tr><tr><td>Human, lenient</td><td>77.4%</td><td>79.9%</td><td>-2.6</td><td>-4.7</td><td>[-5.1, -0.03]</td><td>yes</td></tr></table>

The one-sided p-value is 0.0017. The conversation bootstrap puts the fifth percentile of the difference at −2.3 points. 69 questions were answered correctly only by Turns + Jev, and 73 only by engram v2. Human grading moves the difference to between −1.7 and −2.6 points. It moves the bound to between −3.9 and −4.7. Under lenient grading the two-sided interval lies just below zero. By that grading engram v2 is more accurate, still inside the margin (Figure 3). This depends on keeping the judge’s labels for the one ungraded question; dropping it moves the interval’s upper end to +0.09 (Appendix C). We therefore state the result as non-inferior within a 5-point margin under every grading we applied. At this budget, extraction adds at most 4.7 points. The category comparisons are descriptive; the categories are small and the differences are not tested.

![](images/3b706f9aac8cb954172647a4571e174394e9bdaefd24bb7d50f21102a5327ec4.jpg)  
Figure 3: H1 (registered) as a forest plot: Turns + Jev at k=6 minus engram v2 at k=3, in points, on the 778 questions of the five held-out conversations. Bars are two-sided 95% intervals; the red tick is the onesided 95% lower bound, tested against the −5-point margin (dashed). The judge row is the registered test; the human rows replace the judge’s labels on the 141 graded discordant questions (§5.4); the Llama 3.3 70B row re-answers from the same contexts (the answermodel check of §4).

G Eq. (8) uses Turns + cosine at its own matched k (6), where it scores 68.6%. At the same token budget, one rerank call closes G = 94% of the accuracy gap between Turns + cosine and engram v2 (77.5%). The write-cost ratio uses held-out measurements at list prices. engram v2 costs \$1.865 per 1,000 turns, and Turns + Jev \$0.00061 (embeddings only).

## 5.2 Secondary tests (S1–S7, registered)

S1, S2, S3, S4 and S7 are rejected after Holm correction; S5 and S6 are not. On LoCoMo, at matched context, Turns + Jev was more accurate than similarity search (S1), Jev-Mem (S2) and mem0 (S3). It was non-inferior to the LLM reranker (S4: difference −0.4 points, lower bound −2.0). S3 carries a caveat: mem0’s extraction was served through OpenRouter, about half of it by Azure (Appendix G). On the LongMemEval sample, S5 detected no difference between Turns + Jev and mem0 on 30 knowledge-update questions, which is too few to establish equivalence, and S6 detected none between Turns + Jev and Turns + cosine on 70 questions. On the full set, S7 found Turns + Jev more accurate than Turns + cosine by +9.1 points. By the registered rule, LongMemEval holds: S7 favours Turns + Jev after Holm correction and S5 does not favour mem0. Figure 4 shows the paired differences with their intervals.

![](images/17aaf23a19dc1d919a69f607b6725231bfa5a379b5057d9c0be4b6f81da40f04.jpg)  
Figure 4: Secondary tests S1–S7 (registered): Turns + Jev minus the comparator, in points, with paired 95% intervals; the Holm-adjusted p is printed at the right, and purple rows are rejected after Holm correction (grey rows are not). S4 is a non-inferiority test against the −5- point margin (dashed). LoCoMo tests use 778 questions; S5 and S6 use the LongMemEval sample (30 and 70 questions), S7 the full set (470).

## 5.3 The budget dependence of reranking

The rerank’s gain over similarity search, $\Delta ( k )$ of Eq. (6), depends on how many candidates the budget keeps (Figure 5). On LoCoMo it is +17.4 points at k=3 and +1.5 at k=20. On the full Long-MemEval set it is +9.1 points at k=3 (S7) and +1.1 at k=20. With three of 30 candidates kept, ordering decides which evidence reaches the answer model; with twenty kept, cosine order already includes most of it. The k=3 gains are registered tests (S1, S7); the k=20 differences are descriptive. The k=20 differences also mix the budget with Turns + Jev’s own read-path ceiling. At k=20, Turns + Jev reads 496 tokens against Turns + cosine’s 826. It keeps only turns scored above 0.5, plus the 10- turn cosine floor, so it often cannot fill twenty slots. The post-hoc Turns + Jev (wide), which keeps the top k with no cut-off, scored 81.5% at 2,000 tokens (§5.6). So the decline may be less steep for a wider read path. This is a post-hoc hypothesis, not a result.

Turns + Jev at k=3 is within 1.0 points of full context on LoCoMo while reading 139 tokens per question instead of 23,631. At generous budgets the ordering reverses (Table 3, Figure 6). engram v2 at k=20 was the most accurate system we measured: 82.4% with 1,238 tokens. Jev-Mem at k=40 followed, with 80.3% at 1,987 tokens. mem0 at k=20 scored 78.7% with 839 tokens. Full context scored 78.3%. Turns + Jev stays near 77.6% at any k (§5.6). This comparison is descriptive, not a registered test, and the systems are not token-matched. In particular, Jev-Mem at its default k=40 is more accurate than Turns + Jev’s ceiling (80.3% against 77.6%); Turns + Jev beats it only at matched context (S2).

Table 2: Secondary tests. In each, the comparator runs at k=3 and Turns + Jev at its matched k. “Only Turns + Jev” and “only other” count questions answered correctly by one system. S4 is a non-inferiority test (one-sided p); the rest are exact two-sided McNemar tests. Holm adjustment over S1–S7. LoCoMo tests use 778 questions; LongMemEval ingestion: user turns for S5–S6, user and assistant turns for S7. Bold: rejected after Holm adjustment (family-wise 0.05).
<table><tr><td>Test</td><td>Turns + Jev vs</td><td>Turns + Jev k</td><td>Turns + Jev</td><td>Other</td><td>Only Turns + Jev / only other</td><td>p</td><td>Holm p</td></tr><tr><td>S1</td><td>Turns + cosine (LoCoMo)</td><td>3</td><td>77.2%</td><td>59.9%</td><td>152 / 17</td><td>2.7e-28</td><td>1.9e-27</td></tr><tr><td>S2</td><td>Jev-Mem (LoCoMo)</td><td>4</td><td>77.0%</td><td>70.6%</td><td>98 /48</td><td>4.3e-5</td><td>1.3e-4</td></tr><tr><td>S3</td><td>mem0 (LoCoMo)</td><td>3</td><td>77.2%</td><td>68.5%</td><td>134/66</td><td>1.7e-6</td><td>7.6e-6</td></tr><tr><td>S4</td><td>Turns + LLM (LoCoMo, non-inferiority)</td><td>3</td><td>77.2%</td><td>77.6%</td><td>28 /31</td><td>1.5e-6</td><td>7.6e-6</td></tr><tr><td>S5</td><td>mem0 (LongMemEval, 30 knowledge-update)</td><td>2</td><td>70.0%</td><td>70.0%</td><td>4/4</td><td>1.00</td><td>1.00</td></tr><tr><td>S6</td><td>Turns + cosine (LongMemEval sample, 70)</td><td>3</td><td>68.6%</td><td>65.7%</td><td>7/5</td><td>0.77</td><td>1.00</td></tr><tr><td>S7</td><td>Turns + cosine (LongMemEval, 470)</td><td>3</td><td>66.8%</td><td>57.7%</td><td>61 / 18</td><td>1.3e-6</td><td>7.6e-6</td></tr></table>

![](images/270f57a4f2d2ffd6990db871fbac8318b359d8e389d3027d47e8daa45dec774f.jpg)  
Figure 5: The rerank’s gain over similarity search (Turns + Jev minus Turns + cosine, paired, in points, with 95% intervals) against k. LoCoMo: 778 questions of the five held-out conversations at k=3, k=6 and k=20; LongMemEval: 470 non-abstention questions, user and assistant turns, at k=3 and k=20. The k=3 points are registered tests (S1, S7); the others are descriptive.

## 5.4 Robustness: a second answer model and blind human grading

With Llama 3.3 70B Instruct answering from the same contexts, H1 still passes. Turns + Jev scores 75.3% and engram v2 75.6%. The difference is −0.3, with one-sided bound −2.9. S1 also passes: Turns + Jev scores 74.9% and Turns + cosine 57.6% $( \mathrm { p } = 5 . 8 \mathrm { e } - 2 6 )$ . Both are therefore model-robust by the registered rule. Of 3,112 rebuilt contexts, all but 4 matched their recorded token counts exactly; the four come from near-tie reorderings. OpenRouter served Llama through 9 providers whose numeric precision may differ.

The first author graded, blind, both answers to each of H1’s 142 judge-discordant questions (284 rows; one question was left ungraded). Agreement with the judge was 81% under the strict mapping and 79% under the lenient one (Appendix C). Many judge-discordant pairs were not discordant to the human grader: under the strict mapping both answers were correct for 17 questions and both wrong for 18. The judge credited Turns + Jev’s short answers more readily and engram v2’s list-style answers less (agreement on engram v2’s answers 77% under the lenient mapping, against 82% on Turns + Jev’s), which is why human grading widens the gap.

## 5.5 Long histories (LongMemEval)

LongMemEval compares Turns + Jev with mem0 and Turns + cosine only; engram v2 was not run on it, so these results cannot support any claim that Turns + Jev matches LLM-extracted memory on long histories. What they support is narrower: on histories of about 111,770 rendered tokens, the rerank still beats similarity search (S7), and no difference from mem0 was detected on knowledgeupdate questions (S5).

Full context scored 63.0%, against 66.8% for Turns + Jev at k=3. 72 questions were correct only for Turns + Jev and 54 only for full context $( \mathtt { p } = 0 . 1 3$ , descriptive). Full context read 209× the tokens at 29× the cost per question. For reading and answering, judge excluded, it cost \$0.0169 against \$0.00058. It was weakest on temporal and multi-session questions. On the registered sample (user turns only), Turns + Jev scored 68.6% at k=3 and Turns + cosine 65.7%. mem0 scored 70.0% on the knowledge-update questions at k=3.

Table 3: LoCoMo, five held-out conversations, 778 scored questions: accuracy (%), tokens per question, write cost per 1,000 turns and read cost per query at list prices (as in Table 5: the read cost is the Jev or LLM rerank call, excluding the answer call; ≈0<sup>†</sup>: embedding only, the read path makes no model call, only a query embedding, which is not priced; “–”: none), and accuracy by category. Rows are grouped by budget. Bold: best in column within the budget group (highest accuracy, lowest write cost); read costs are not bolded, because the lowest are the unpriced embedding-only reads.
<table><tr><td>System, setting</td><td>Accuracy</td><td>Tokens</td><td>Write $/1k</td><td>Read $/query</td><td>Multi-hop</td><td>Temporal</td><td>Open-domain</td><td>Single-hop</td></tr><tr><td colspan="9"> ${ T i g h t b u d g e t } ( k { = } 3 o r k { = } 6 )$ </td></tr><tr><td> $\mathrm { T u r n s } + \mathrm { c o s i n e } , k { = } 3$ </td><td>59.9%</td><td>130</td><td>$0.0006</td><td>≈0†</td><td>54.3</td><td>55.2</td><td>42.0</td><td>65.7</td></tr><tr><td> $\mathrm { T u r n s } + \mathrm { c o s i n e } , k { = } 6$ </td><td>68.6%</td><td>254</td><td>$0.0006</td><td>≈0†</td><td>61.4</td><td>60.6</td><td>54.0</td><td>75.9</td></tr><tr><td> $\mathrm { T u r n s } + \mathrm { J e v } , k { = } 3$ </td><td>77.2%</td><td>139</td><td>$0.0006</td><td>$0.00020</td><td>75.7</td><td>69.1</td><td>58.0</td><td>83.2</td></tr><tr><td> $\mathrm { T u r n s } + \mathrm { J e v } , k { = } 6$ </td><td>77.0%</td><td>265</td><td>$0.0006</td><td>$0.00020</td><td>74.3</td><td>72.1</td><td>56.0</td><td>82.3</td></tr><tr><td> $\mathrm { T u r n s } + \mathrm { L L M } , k { = } 3$ </td><td>77.6%</td><td>143</td><td>$0.0006</td><td>$0.00023</td><td>73.6</td><td>72.7</td><td>58.0</td><td>83.2</td></tr><tr><td> $\mathrm { e n g r a m } \mathrm { v } 2 , k { = } 3$ </td><td>77.5%</td><td>251</td><td>$1.865</td><td>$0.00018</td><td>72.1</td><td>77.6</td><td>60.0</td><td>81.3</td></tr><tr><td>mem0, k=3</td><td>68.5%</td><td>129</td><td>$1.321</td><td>≈0†</td><td>56.4</td><td>67.3</td><td>56.0</td><td>74.5</td></tr><tr><td>Jev-Mem, k=3</td><td>70.6%</td><td>162</td><td>$0.212</td><td>$0.00120</td><td>57.9</td><td>64.2</td><td>50.0</td><td>79.7</td></tr><tr><td colspan="9">Generous budget (k=20 or k=40, and full context)</td></tr><tr><td> $\mathrm { T u r n s } + \mathrm { c o s i n e } , k { = } 2 0$ </td><td>76.1%</td><td>826</td><td>$0.0006</td><td> ${ \approx } 0 ^ { \dagger }$ </td><td>68.6</td><td>68.5</td><td>58.0</td><td>83.7</td></tr><tr><td> $\mathrm { T u r n s } + \mathrm { J e v } , k { = } 2 0$ </td><td>77.6%</td><td>496</td><td>$0.0006</td><td>$0.00020</td><td>75.0</td><td>72.7</td><td>58.0</td><td>82.7</td></tr><tr><td> $\mathrm { T u r n s } + \mathrm { L L M } , k { = } 2 0$ </td><td>78.0%</td><td>456</td><td>$0.0006</td><td>$0.00023</td><td>72.9</td><td>71.5</td><td>60.0</td><td>84.4</td></tr><tr><td> $\mathrm { e n g r a m } \mathrm { v } 2 , k { = } 2 0$ </td><td>82.4%</td><td>1,238</td><td>$1.865</td><td>$0.00018</td><td>77.9</td><td>80.0</td><td>64.0</td><td>87.0</td></tr><tr><td>mem0, k=20</td><td>78.7%</td><td>839</td><td>$1.321</td><td>≈0†</td><td>74.3</td><td>76.4</td><td>54.0</td><td>83.9</td></tr><tr><td> $\mathbf { J e v - M e m } , k { = } 4 0$ </td><td>80.3%</td><td>1,987</td><td>$0.212</td><td>$0.00174</td><td>76.4</td><td>73.9</td><td>54.0</td><td>87.2</td></tr><tr><td>Full context</td><td>78.3%</td><td>23,631</td><td></td><td></td><td>78.6</td><td>57.0</td><td>64.0</td><td>88.2</td></tr></table>

![](images/e6c94bc04092fb74ed38c7dd6aabd7d93a9174558fcf73cbf56eebc5e4a84d84.jpg)

![](images/303d3fc3fb77fe3451044fcac97cdd2198d029ea64bef1b4f43dfa879e507e7c.jpg)  
Figure 6: Accuracy against retrieved tokens per question (log scale), with Wilson 95% intervals. Left: LoCoMo, 778 questions of the five held-out conversations, each system at each k it was run. Right: LongMemEval, 470 non-abstention questions, user and assistant turns; only Turns + Jev, Turns + cosine and full context ran on all 500 LongMemEval questions (mem0 ran only on the 30 knowledge-update questions of the registered sample, and engram v2 and Jev-Mem not at all), which is why the right panel has three systems. Shaded: the tight budget (a most 300 tokens). the hollow point, Turns + Jev (wide), is post-hoc; the other points are registered runs, compared descriptively except in the tests of §5.1–§5.3.

## 5.6 Where Turns + Jev’s accuracy stops

Turns + Jev levels off near 77.6%. At k=20 it reads only 496 tokens, because its read path keeps the shortlisted turns scored above 0.5 plus a 10-turn cosine floor (§3). The floor is part of why Turns + Jev cannot fill k=20: when few turns clear the threshold, the context stops near the floor.

Shortlist recall (exploratory as registered) locates the loss on all nine held-out conversations (1,388 questions). All evidence turns were in the 30-turn shortlist for 76.9% of questions. At least one was there for 88.5%. Among questions with evidence in the shortlist, the rerank kept none of it for 10.4%. So about 11.5% of questions are lost to the shortlist, and another 9.2% to the rerank.

Table 4: LongMemEval, all 500 questions, user and assistant turns ingested: accuracy (%) on the 470 non-abstention questions and by question type (n in parentheses), and the share of the 30 abstention questions answered by abstaining. KU: knowledge update; MS: multi-session; SS-A, SS-P, SS-U: single-session assistant, preference and user; TR: temporal reasoning; Abs: abstention. Bold: best in column within the budget group.
<table><tr><td>System</td><td>Accuracy</td><td>Tokens</td><td>KU (72)</td><td>MS (121)</td><td>SS-A (56)</td><td>SS-P (30)</td><td>SS-U (64)</td><td>TR (127)</td><td>Abs</td></tr><tr><td>Tight budget (k=3)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Turns + cosine, k=3</td><td>57.7%</td><td>522</td><td>58.3</td><td>31.4</td><td>85.7</td><td>53.3</td><td>95.3</td><td>52.0</td><td>43.3%</td></tr><tr><td>Turns + Jev, k=3</td><td>66.8%</td><td>534</td><td>77.8</td><td>47.9</td><td>98.2</td><td>46.7</td><td>98.4</td><td>53.5</td><td>43.3%</td></tr><tr><td colspan="8">Generous budget (k=20, and full context)</td><td></td><td></td></tr><tr><td>Turns + cosine, k=20</td><td>72.8%</td><td>4,352</td><td>84.7</td><td>58.7</td><td>98.2</td><td>40.0</td><td>96.9</td><td>63.8</td><td>70.0%</td></tr><tr><td>Turns + Jev, k=20</td><td>73.8%</td><td>2,278</td><td>84.7</td><td>67.8</td><td>96.4</td><td>46.7</td><td>96.9</td><td>58.3</td><td>63.3%</td></tr><tr><td>Full context</td><td>63.0%</td><td>111,770</td><td>81.9</td><td>45.5</td><td>91.1</td><td>56.7</td><td>92.2</td><td>43.3</td><td>70.0%</td></tr></table>

![](images/1fbf4f17ea4abe2970372f9ba900917dfe44f061d673c74757696395b8e4618d.jpg)  
Figure 7: Where Turns + Jev’s accuracy stops, by category, on the 778 questions of the five held-out conversations: the rerank kept at least one evidence turn (purple), evidence was in the shortlist but the rerank kept none of it (amber), or no evidence turn was in the shortlist (grey). Upper bar of each pair: Turns + Jev’s 30-turn shortlist (shortlist recall, exploratory as registered). Lower, lighter bar: the wide variant’s 150-turn shortlist (post-hoc). Percentages are printed where the segment is wide enough.

Figure 7 shows the same decomposition by category on the five held-out conversations. Figure 9 (Appendix J) is an example of a shortlist miss: the evidence turn lies outside Turns + Jev’s 30-turn shortlist, while engram v2’s fact extracted from it reaches the answer model.

The categories differ. Open-domain evidence reaches the shortlist least often (63.0%) and is dropped most often (31.4%), consistent with Turns + Jev trailing engram v2 on open-domain questions. Temporal evidence usually reaches the shortlist but is dropped by the rerank for 22.7% of questions: a turn that only establishes when something happened does not look relevant to the question on its own. Multi-hop questions usually get some evidence into the shortlist (88.4%) but rarely all of it (42.8%).

A wider read path (post-hoc exploratory). After the registered results were in, we tested one variant once, outside the Holm family and in its own ledger. Turns + Jev (wide) takes a 150-turn cosine shortlist and asks Jev about every shortlisted turn. It keeps the top k by Jev’s score, with no cut-off. Its k=47 was matched to Jev-Mem at k=40 (2,000 tokens against 1,987). It scored 81.5%. Jev-Mem at k=40 scored 80.3%, and engram v2 at k=20 scored 82.4% with 1,238 tokens. All-evidence recall rose to 91.9%, and the rerank’s losses fell to 0.9% (Figure 7, lower bars). This suggests the ceiling comes from Turns + Jev’s read path rather than from storing raw turns. It is a hypothesis for new data, not a finding of this study.

## 5.7 Cost and latency

Jev reads about 3.0× faster than the LLM reranker and 4.9× faster than Jev-Mem at k=3. engram v2 reads as fast as Turns + Jev, because its read path is the same one request. The write cost is where the systems differ. Extraction makes engram v2 and mem0 thousands of times more expensive to write than Turns + Jev. Jev-Mem’s two Jev requests per turn cost \$0.211 per 1,000 turns. At k=40, Jev-Mem averaged 3.0 Jev calls per query, up to 11. Its read cost was \$0.00174 per query. Mem0’s read cost is a query embedding only.

Table 5: Cost and latency (exploratory as registered), all at k=3 (the tight budget). Write cost per 1,000 turns at list prices, split by LLM, Jev and embeddings; read cost per query; read latency measured live on a fixed sample of 40 questions at $k { = } 3 ,$ , one query at a time, including the query-embedding call. Jev-Mem’s latency comes from its reads of the same questions, measured live when they ran (42 reads: two question texts repeat). A write-latency range spans the conversations and is compared by its lower end. ${ \approx } 0 ^ { \dagger }$ : embedding only, no model call on the read path, only an unpriced query embedding. Bold: best in column within the budget group (lowest cost and latency); read costs are not bolded, as in Table 3.
<table><tr><td>System</td><td>Write $/1k turns</td><td>LLM</td><td>Jev</td><td>Embeddings</td><td>Write p50 (s)</td><td>$/query</td><td>Read Jev calls/query</td><td>Read p50 (ms)</td><td>Read p90 (ms)</td></tr><tr><td>Turns + Jev</td><td>$0.0006</td><td></td><td></td><td>$0.0006</td><td>0.2</td><td>$0.00020</td><td>one</td><td>273</td><td>359</td></tr><tr><td>Turns + cosine</td><td>$0.0006</td><td></td><td></td><td>$0.0006</td><td>0.2</td><td>≈0†</td><td>none</td><td>219</td><td>263</td></tr><tr><td>Turns + LLM</td><td>$0.0006</td><td></td><td></td><td>$0.0006</td><td>0.2</td><td>$0.00023</td><td>one (no-op)</td><td>816</td><td>1,107</td></tr><tr><td>engram v2</td><td>$1.865</td><td>$1.322</td><td>$0.542</td><td>$0.0015</td><td>2.0–2.1</td><td>$0.00018</td><td>one</td><td>276</td><td>319</td></tr><tr><td>mem0</td><td>$1.321</td><td>$1.319</td><td></td><td>$0.0017</td><td>2.1–2.3</td><td>≈0†</td><td>none</td><td>486</td><td>718</td></tr><tr><td>Jev-Mem</td><td>$0.212</td><td></td><td>$0.211</td><td>$0.0011</td><td>0.5</td><td>$0.00120</td><td>5.3</td><td>1,329</td><td>1,622</td></tr></table>

![](images/2c1d671f91e816c7ca1436f0144db8e39131522dc954cf1c055774a5375c00b3.jpg)  
Figure 8: Accuracy against total cost per question (log scale) at $k { = } 3$ , on the 778 questions of the five heldout conversations (exploratory as registered): write cost amortised at the benchmark’s 4.0 turns written per question, plus read cost and answer cost (judge excluded), at list prices; full context has no write or read cost. In a read-heavy use with one turn written per question, engram $\mathbf { v } 2 \mathbf { \bar { s } }$ total falls to \$0.00216, mem0’s to \$0.00142 and Jev-Mem’s to \$0.00151; the other systems’ totals do not change at this precision.

Figure 8 plots the cost per question of Eq. (9) at the benchmark’s own ratio, $r _ { w } = 4 . 0 .$ . At that ratio, Turns + Jev’s total cost per question is \$0.00030 and engram $\mathbf { v } 2 \mathbf { \bar { s } }$ is \$0.00778. Full context costs \$0.00362. In a read-heavy use, with one turn written per question, the write cost weighs less: engram v2’s total falls to \$0.00216.

## 5.8 Abstention

Reranking lowered correct abstention at k=3. Similarity search abstained correctly on 63.6% of adversarial questions. With Jev it was 54.1%, and with an LLM reranker 48.8%. Relevant-looking context makes the answer model less willing to say that something was not mentioned. The exploratory conversations show the same (54.2% against 65.3%). So does LongMemEval at k=20: 63.3% for Turns + Jev against 70.0% for Turns + cosine. Fidelity Before Structure reports that verbatim chunks abstain worse than extracted artifacts; we find that reranking adds to that.

Table 6: Share of LoCoMo adversarial questions (209, five held-out conversations) answered by abstaining (exploratory as registered). Each column is a budget group. Bold: best in column within the budget group (highest share).
<table><tr><td>System</td><td>k=3</td><td> ${ k } = \mathbf { 2 0 }$ </td></tr><tr><td>Turns + cosine</td><td>63.6%</td><td>47.8%</td></tr><tr><td>engram v2</td><td>59.8%</td><td>52.2%</td></tr><tr><td>mem0</td><td>59.8%</td><td>53.6%</td></tr><tr><td> $\mathrm { T u r n s } + \mathrm { J e v }$ </td><td>54.1%</td><td>49.3%</td></tr><tr><td> $\mathrm { T u r n s } + \mathrm { L L M }$ </td><td>48.8%</td><td>53.1%</td></tr></table>

## 6 Discussion

A budget reading of two prior findings (interpretation). SmartSearch finds ranking to be the bottleneck; Fidelity finds reranking marginal. Our within-study results suggest the difference is the budget. Reranking matters in proportion to how hard truncation cuts the candidate set. In Smart-Search a question has about 431 grep candidates on average. About 62 passages fit its 2,000-word budget. Without ranking, only 22.5% of gold evidence survives truncation. Our k=3 likewise keeps three of 30, and both show large ranking gains. Fidelity reranks a top-30 pool to 15 with bge-reranker-v2- m3, under a 5,000-token cap. Its gains are 2.9 points on LoCoMo and 0.6 on LongMemEval-S. Our k=20 likewise keeps twenty of 30, and both show small gains. This is an interpretation across pipelines that differ in retrievers, rerankers, answer models and judges. Inside our study it is supported by the k=3 to k=20 comparison on two benchmarks; it is not a tested claim across papers. Read together with Kang et al. (2026), the two studies suggest that compression is favoured when the budget cannot fit the relevant raw evidence, a regime our budgets did not reach, and that raw evidence with good selection is competitive once it can, as at our tight budgets.

When extraction is worth it. At the tight budget of H1, extraction adds little and costs thousands of times more to write. At generous budgets it is more accurate and more compact: engram v2 at k=20 was the most accurate system we measured, with fewer tokens than Jev-Mem at k=40. Opendomain and temporal questions are where Turns + Jev trailed engram v2, and where its shortlist and rerank lose the most evidence.

What a typed decision model contributes. In this study, it contributed speed and cost at equal selection quality, not higher accuracy. At matched context, Jev selected as accurately as the gpt-4omini reranker (S4) at about a third of the latency, in one request per question. One Jev request also beat Jev-Mem’s multi-request graph walk at matched context (S2). A typed question returns a probability over fixed options in one short call, which is what a reranker needs.

Judge leniency and answer style. mem0’s Lo-CoMo judge is lenient, and in our audit it credited short answers more readily than list-style ones, so its agreement with human grading differed by system. Fidelity Before Structure’s human study found no such dependence on answer length, but its LoCoMo judge is instructed to be strict (“Binary - strict”: “partial answers or answers with significant missing information should be marked INCORRECT”, their Appendix J.4), while mem0’s asks the judge to “be generous with your grading - as long as it touches on the same topic as the gold answer”. A strict instruction leaves less room for style to matter, which may explain why their audit found no short-answer bias and ours did. Other audits point the same way. On multimodal memory questions, MemLens (Ren et al., 2026) finds that its LLM judge’s leniency inflates closed-form accuracy by about 5 points, without reordering its leaderboard. An audit of LoCoMo by Penfield Labs (Penfield Labs, 2026) reports 99 answer-key errors in 1,540 questions (6.4%) and a gpt-4o-mini judge that accepted 62.81% of deliberately wrong but topically adjacent answers. Memory benchmarks that compare systems with different answer styles should report judge–human agreement by system.

Not state of the art. SmartSearch reports 91.9% on LoCoMo under its own protocol. That protocol uses gpt-4o-mini to answer and judge, binary judgments, all ten conversations and 1,540 questions in categories 1–4, at 3,141 tokens per question. Our numbers come from a different protocol on five held-out conversations and are not comparable to it. Our best result, the post-hoc Turns + Jev (wide), is below that figure.

## Limitations

• Benchmarks. One benchmark family per setting: LoCoMo, whose dialogues are LLM-generated, and LongMemEval. Five primary conversations give 778 questions; categories are small.

• LongMemEval scope. engram v2 was not run on LongMemEval, so no claim about extraction on long histories follows from this study.

• Human grading. The grader, the first author, built the systems evaluated; the mapping of partial grades was not pre-specified (two are reported); one question was ungraded; and only judge-discordant questions were re-graded, so judge errors on questions where the judge agreed across systems remain.

• Adversarial content. Turns + Jev passes raw, user-written turns to Jev’s relevance question, so text injected into a conversation could shift which turns are selected; prompt injection shifts Jev’s decision probabilities (Wu and Lim, 2026). We did not test adversarial content.

• A closed decision model. Jev is a closed, versioned model; results hold for jev-1.13.0.

• mem0 serving. mem0’s extraction calls were served through OpenRouter, about half by Azure, not the OpenAI API as registered; only S3 involves mem0.

• Post-hoc variant. Turns + Jev (wide) was designed after the registered results and tested once on the same questions.

• Budget and read path. The k=20 comparisons mix the budget with Turns + Jev’s read-path ceiling: its threshold and cosine floor keep it at 496 tokens at k=20, against Turns + cosine’s 826. The budget dependence at generous budgets may be less steep for a wider read path; Turns + Jev (wide) suggests so, post-hoc.

• Absolute accuracy. Below SmartSearch’s reported figures at generous budgets, under a different protocol.

• Development data. Every design choice was made on one conversation, conv-26.

## 7 Conclusion

Within this study, reranking’s gain over similarity search shrinks as the context budget grows. Reranking added 17.4 points on LoCoMo and 9.1 on Long-MemEval when three of 30 candidates were kept. At k=20 it added 1.5 and 1.1, and extraction systems were more accurate. This suggests an explanation for the published disagreement, which remains an interpretation across papers; Kang et al. (2026) find a complementary budget dependence for consolidation. A pre-registered non-inferiority test on held-out conversations bounds what extraction adds at a tight budget to at most 4.7 points. That test holds with a second answer model and under blind human grading, at 3,061× lower write cost. A typed decision model is an effective selector. At matched context, Jev was non-inferior to an LLM reranker (bound −2.0) at about a third of the latency. It was also more accurate than a multi-call Jev graph traversal at matched context. The diagnostics locate where selection stops: in shortlist misses (11.5%) and rerank drops (9.2%), most for temporal evidence. They also show that an LLM judge’s leniency, documented elsewhere, interacts with answer length. Memory benchmarks that compare systems with different answer styles should therefore report judge–human agreement by system.

## Author Contributions

Rishabh Sharma designed the study and its preregistered plan, built the systems, ran and orchestrated the experiments, and did the blind human audit; the human grader is therefore the author of the systems evaluated. Rishika Lall contributed to the analysis and interpretation of the results. Both authors drafted, reviewed and edited the paper.

## AI Assistance

The code, run orchestration and drafting of this paper were done with Claude Code (Anthropic) under

the first author’s direction. The first author made every methodological decision, approved each stage of the registered plan and did the human audit.

## Artifacts

Code, plans, per-question answers and judge labels, and the human-audit grades with their key are at github.com/ris3abh/Engram: tags v3-frozen, v3-amended and the paper tag; results in bench/results/v3/ (per-question files, reports, ledgers, human\_audit/) and bench/results/v3\_posthoc/. The plan and its amendment are deposited at 10.5281/zenodo.22970745 and 10.5281/zenodo.22977848; this paper is 10.5281/zenodo.22985242 (release tag paper-v3-preprint-r3); the earlier engram preprint is 10.5281/zenodo.22941757 (Sharma, 2026c).

## References

Diogo Almeida. 2026. Introducing system one models & Jev. TypeSafe AI blog. Published 15 September 2026.

Tao An. 2026. Fidelity before structure: Verbatim chunks beat lossy artifact extraction in long-conversation LLM memory. Preprint, arXiv:2601.00821. V1 submitted December 2025; version 4, July 2026.

Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. 2025. Mem0: Building production-ready AI agents with scalable long-term memory. Preprint, arXiv:2504.19413.

Jesper Derehag, Carlos Calva, and Timmy Ghiurau. 2026. SmartSearch: How ranking beats structure for conversational memory retrieval. Preprint, arXiv:2603.15599.

Jizhan Fang, Xinle Deng, Haoming Xu, Ziyan Jiang, Yuqi Tang, Ziwen Xu, Shumin Deng, Yunzhi Yao, Mengru Wang, Shuofei Qiao, Huajun Chen, and Ningyu Zhang. 2026. LightMem: Lightweight and efficient memory-augmented generation. In International Conference on Learning Representations (ICLR).

Qi Feng, Chris Ding, and Jicong Fan. 2026. The retriever should remember: Experience-amortized reranking for long-term agent memory. Preprint, arXiv:2608.22767.

Chuanrui Hu, Xingze Gao, Zuyi Zhou, Dannong Xu, Yi Bai, Xintong Li, Hui Zhang, Tong Li, Chong Zhang, Lidong Bing, and Yafeng Deng. 2026. Ever-MemOS: A self-organizing memory operating system for structured long-horizon reasoning. Preprint, arXiv:2601.02163.

Wei-Chieh Huang, Weizhi Zhang, Yuchen Wu, Yankai Chen, Eric Hanchen Jiang, Wooseong Yang, Yiwei Yang, Henry Peng Zou, Hanrong Zhang, Ying Nian Wu, Haolun Wu, Kai-Wei Chang, Philip S. Yu, Xue Liu, and Aylin Caliskan. 2026. Harness the memory: A holistic evaluation of memory substrates in memory agents. Preprint, arXiv:2608.15008.

Dongming Jiang, Yi Li, and Bingzhe Li. 2026a. Jev-Mem: System-one-controlled agentic memory for efficient AI agents. Preprint, arXiv:2609.23986.

ZhiShu Jiang, Haibo Liu, Xin Shen, Guanqiang Qi, Chenxi Miao, Weikang Li, Liwei Qian, Xin Pei, and Jizhou Huang. 2026b. Learning user-aware recall: Personalized retrieval in long-term conversational memory. Preprint, arXiv:2607.00017.

Qingcan Kang, Mingyang Liu, Shixiong Kai, Kaichao Liang, Zhentao Tang, Yuqi Cui, Tao Zhong, and Mingxuan Yuan. 2026. Retain or consolidate? budget-dependent operator selection for language agent memory. Preprint, arXiv:2607.17545.

Letta. 2025. Benchmarking AI agent memory: Is a filesystem all you need? Letta Blog. Published 12 August 2025.

Chunyu Li, Mengyuan Zhang, Jingyi Kang, Ding Chen, Jiajun Shen, Bo Tang, Xuanhe Zhou, Feiyu Xiong, and Zhiyu Li. 2026a. MemReranker: Reasoningaware reranking for agent memory retrieval. Preprint, arXiv:2605.06132.

Yilong Li, Suman Banerjee, and Tong Che. 2026b. EMBER: Efficient memory via budgeted evidence retention for long-horizon agents. Preprint, arXiv:2606.05894.

Yuxin Liao, Le Wu, Min Hou, Hao Liu, Han Wu, and Zishu Wang. 2026. LeanMem: Simple and efficient long-term memory for LLM agents. Preprint, arXiv:2608.03463.

Jiaqi Liu, Yaofeng Su, Peng Xia, Siwei Han, Zeyu Zheng, Cihang Xie, Mingyu Ding, and Huaxiu Yao. 2026. SimpleMem: Efficient lifelong memory for LLM agents. Preprint, arXiv:2601.02553.

Christian Lysenstøen. 2026. Training-free lexical-dense fusion for conversational-memory retrieval. Preprint, arXiv:2606.04194.

Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. 2024. Evaluating very long-term conversational memory of LLM agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13851– 13870, Bangkok, Thailand. Association for Computational Linguistics.

Mem0 Engineering Team. 2026. State of AI agent memory 2026: Benchmarks & trends. Mem0 Blog. Published 1 April 2026, updated 22 September 2026.

Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. 2023. MemGPT: Towards LLMs as operating systems. Preprint, arXiv:2310.08560.

Taiheng Pan. 2026. ConvMemory v2: A recallpreserving top-10 evidence reranker for conversational memory retrieval. Preprint, arXiv:2606.10842.

Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, Xufang Luo, Hao Cheng, Dongsheng Li, Yuqing Yang, Chin-Yew Lin, H. Vicky Zhao, Lili Qiu, and Jianfeng Gao. 2025. On memory construction and retrieval for personalized conversational agents. In International Conference on Learning Representations (ICLR).

Sugam Panthi and Rabab Abdelfattah. 2026. Same ranking, different winner: How scoring targets shape LLM memory benchmarks. Preprint, arXiv:2605.24060.

Penfield Labs. 2026. We audited LoCoMo: 6.4% of the answer key is wrong and the judge accepts up to 63% of intentionally wrong answers. Blog post. Published 8 April 2026; accessed 26 September 2026.

Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, and Daniel Chalef. 2025. Zep: A temporal knowledge graph architecture for agent memory. Preprint, arXiv:2501.13956.

Xiyu Ren, Zhaowei Wang, Yiming Du, Zhongwei Xie, Chi Liu, Xinlin Yang, Haoyue Feng, Wenjun Pan, Tianshi Zheng, Baixuan Xu, Zhengnan Li, Yangqiu Song, Ginny Wong, and Simon See. 2026. MemLens: Benchmarking multimodal long-term memory in large vision-language models. Preprint, arXiv:2605.14906.

Rishabh Sharma. 2026a. engram v3: Selection over extraction: pre-registered analysis plan. Version v3.

Rishabh Sharma. 2026b. engram v3: Selection over extraction: pre-registered analysis plan. Version v3- amended.

Rishabh Sharma. 2026c. Typed decisions in agent memory: Where they help, where they don’t, and what it costs. Preprint (concept DOI; latest version 10.5281/zenodo.22948964).

Javad Taghia. 2026. Governed agent memory with structured judgment: An AtMem–Jev retrieval study. Hugging Face Community Article. Published 19 September 2026.

TypeSafe AI. 2026. Jev documentation. Accessed 26 September 2026.

Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. 2025. LongMemEval: Benchmarking chat assistants on long-term interactive memory. In International Conference on Learning Representations (ICLR).

Tiantong Wu and Wei Yang Bryan Lim. 2026. Decision hijacking: Prompt injection attacks on Jev’s typed probabilistic decisions. Preprint, arXiv:2609.28613.

Yuqian Wu, Wei Chen, Zhengjun Huang, Junle Chen, Qingxiang Liu, Kai Wang, Xiaofang Zhou, and Yuxuan Liang. 2026. Back to basics: Let conversational agents remember with just retrieval and generation. Preprint, arXiv:2604.11628.

Menglin Xia, Xuchao Zhang, Shantanu Dixit, Paramaguru Harimurugan, Rujia Wang, Victor Ruhle, Robert Sim, Chetan Bansal, and Saravan Rajmohan. 2026. Memora: A harmonic memory representation balancing abstraction and specificity. Preprint, arXiv:2602.03315. ICML 2026 (per arXiv comment).

Yilin Xiao, Zhehan Zhu, Yujing Zhang, Jin Chen, Zijin Hong, Luyao Zhuang, Qinggang Zhang, Shengyuan Chen, Xiaocao Ouyang, Lingfei Ren, and Xiao Huang. 2026. Zero-Mem: Zero-token memory operations for LLM agents. Preprint, arXiv:2607.29377.

Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. 2025. A-MEM: Agentic memory for LLM agents. In Advances in Neural Information Processing Systems (NeurIPS).

Sikuan Yan, Xiufeng Yang, Zuchao Huang, Ercong Nie, Zifeng Ding, Zonggen Li, Xiaowen Ma, Jinhe Bi, Kristian Kersting, Jeff Z. Pan, Hinrich Schütze, Volker Tresp, and Yunpu Ma. 2025. Memory-R1: Enhancing large language model agents to manage and utilize memories via reinforcement learning. Preprint, arXiv:2508.19828.

Jing Yu, Yibo Zhao, Jiaming Zhang, and Xiang Li. 2026. LazyMem: Retrieve broadly, construct selectively for efficient long-term agent memory. Preprint, arXiv:2607.22690.

Ruihong Zeng, Jinyuan Fang, Siwei Liu, and Zaiqiao Meng. 2024. On the structural memory of LLM agents. Preprint, arXiv:2412.15266.

Sizhe Zhou and Jiawei Han. 2025. A simple yet strong baseline for long-term conversational memory of LLM agents. Preprint, arXiv:2511.17208.

## A Per-conversation results

Table 7: Accuracy (%) per conversation, LoCoMo scored categories.
<table><tr><td>System, setting</td><td>conv-44 (123)</td><td>conv-47 (150)</td><td>conv-48 (191)</td><td>conv-49 (156)</td><td>conv-50 (158)</td></tr><tr><td>Turns + Jev, k=3</td><td>78.0</td><td>76.7</td><td>79.1</td><td>73.7</td><td>78.5</td></tr><tr><td>Turns + Jev, k=6</td><td>79.7</td><td>76.7</td><td>78.5</td><td>73.7</td><td>76.6</td></tr><tr><td>Turns + Jev, k=20</td><td>79.7</td><td>74.7</td><td>81.7</td><td>75.0</td><td>76.6</td></tr><tr><td>Turns + cosine, k=3</td><td>65.9</td><td>57.3</td><td>62.8</td><td>58.3</td><td>55.7</td></tr><tr><td>Turns + cosine, k=20</td><td>78.0</td><td>74.0</td><td>79.1</td><td>74.4</td><td>74.7</td></tr><tr><td>Turns + LLM, k=3</td><td>76.4</td><td>72.7</td><td>80.6</td><td>77.6</td><td>79.7</td></tr><tr><td>engram v2, k=3</td><td>77.2</td><td>80.7</td><td>78.5</td><td>76.3</td><td>74.7</td></tr><tr><td>engram v2, k=20</td><td>84.6</td><td>84.7</td><td>82.2</td><td>80.8</td><td>80.4</td></tr><tr><td>mem0, k=3</td><td>68.3</td><td>65.3</td><td>69.6</td><td>72.4</td><td>66.5</td></tr><tr><td>mem0, k=20</td><td>79.7</td><td>78.0</td><td>81.2</td><td>78.8</td><td>75.3</td></tr><tr><td>Jev-Mem, k=3</td><td>69.1</td><td>66.7</td><td>74.3</td><td>69.2</td><td>72.2</td></tr><tr><td>Jev-Mem, k=40</td><td>81.3</td><td>78.7</td><td>80.6</td><td>82.7</td><td>78.5</td></tr><tr><td>Full context</td><td>82.1</td><td>72.7</td><td>82.7</td><td>76.3</td><td>77.2</td></tr></table>

The exploratory replication on conv-30, conv-41, conv-42 and conv-43 (610 scored questions): Turns + cosine 60.7% at k=3 and 76.2% at k=20; Turns + Jev 77.2% at k=3 and 76.6% at k=20. Turns + Jev’s matched k against Turns + cosine was 3, with 116 questions correct only for Turns + Jev and 15 only for Turns + cosine (p = 1.6e−20, exploratory).

## B Registered plan and deviations

The plan’s guarded sections in docs/V3\_PLAN.md, checked by a test that fails on any undated change, fixed the systems, data, token-matching rule, tests, predictions, human check, run order and budget before any run. Every later change is a dated entry in its Deviations section, one row each below; presentation changes share a row.

Table 8: Deviations from the registered plan.
<table><tr><td>Date</td><td>Change</td><td>Reason</td><td>Effect on results</td></tr><tr><td>2026-09-26</td><td>Amendment, deposited after Batch A (S1 and S2 known) and before any H1 result: shortlist recall; LongMemEval on all 500 questions with user and assistant turns, with the new test S7; the second answer model; the outcome paragraphs and the</td><td>Extend the study before the primary test was run</td><td>S7 joins the Holm family; new robustness checks; H1, its margin and the other tests unchanged</td></tr><tr><td>2026-09-26 2026-09-26</td><td>caps Outcome paragraphs revised before upload mem0&#x27;s extraction calls went through OpenRouter, about half served by Azure, instead of the OpenAI API; the ledger was</td><td>The first author&#x27;s own wording A mem0 library default routes calls to OpenRouter when its key is set (Appendix G)</td><td>None; made before any H1 result S3 is reported with a caveat; no other test involves mem0</td></tr><tr><td>2026-09-26</td><td>corrected and guards added before any later run Runs repeated after OpenAI rate limits, with more retries and a Jev throttle; completed calls replayed from the call</td><td>Rate limits</td><td>None: replayed calls are identical</td></tr><tr><td>2026-09-26</td><td>cache Turns + LLM also asks Jev&#x27;s</td><td>Shared read-path code</td><td>None: the question is a no-op on turns</td></tr><tr><td>2026-09-26</td><td>query-relation question once per query Token counting treats text that spells a special token (&lt;|endoftext |&gt;, in one</td><td>The tokenizer refused that text</td><td>One question re-run; every other count unchanged</td></tr><tr><td>2026-09-27</td><td>LongMemEval haystack) as ordinary text Human-check grades: the sheet asked for CORRECT or WRONG, and partial grades appeared, so two mappings are</td><td>The plan fixed no rule for partial grades</td><td>Both mappings reported (Table 1, Appendix C); H1 is decided by the judge</td></tr><tr><td>2026-09-26 and 2026-09-27</td><td>reported Presentation only: the paper title replaced; systems renamed Turns + Jev, Turns + cosine, Turns + LLM and Turns + Jev (wide) (registered as T0R, L0, TOR-LLM and TOR-wide); the audit sheet put each answer on its own row and shuffled all 284</td><td>The selected title presented a published idea as new and its “matches&quot; overstated a non-inferiority result; readability; sheet layout</td><td>None</td></tr></table>

An implementation note not in the Deviations section: two retrieval-only sweeps were stopped by mistake and re-run from the cache.

## C Human audit

Protocol. The sheet held every question on which the judge found exactly one of H1’s two answers correct (142 questions), each answer as its own row (284 rows), shuffled with seed 0, with the question and gold answer shown and no system name or judge label. The sheet asked for CORRECT or WRONG; the plan’s notes had listed CORRECT, WRONG or UNCLEAR. The first author’s grades included partial and hedged labels, and one question was left ungraded. Two mappings are reported: strict (only grades starting with CORRECT count as correct) and lenient (partial and hedged-correct grades also count); any grade containing WRONG counts as wrong under both. H1 is decided by the judge.

<table><tr><td>Mapping</td><td>Agreement with judge</td><td>On Turns + Jev&#x27;s answers</td><td>On engram v2&#x27;s answers</td><td>Only Turns + Jev right</td><td>Only engram v2 right</td><td>Both right</td><td>Both wrong</td></tr><tr><td>Strict</td><td>81%</td><td>81%</td><td>82%</td><td>47</td><td>59</td><td>17</td><td>18</td></tr><tr><td>Lenient</td><td>79%</td><td>82%</td><td>77%</td><td>41</td><td>60</td><td>31</td><td>9</td></tr></table>

The ungraded question. One discordant question, in conv-47, has an ungraded row. H1 with human grades can treat it two ways: (a) it keeps the judge’s labels, or (b) it is dropped. The paper reports (a) in Table 1 and §5.4. (a) keeps all 778 questions, and it is the conservative choice: the judge scored only engram v2 correct on this question. The strict 78.0% and lenient 79.9% for engram v2 are the (a) values.

Table 9: H1 with human grades under both treatments of the ungraded question. Differences and bounds in points.
<table><tr><td>Mapping, treatment</td><td>Questions</td><td>Turns + Jev</td><td>engram v2</td><td>Difference</td><td>One-sided 95% bound</td><td>Two-sided 95% CI</td></tr><tr><td>Strict, (a) judge&#x27;s labels</td><td>778</td><td>76.3%</td><td>78.0%</td><td>-1.7</td><td>-3.9</td><td>[-4.3, +0.9]</td></tr><tr><td>Strict, (b) dropped</td><td>777</td><td>76.4%</td><td>78.0%</td><td>-1.5</td><td>-3.7</td><td>[-4.1, +1.1]</td></tr><tr><td>Lenient, (a) judge&#x27;s labels</td><td>778</td><td>77.4%</td><td>79.9%</td><td>-2.6</td><td>-4.7</td><td>[-5.1, -0.03]</td></tr><tr><td>Lenient, (b) dropped</td><td>777</td><td>77.5%</td><td>79.9%</td><td>-2.4</td><td>-4.6</td><td>[−5.0, +0.09]</td></tr></table>

No conclusion changes. H1 is non-inferior under all four. The worst bound is −4.7 under (a) and −4.6 under (b). One statement depends on the choice. Under lenient grading with (a), the two-sided interval lies just below zero, so by that grading engram v2 is more accurate. With (b), the interval reaches +0.09, so the difference is not detected.

The grades, the key and the analysis are in bench/results/v3/human\_audit/ and bench/v3\_human\_ audit.py.

## D Shortlist recall

For every scored question of the nine held-out conversations, the 30-turn cosine shortlist (shared by Turns + cosine and Turns + Jev) and the turns Turns + Jev’s rerank keeps were rebuilt from the frozen stores through the call cache. Each turn’s id is its LoCoMo dialogue id, so the question’s evidence ids can be located. Reported: the share of questions with all, and with at least one, evidence turn in the shortlist, and, among questions with an evidence turn in the shortlist, the share where the rerank keeps none. Recall is scored on raw turns only (Panthi and Abdelfattah, 2026). The five fresh conversations (76.9% all-evidence recall, 10.4% dropped by the rerank) and the four exploratory ones (76.9%, 10.3%) agree.

<table><tr><td>Category</td><td>Questions</td><td>All evidence in shortlist</td><td>At least one</td><td>Rerank keeps none</td></tr><tr><td>Multi-hop</td><td>250</td><td>42.8%</td><td>88.4%</td><td>5.9%</td></tr><tr><td>Temporal</td><td>284</td><td>83.5%</td><td>88.4%</td><td>22.7%</td></tr><tr><td>Open-domain</td><td>83</td><td>42.0%</td><td>63.0%</td><td>31.4%</td></tr><tr><td>Single-hop</td><td>771</td><td>89.2%</td><td>91.2%</td><td>5.8%</td></tr><tr><td>All</td><td>1,388</td><td>76.9%</td><td>88.5%</td><td>10.4%</td></tr></table>

## E Turns + Jev (wide), post-hoc exploratory

Designed after the registered results were seen, tested once on the 778 questions of the five held-out conversations, outside the Holm family, in its own ledger (\$0.31 OpenAI and \$0.58 Jev). Design: Turns + Jev’s store; a 150-turn cosine shortlist; Jev’s relevance question on every shortlisted turn, thirty per request; the top k by Jev’s probability, with no cut-off, floor or expansion; k=47 matched to Jev-Mem at k=40 by the registered rule, saved before answering. Results: accuracy 81.5% at 2,000 tokens (multi-hop 78.6, temporal 76.4, open-domain 64.0, single-hop 86.5); read cost \$0.00089 per query; live read latency 720 ms at p50 and 776 ms at p90. With the same recall method on the 150-turn shortlist, all evidence was in the shortlist for 91.9% of questions and at least one turn for 97.9%, and the top k kept none of the shortlisted evidence for 0.9%.

## F Prompts and the Jev question

The answer prompt is mem0’s LoCoMo answer prompt adapted to one memory list, and the judge is mem0’s LoCoMo accuracy prompt (bench/locomo\_subset.py, ANSWER\_PROMPT and ACCURACY\_ PROMPT). LongMemEval questions carry their question date in the question slot. Turns + Jev’s rerank asks Jev one yes/no question per shortlisted turn (src/engram/decide/questions.py, RELEVANT\_TO\_ QUERY):

• instructions: "Does memory help answer query?"

• true: “It states or directly implies part of the answer.”

• false: “It is off-topic or only shares a keyword.”

## G The mem0 serving incident

mem0 2.1.0 sends its OpenAI LLM calls to OpenRouter whenever an OpenRouter key is present in the environment, without warning (mem0/llms/openai.py). A key added for the second answer model therefore routed all 3,122 of mem0’s extraction calls through OpenRouter, which served 1,599 of them by OpenAI and 1,523 by Azure, all as gpt-4o-mini, the registered model. OpenRouter billed \$2.42; at OpenAI list price the same calls cost \$4.12. S3’s difference is far larger than a serving difference could explain, and S3 is reported with this caveat. Before any later run, mem0 was pinned to the OpenAI endpoint and the spend guard was extended to OpenRouter. Anyone benchmarking mem0 2.1.0 with an OpenRouter key in their environment will get routed calls without warning.

## H Cost accounting

Every system’s gpt-4o-mini, embedding and Jev calls are priced at list price (gpt-4o-mini \$0.15 and \$0.60 per million input and output tokens; Jev \$0.042 per million input tokens), so no system looks cheaper because of a discount the others did not get. Billed amounts are reported separately: the registered ledger records \$23.16 of OpenAI, \$5.55 of Jev and \$0.31 of OpenRouter for the second answer model (at \$0.10 and \$0.32 per million input and output tokens), plus mem0’s \$2.42 through OpenRouter. Write-cost ratios use the held-out measurements; read costs are per query; totals per question state their reads-per-write assumption (Figure 8).

## I Reproduction

Each table and figure is rebuilt from the committed result files, with no API calls:

• numbers: uv run --extra bench python paper\_v3/make\_numbers.py

• figures: uv run --with matplotlib --with pymupdf python paper\_v3/figures.py (Figures 1 and 2 and the Appendix J figure are TikZ, from paper\_v3/diagram.py, compiled with pdflatex)

• the worked examples of Figure 2 and Appendix J: bench/v3\_worked\_example.py, which replays both read paths from the frozen stores through the call cache opened read-only (a cache miss stops it, so it cannot call an API)

• paper: make -C paper\_v3 paper (renders main.md and main.tex, builds the PDF, runs paper\_v3/ check.py)

• reports behind the tables: bench/v3\_report.py (Batches A–C), bench/v3\_human\_audit.py, bench/v3\_shortlist\_recall.py, bench/v3\_latency.py, bench/v3\_second\_model.py, bench/ v3\_posthoc.py

The runs themselves are bench/run.py with --study v3, bench/jevmem\_run.py and bench/v3\_ batch\_c.py, from tag v3-frozen onward, as recorded in the plan.

## J A counter-example

Figure 2 shows a question where selection wins. Figure 9 shows the opposite. It was chosen by the same kind of rule and replayed the same way, with no API call. The question must:

1. be an H1 question that the judge and the human grader both scored correct for engram v2 and wrong for Turns + Jev (44 questions);

2. have replayed contexts that match the recorded ones;

3. come first by conversation and question index among those left.

The question asks where Audrey got Pixie. The answer, a breeder, is in a turn that does not name Pixie (“I got lucky finding a breeder nearby that has the dogs I wanted”). That turn did not reach Turns + Jev’s 30-turn cosine shortlist, which turns about Pixie fill. Jev kept the turn about her adoption (P = 0.65) and one unrelated turn (P = 0.66). Turns + Jev answered that the memories do not say. engram v2’s extraction had rewritten the turn as a fact: “Audrey found a nearby breeder that had the dogs she wanted”. That fact ranked 16th in its fact shortlist. Jev kept it (P = 0.74), and it reached the answer model at k=3. This is the shortlist-miss failure of §5.6: a fact extracted from a turn can be retrieved when the turn itself is not.

kept by Jev (P > 0.5) cosine floor kept by Jev, then cut at k not read

Turns + Jev: raw turns k=6, 279 tokens

rank Jev P shortlisted turn, in cosine order

2 0.65 evidence Audrey: Hey Andrew, I got a surprise for you! We adopted another puppy called Pixie. She’s SO cute! . . .

3 0.05 Audrey: Thanks! I know right? She’s so cute! Pixie’s been keeping us busy, so I haven’t had a

4 0.04 Audrey: Yeah, for sure. They each have their favorite spot to chill. Pepper loves lounging on the . . .

5 0.09 Audrey: Thanks! They’re all mutts, but Pepper and Panda are Lab mixes, and Precious and Pixie are . . .

18 0.66 Audrey: Thanks! Yeah, I made them myself. I wanted each one to be special and fit their . . .

evidence D11:4 is not in the 30-turn shortlist; 24 more shortlisted turns not read (highest P 0.23)

engram v2: extracted facts k=3, 199 tokens

rank Jev P shortlisted fact, in cosine order

2 0.06 Andrew expressed excitement about Audrey’s new puppy named Pixie, describing her as very cute.

3 0.06 Audrey’s new puppy Pixie is fitting in great and took a few days to get used to the other dogs, but . . .

4 0.08 Audrey’s dog Pepper took some time to get used to the new puppy Pixie, but now they are always

16 0.74 evidence Audrey found a nearby breeder that had the dogs she wanted, which she considers lucky.

25 more shortlisted facts not read (highest P 0.17)

Figure 9: The counter-example of Appendix J, drawn as Figure 2 (both contexts match the recorded token counts: Turns + Jev 279, engram v2 199). The evidence turn for “a breeder” is not in Turns + Jev’s 30-turn shortlist; engram v2’s fact from it is, and Jev keeps it. An illustration chosen by the rule above, not evidence.