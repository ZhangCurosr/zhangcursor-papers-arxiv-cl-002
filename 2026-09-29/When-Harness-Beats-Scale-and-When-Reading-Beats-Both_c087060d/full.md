# When Harness Beats Scale, and When Reading Beats Both

Ivan Bondarenko<sup>1</sup> Nikolay O. Nikitin<sup>2</sup>

<sup>1</sup>Novosibirsk State University <sup>2</sup>ITMO University

i.bondarenko@g.nsu.ru nicl.nno@gmail.com

## Abstract

We describe our system for DocSem, the document-grounded quantitative reasoning shared task at DocInsights 2026, and analyze why it succeeded on labeled data and failed on the test set. The pipeline pairs hybrid block retrieval with Program-of-Thoughts (PoT) generation executed in a sandboxed interpreter, selfconsistency sampling, and entity enrichment from chunk-level knowledge graphs. On our held-out split, application architecture moved the metrics far more than model scale did: PoT added 0.282 joint accuracy to a compact 7B model but at most 0.005 to a 72B model, and a 27B model with the full harness matched the 72B (0.884 vs. 0.873) at roughly 2.7× fewer parameters and a quarter of the CO<sub>2</sub>. We read this through a distinction between world knowledge, which scales steeply with parameters, and language knowledge, which scales gently, and show that structured-output training makes a compact model harness-ready rather than merely small. On the raster, watermarked test PDFs the same system collapsed to 13.58% joint (rank 149 of 163); a controlled rerendering of the validation set reproduces the OCR half of the collapse while bounding what the simulation misses. Auditing the physical nature of evaluation inputs precedes architecture, and the leaderboard’s bimodality is consistent with reading quality, not reasoning, having separated the field.

## 1 Introduction

DocSem asks for two things at once: given a PDF and a paraphrased query, compute a numeric answer strictly from one embedded quantitative passage, and name the exact layout block identifiers that justify it. Scoring is strict on both axes, and the primary ranking metric, joint exact accuracy, rewards systems that find the passage and reproduce its arithmetic without contamination from the dozens of unrelated numbers that surround it.

Our system builds on RAGU, a graph retrievalaugmented generation toolkit, and entered the test phase at 0.901 joint exact accuracy on our internal held-out split and 0.853 on the organizers’ validation portal. The test phase returned 13.58%, rank 149 of 163 teams. This paper documents both halves of that gap, because each half carries a transferable lesson.

On labeled data, application architecture dominated model scale. Switching a compact 7B model from direct answering to Program-of-Thoughts (PoT), where the model writes a program that a sandboxed interpreter executes (Chen et al., 2023), raised its joint accuracy by 0.282, while the same switch moved a 72B model by at most 0.005; a 27B model with PoT then matched the 72B answering directly (0.884 vs. 0.873 joint). A lineage comparison against the base Qwen2.5-7B-Instruct (Yang et al., 2024) qualifies the lever: the same harness that rescues the domain-adapted 7B model taxes the base model’s citation discipline, so PoT is modeldependent; the divergence is portable, reappearing in every rigid output format we tested (JSON validity, answer markers, evidence identifiers). Graph artifacts helped only in distilled form, and only at scale: appending the entities nearest to the query, out of chunk-level graphs built by a 7B extractor, added 0.017 joint on the 27B line and 0.012 on the 72B, but the same enrichment hurt the compact 7B line, dropping the base from 0.497 to 0.448 joint and the adapted model from 0.641 to 0.011, while fuller graph views subtracted up to 0.011.

We read this asymmetry through a distinction between two kinds of competence an LLM can supply: knowledge about the world, which scales steeply with parameter count, and knowledge about language—comprehension, extraction, faithful reformatting—which scales far more gently. A harness that routes world knowledge to external tools (an interpreter for arithmetic, a retriever for passage search, an ontology for fact structure)

leaves the model only the linguistic residue, and it is on that residue that a compact model is competitive: our 7B line trails the 72B baseline by 0.10 on instruction following and 0.05 on flexible arithmetic, gaps far smaller than any world-knowledge comparison would show. The same harness reaches 0.64 joint on a single GPU at 0.14 kgCO<sub>2</sub>e for its check runs, against eight GPUs and 0.53 kgCO<sub>2</sub>e for the 72B line (Appendix C).

On the test set, none of this mattered, because the test PDFs are raster scans with watermarks, unlike the born-digital training and validation documents, and our reading pipeline degraded their text beyond what retrieval and arithmetic could recover. Three facts pin down the diagnosis. Answers flipped on 89.6% of tasks when we replaced OCR with page-level vision transcription: the flip measures the input, with the reasoning stack untouched. Evidence sets, singleton in all training labels, grew to three or more blocks on 32.5% of test tasks, a signature of a model that cannot locate the passage and hedges. And the final leaderboard is bimodal, with a 22-team spike at 67.46% and a long reading-failure tail in which we sit.

Our contributions:

• a DocSem system in which architectural choices (PoT execution, self-consistency, distilled graph context) moved the metrics further than any model-scale increase we could afford;

• a 7B lineage comparison (Meno-Lite-0.1 against its base Qwen2.5-7B-Instruct) showing the PoT lever is model-dependent: it rescued the domain-adapted model and taxed the base model’s citation discipline;

• a cross-task reading of that lineage (instruction following, arithmetic, world knowledge, JSON generation, extraction) showing that structured-output training confers a languageindependent discipline in rigid formats, while the base model keeps an edge in free-form instruction;

• retrieval and context ablations showing that only the top-k distillation of chunk-level graphs helps, while fuller graph views hurt;

• a submission-level post-mortem of three test attempts, plus a controlled degradation study that reproduces the OCR half of the test collapse on validation inputs and bounds what the simulation misses.

## 2 Task and the Data Shift We Missed

Each task provides a PDF and a user query paraphrasing an embedded GSM-style quantitative passage (Singh et al., 2026; Cobbe et al., 2021). Documents contain a background narrative, tables, dates, and numeric facts; roughly one block in ten holds tabular content. Blocks carry visible identifiers (a token ending in a colon); a submission must return the answer and the set of justifying identifiers, matched exactly against the gold set. In all 908 training labels the gold evidence is a single block and the answer is numeric, which fixed two design choices: output exactly one identifier unless the passage demands more, and emit canonical decimal strings.

Our exploratory analysis of the training data verified the things we thought to check: no duplicate documents across splits (908/217/1730 unique PDFs), 99.78% recoverability of gold blocks by our parser, a block-length tail short enough (p99 = 638 tokens) to avoid truncation in an 8k context. The test snapshot differed in one more way, which we measured late: its PDFs are low-resolution raster scans, one image per page, stamped with a diagonal “TESTING COPY” watermark and running headers, while training and validation pages are clean renderings of born-digital documents. Tesseract (Smith, 2019) on these pages yields a median service-word share of 0.091 versus 0.224 on validation (Figure 1), and our later audit found pages where the vision model transcribed the watermark in loops until it hit the token ceiling. We verified checksums before the deadline; we did not look at the pixels. What we now audit before designing the reading stack: born-digital versus raster pages, the presence of a text layer, image resolution, watermarks and running headers, and the distribution of page and block counts—a check that would have caught the test shift in minutes.

## 3 System

## 3.1 Document ingestion

Figure 2 sketches the pipeline. Blocks are segmented by the generic rule “identifier token up to the first colon”, which copies opaque test tokens verbatim; tables are linearized row-wise as column: value lines inside their block. Training and validation PDFs are parsed directly; raster test pages were transcribed page-by-page by Qwen2.5-VL-7B (Bai et al., 2025) at 2× zoom, with pages distributed across service instances and stitched by page index. A targeted repair pass with Qwen3.8-27B (Qwen Team, 2026) re-read 38 damaged pages (26 documents): three empty ones and 35 degenerate repetition loops on near-empty watermarked pages.

![](images/c222046fe6a3536a18e63809708d5147d44bba78b1af287d43b1d6c08a7e1460.jpg)  
Figure 1: The same task under two input regimes. (a) A validation page renders cleanly: block identifiers survive and Tesseract recovers the text. (b) A test page is a low-resolution raster scan stamped with a diagonal “TESTING COPY” watermark and running headers; Tesseract garbles the identifiers (#72:, H&62<sup>◦</sup>, 2-713K:) so evidence can no longer match gold labels.

## 3.2 Retrieval

Each deduplicated document gets its own index; a block is a chunk. Dense vectors come from gtemultilingual-base (Zhang et al., 2024), sparse from BM42, merged by reciprocal rank fusion (Cormack et al., 2009) in Qdrant; a gte-multilingual-reranker cross-encoder cuts the fused top-8 to the 3 blocks shown to the generator. The pair was chosen by an evidence-recall benchmark on our held-out split: bge-m3 reached recall@1 of 0.348, gte 0.901; after reranking, gte saturates at 1.000 and bge reaches 0.807 (Appendix B).

## 3.3 Program-of-Thoughts generation

The generator writes a Python program that reads only the displayed blocks, and a sandboxed interpreter executes it; the final variable becomes the answer, and the cited identifiers become evidence. Structured output (a JSON schema over reasoning, program, and evidence) is enforced server-side by vLLM (Kwon et al., 2023). Self-consistency (Wang et al., 2023) samples k=5 programs at T=0.8 and votes on the executed answers; identifiers snap to valid block tokens when transcription noise corrupts them.

## 3.4 Graph context and final assembly

Chunk-level mini-graphs are extracted from the retrieved passages by Meno-Lite-0.2, a compact domain-adapted Qwen2-family model and the successor of the released Meno-Lite-0.1, with a numeric ontology (NEREL entity types (Loukachevitch et al., 2021) extended with QUAN-TITY and RATE). The engine underneath, from block indexing to search, is RAGU (Komarov et al., 2026).<sup>2</sup> The generator prompt is enriched with the entities nearest to the query in embedding space. The final test submission merged three votes by majority (the base PoT run, the key-entity run, and a multimodal escalation that shows page images of non-unanimous tasks to Qwen3.8-27B). While test scores were still hidden, an external GLM-5.3-Flash judge probed 60 test tasks; the released leaderboard later superseded these estimates.

![](images/cb1c0f55097a1f61b3d29fcb7f792e787de5aae7fc0cd9ea940fef36113b74d5.jpg)  
Figure 2: The DocSem pipeline. Raster test pages are transcribed by a VLM before block segmentation; a hybrid retriever selects relevant passages; RAGU builds chunk-level graphs whose entities enrich the prompt; Program-of-Thoughts returns the answer and evidence.

<table><tr><td>Configuration (check, 181 tasks)</td><td>Ans.</td><td>Evid. EM</td><td>Joint</td></tr><tr><td>Qwen2.5-7B-Instruct, direct</td><td>0.735</td><td>0.956</td><td>0.691</td></tr><tr><td>Qwen2.5-7B-Instruct, PoT</td><td>0.757</td><td>0.646</td><td>0.497</td></tr><tr><td>Meno-Lite-0.1 7B, direct</td><td>0.420</td><td>0.895</td><td>0.359</td></tr><tr><td>Meno-Lite-0.1 7B, PoT</td><td>0.674</td><td>0.934</td><td>0.641</td></tr><tr><td>Qwen2.5-72B, direct, k=1</td><td>0.867</td><td>1.000</td><td>0.867</td></tr><tr><td>Qwen2.5-72B, direct, k=3</td><td>0.873</td><td>1.000</td><td>0.873</td></tr><tr><td>full document, k=3</td><td>0.884</td><td>1.000</td><td>0.884</td></tr><tr><td>Qwen2.5-72B, PoT, k=3</td><td>0.873</td><td>1.000</td><td>0.873</td></tr><tr><td>Qwen2.5-72B, PoT, k=5, T=0.8</td><td>0.878</td><td>1.000</td><td>0.878</td></tr><tr><td>Qwen3.8-27B, PoT, k=5</td><td>0.884</td><td>1.000</td><td>0.884</td></tr><tr><td>+ key entities (top-3)</td><td>0.901</td><td>1.000</td><td>0.901</td></tr><tr><td>+ mini-graph (12 ent.)</td><td>0.878</td><td>1.000</td><td>0.878</td></tr><tr><td>+ entities, edges</td><td>0.890</td><td>1.000</td><td>0.890</td></tr></table>

Table 1: Main line on the internal check split. PoT moves a compact model further than any scale jump; graph context helps only in top-3 distilled form.

## 4 Results on Labeled Data

Table 1 traces the generation line, and its reading is asymmetric by model size and by model origin. Consider the 7B pair first, where the same two protocols run on Meno-Lite-0.1 and on its root ancestor Qwen2.5-7B-Instruct. Direct answering splits them: the unmodified base reaches 0.691 joint (0.735 answers, 0.956 evidence exact match), while the domain-adapted model lands at 0.359, a drop consistent with its declared focus on Russianlanguage RAG and extraction skills. Program-of-Thoughts then moves them in opposite directions: it lifts Meno-Lite by 0.282 joint (0.359 to 0.641, with evidence discipline rising to 0.934) and pulls the base down by 0.194 (0.691 to 0.497), because the base keeps computing (0.757 answers) but stops citing carefully (0.646 evidence exact match) once it writes programs. The harness lever is real but model-dependent: it pays most where the model is weakest, and it can tax a strength.

At the top of the scale the same lever moves nothing: the 72B model scores 0.873 with direct answering and 0.873 with PoT at identical sampling (k=3, T=0.7), and 0.878 at the champion sampling (k=5, T=0.8). PoT is an equalizer, and the 27B model with the full harness (0.884, or 0.901 with key entities) matches or passes every 72B row in the table at roughly 2.7× fewer parameters. Self-consistency behaves the same way: k=3 over k=1 adds 0.006 on the 72B line, against 0.022 for k=5 over k=3 on the compact line. As shown in Appendix A, PoT demonstrations trade answer accuracy for perfect evidence.

The asymmetry has an explanation that guided our design. DocSem documents are synthetic: cities, agencies, and numbers are invented for the benchmark, so world knowledge memorized in parameters buys nothing, and the residual demands on the LLM are linguistic (comprehension of a paraphrased query, selection of stated facts, faithful program writing), while arithmetic, passage search, and fact structuring are routed to the interpreter, the retriever, and the ontology. This is the design hypothesis behind Meno-Lite, a 7B line trained to read rather than to memorize (Bondarenko, 2026a). DocSem, an English benchmark outside that model’s primary domain, tests the hypothesis from its weak side: the harness recovers most of the distance the domain adaptation had cost (0.359 to 0.641 joint), while the same harness adds at most 0.005 to a model ten times larger. Retrieval ablations and the full-document baseline complete the picture in the appendix. The dense retriever does almost all the selection work: recall@3 is 0.9945 for dense alone and 0.0663 for BM42, because the paraphrase queries are built to avoid the target passage’s wording; the cross-encoder closes the residual gap (1.000 at top-3). The fulldocument row of Table 1 adds an uncomfortable fact: with the 72B generator, feeding all 23–42 blocks matches curated top-3 selection (0.884 vs. 0.873, within noise). Block selection earned its place on the compact line, where context discipline and cost bind, and it kept evidence attribution exact on every run; for a strong generator on short documents it was accuracy-neutral.

<table><tr><td>Validation portal (217 tasks)</td><td>Ans.</td><td>Evid. EM</td><td>Joint</td></tr><tr><td>Meno-Lite-0.1 + PoT</td><td>0.456</td><td>0.899</td><td>0.438</td></tr><tr><td>Qwen2.5-7B-Instruct, direct</td><td>0.659</td><td>0.959</td><td>0.636</td></tr><tr><td>Meno-Lite-0.2 (unreleased) + PoT, track C (App. B)</td><td>0.682</td><td>0.949</td><td>0.641</td></tr><tr><td>Qwen2.5-72B, prompt v1</td><td>0.774</td><td>1.000</td><td>0.774</td></tr><tr><td>Qwen2.5-72B, prompt v2</td><td>0.825</td><td>1.000</td><td>0.825</td></tr><tr><td>Qwen3.8-27B + PoT</td><td>0.848</td><td>1.000</td><td>0.848</td></tr><tr><td>+ permutation ensemble</td><td></td><td></td><td>0.853</td></tr><tr><td>+ key entities</td><td></td><td>1</td><td>0.853</td></tr></table>

Table 2: Portal-scored validation submissions. The 27B PoT line overtakes the 72B direct line; graph enrichment adds 0.005 at portal precision.

The validation portal (Table 2) confirms the ordering on organizer-held labels: the 27B PoT configuration passes the 72B direct one, and the keyentity enrichment holds a small positive effect. The compact line’s own row carries a caution about offdomain generalization: Meno-Lite-0.1 with PoT drops from 0.641 joint on check to 0.438 on the portal, a 0.203 gap nearly four times the base model’s 0.055 (0.691 to 0.636), and the loss is concentrated in answers (0.674 to 0.456) while evidence discipline barely moves (0.934 to 0.899). Majority voting over four context variants (base, key entities, full mini-graph, entities-plus-edges) scored 0.890– 0.901 joint on check against 0.901 for the best single vote, which is why the three-vote merge of the final submission stayed a hedge against transcription noise rather than an accuracy device. Singlerow differences on 181 tasks carry 95% confidence intervals of roughly ±4.5 points (0.901 maps to 0.849–0.937), so we read the table by its ordering across model lines and samplers rather than by any one delta.

<table><tr><td>Benchmark</td><td>Qwen2.5-7B-Instruct</td><td>Meno-Lite-0.1</td></tr><tr><td>IFEval, strict</td><td>0.791</td><td>0.679</td></tr><tr><td>GSM8K, flexible / strict</td><td>0.719 / 0.187</td><td>0.759 / 0.619</td></tr><tr><td>MMLU, 5-shot</td><td>0.743</td><td>0.721</td></tr><tr><td>MMLU, Russian</td><td>0.649</td><td>0.653</td></tr><tr><td>JSONSchemaBench (easy), valid JSON</td><td>0.302</td><td>0.981</td></tr><tr><td>Evidence exact match under PoT (DocSem)</td><td>0.646</td><td>0.934</td></tr><tr><td>Librusec history (Russian QA)</td><td>0.781</td><td>0.906</td></tr></table>

Table 3: Cross-task comparison of the base and the adapted 7B model. Rigid output formats split the pair sharply in favor of the adapted model, while free-form instruction and English world knowledge favor the base; Russian narrative knowledge favors the adapted model.

## 4.1 Structured-Output Training as a Harness Prerequisite

The 7B comparison is not confined to DocSem: we ran the base and the adapted model through instruction following (IFEval (Zhou et al., 2023)), arithmetic (GSM8K (Cobbe et al., 2021)), world knowledge (MMLU (Hendrycks et al., 2021), English and Russian), and structured generation (JSONSchemaBench (Geng et al., 2025)) with lmevaluation-harness (Gao et al., 2024) at temperature 0 and chat templates applied server-side. Table 3 condenses the outcome, and two patterns stand out.

First, the pair diverges most on rigid output formats, and the divergence is language-independent: the base model emits valid JSON only 30% of the time on the easy JSONSchemaBench split (0.302 against 0.981), follows the GSM8K boxed-answer convention (“####”) in 19% of cases (0.187 against 0.619), and, once it writes programs, drops to 0.646 evidence exact match where the adapted model holds 0.934. The citation and format discipline that PoT rewards inside DocSem is thus a transferable property of structured-output training, in English as much as in Russian. Second, the language trade-off is real but narrower than the design hypothesis: the base keeps a small edge in English world knowledge (0.743 vs. 0.721 MMLU), the two are within noise on Russian MMLU (0.649 vs. 0.653), and the adapted model leads only in Russian narrative knowledge (0.906 vs. 0.781 on Librusec history), while the base wins free-form instruction (0.791 vs. 0.679). Extraction is the one axis without a clean winner (NEREL-Bench (Bondarenko, 2026b): the base leads entity recognition 0.473 vs. 0.437 F1, the adapted model relation extraction 0.248 vs. 0.192). What the adaptation bought, then, is not “more language” in general but a specific, portable competence in structured output—exactly what a program-writing harness consumes. Domain adaptation cost the model 0.022 of English world knowledge (0.743 to 0.721 MMLU) and bought 0.679 of JSON validity and 0.432 of strict-format compliance on GSM8K. Structured-output competence is therefore not a side benefit but a precondition for harness-based scaling: PoT pays where that discipline exists (Meno-Lite: +0.282) and taxes where it does not (base Qwen2.5-7B: −0.194). JSON-SchemaBench validity (0.981 vs. 0.302) predicts the sign of the PoT effect better than parameter count does.

<table><tr><td>Test attempt</td><td>Joint (%)</td><td>Ans. (%)</td><td>Evid. F1</td></tr><tr><td>1: Tesseract reading</td><td>0.29</td><td>2.54</td><td>1.50</td></tr><tr><td>2: VLM page transcription</td><td>13.58</td><td>17.57</td><td>20.34</td></tr><tr><td>3: + merge, mm escalation, repair</td><td>13.41</td><td>17.34</td><td>20.11</td></tr></table>

Table 4: Portal truth for the three accepted attempts, in percent. The leaderboard selected attempt 2; rank 149 of 163.
<table><tr><td>Diagnostic per attempt</td><td>1</td><td>2</td><td>3</td></tr><tr><td>Singleton evidence, %</td><td>93.9</td><td>64.5</td><td>71.9</td></tr><tr><td>Three or more blocks, %</td><td>3.8</td><td>32.5</td><td>25.3</td></tr><tr><td>Evidence codes outside doc, %</td><td>8.7</td><td>3.5</td><td>1.7</td></tr></table>

Table 5: Submission diagnostics. Gold evidence is always a single block; inflated sets mark a model that cannot locate the passage.

## 5 What Happened on Test

Table 4 shows the portal truth; Table 5 the diagnostics we computed afterwards. The first attempt, built on Tesseract, was effectively a random draw (2.54% answers). Replacing the reader with pagewise vision transcription multiplied joint accuracy by five, and answers changed on 89.6% of tasks between the two attempts: the difference came from re-reading the documents, with the reasoning stack unchanged. The third attempt bundled everything that had passed our check-split gates: the three-vote merge with multimodal escalation, plus a deadline repair of the 38 pages the audit had flagged. It altered 14.3% of submission rows relative to attempt 2 and scored 0.17 percentage points of joint accuracy lower, a difference within sampling noise on 1,730 tasks; the informative part is that a 14% turnover of predictions produced no gain to trade against. The repair itself changed nine rows, all inside the 26 re-read documents, several of them degrading a nonzero answer to zero on the repaired text.

Format sanitation, the one axis that clearly improved across attempts (invalid evidence codes fell from 8.7% to 1.7% of tasks), did not move the metrics, which locates the remaining failure in semantics: wrong passage, wrong numbers, or both. Evidence inflation points the same way: on a third of tasks the model returned three or more blocks where the gold set has exactly one.

The leaderboard is bimodal. A smooth distribution would follow if reasoning architecture separated teams; instead the field splits into a majority that read the raster documents well enough to reason, a dense 22-team spike at 67.46% joint accuracy, and a long tail that did not—a shape consistent with reading quality, not reasoning, having separated the field, though we cannot verify other teams pipelines. The top-ranked team reached 85% with the same inputs, so the task was solvable and the failure sat in our reading stage.

## 6 A Controlled Degradation Study

The post-mortem is observational; we can also run the experiment it implies. We re-rendered the validation PDFs to match the measured physical properties of the test scans: 72-dpi grayscale JPEG (the test pages embed 69–78-dpi JPEG images), with a diagonal “TESTING COPY” stamp and a running header, both baked into the pixels as in the genuine article. A median document in this degraded corpus shows a stopword share of 0.237, between the born-digital originals (0.349) and the test scans as read by the same OCR (0.153), so the treatment lands between the two regimes it connects. The same pipeline, the same 72B generator, and the same sampling as the 0.825 validation row then run over the degraded rendering; the organizers’ portal, which holds the validation labels, scores the result.

Table 6 reports the outcome, and it splits cleanly by reader. Tesseract collapses exactly as on the real test: answers fall from 0.825 to 0.244, and evidence falls to exactly zero, because OCR garbles the printed b06: prefixes into tokens like 0%: that no longer match the gold identifiers at all. The vision-language reader recovers almost everything on our synthetic scans: 0.756 joint against 0.825 on born-digital input, with perfect evidence identifiers, because page-wise transcription restores the canonical prefixes it can read from context. The genuine test told a different story for the same reader (17.6% answers), and the distance between these two numbers is itself a finding: a clean re-render at test-like resolution reproduces the OCR failure mode but underestimates what the real scans did to vision transcription, whose damage on test came from dirtier scans, eight-page documents with four times the blocks, and identifier tokens that cannot be restored from convention. The controlled half of the claim stands (rendering alone can zero out the primary metric through the identifier channel), and the uncontrolled half sharpens the lesson of Section 2: simulate the inputs, then verify the simulation against the real thing before trusting either.

<table><tr><td>Reading regime</td><td>Joint</td><td>Ans.</td><td>Evid. F1</td></tr><tr><td>val, born-digital (reference)</td><td>0.825</td><td>0.825</td><td>1.000</td></tr><tr><td>val, degraded raster + Tesseract</td><td>0.000</td><td>0.244</td><td>0.000</td></tr><tr><td>val, degraded raster + VLM pages</td><td>0.756</td><td>0.756</td><td>1.000</td></tr><tr><td>test, Tesseract (attempt 1)</td><td>0.29</td><td>2.54</td><td>1.50</td></tr><tr><td>test, VLM pages (attempt 2)</td><td>13.58</td><td>17.57</td><td>20.34</td></tr></table>

Table 6: The same tasks and model under three input renderings, scored by the validation portal (top, fractions) against the two real test attempts (bottom, percent). Synthetic degradation reproduces the OCR collapse; the residual gap to the real test rows measures what the simulation does not capture.

## 7 Related Work

DocSem instantiates the GSM-SEM recipe (Singh et al., 2026) over synthetic documents; our reasoning stack follows Program-of-Thoughts (Chen et al., 2023) and self-consistency (Wang et al., 2023), built on chain-of-thought prompting (Wei et al., 2022) and the program-aided line of work (Gao et al., 2023), with structured output served by vLLM (Kwon et al., 2023). Our system is a retrieval-augmented generation pipeline (Lewis et al., 2020); graph-based retrieval descends from GraphRAG and its lightweight variants (Edge et al., 2024; Guo et al., 2025; Gutiérrez et al., 2024), evaluated for multi-hop settings by Xiang et al. (2026); our finding is a boundary condition: on singlepassage arithmetic over short documents, only the distilled top-k of the graph earns its tokens. Ensemble selection by a lightweight judge follows our SemEval-2026 system (Bondarenko et al., 2026); here the analogous device, majority voting over context variants, did not beat the best single vote. Grounding over tabular and textual evidence connects to HybridQA and TaPas (Chen et al., 2020; Herzig et al., 2020); the DocSem twist is the exactmatch block identifier, which punishes any segmentation drift.

## 8 Conclusion: Lessons with Numbers

First, audit the physical nature of evaluation inputs before the architecture. We verified checksums, duplicates, and label recoverability, and skipped the pixel-level look that would have shown watermarked raster scans; the cost was the gap between 0.853 and 0.136 joint accuracy. Second, application architecture is worth more than model scale, because the two kinds of competence a model can supply do not scale alike: routing world knowledge to external tools leaves the model only the linguistic residue, where a 27B model with the full harness matches a 72B model without it at roughly 2.7× fewer parameters, on one GPU and 0.14 kgCO<sub>2</sub>e against eight GPUs and 0.53 kgCO<sub>2</sub>e. None of this transfers from born-digital validation to damaged test reading, which is why the first lesson comes first. Third, the lever is model-dependent in a predictable way: PoT pays where format discipline exists and taxes where it does not, which is how a compact model is made harness-ready rather than merely small. Fourth, deadline repairs treat symptoms. The re-read of 38 damaged pages changed nine submission rows and slightly lowered the score, because the visible artifacts were a small sample of the systemic transcription damage.

## Limitations

This is a single-shared-task study; the readingfailure analysis rests on portal scores and submission diagnostics, not on human re-annotation of test documents. Our vision reader was a 7B model transcribing pages independently; we did not evaluate stronger document-parsing VLMs, so the ceiling of our pipeline on raster inputs is unknown. The 7B lineage comparison runs an English benchmark through a Russian-primary model, which is the weaker side of its declared domain; the comparison bounds the harness, not the model. The externaljudge probe referenced in the system description covers 60 test tasks only. Validation-portal numbers rest on 217 tasks (95% confidence intervals of four to five points at the observed accuracies); differences below one point between validation rows should be read accordingly.

## Ethics Statement

The system processes only the shared-task PDFs and outputs numeric answers with block identifiers; no personal data is processed. Model serving used institutional cluster resources.

## References

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, and 8 others. 2025. Qwen2.5-vl technical report. Preprint, arXiv:2502.13923.

Ivan Bondarenko. 2026a. Meno-lite-0.1: A 7b language model optimized for russian rag pipelines.

Ivan Bondarenko. 2026b. Nerel-bench: A benchmark for evaluating llms on russian knowledge graph construction tasks. https://huggingface.co/ datasets/bond005/NEREL\_bench.

Ivan Bondarenko, Roman Derunets, Oleg Sedukhin, Mikhail Komarov, Ivan Chernov, and Mikhail Kulakov. 2026. RaguTeam at SemEval-2026 task 8: Meno and Friends in a judge-orchestrated LLM ensemble for faithful multi-turn response generation. In Proceedings ofthe 20th International Workshop on Semantic Evaluation (2026), pages 1678–1694, San Diego, California, USA. Association for Computational Linguistics.

Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W. Cohen. 2023. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. Transactions on Machine Learning Research.

Wenhu Chen, Hanwen Zha, Zhiyu Chen, Wenhan Xiong, Hong Wang, and William Yang Wang. 2020. HybridQA: A dataset of multi-hop question answering over tabular and textual data. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2020, pages 1026–1036, Online. Association for Computational Linguistics.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. Preprint, arXiv:2110.14168.

Gordon V. Cormack, Charles L A Clarke, and Stefan Buettcher. 2009. Reciprocal rank fusion outperforms condorcet and individual rank learning methods. In Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’09, pages 758–759, New York, NY, USA. Association for Computing Machinery.

Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, and Jonathan Larson. 2024. From local to global: A graph RAG approach to query-focused summarization. arXiv preprint arXiv:2404.16130.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster,

Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, and 5 others. 2024. The language model evaluation harness.

Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Pal: Program-aided language models. In Proceedings of the 40th International Conference on Machine Learning, pages 10764– 10799.

Saibo Geng, Hudson Cooper, Michał Moskal, Samuel Jenkins, Julian Berman, Nathan Ranchin, Robert West, Eric Horvitz, and Harsha Nori. 2025. Jsonschemabench: A rigorous benchmark of structured outputs for language models. Preprint, arXiv:2501.10868.

Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. 2025. LightRAG: Simple and fast retrievalaugmented generation. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 10746–10761, Suzhou, China. Association for Computational Linguistics.

Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. 2024. Hipporag: Neurobiologically inspired long-term memory for large language models. In Advances in Neural Information Processing Systems, volume 37, pages 59532–59569. Curran Associates, Inc.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring massive multitask language understanding. In International Conference on Learning Representations.

Jonathan Herzig, Pawel Krzysztof Nowak, Thomas Müller, Francesco Piccinno, and Julian Eisenschlos. 2020. TaPas: Weakly supervised table parsing via pre-training. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 4320–4333, Online. Association for Computational Linguistics.

Mikhail Komarov, Ivan Bondarenko, Stanislav Shtuka, Oleg Sedukhin, Roman Shuvalov, Yana Dementyeva, Matvey Solovyov, and Nikolay O. Nikitin. 2026. Ragu: A multi-step graphrag engine with a compact domain-adapted llm. Preprint, arXiv:2607.11683.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In Proceedings ofthe 29th Symposium on Operating Systems Principles, SOSP ’23, pages 611–626, New York, NY, USA. Association for Computing Machinery.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for knowledgeintensive nlp tasks. In Advances in Neural Information Processing Systems, volume 33, pages 9459– 9474. Curran Associates, Inc.

Natalia Loukachevitch, Ekaterina Artemova, Tatiana Batura, Pavel Braslavski, Ilia Denisov, Vladimir Ivanov, Suresh Manandhar, Alexander Pugachev, and Elena Tutubalina. 2021. NEREL: A russian dataset with nested named entities, relations and events. In Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP 2021), pages 876–885, Held Online. IN-COMA Ltd.

Qwen Team. 2026. Qwen3.8-Max: A new bar for coding and cowork.

Jyotika Singh, Fang Tu, Aziza Mirsaidova, Amit Agarwal, Hitesh Laxmichand Patel, Sandip Ghoshal, Miguel Ballesteros, Karan Dua, Yassine Benajiba, Weiyi Sun, Tao Sheng, Graham Horwood, Sujith Ravi, and Dan Roth. 2026. Gsm-sem: Benchmark and framework for generating semantically variant augmentations. Preprint, arXiv:2605.07053.

Ray Smith. 2019. Tesseract open source ocr engine. https://github.com/tesseract-ocr/ tesseract.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc V Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems, volume 35, pages 24824–24837. Curran Associates, Inc.

Zhishang Xiang, Chuanjie Wu, Qinggang Zhang, Shengyuan Chen, Zijin Hong, Xiao Huang, and Jinsong Su. 2026. When to use graphs in RAG: A comprehensive analysis for graph retrieval-augmented generation. In International Conference on Learning Representations (ICLR 2026).

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, and 22 others. 2024. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115.

Xin Zhang, Yanzhao Zhang, Dingkun Long, Wen Xie, Ziqi Dai, Jialong Tang, Huan Lin, Baosong Yang,

Pengjun Xie, Fei Huang, Meishan Zhang, Wenjie Li, and Min Zhang. 2024. mGTE: Generalized longcontext text representation and reranking models for multilingual text retrieval. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 1393–1412, Miami, Florida, US. Association for Computational Linguistics.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. 2023. Instruction-following evaluation for large language models. Preprint, arXiv:2311.07911.

## A Run Registry

All 181-task check rows referenced in Table 1, with sampling parameters, appear in the internal run registry; the val rows of Table 2 are portal-scored. PoT demonstrations (34 verified programs) lift evidence exact match to 1.000 and cost 0.02–0.03 answer accuracy: a trade we declined for the final system. A verification pass that asks the model to re-derive constants from cited blocks subtracts 0.013 joint by rejecting honest fractions.

## B Retrieval Benchmark

Evidence recall on the check split: the bge-m3 pair reaches 0.348/0.646/0.983 at top-1/3/8 and 0.807 after reranking to top-3; the gte pair reaches 0.901/0.995/1.000 and 1.000. A component ablation of the gte pair shows the dense vectors carrying the selection: BM42 alone reaches recall@3 of 0.066, because paraphrase queries are built to avoid the wording of the target passage, and the reranker lifts dense top-8 to a perfect top-3. Track C (mixing vector and graph search contexts) won by 0.022 joint at k=3 on the compact line and was neutral at k=5; the graph-native track B lost 0.11 joint to vector retrieval on these short documents.

## C Cost Accounting

Mini-graph extraction for the test submission (1,730 tasks, two extractor calls per retrieved passage) ran at about four tasks per minute on a teninstance 7B fleet, roughly seven hours end to end; generation over the same tasks at k=5 took thirteen minutes per four-document shard on a single 27B server. The full-document baseline of Table 1 skips the retrieval stack entirely, at the price of prompts that grow with document length.

Measured serving costs for the 181-task check runs of Table 1 (wall clock on our infrastructure, A100 80GB; energy estimated at 400 W TDP per GPU, PUE 1.1, grid intensity 0.35 kgCO<sub>2</sub>e/kWh):

<table><tr><td>Configuration</td><td>GPUs</td><td>Wall</td><td>GPU-h</td><td> $\mathrm { k g C O _ { 2 } e }$ </td></tr><tr><td>72B direct, k=1</td><td>8</td><td>15 min</td><td>2.0</td><td>0.31</td></tr><tr><td>72B PoT, k=3</td><td>8</td><td>26 min</td><td>3.5</td><td>0.53</td></tr><tr><td>72B PoT, k=5</td><td>8</td><td>26 min</td><td>3.5</td><td>0.53</td></tr><tr><td>72B full doc, k=3</td><td>8</td><td>29 min</td><td>3.9</td><td>0.60</td></tr><tr><td>7B PoT (Meno-0.1), k=5</td><td>1</td><td>53 min</td><td>0.9</td><td>0.14</td></tr></table>

Table 7: Serving cost of the check runs. The compact line reaches 0.64 joint on a single GPU in about twice the wall time the 72B line spends on eight of them.

## D Prompts and Ontology

The generator instruction fixes five rules: reason only from displayed blocks; never mix numbers across passages; the program may only read displayed block contents; the answer is a canonical decimal string; evidence identifiers are copied character by character. In the PoT mode the same instruction asks for a Python program whose last assignment holds the answer, executed by a sandboxed interpreter with a 2 s budget; samples whose programs fail to execute are dropped from the vote. The extractor receives 18 entity types (the NEREL numeric and object core (Loukachevitch et al., 2021) plus QUANTITY and RATE) and 13 relation types (PRICE\_OF, INCOME, EXPENDITURE, AGE\_IS, POINT\_IN\_TIME, START\_TIME, END\_TIME, PART\_OF, LO-CATED\_IN, TAKES\_PLACE\_IN, AGENT, OWNER\_OF, HAS\_QUANTITY), with MEA-SURED\_IN reserved for unit relations.

## E Degradation Protocol

Each validation page is rendered to 72-dpi grayscale, stamped, re-rendered 1:1, and stored as JPEG quality 55 in a one-image-per-page PDF, so the watermark and header live in the pixels and the text layer is empty, matching the genuine test scans (69–78 dpi JPEG). The shadow snapshot keeps original file names, so the pipeline resolves tasks unchanged; only the parse differs. Parsing runs in generic identifier mode, because OCR corrupts the printed b01: prefixes into tokens like 0%: or ‘b12; strict known-mode parsing then returns zero blocks. Unchanged paths also invalidate the pathto-hash index unless cleared.