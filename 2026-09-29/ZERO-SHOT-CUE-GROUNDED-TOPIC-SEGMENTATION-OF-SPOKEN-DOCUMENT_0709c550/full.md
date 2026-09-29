# ZERO-SHOT CUE-GROUNDED TOPIC SEGMENTATION OF SPOKEN DOCUMENTS

Suhwan Choi<sup>1</sup>, Myeongho Jeon<sup>2</sup>, Myungjoo Kang<sup>1</sup>

<sup>1</sup>Seoul National University, <sup>2</sup>KAIST

## ABSTRACT

Topic segmentation structures spoken documents into coherent sections, facilitating navigation and downstream understanding. The appropriate granularity can vary substantially, ranging from broad thematic shifts to fine-grained subtopics. Existing LLM-based segmenters, however, often struggle to adapt to this variation, causing them to either merge distinct subtopics or over-segment coherent themes. To address this, we introduce Cue-Grounded Segmentation (CGS), a training-free framework that operates without any taskspecific supervision. CGS first identifies phrases that explicitly signal the start of a new topic and uses their sentence positions as segment boundaries. When such cues are insufficient, it falls back to semantic segmentation, guided by the document structure inferred during cue extraction. Across six benchmarks and six LLM backbones, CGS consistently outperforms existing baselines, remains robust to noisy ASR transcripts, and achieves these gains with low API cost on proprietary models.

Index Terms— topic segmentation, spoken language processing, large language models, zero-shot learning, adaptive inference

## 1. INTRODUCTION

Topic segmentation organizes spoken documents into coherent sections by identifying where new topics begin. These topic-level units are essential for efficiently navigating, retrieving, and summarizing long-form recordings. Unlike written documents, however, spoken content rarely provides explicit structural cues such as headings or section titles, requiring topic boundaries to be inferred from the discourse itself, including transition phrases and shifts in subject matter.

Classical unsupervised methods place boundaries where lexical cohesion drops [1, 2], and speech segmentation methods use prosody and discourse markers [3, 4]. Representation-based methods learn boundary decisions from labeled corpora [5, 6], train per-document segment representations without labels [7], or hierarchically cluster transcript embeddings [8]. However, these methods determine section granularity through thresholds, supplied segment counts, or learned boundary rules, which can limit generalization across diverse document structures. Recent LLM systems predict topic boundaries through boundary lists, sentence-level topic labels, recursive splitting, table-of-contents planning, topic-shift predictions based on utterance intent, or iterative chunking [9, 10, 11, 12, 13]. These methods directly predict boundaries or segment structures without explicitly grounding them in topic-opening cues. To make these predictions, the model must decide whether each change in the discussion marks a new topic or is only a minor shift within the current topic. Treating minor shifts as new topics leads to over-segmentation, while overlooking meaningful topic changes leads to under-segmentation.

Spoken documents vary substantially in their topical structure. Some consist of short, clearly separated topics, whereas others sustain extended discussions around a single subject. An effective topic segmenter must therefore adapt to a wide range of segment granularities. This variation is evident across six benchmarks. The median ground-truth segment length ranges from 12.9 sentences in chaptered YouTube videos to 241 sentences in research-group meetings, an 18.6-fold difference. Despite this substantial variation, existing LLM-based segmenters often produce segments that are systematically too short or too long for the target corpus (Fig. 1).

![](images/1007acaac33f541bee61fb1d8f56c58f2652b64f44afc5719d94e23dd7ab197c.jpg)  
Fig. 1. Predicted versus ground-truth section length. CGS more closely tracks ground-truth lengths across corpora, while baselines often produce sections that are too short or too long. Lengths are defined in Table 1; predictions are averaged across six backbones. Vertical lines mark corpora; colored lines show log–log fits.

Table 1. Evaluated corpus subsets. n counts documents. For each document, we calculate the average ground-truth segment length in sentences. Length reports the median across documents.
<table><tr><td>Dataset</td><td>n Length</td><td></td><td>Description</td></tr><tr><td>YTSeg [14]</td><td>1,448</td><td>12.9</td><td>Creator-authored chapters</td></tr><tr><td>ICSI [15, 16]</td><td>75</td><td>241.0</td><td>Research meetings</td></tr><tr><td>AMI [17]</td><td>129</td><td>66.0</td><td>Multi-party meetings</td></tr><tr><td>MeetingBank [18, 16]</td><td>28</td><td>72.6</td><td>City-council meetings</td></tr><tr><td>QMSum [19, 16]</td><td>20</td><td>28.6</td><td>Parliamentary committees</td></tr><tr><td>SIM [16]</td><td>100</td><td>91.6</td><td>Spliced meeting excerpts</td></tr></table>

![](images/6ad433cb8879372ba8e9a5585fd0f840f277d37e7cf73c892a8a8fcca29e7099.jpg)  
Fig. 2. CGS overview. Each cue pairs a topic-opening quote with its sentence index. Both stages use the same LLM with task-specific prompts. Structural guidance covers phases, boundary signals, and expected segment durations.

In this regard, we propose Cue-Grounded Segmentation (CGS)<sup>1</sup>, a zero-shot framework that grounds boundary prediction in topicopening cues. Such cues can indicate which parts of a discussion are presented as distinct topics. When these cues are sufficiently informative, CGS uses their sentence positions as boundaries, allowing section lengths to reflect the spacing between topic openings. When cue evidence is insufficient, CGS falls back to semantic segmentation guided by the document structure inferred during cue extraction. This design enables CGS to better adapt to the wide variation in ground-truth segment lengths across the six benchmarks (Fig. 1).

More specifically, CGS is implemented in two stages (Fig. 2). Stage 1 reads the full numbered transcript, quotes topic-opening phrases, and reports their sentence indices. On the direct route, those indices define the boundaries, so each cue-derived boundary is paired with the transcript phrase that motivated it. When Stage 1 requests fallback or judges the cues unreliable, Stage 2 re-reads the full transcript, guided by the expected topic count, structural phases, and boundary signals inferred in Stage 1. It identifies topic changes from the transcript’s semantic content, without explicit transition phrases.

Our main contributions and findings are:

• We introduce CGS, a training-free, zero-shot method that uses quoted topic openings to locate boundaries and invokes structure-guided semantic segmentation when those cues are insufficient.

• Across six benchmarks and six LLM backbones, CGS achieves the best average performance on all metrics, with low API cost on proprietary models.

• Ablations show the benefit of combining cue extraction with semantic fallback and supplying segment-count guidance. Automatic speech recognition (ASR) experiments show that CGS retains its performance advantage under speech recognition noise.

## 2. METHODOLOGY

CGS consists of two stages. First, it identifies explicit transition cues in the transcript and determines whether they provide sufficient evidence for topic boundaries. When such cues are insufficient, CGS falls back to semantic segmentation guided by the document structure inferred during cue extraction (Fig. 2). The two stages are implemented as separate calls to the same LLM. We use a single shared set of prompts across all datasets, without providing dataset names, task-specific tuning, or labeled examples.

## 2.1. Stage 1: Indexed Transition Evidence

Stage 1 processes the full transcript and returns a structured JSON output. The prompt asks the model to identify and quote each topicopening phrase together with its sentence index, forming the cue list C. In the same response, the model assigns an overall cue-reliability label—reliable, uncertain, or unreliable—and indicates whether semantic fallback is required. Stage 2 is invoked when fallback is explicitly requested or when the cue list is labeled unreliable; an uncertain label alone does not trigger fallback.

Stage 1 also predicts a segment-count range from the extracted cues. Given n cues, the lower and upper bounds are set to max(1, n − 1) and n + 1, respectively. When fallback is required, the model additionally produces structural guidance for Stage 2: phases summarizing the document’s major parts and their order, boundary signals describing likely indicators of topic transitions, and a duration range estimating plausible segment lengths in seconds from the transcript. When fallback is not required, these fields are omitted. Table 3 illustrates the Stage 1 output using an example from an AMI meeting.

The direct route completes segmentation after Stage 1. CGS discards out-of-range cue indices and returns $B _ { \mathrm { c u e } } ~ = ~ \{ 0 \}$ ∪ {i : $( i , q ) \in C \}$ over the remaining pairs.

## 2.2. Stage 2: Conditioned Semantic Fallback

Stage 2 reads the full numbered transcript together with the segmentcount range and structural guidance from Stage 1. The LLM uses this guidance to identify coherent sections and places boundaries where the discussion moves to a new topic. It can identify these transitions without an explicit phrase such as “Next, let’s discuss staffing.” Stage 2 predicts a new set of topic-start sentence indices from the transcript; it does not receive the Stage-1 cue list. The suggested segment-count range guides this prediction rather than imposing a strict constraint. To reduce the number of very short segments, CGS merges neighboring Stage 2 boundary predictions whose sentence indices differ by at most two.

Table 2. Performance on the six benchmark corpora. Bold and underline mark the best and second-best values per column. †Training-free, but uses self-supervised BERT pretraining. ‡Def-DTS uses only valid outputs completed within the output-token limit of backbones.
<table><tr><td></td><td></td><td colspan="7"> $P _ { k } \left( \downarrow \right)$ </td><td colspan="7">WindowDiff  $( \downarrow )$ </td><td colspan="7">Boundary  $F _ { 1 }$  (↑)</td></tr><tr><td></td><td>Method</td><td>YT</td><td>IC</td><td>MB</td><td>SIM</td><td>AMI</td><td>QM</td><td> $\operatorname { A v g } .$ </td><td>YT</td><td>IC</td><td>MB</td><td>SIM</td><td>AMI</td><td>QM</td><td> $\operatorname { A v g } .$ </td><td>YT</td><td>IC</td><td>MB</td><td>SIM</td><td>AMI</td><td>QM</td><td>Avg.</td></tr><tr><td rowspan="2">Classical</td><td>TEXTTILING [1]</td><td>.513</td><td>.716</td><td>.648</td><td>.604</td><td>.611</td><td>.630</td><td>.620</td><td>.653</td><td>.996</td><td>.994</td><td>1.00</td><td>.816</td><td>.999</td><td>.910</td><td>.433</td><td>.036</td><td>.086</td><td></td><td>.057</td><td>.121 .129</td><td>.144</td></tr><tr><td>BERT-TT [20]†</td><td>.402</td><td>.473</td><td>.420</td><td>.378</td><td>.420</td><td>.407</td><td>.417</td><td>.409</td><td>.514</td><td>.464</td><td>.434</td><td>.442</td><td>.432</td><td>.449</td><td>.189</td><td>.055</td><td>.133</td><td>.197</td><td>.102</td><td>.188</td><td>.144</td></tr><tr><td rowspan="5">LLM-based</td><td>LUMBERCHUNKER [12]</td><td>.378</td><td>.714</td><td>.641</td><td>.604</td><td>.540</td><td>.623</td><td>.583</td><td>.473</td><td>.983</td><td>.952</td><td>.994</td><td>.732</td><td>.972</td><td>.851</td><td>.539</td><td>.121</td><td>.182</td><td>.119</td><td>.291</td><td>.233</td><td>.247</td></tr><tr><td>Mackenzie et al. [10]</td><td>.349</td><td>.533</td><td>.460</td><td>.540</td><td>.459</td><td>.511</td><td>.476</td><td>.421</td><td>.722</td><td>.692</td><td>.844</td><td>.599</td><td>.712</td><td>.665</td><td>.519</td><td>.182</td><td>.293</td><td>.141</td><td>.302</td><td>.226</td><td>.277</td></tr><tr><td>TOC PROMPT [11]</td><td>.414</td><td>.516</td><td>.472</td><td>.574</td><td>.511</td><td>.495</td><td>.497</td><td>.483</td><td>.632</td><td>.648</td><td>.840</td><td>.606</td><td>.693</td><td>.650</td><td>.453</td><td>.098</td><td>.181</td><td>.090</td><td>.158</td><td>.322</td><td>.217</td></tr><tr><td>DEF-DTS‡ [13]</td><td>.338</td><td>.400</td><td>.287</td><td>.317</td><td>.304</td><td>.406</td><td>.340</td><td>.368</td><td>.490</td><td>.402</td><td>.430</td><td>.387</td><td>.536</td><td>.433</td><td>.325</td><td>.170</td><td>.272</td><td>.279</td><td>.358</td><td>.251</td><td>.289</td></tr><tr><td>SEGMENTLLM [9]</td><td>.271</td><td>.333</td><td>.206</td><td>.306</td><td>.309</td><td>.416</td><td>.307</td><td>.324</td><td>.402</td><td>.275</td><td>.370</td><td>.367</td><td>.535</td><td>.379</td><td>.590</td><td>.200</td><td>.618</td><td>.350</td><td>.330</td><td>.382</td><td>.412</td></tr><tr><td>Ours</td><td>CGS</td><td>.245</td><td>.212</td><td>.069</td><td>.206</td><td>.216</td><td>.301</td><td>.208</td><td>.274</td><td>.272</td><td>.155</td><td>.241</td><td>.269</td><td>.333</td><td>.257</td><td>.604</td><td>.395</td><td>.751</td><td>.471</td><td>.438</td><td>.391</td><td>.508</td></tr></table>

Table 3. Stage-1 guidance supplied to Stage 2: an example from an AMI meeting. Phase and signal entries are excerpts; both ranges are shown in full.
<table><tr><td>Guidance</td><td>Saved example</td></tr><tr><td>Segment count</td><td>3–5 segments</td></tr><tr><td>Structural phases</td><td>Prototype Discussion: Follows the introduction; covers design ergonomics,</td></tr><tr><td>Boundary signals</td><td>materials, and cost-saving measures. Introduction of new documents or</td></tr><tr><td>Segment duration</td><td>evaluation criteria 120–300 seconds</td></tr></table>

## 3. EXPERIMENTAL SETUP

Corpora. Table 1 summarizes the six benchmarks, covering 1,800 documents. We use their supplied transcripts, treating each sentence or utterance as one indexed sentence. SIM tests topic changes introduced by splicing unrelated meeting excerpts. YTSeg, AMI, and ICSI provide audio recordings and timestamped topic boundaries. For each evaluated configuration, prompts and settings remain fixed across corpora. No model is fine-tuned for segmentation or given labeled examples from the target corpora.

Metrics. $P _ { k }$ metric [21] measures how often sentences i and i + k share a predicted segment but not a ground-truth segment, or vice versa. WindowDiff [22] measures how often predicted and ground-truth boundary counts differ within a window. Both use $\bar { k } = \operatorname* { m a x } ( 1 , \lfloor \bar { L } / 2 \rfloor )$ , where L<sup>¯</sup> is mean ground-truth segment length per document. Boundary $F _ { 1 }$ [14] is the harmonic mean of precision and recall. We count a predicted boundary as correct if it is within two sentences of a ground-truth boundary. Each boundary can be matched at most once. Macro scores average documents within corpora, then corpora equally.

Baselines. TEXTTILING [1] detects changes in lexical cohesion, while BERT-TT [20] uses contextual embeddings to detect drops in similarity between adjacent sentence blocks. The LLM baselines include a TOC PROMPT [11], which generates section headings and start indices; the recursive segmentation method of

Table 4. Performance by backbone, averaged equally over the six corpora. Table 2 averages LLM results over these backbones. Bold marks the better value.
<table><tr><td></td><td colspan="3">SEGMENTLLM</td><td colspan="3">CGS (ours)</td></tr><tr><td>Backbone</td><td> $P _ { k }$  (↓)</td><td>WD (↓)</td><td>F1 (↑)</td><td>Pk (↓)</td><td>WD (↓)</td><td>F1 (↑)</td></tr><tr><td>Gemini-3.1-FL</td><td>.234</td><td>.285</td><td>.539</td><td>.177</td><td>.207</td><td>.557</td></tr><tr><td>GPT-5.6-Terra</td><td>.328</td><td>.415</td><td>.525</td><td>.181</td><td>.218</td><td>.550</td></tr><tr><td>Qwen3.6-27B</td><td>.241</td><td>.278</td><td>.495</td><td>.175</td><td>.208</td><td>.543</td></tr><tr><td>Gemma-4-26B</td><td>.411</td><td>.605</td><td>.363</td><td>.221</td><td>.296</td><td>.486</td></tr><tr><td>Qwen3.5-9B</td><td>.309</td><td>.334</td><td>.299</td><td>.230</td><td>.279</td><td>.464</td></tr><tr><td>Qwen3.5-4B</td><td>.319</td><td>.356</td><td>.248</td><td>.264</td><td>.337</td><td>.452</td></tr></table>

Mackenzie et al. [10], which predicts boundary indices and recursively divides long segments using fixed illustrative examples; and LUMBERCHUNKER [12], which iteratively identifies topic shifts, adapted here to sentence units. SEGMENTLLM [9] uses a zeroshot prompt to group consecutive sentences by topic, listing every sentence index in each group. Segment boundaries are recovered from these groups after resolving gaps and overlaps. DEF-DTS [13] summarizes the preceding and following context of each utterance, classifies its intent, and predicts whether it starts a new topic. Generating these intermediate outputs for every utterance leads to substantial output-token usage on long transcripts. We apply Def-DTS to the full transcript, including speaker labels when available.

Backbones. We evaluate six LLM backbones: Gemini-3.1- Flash-Lite [23], GPT-5.6-Terra [24], Qwen3.6-27B [25], Gemma-4- 26B-A4B [26], Qwen3.5-9B [27], and Qwen3.5-4B [27]. We use temperature T=0.3 for Flash-Lite and the open-weight models. Terra uses the API’s default temperature because non-default values were not supported.

## 4. RESULTS AND ANALYSIS

Table 2 shows that, averaged over six backbones, CGS outperforms all six published baselines on every corpus under all three metrics. Figure 1 summarizes section lengths from the same predictions. CGS also outperforms SEGMENTLLM, the strongest baseline in Table 2, on all three macro metrics for each backbone (Table 4). Across the six backbones, CGS completes segmentation after Stage 1 for 80–93% of documents.

Cost and token use. Table 5 shows that CGS combines the lowest segmentation error with the second-lowest API cost. CGS generates about one-quarter as many output tokens as SEG-MENTLLM, averaged across six backbones. On the two proprietary backbones, output tokens are priced six times higher than input tokens, allowing the output savings to offset the additional input cost.

Table 5. P from Table 2 and recorded token use (thousands per document). Per-document means are averaged across the six corpora and six backbones. Costs are calculated from recorded token usage at standard API rates and averaged over the two proprietary backbones (USD/1,000 documents). Bold and underline mark the best and second-best values.
<table><tr><td>Method</td><td> $P _ { k } \downarrow$ </td><td>Input (k)↓</td><td>Output (k)↓</td><td>Cost↓</td></tr><tr><td>LUMBERCHUNKER</td><td>.583</td><td>32.36</td><td>0.825</td><td>35.59</td></tr><tr><td>Mackenzie et al.</td><td>.476</td><td>46.25</td><td>0.233</td><td>53.03</td></tr><tr><td>TOC PROMPT</td><td>.497</td><td>14.94</td><td>0.306</td><td>17.17</td></tr><tr><td>DEF-DTS</td><td>.340</td><td>16.37</td><td>42.221</td><td>317.48</td></tr><tr><td>SEGMENTLLM</td><td>.307</td><td>14.03</td><td>2.428</td><td>26.13</td></tr><tr><td>CGS</td><td>.208</td><td>19.63</td><td>0.603</td><td>21.87</td></tr></table>

CGS ablations. Table 6 reports ablations on Flash-Lite and Qwen3.6-27B, which achieve the strongest CGS performance among the evaluated proprietary and open-weight backbones, respectively.

Stage 1 only (FORCED CUES) places boundaries at the extracted topic openings and skips semantic fallback. Stage 2 only skips Stage 1 and segments every transcript directly. These comparisons test whether either stage can replace the complete pipeline. Cue indices only keeps the two-stage procedure but asks Stage 1 to report topic-opening sentence indices without quoting the corresponding phrases. This tests whether generating the quotes helps boundary prediction. CGS performs best among the variants in Table 6 on both backbones.

We also examine how much Stage-1 guidance improves Stage-2 segmentation. For the same documents routed to Stage 2, Countonly fallback supplies the transcript and suggested segment-count range, but removes phase descriptions, boundary signals, and duration estimates. Transcript-only fallback also removes the count range. Retaining the count range provides most of the improvement over transcript-only fallback. Structural guidance reduces $P _ { k }$ further, with a larger effect on Qwen-27B.

Table 6. CGS ablations $( P _ { k }$ ↓, averaged equally over six corpora; T=.3). Bold marks the lowest displayed value.
<table><tr><td>Configuration</td><td>Flash-Lite</td><td>Qwen-27B</td></tr><tr><td>CGS</td><td>.177</td><td>.175</td></tr><tr><td>Stage 1 only (FORCED CUES)</td><td>.197</td><td>.187</td></tr><tr><td>Stage 2 only</td><td>.306</td><td>.285</td></tr><tr><td>Cue indices only</td><td>.197</td><td>.180</td></tr><tr><td>Count-only fallback</td><td>.192</td><td>.183</td></tr><tr><td>Transcript-only fallback</td><td>.249</td><td>.218</td></tr></table>

Boundary errors. Table 7 summarizes the correct, missed, and incorrect boundary counts used to calculate Boundary F for each document. CGS detects about as many ground-truth boundaries as SEGMENTLLM and makes the fewest incorrect boundary predictions overall. Some baselines detect more ground-truth boundaries but also make many more incorrect predictions.

Table 7. Counts are normalized to 100 ground-truth boundaries. Incorrect predictions can exceed 100 when a method predicts too many boundaries. Correct counts predictions that match a groundtruth boundary within two sentences. Each boundary is matched at most once. Missed reports how many ground-truth boundaries are not detected. Incorrect counts wrong boundary predictions. Values average six corpora and, for LLMs, six backbones.
<table><tr><td>Method</td><td>Correct↑</td><td>Missed↓</td><td>Incorrect↓</td></tr><tr><td>TEXTTILING</td><td>61.6</td><td>38.4</td><td>1361.8</td></tr><tr><td>BERT-TT</td><td>13.3</td><td>86.7</td><td>71.6</td></tr><tr><td>LUMBERCHUNKER</td><td>79.3</td><td>20.7</td><td>750.6</td></tr><tr><td>Mackenzie et al.</td><td>62.4</td><td>37.6</td><td>476.2</td></tr><tr><td>TOC PROMPT</td><td>41.4</td><td>58.6</td><td>262.4</td></tr><tr><td>DEF-DTS</td><td>46.4</td><td>53.6</td><td>351.0</td></tr><tr><td>SEGMENTLLM</td><td>54.6</td><td>45.4</td><td>148.3</td></tr><tr><td>CGS</td><td>55.5</td><td>44.5</td><td>58.2</td></tr></table>

CGS retains its advantage on ASR transcripts. Speech recognition errors and missing punctuation can make topic boundaries harder to locate. We use YTSeg’s released Whisper-Large transcripts and generate AMI and ICSI transcripts with Whisper large-v3 [28]. For AMI and ICSI, we compare the audio timestamp of each ground-truth boundary with Whisper’s chunk start timestamps. A chunk starting at the boundary timestamp begins the new segment. If there is no exact timestamp match, the new segment begins with the first chunk after the boundary timestamp. Averaged across six backbones, CGS outperforms SEGMENTLLM on all metrics for all three corpora (Table 8).

Table 8. ASR segmentation performance, averaged over six backbones. Sentence divisions differ between the original and ASR transcripts.
<table><tr><td>Corpus</td><td>Method</td><td> $P _ { k } \downarrow$ </td><td>WD↓</td><td> $F _ { 1 } \uparrow$ </td></tr><tr><td rowspan="2">AMI</td><td>SegmentLLM</td><td>.278</td><td>.359</td><td>.420</td></tr><tr><td>CGS</td><td>.202</td><td>.263</td><td>.512</td></tr><tr><td rowspan="2">ICSI</td><td>SegmentLLM</td><td>.310</td><td>.391</td><td>.355</td></tr><tr><td>CGS</td><td>.244</td><td>.307</td><td>.456</td></tr><tr><td rowspan="2">YTSeg</td><td>SegmentLLM</td><td>.277</td><td>.335</td><td>.577</td></tr><tr><td>CGS</td><td>.253</td><td>.286</td><td>.592</td></tr></table>

## 5. CONCLUSION

CGS grounds topic boundaries in quoted transition cues and invokes structure-guided semantic segmentation when these cues are insufficient. Averaged across the six corpora, it outperforms the evaluated baselines on all metrics with each of the six backbones. Ablations show the benefit of combining cue extraction with selective semantic fallback and identify segment-count guidance as an important contributor to fallback performance. CGS also retains its performance advantage on noisy ASR transcripts, supporting robustness to speech recognition errors. By combining accurate segmentation with low API cost, CGS offers a practical approach to segmenting spoken documents without task-specific training.

## 6. ACKNOWLEDGMENTS

This work was supported by (1) the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2026-25477522), (2) Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) [NO.RS-2021-II211343, Artificial Intelligence Graduate School Program (Seoul National University)], and (3) the Starting growth Technological R&D Program (RS-2024- 00506994) funded by the Ministry of SMEs and Startups (MSS, Korea).

## 7. COMPLIANCE WITH ETHICAL STANDARDS

This study uses publicly released research corpora under their respective terms and involves no new human-subject data collection.

## 8. REFERENCES

[1] M. A. Hearst, “TextTiling: Segmenting text into multiparagraph subtopic passages,” Computational Linguistics, vol. 23, no. 1, pp. 33–64, 1997.

[2] J. Eisenstein and R. Barzilay, “Bayesian unsupervised topic segmentation,” in Proceedings of EMNLP, 2008, pp. 334–343.

[3] E. Shriberg, A. Stolcke, D. Hakkani-Tur, and G. T¨ ur,¨ “Prosody-based automatic segmentation of speech into sentences and topics,” Speech Communication, vol. 32, no. 1–2, pp. 127–154, 2000.

[4] M. Galley, K. R. McKeown, E. Fosler-Lussier, and H. Jing, “Discourse segmentation of multi-party conversation,” in Proceedings ofACL, 2003, pp. 562–569.

[5] M. Lukasik, B. Dadachev, K. Papineni, and G. Simoes, “Text˜ segmentation by cross segment attention,” in Proceedings of EMNLP, 2020, pp. 4707–4716.

[6] K. Lo, Y. Jin, W. Tan, M. Liu, L. Du, and W. Buntine, “Transformer over pre-trained transformer for neural text segmentation with enhanced topic coherence,” in Findings of EMNLP, 2021, pp. 3334–3340.

[7] K. Wang, X. Zhao, Y. Li, and W. Peng, “M3Seg: A maximumminimum mutual information paradigm for unsupervised topic segmentation in ASR transcripts,” in Proceedings of EMNLP, 2023, pp. 7928–7934.

[8] D. C. Gklezakos, T. Misiak, and D. Bishop, “TreeSeg: Hierarchical topic segmentation of large transcripts,” arXiv preprint arXiv:2407.12028, 2024.

[9] Y. Fan, F. Jiang, P. Li, and H. Li, “Uncovering the potential of ChatGPT for discourse analysis in dialogue: An empirical study,” in Proceedings of LREC-COLING, 2024, pp. 16998– 17010.

[10] P. Mackenzie, M. Shah, and P. Frenett, “Topic segmentation using generative language models,” arXiv preprint arXiv:2601.03276, 2025.

[11] S. Freisinger, P. Seeberger, T. Ranzenberger, T. Bocklet, and K. Riedhammer, “Towards multi-level transcript segmentation: LoRA fine-tuning for table-of-contents generation,” in Proceedings ofInterspeech, 2025, pp. 276–280.

[12] A. V. Duarte, J. D. S. Marques, M. Grac¸a, M. Freire, L. Li, and A. L. Oliveira, “LumberChunker: Long-form narrative document segmentation,” in Findings of EMNLP, 2024, pp. 6473–6486.

[13] S. Lee, Y. Yoo, M. Jung, and M. Song, “Def-DTS: Deductive reasoning for open-domain dialogue topic segmentation,” in Findings of ACL, 2025, pp. 20736–20753.

[14] F. Retkowski and A. Waibel, “From text segmentation to smart chaptering: A novel benchmark for structuring video transcriptions,” in Proceedings ofEACL, 2024, pp. 406–419.

[15] A. Janin, D. Baron, J. Edwards, D. Ellis, D. Gelbart, N. Morgan, B. Peskin, T. Pfau, E. Shriberg, A. Stolcke, and C. Wooters, “The ICSI meeting corpus,” in Proceedings of ICASSP, 2003, vol. 1, pp. 364–367.

[16] Y. Fan, J. Pool, S. Filipi, and R. Cutler, “Topic-conversation relevance (TCR) dataset and benchmarks,” in Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, 2024, vol. 37, pp. 140159–140174.

[17] J. Carletta, “Unleashing the killer corpus: experiences in creating the multi-everything AMI meeting corpus,” Language Resources and Evaluation, vol. 41, no. 2, pp. 181–190, 2007.

[18] Y. Hu, T. Ganter, H. Deilamsalehy, F. Dernoncourt, H. Foroosh, and F. Liu, “MeetingBank: A benchmark dataset for meeting summarization,” in Proceedings ofACL, 2023, pp. 16409– 16423.

[19] M. Zhong, D. Yin, T. Yu, A. Zaidi, M. Mutuma, R. Jha, A. H. Awadallah, A. Celikyilmaz, Y. Liu, X. Qiu, and D. Radev, “QMSum: A new benchmark for query-based multi-domain meeting summarization,” in Proceedings of NAACL-HLT, 2021, pp. 5905–5921.

[20] A. Solbiati, K. Heffernan, G. Damaskinos, S. Poddar, S. Modi, and J. Cali, “Unsupervised topic segmentation of meetings with BERT embeddings,” arXiv preprint arXiv:2106.12978, 2021.

[21] D. Beeferman, A. Berger, and J. Lafferty, “Statistical models for text segmentation,” Machine Learning, vol. 34, no. 1–3, pp. 177–210, 1999.

[22] L. Pevzner and M. A. Hearst, “A critique and improvement of an evaluation metric for text segmentation,” Computational Linguistics, vol. 28, no. 1, pp. 19–36, 2002.

[23] Google DeepMind, “Gemini 3.1 Flash-Lite model card,” https: //deepmind.google/models/model-cards/gemini-3-1-flash-lit e/, Mar. 2026.

[24] OpenAI, “GPT-5.6 Terra,” https://developers.openai.com/api/ docs/models/gpt-5.6-terra, 2026, API model documentation, accessed September 17, 2026.

[25] Qwen Team, “Qwen3.6-27B: Flagship-level coding in a 27B dense model,” https://qwen.ai/blog?id=qwen3.6-27b, Apr. 2026.

[26] Gemma Team, “Gemma 4 technical report,” arXiv preprint arXiv:2607.02770, 2026.

[27] Qwen Team, “Qwen3.5: Towards native multimodal agents,” https://qwen.ai/blog?id=qwen3.5, Feb. 2026.

[28] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in Proceedings of ICML, 2023, pp. 28492– 28518.