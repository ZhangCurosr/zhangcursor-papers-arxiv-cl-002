# VOSSA: Voiceprint Optimization for Streaming Speech Architectures

Mu-Ruei Tseng, Waris Quamer, Ghady Nasrallah, Ricardo Gutierrez-Osuna

Department of Computer Science & Engineering, Texas A&M University, College Station, US {mtseng,quamer.waris,ghadynasrallah,rgutier}@tamu.edu

## Abstract

Real-time voice conversion (VC) systems commonly rely on pretrained speaker embeddings from automatic speaker verification (ASV) models. While effective for speaker discrimination, these embeddings are trained to remain stable across phonetic and prosodic variations within-speaker, which may conflict with frame-level acoustic generation in streaming constraints. To address this issue, we propose VOSSA (Voiceprint Optimization for Streaming Speech Architectures), a speaker representation framework that extracts speaker information from intermediate content encoder layers and aggregates using attentive statistics pooling. The embedding is trained jointly with VC objectives, removing the need for a separate speaker encoder. Across six datasets, VOSSA improves F0 dynamics and vowel-discriminative acoustic cues while maintaining comparable NISQA-MOS, WER, and speaker similarity. Perceptual tests further indicate improvements in naturalness, speaker similarity, intelligibility, and vibrancy.

Index Terms: voice conversion, speaker embedding, vowel formants, acoustic analysis

## 1. Introduction

Real-time voice conversion (VC) aims to transform speech from a source speaker to match the identity of a known target speaker while generating natural and intelligible speech under strict streaming and latency constraints. Recent streaming architectures achieve latencies as low as 80 ms by combining lightweight content encoders with causal decoders while explicitly disentangling linguistic and speaker representations [1, 2, 3]. As streaming VC moves toward practical applications, e.g., speech anonymization, VoIP, the design of speaker representations becomes increasingly critical.

VC systems typically decompose a speech signal into linguistic content and a speaker-identity representation, then resynthesize the content conditioned on a desired target identity. Accordingly, a central design choice is how speaker identity is encoded and injected into the generative model. Existing methods generally follow two approaches. The conventional approach is to condition the decoder on pretrained speaker verification embeddings, e.g., x-vectors [4], ECAPA-TDNN [5], which remain frozen during VC training. More recent approaches learn speaker or style encoders jointly within the generative framework. For instance, GenVC [6] extracts a fixedlength style embedding through a Perceiver encoder to guide autoregressive generation. Conan [7] and RT-VC [8] adopt streaming architectures that pair content extraction with adaptive style or speaker encoders for low-latency zero-shot VC.

Despite their effectiveness, both paradigms present limitations in streaming VC. First, ASV-based speaker embeddings are explicitly trained to suppress intra-speaker variability (phonetic, prosodic), but generative speech modeling requires speaker conditioning to be consistent with frame-level phonetic realizations, such as formant structure. This introduces a representational mismatch between (intra-speaker) invariant identity modeling and phoneme-conditioned acoustic generation. On the other hand, approaches that jointly train speaker encoders with the VC model in a self-supervised fashion can reduce this mismatch but they still retain the additional encoder branch for speaker representation learning, increasing model complexity and memory footprint which could be critical in resourceconstrained settings. A potential solution to these issues comes from the analysis of large self-supervised speech models such as WavLM [9, 10], which suggests that intermediate encoder layers retain substantial speaker and acoustic information, while deeper layers become increasingly content-focused. This finding suggests a third, unified, alternative: rather than introducing an additional speaker encoder, speaker identity can be extracted directly from intermediate representations of the content encoder.

To explore this possibility, we present VOSSA (Voiceprint Optimization for Streaming Speech Architectures), an integrated model for speaker representation learning that is suitable for real-time streaming VC. VOSSA derives speaker embeddings from intermediate layers of the content encoder and optimizes them jointly within a generative framework via reconstruction and cross-speaker conversion objectives. By leveraging representations already computed for content modeling, our approach eliminates the need for a separate speaker encoder while preserving strict causality and low latency.

We evaluate VOSSA against four streaming VC baselines: three systems that rely on frozen ASV-based speaker embeddings [1, 2, 3], and a jointly trained speaker representation model (GenVC [6]) that learns speaker embeddings within the generative framework. Our formulation achieves performance comparable to state-of-the-art streaming models in terms of NISQA-MOS [11], word error rate (WER), and harmonicity, while improving target-speaker similarity under normalized embedding evaluation. Moreover, to examine the representational characteristics of our model, we perform additional diagnostics based on large-scale spectral statistics, pitch and harmonics analysis, and formant distributions. Across six datasets, the learned representations in VOSSA exhibit richer pitch variability and better preservation of speaker- and voweldependent formant distributions, indicating increased phonemeconditioned acoustic flexibility that is not captured by standard objective scores.

Audio samples are available at our demo page.

![](images/ff8ef2afe0e191db893c90d24bb92a3bb50563af9f915af0b79e5b553cfb7390.jpg)  
Figure 1: Training workflow for the TVTSyn backbone. (a) content encoder trained against HuBERT k-means pseudo-labels, and (b) decoder conditioned on speaker embedding trained with self-supervision and discriminator objectives. (c) Overview ofthe training protocol in VOSSA. The self-reconstruction path (bottom, purple) uses segmentsfrom the same LibriTTS speaker to providefully supervised training, while the non-parallel VC path (top, orange) uses VoxCeleb targets to condition conversion. (d) Speaker embedding extraction from the frozen content encoder. We collect features from the last CNN layer and every other layer of the MHSA stack, concatenate them to form $\mathbf { H } \in \mathbb { R } ^ { T \times L d }$ , and apply attentive statistics pooling and an MLP to obtain a global speaker embedding.

## 2. Methods

Our proposed model (VOSSA) uses TVTSyn as a backbone [3]. TVTSyn is a streaming speech synthesizer designed for lowlatency VC and anonymization –see Fig. 1a-b. Its content encoder uses a causal CNN followed by transformer layers with a 2 s look-back window and 80 ms look-ahead, producing 50 Hz frame embeddings quantized through a factorized VQ bottleneck and trained with cross-entropy against HuBERT k-means pseudo-labels $( N { = } 2 0 0 )$ . The speaker embeddings generated through external speaker encoders are expanded into a global timbre memory that serves as key and value pairs for input content features to attend to, generating a time-varying speaker embedding representation, which is then passed to the decoder along with the content features. The decoder mirrors this architecture of the content encoder and reconstructs waveforms via causal transposed convolutions. TVTSyn generates timevarying timbre (TVT) embeddings based on a global speaker embedding conditioned on content tokens. TVTSyn achieves latencies as low as 80 ms on GPUs. For full architectural details, please refer to [3]. VOSSA adopts these components and reformulates how speaker identity is extracted during training as detailed below.

## 2.1. Joint speaker representation learning

VOSSA replaces the external speaker embeddings used in TVTSyn (concatenated x-vectors and ECAPA-TDNN) with a speaker representation derived from the same encoder backbone used for content extraction. Let $E _ { c }$ denote the frozen streaming content encoder in TVTSyn. Given waveform x, we extract content $c = E _ { c } ( x )$ and multi-layer speaker features from the last CNN layer and every other layer of the 8-layer MHSA stack (see Figure 1d). At each time step t, we stack the selected layer features as $\mathbf { H } _ { t } ~ \in ~ \mathbb { R } ^ { L d }$ , where L is the number of selected layers and d the feature dimension per layer. The full sequence is $\mathbf { H } \ = \ \{ \mathbf { H } _ { t } \} _ { t = 1 } ^ { T } \ \in \ \mathbb { R } ^ { T \times L d }$ We aggregate the resulting feature sequence H via attentive statistics pooling (ASP) [12]. Specifically, we compute attention weights as $\alpha _ { t } = \mathrm { s o f t m a x } \big ( \bar { f } ( \mathbf { H } _ { t } ) \big )$ , where $f ( \cdot )$ denotes a lightweight fully-connected (FC) two-layer network. Finally, we compute the attention-weighted mean and variance:

$$
\mu = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } { \alpha _ { t } \mathbf { H } _ { t } } , \qquad \sigma ^ { 2 } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } { \alpha _ { t } ( \mathbf { H } _ { t } - \mu ) ^ { 2 } } .\tag{1}
$$

and compute the global speaker embedding as $\begin{array} { r l } { g ( x ) } & { { } = } \end{array}$ $\phi ( [ \mu ; \sigma ] )$ , where $\phi ( \cdot )$ is a two-layer projection network. This design encourages the embedding to capture utterance-level speaker characteristics while allowing frame-level variability to be weighted adaptively.

## 2.2. Dual-path speaker-aligned training

The training protocol for VOSSA is illustrated in Fig. 1c. Let us denote an utterance from speaker $S _ { n }$ with content $C _ { i }$ as $x _ { n , i } ~ = ~ ( S _ { n } , C _ { i } )$ , the corresponding speaker embedding as $g ( x _ { n , i } ) ~ = ~ g _ { n , i } ,$ and the reconstructed utterance as $\begin{array} { r l } { \hat { x } _ { n , i } } & { { } = } \end{array}$ $( S _ { n } , C _ { i } ) ^ { \prime }$ . During training, we randomly select two utterances from each of two random speakers A and B: $x _ { A , 1 } , x _ { A , 2 }$ and $x _ { B , 1 } , x _ { B , 2 }$ The training protocol proceeds in two parallel paths: (1) a self-reconstruction path on the lower branch that reconstructs utterance $x _ { A , 1 }$ at the output, i.e., $( S _ { A } , C _ { 1 } ) ^ { \prime }$ , and (2) a voice-conversion path that performs voice conversion to speaker S<sub>B</sub> using the content from $x _ { A , 1 } , \mathrm { i . e . , } ( S _ { B } , C _ { 1 } ) ^ { \prime }$

We extract content features $c = E _ { c } ( x )$ and synthesize both self-reconstruction and voice conversion utterances as $\hat { x } _ { A } ~ =$ $D ( c , { \bar { g } } _ { A } )$ and $\hat { x } _ { B } = D ( c , \bar { g } _ { B } )$ where $D ( )$ denotes the decoder of TVTSyn , and ${ \bar { g } } _ { A } , { \bar { g } } _ { B }$ are the average speaker embedding for the two speakers (to promote stable identity targets):

$$
\bar { g } _ { A } = \frac { g _ { A , 1 } + g _ { A , 2 } } { 2 } , \quad \bar { g } _ { B } = \frac { g _ { B , 1 } + g _ { B , 2 } } { 2 } ,\tag{2}
$$

To enforce consistency within-speakers after re-synthesis, we add a loss term that minimizes the cosine distance $d _ { \cos } ( \mathbf { u } , \mathbf { v } ) =$ $\mathrm { ~ 1 ~ - ~ } \cos ( \mathbf { u } , \mathbf { v } )$ between the original speaker embeddings $\left( { \bar { g } } _ { A } , { \bar { g } } _ { B } \right)$ and those obtained from the reconstructed utterances:

$$
\mathcal { L } _ { \mathrm { s p k } } = d _ { \mathrm { c o s } } \big ( g ( \hat { x } _ { A } ) , \bar { g } _ { A } \big ) + d _ { \mathrm { c o s } } \big ( g ( \hat { x } _ { B } ) , \bar { g } _ { B } \big ) .\tag{3}
$$

As an auxiliary regularization term, we adopt a symmetric NT-Xent (InfoNCE) [13] objective to encourage embeddings from the same speaker to remain consistent across different utterances. Namely, for each mini-batch, we construct two views by concatenating the source and target speaker representations, yielding $\mathbf { z } _ { 1 } = \big [ \mathbf { g } _ { A } ^ { ( 1 ) } ; \mathbf { g } _ { B } ^ { ( 1 ) } \big ]$ and $\mathbf { z } _ { 2 } = [ \bar { \mathbf { g } } _ { A } ^ { ( 2 ) } ; \mathbf { g } _ { B } ^ { ( 2 ) } ]$ ], with corresponding speaker labels $\mathbf { y } = [ y _ { A } ; y _ { B } ]$ . We minimize a bidirectional InfoNCE loss b/w z<sub>1</sub> and z<sub>2</sub>. To prevent false negatives, samples sharing the same speaker label are excluded from the negative set, except for the matched positive pair.

Table 1: Evaluation of VOSSA against SOTA streaming VC baselines. $\mathrm { S i m } _ { s r c } ^ { s y n }$ can be interpreted as anonymization strength (lower is better), whereas $\mathrm { S i m } _ { t r g } ^ { s y n }$ represents VC strength (higher is better). Best results are bolded; second-best are underlined.
<table><tr><td>Models</td><td>Ground truth</td><td>slt24 [1]</td><td>DarkStream [2]</td><td> $\mathbf { G e n V C - s } \left[ 6 \right]$ </td><td></td><td>TVTSyn [3] | VOSSA (Ours)</td></tr><tr><td colspan="7">Standard VC metrics and harmonics</td></tr><tr><td>NISQA-MOS (↑)</td><td> $4 . 0 6 \pm 0 . 8 4$ </td><td> $3 . 4 6 \pm 0 . 8 0$ </td><td> $3 . 1 2 \pm 0 . 8 1$ </td><td> $3 . 0 4 \pm 0 . 8 3$ </td><td> ${ \bf 3 . 5 2 \pm 0 . 8 1 }$ </td><td> $3 . 4 8 \pm 0 . 9 1$ </td></tr><tr><td>WER (↓)</td><td> $0 . 0 7 \pm 0 . 1 6$ </td><td> $0 . 1 8 \pm 0 . 2 5$ </td><td> $0 . 2 5 \pm 0 . 2 7$ </td><td> $0 . 2 0 \pm 0 . 2 3$ </td><td> $\underline { { 0 . 1 7 \pm 0 . 2 3 } }$ </td><td> $\mathbf { 0 . 1 7 \pm 0 . 2 3 }$ </td></tr><tr><td>Simsrc )</td><td></td><td> $0 . 2 1 \pm 0 . 1 1$ </td><td> ${ \bf 0 . 0 6 \pm 0 . 1 1 }$ </td><td></td><td> $\overline { { 0 . 1 0 \pm 0 . 1 0 } }$ </td><td> $0 . 1 1 \pm 0 . 3 1$ </td></tr><tr><td> $\mathrm { { S i m } _ { t r g } ^ { s y n } \left( \uparrow \right) }$ </td><td></td><td> $0 . 4 6 \pm 0 . 1 1$ </td><td> $0 . 5 4 \pm 0 . 1 2$ </td><td></td><td> $\underline { { 0 . 5 9 \pm 0 . 1 1 } }$ </td><td> ${ \bf 0 . 8 6 \pm 0 . 1 9 }$ </td></tr><tr><td>HNR(↑)</td><td></td><td> $9 . 5 2 \pm 2 . 9 0$ </td><td> $7 . 9 3 \pm 3 . 6 9$ </td><td> $8 . 6 7 \pm 3 . 0 4$ </td><td> $\mathbf { 9 . 9 0 \pm 2 . 7 5 }$ </td><td> $9 . 7 2 \pm 2 . 8 0$ </td></tr><tr><td colspan="7">Pitch prediction</td></tr><tr><td>Pitch MAE (W) (↓)</td><td></td><td> $3 1 . 4 \pm 3 0 . 4$ </td><td> $3 3 . 3 \pm 2 2 . 5$ </td><td> $3 0 . 8 \pm 2 7 . 0$ </td><td> $2 9 . 0 \pm 2 0 . 5$ </td><td> ${ \bf 2 7 . 8 \pm 2 1 . 5 }$ </td></tr><tr><td>Voicing Mismatch (W) (↓) Pearson&#x27;s CC (↑)</td><td></td><td> $\underline { { 0 . 2 5 \pm 0 . 1 0 } }$ </td><td> $0 . 2 7 \pm 0 . 1 0$ </td><td> $0 . 4 3 \pm 0 . 1 0$ </td><td> $\mathbf { 0 . 2 3 \pm 0 . 0 9 }$ </td><td> $0 . 2 6 \pm 0 . 0 9$ </td></tr><tr><td></td><td></td><td> $0 . 4 0 \pm 0 . 3 3$ </td><td> $0 . 2 1 \pm 0 . 3 0$ </td><td> $0 . 3 3 \pm 0 . 3 9$ </td><td> $\underline { { 0 . 4 0 \pm 0 . 3 1 } }$ </td><td> ${ \bf 0 . 4 3 \pm 0 . 3 3 }$ </td></tr><tr><td colspan="7">Wasserstein distance for vowels (F1), synthesized vs. target</td></tr><tr><td>High Vowels (↓)</td><td></td><td> $4 2 . 3 \pm 3 1 . 2$ </td><td> $2 8 . 9 \pm 1 6 . 7$ </td><td> $3 2 . 7 \pm 2 0 . 8$ </td><td> $2 9 . 2 \pm 2 1 . 7$ </td><td> ${ \bf 2 4 . 6 \pm 1 3 . 8 }$ </td></tr><tr><td>Mid Vowels (↓)</td><td></td><td> $4 4 . 5 \pm 2 9 . 5$ </td><td> $\overline { { 3 1 . 9 \pm 1 5 . 5 } }$ </td><td> $4 9 . 2 \pm 2 5 . 3$ </td><td> $3 2 . 9 \pm 2 1 . 8$ </td><td> ${ \bf 2 9 . 7 \pm 1 6 . 8 }$ </td></tr><tr><td>Low Vowels (↓)</td><td></td><td> $5 7 . 1 \pm 3 2 . 0$ </td><td> $3 7 . 4 \pm 2 4 . 3$ </td><td> $6 0 . 7 \pm 2 8 . 2$ </td><td> $3 6 . 6 \pm 2 1 . 5$ </td><td> ${ \bf 3 1 . 0 \pm 2 4 . 5 }$ </td></tr><tr><td colspan="7">Mean opinion score (from perceptual listening tests)</td></tr><tr><td>MOS (↑)</td><td> $4 . 6 5 \pm 0 . 6 9$ </td><td> $3 . 6 2 \pm 1 . 0 2$ </td><td> $2 . 5 1 \pm 1 . 1 4$ </td><td> $3 . 1 6 \pm 1 . 1 8$ </td><td> $3 . 6 5 \pm 0 . 9 1$ </td><td> ${ \bf 3 . 7 9 \pm 1 . 0 1 }$ </td></tr></table>

The overall training objective is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { t o t a l } } = \lambda _ { \mathrm { m e l } } \mathcal { L } _ { \mathrm { m e l } } + \lambda _ { \mathrm { a d v } } \mathcal { L } _ { \mathrm { a d v } } + \lambda _ { \mathrm { f m } } \mathcal { L } _ { \mathrm { f m } } } \\ { + \lambda _ { \mathrm { s p k } } \mathcal { L } _ { \mathrm { s p k } } + \lambda _ { \mathrm { c o n } } \mathcal { L } _ { \mathrm { c o n } } } \end{array}\tag{4}
$$

where $\mathcal { L } _ { \mathrm { m e l } }$ is the L1 reconstruction loss, ${ \mathcal { L } } _ { \mathrm { a d v } }$ and ${ \mathcal L } _ { \mathrm { f m } }$ are the adversarial and feature-matching loss from the discriminator, respectively [3], and $\mathcal { L } _ { \mathrm { s p k } }$ and $\mathcal { L } _ { \mathrm { c o n } }$ enforce speaker consistency and supervised contrastive regularization. Gradients propagate through the speaker encoder, the TVT module, and the decoder, aligning the learned speaker space with streaming VC.

## 3. Experiments

## 3.1. Voice conversion protocol and diagnostics

To evaluate the effectiveness of the speaker representation learning in VOSSA, we employ two complementary protocols: voice conversion and self-reconstruction.<sup>2</sup> For the VC protocol, source utterances are sampled from the test and dev set of LibriTTS [14]. Target speakers are drawn from LibriTTS, Vox-Celeb [15, 16], EMIME [17], ARCTIC [18], L2-ARCTIC [19], and VCTK [20]. For each target speaker t, we sample two disjoint subsets of utterances: a reference set $S _ { t } ^ { r e f }$ and an evaluation subset $S _ { t } ^ { e v a l }$ . Reference utterances provide the target speaker identity for VC, while evaluation utterances are used to extract the speaker’s natural formant distributions. All systems are evaluated on identical source–target pairs. We evaluate VC performance using conventional objective metrics: speaker similarity, content preservation, and acoustic quality. We measure speaker similarity as the cosine between speaker embeddings of synthesized utterances and both (1) source utterances cos $( \mathbf { e } _ { s y n } , \mathbf { e } _ { s r c } )$ and (2) target utterances $\cos ( { \bf e } _ { s y n } , { \bf e } _ { t g t } )$ . Because the range of cosine similarities for each system is different (e.g., TVTSyn uses pretrained x-vector/ECAPA embeddings while VOSSA learns speaker representations internally), direct comparison of raw cosine similarity is difficult. Instead, we report normalized speaker similarity:

$$
\mathrm { S i m } _ { \mathrm { t r g } } ^ { \mathrm { s y n } } = \frac { \cos ( \mathbf e _ { s y n } , \mathbf e _ { t g t } ) - \cos ( \mathbf e _ { s r c } , \mathbf e _ { t g t } ) } { 1 - \cos ( \mathbf e _ { s r c } , \mathbf e _ { t g t } ) } ,\tag{5}
$$

Thus, $\mathrm { S i m } _ { \mathrm { t r g } } ^ { \mathrm { s y n } }$ represents how close the timbre of synthesized utterance is to that of target speaker relative to the timbre similarity between the source and target speakers. Following conventions, we also report word error rate (WER) from a pretrained ASR model [21], and acoustic quality using NISQA-MOS [11].

Additionally, we evaluate voice quality using the harmonics-to-noise ratio (HNR), which is indicative of clearer and more stable vocal production, using Praat’s cross-correlation harmonicity method [22, 23]. We compute frame-level harmonicity values with a 10 ms time step and report the mean HNR (in dB) for each utterance.

Results are summarized in Table 1. VOSSA achieves comparable NISQA-MOS to TVTSyn, which conditions on frozen speaker embeddings, and outperforms all other baselines, including GenVC-s, which jointly learns speaker representations through a separate encoder branch. VOSSA also matches TVT-Syn in WER, confirming preserved intelligibility. In addition, VOSSA achieves higher HNR than most baselines and remains close to TVTSyn, indicating stable harmonic structure and reduced noise in the synthesized speech. Notably,

![](images/f1489cf03575a9a0c0d3430651cd7bde06cc04846bd4d597dd7b97b826a22004.jpg)  
Figure 2: F1 distribution by vowel height for target speaker id00061 (Voxceleb). GT: original speech.

VOSSA achieves significantly higher normalized target similarity, indicating stronger speaker identity transfer relative to the source–target similarity. These findings indicate that the intermediate content representations provide sufficient speakerdiscriminative information, eliminating the need for an additional speaker encoder during inference.

## 3.2. Self-reconstruction protocol and diagnostics

To isolate acoustic modeling from cross-speaker effects, we also evaluate each system in a self-reconstruction setting, where the source and target speakers are identical. This protocol removes cross-speaker differences in pitch range and allows direct comparison between input and reconstructed speech. Selfreconstruction is conducted on LibriTTS.

Pitch prediction. We extract pitch contours using WORLD’s DIO algorithm with StoneMask refinement [24, 25]. We report the pitch mean absolute error (MAE) in Hz, the voiced/unvoiced mismatch (V/UV) ratio, and the Pearson correlation coefficient (PCC) between predicted and ground-truth pitch trajectories (on frames where both signals are voiced).

Shown in Table 1, VOSSA has the lowest pitch MAE and the highest PCC across systems, with V/UV mismatches close to those of the slt24 baseline (2nd best) and significantly lower than GenVC, which also uses an internal speaker representation. This indicates that VOSSA not only improves absolute pitch accuracy but also better preserves frame-level pitch dynamics.

Acoustic phonetics. As a final objective measure, we also evaluate the ability of each system to preserve formant distributions, which convey both phonetic acoustic cues and speakerdependent cues (i.e., formant frequencies depend on vocal tract length). For each utterance, we obtain phone-level alignments using the Montreal Forced Aligner [26], extract time-aligned phoneme boundaries, and select vowel segments. We restrict the analysis to monophthongs, as diphthongs involve timevarying formant movement. Due to space constraints, we only report analysis of the first formant (F1), which is associated with vowel height. To reduce boundary and coarticulation effects, we measure F1 at the temporal midpoint of each vowel. The resulting F1 values are aggregated across utterances to form empirical distributions, which are compared across systems and ground truth to assess differences in vowel height realization.

Table 1 reports the Wasserstein (WS) distance between resynthesized and original utterances for F1 distributions for high, mid, and low vowels on a speaker-by-speaker basis to eliminate differences in vocal tract length across speakers. VOSSA achieves the lowest WS distance for all high, mid, and low vowels as compared to all baselines, which indicates better preservation of vowel height realizations. This result is illustrated in Fig. 2 for VOSSA against its TVTSyn backbone.

Table 2: Perceptual listening test results.
<table><tr><td>Metric</td><td>TVTSyn</td><td>VOSSA (Ours)</td></tr><tr><td>Speaker similarity (↑, %)</td><td>46</td><td>54</td></tr><tr><td>Intelligibility (↑, %)</td><td>44</td><td>56</td></tr><tr><td>Vibrancy (↑, %)</td><td>48</td><td>52</td></tr></table>

## 3.3. Perceptual evaluations

To corroborate these findings, we conducted listening tests comparing VOSSA against baselines. To evaluate synthesis quality, we used a 5-point Mean Opinion Score (MOS) test (N=20 listeners; 15 utterances/model). Shown at the bottom of Table 1, subjective MOS ratings follow a similar trend to the NISQA-MOS results: VOSSA achieves the highest MOS, closely followed by TVTSyn and SLT24, and notably higher than GenVCs and DarkStream. As expected, ground-truth (original) speech remains substantially higher.

For the remaining tests we compared VOSSA against its TVTSyn backbone, the overall best baseline in Table 1. We measured speaker similarity using an ABX test. In each trial, listeners were presented with a target utterance X and samples from VOSSA and TVTSyn, A and B, and were asked to select the sample whose speaker identity was closest to that of X. We also measured intelligibility and vocal vibrancy using AB preference tests. For intelligibility we asked listeners to choose the one which was easier to understand whereas for the vibrancy test they selected utterances which sounded more vibrant and/or fuller. The assignment of TVTSyn and VOSSA to A and B was randomized across trials to mitigate order bias. Shown in Table 2, listeners prefer VOSSA across the three perceptual measures, indicating its superiority to the TVTSyn backbone.

## 3.4. Model size and streaming efficiency

VOSSA contains 19% fewer parameters than TVTSyn (132.4M vs. 162.8M), reflecting the removal of a separate external speaker encoder. Under streaming inference (batch size 1, 16 kHz, single NVIDIA RTX 5000 Ada GPU, 60 ms chunk size), both VOSSA and TVTSyn achieve a real-time factor (RTF) of approximately 0.25 and an end-to-end latency of about 73 ms. These results indicate that integrating speaker representation learning within the content encoder reduces overall model size without introducing additional streaming overhead.

## 4. Discussion

We proposed VOSSA, a streaming VC system that learns speaker representations directly from intermediate content features. By eliminating a standalone ASV-based speaker encoder, VOSSA reduces model size by 19% while maintaining comparable streaming efficiency. Further, VOSSA attains the highest normalized target-speaker similarity and is preferred in ABX speaker similarity tests, indicating that the learned representation guides identity realization more effectively during generation. Importantly, these improvements are not limited to cosine similarity at the speaker-embedding level. Vowel-level evaluation shows reduced Wasserstein distance to ground-truth F1 distributions across vowel heights, suggesting that VOSSA captures speaker-dependent phonetic-acoustics beyond overall speaker timbre. Finally, VOSSA achieves lower pitch error and V/UV mismatch while preserving high contour correlation, suggesting more accurate and dynamically consistent F0 modeling.

## 5. Acknowledgments

Supported by the Intelligence Advanced Research Projects Activity (IARPA) via Department of Interior/Interior Business Center (DOI/IBC) contract number 140D0424C0066. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright annotation thereon. Disclaimer: The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of IARPA, DOI/IBC, or the U.S. Government.

## 6. Use of Generative AI Disclosure

LLMs were used only minimally to improve writing clarity and presentation. All experimental design, implementation, data analysis, and interpretation of results were conducted by the authors.

## 7. References

[1] W. Quamer and R. Gutierrez-Osuna, “End-to-end streaming model for low-latency speech anonymization,” in 2024 IEEE Spoken Language Technology Workshop (SLT). IEEE, 2024, pp. 727–734.

[2] ——, “Darkstream: real-time speech anonymization with low latency,” in 2025 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), 2025, pp. 1–7.

[3] W. Quamer, M.-R. Tseng, G. Nasrallah, and R. Gutierrez-Osuna, “TVTSyn: Content-synchronous time-varying timbre for streaming voice conversion and anonymization,” in The Fourteenth International Conference on Learning Representations, 2026.

[4] D. Snyder, D. Garcia-Romero, G. Sell, D. Povey, and S. Khudanpur, “X-vectors: Robust dnn embeddings for speaker recognition,” in 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2018, pp. 5329–5333.

[5] B. Desplanques, J. Thienpondt, and K. Demuynck, “Ecapa-tdnn: Emphasized channel attention, propagation and aggregation in tdnn based speaker verification,” in Proc. INTERSPEECH, 2020, pp. 3830–3834.

[6] Z. Cai, H. L. Xinyuan, A. Garg, L. P. Garc´ıa-Perera, K. Duh, S. Khudanpur, M. Wiesner, and N. Andrews, “Genvc: Selfsupervised zero-shot voice conversion,” in 2025 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), 2025, pp. 1–8.

[7] Y. Zhang, B. Tian, and Z. Duan, “Conan: A chunkwise online network for zero-shot adaptive voice conversion,” in 2025 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), 2025, pp. 1–8.

[8] Y. Liu, C. Wang, H. Kim, R. Khan, and G. Anumanchipalli, “Rtvc: Real-time zero-shot voice conversion with speech articulatory coding,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), 2025, pp. 385–393.

[9] W.-N. Hsu, B. Bolte, Y.-H. H. Tsai, K. Lakhotia, R. Salakhutdinov, and A. Mohamed, “Hubert: Self-supervised speech representation learning by masked prediction of hidden units,” IEEE/ACM transactions on audio, speech, and language processing, vol. 29, pp. 3451–3460, 2021.

[10] S. Chen, C. Wang, Z. Chen, Y. Wu, S. Liu, Z. Chen, J. Li, N. Kanda, T. Yoshioka, X. Xiao, J. Wu, L. Zhou, S. Ren, Y. Qian, Y. Qian, J. Wu, M. Zeng, X. Yu, and F. Wei, “Wavlm: Large-scale self-supervised pre-training for full stack speech processing,” IEEE Journal of Selected Topics in Signal Processing, vol. 16, no. 6, p. 1505–1518, Oct. 2022.

[11] G. Mittag, B. Naderi, A. Chehadi, and S. Moller, “Nisqa: A deep ¨ cnn-self-attention model for multidimensional speech quality prediction with crowdsourced datasets,” in Proc. Interspeech 2021, 2021, pp. 2127–2131.

[12] K. Okabe, T. Koshinaka, and K. Shinoda, “Attentive Statistics Pooling for Deep Speaker Embedding,” in Interspeech 2018, 2018, pp. 2252–2256.

[13] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for contrastive learning of visual representations,” in International conference on machine learning, 2020, pp. 1597– 1607.

[14] H. Zen, V. Dang, R. Clark, Y. Zhang, R. J. Weiss, Y. Jia, Z. Chen, and Y. Wu, “LibriTTS: A Corpus Derived from LibriSpeech for Text-to-Speech,” in Interspeech 2019, 2019, pp. 1526–1530.

[15] A. Nagrani, J. S. Chung, and A. Zisserman, “VoxCeleb: A Large-Scale Speaker Identification Dataset,” in Interspeech 2017, 2017, pp. 2616–2620.

[16] J. S. Chung, A. Nagrani, and A. Zisserman, “Voxceleb2: Deep speaker recognition,” in Proceedings of the Annual Conference of the International Speech Communication Association, INTER-SPEECH, vol. 2018, 2018, pp. 1086–1090.

[17] M. Wester, “The emime bilingual database,” The University of Edinburgh, Tech. Rep., 2010.

[18] J. Kominek and A. W. Black, “The cmu arctic speech databases,” in Proc. SSW 2004, 2004, pp. 223–224.

[19] G. Zhao, S. Sonsaat, A. Silpachai, I. Lucic, E. Chukharev-Hudilainen, J. Levis, and R. Gutierrez-Osuna, “L2-arctic: A nonnative english speech corpus,” in Proc. Interspeech 2018, 2018, pp. 2783–2787.

[20] C. Veaux, J. Yamagishi, K. MacDonald et al., “Cstr vctk corpus: English multi-speaker corpus for cstr voice cloning toolkit,” University ofEdinburgh. The Centrefor Speech Technology Research (CSTR), vol. 6, p. 15, 2017.

[21] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in International conference on machine learning, 2023, pp. 28 492–28 518.

[22] P. Boersma, “Praat: doing phonetics by computer [computer program],” http://www. praat. org/, 2011.

[23] Y. Jadoul, B. Thompson, and B. de Boer, “Introducing Parselmouth: A Python interface to Praat,” Journal of Phonetics, vol. 71, pp. 1–15, 2018.

[24] M. Morise, F. Yokomori, and K. Ozawa, “World: a vocoder-based high-quality speech synthesis system for real-time applications,” IEICE TRANSACTIONS on Information and Systems, vol. 99, no. 7, pp. 1877–1884, 2016.

[25] M. Morise, H. Kawahara, and H. Katayose, “Fast and reliable f0 estimation method based on the period extraction of vocal fold vi bration of singing voice and speech,” in Audio Engineering Society Conference: 35th International Conference: Audiofor Games. Audio Engineering Society, 2009.

[26] M. McAuliffe, M. Socolof, S. Mihuc, M. Wagner, and M. Sonderegger, “Montreal forced aligner: Trainable text-speech alignment using kaldi.” in Interspeech, vol. 2017, 2017, pp. 498–502.