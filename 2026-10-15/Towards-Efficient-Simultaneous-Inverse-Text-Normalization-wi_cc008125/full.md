# Towards Efficient Simultaneous Inverse Text Normalization with Pretrained Text-to-Text Language Model and Read–Tag–Write Policy

Kiet Anh Hoang <sup>ID</sup> <sup>1,∗,∗∗</sup>, Khanh Le <sup>ID</sup> <sup>1,∗</sup>, Bao Nguyen <sup>ID</sup> <sup>1</sup>, Linh Pham <sup>ID</sup> <sup>1</sup>, Dung Vo <sup>ID</sup> <sup>1</sup>,

Thai Tran <sup>ID</sup> <sup>1</sup>, Mai Nguyen<sup>1</sup>, Tri Nguyen <sup>ID</sup> <sup>1</sup>, Vu Le<sup>1</sup>

<sup>1</sup> UNEY, Switzerland

kiet.ha@uney.com, khanh.ld@uney.com

## Abstract

Streaming Inverse Text Normalization (ITN) is vital for converting spoken-form outputs from streaming Automatic Speech Recognition into formatted written text. Existing streaming ITN methods rely on hybrid systems combining neural tagging with expert-crafted finite-state transducer rules, limiting scalability across domains and languages. While endto-end models offer superior scalability by learning directly from data, standard encoder-decoder architectures are inherently non-streaming due to global attention. In this paper, we propose an efficient streaming end-to-end ITN system, adapting a pretrained text-to-text model to leverage its robust linguistic knowledge. To enable streaming, we introduce architectural adaptations, a specialized training strategy, a Read–Tag–Write decoding policy, and inference optimizations. Experiments on Vietnamese datasets show accuracy comparable to nonstreaming baselines, outperforming hybrid methods while satisfying real-time latency requirements.

Index Terms: inverse text normalization, speech recognition, streaming, end-to-end

## 1. Introduction

Inverse Text Normalization (ITN) converts spoken-form transcriptions produced by Automatic Speech Recognition (ASR) systems into readable written-form representations [1]. While ASR models typically train on verbalized labels to maintain phonetic alignment with audio, their outputs often lack readability and impair downstream tasks such as machine translation and natural language understanding. ITN addresses this by restoring punctuation, applying capitalization, and performing phrase-level transformations, including semiotic and phonetic phrases [2]. For example, it converts expressions like “one point two dollars” to “\$1.20”. While spoken and written forms are frequently identical, certain structured phrases differ substantially and require contextual reasoning to resolve ambiguities. This makes ITN inherently context-dependent and necessitates robust contextual representations to ensure accurate disambiguation across diverse linguistic patterns [3].

Recent advances in ITN have focused on end-to-end approaches, which achieve strong performance across languages and domains [1] by implicitly learning patterns from data, thereby offering greater scalability and enabling effective model compression [3]. These methods primarily use sequence-tosequence (seq2seq) architectures to map spoken-form inputs directly to written-form outputs [1, 2, 4, 5, 6]. However, standard seq2seq models often produce unrecoverable hallucination errors, yielding fluent yet semantically incorrect results that alter original meaning [1]. Additionally, processing full sentences increases computational cost and risks unnecessary changes to unchanged tokens [2]. Hybrid methods alleviate these problems by combining neural tagging models with finite-state transducer (FST) rules [7, 8, 9]. They excel at preserving unmodified text but depend on expert-crafted FST grammars and extensive inside-outside-beginning (IOB) tag annotation for each ITN category. This dependence limits scalability and performance improvement due to the substantial manual effort required for IOB labeling and FST rule creation [3]. Pretrained language models [1, 6] and data-driven techniques [5, 10] have also been explored to address data scarcity, improve generalization, and enhance ITN performance across diverse languages and domains.

Despite advances in offline ITN, simultaneous or streaming ITN remains underexplored yet essential for real-time applications such as live ASR, meeting transcription, and speech translation, where formatted text must be displayed instantaneously to preserve conversational flow. Industrial systems often rely on cascaded architectures, applying incremental punctuation and casing models as post-processing steps [7, 8]. While these modular pipelines provide stability, they struggle with complex semiotic ambiguities that require joint modeling of capitalization, punctuation, and structural transformations. Pioneering work in this area [3] introduced a hybrid streaming solution employing a chunk-based Transformer [11] tagger to detect ITN categories in incoming ASR tokens, followed by categoryspecific FSTs for targeted conversion. A subsequent study [12] adapted a pretrained language model with dynamic, contextaware processing to handle variable ASR chunk sizes and accumulated tokens, achieving low latency and accuracy comparable to non-streaming ITN on a limited dataset. However, current streaming methods rely exclusively on hybrid architectures and inherit limited scalability due to the need for manual FST rule development and extensive IOB labeling. While seq2seq models offer greater scalability, they cannot operate in a streaming fashion because their encoder-decoder architectures rely on global cross-attention, which prevents incremental processing.

To adapt offline seq2seq models for streaming inference, the field of Simultaneous Machine Translation (SiMT) has introduced various techniques to enable incremental processing [13]. A core component of SiMT is the read-write policy, which decides whether to wait for new source tokens (READ) or generate target tokens (WRITE) to balance quality and latency. For instance, the wait-k policy [14, 15, 16] enforces a fixed lag by alternating WRITE/READ operations after an initial k-token delay, while monotonic attention mechanisms [17] restrict attention to left-to-right alignments. Other adaptive strategies involve learned thresholds for dynamic read-write decisions [18], Hidden Markov Transformers for simultaneous generation [19], or leveraging large language models as policy-makers [20].

![](images/c9faa4eb94862f8eb4db86904f6acc2e6481f620cb0582ff0b22035e4c20443d.jpg)  
Figure 1: Overview of the proposed streaming end-to-end ITNframework using the Read–Tag–Write policy. The streaming encoder utilizes Dynamic Right Context masking to process incoming chunks of spoken-form tokens, while a joint tagger predicts {B, I, O} labels. Verbatim tokens (O) are passed through instantly to minimize latency. Upon the completion of a normalization span, the autoregressive decoder is selectively triggered, generating normalized text by cross-attending to the relevant encoder hidden states (H<sub>0:m</sub>) and conditioning on the written-form prefix yˆ<sub>pre</sub>.

However, while effective for translation, these strategies introduce redundant computation for ITN. In SiMT, the reordering problem represents a primary conflict between translation quality and real-time latency. Unlike SiMT, which requires global structural restructuring, ITN is characterized by local transformations that maintain the original word order. Because ASR outputs are largely verbatim, the fixed lags and generative overhead typical of SiMT policies may be counterproductive for ITN. Consequently, the applicability of complex SiMT scheduling to the localized domain of ITN remains unexplored.

In this paper, we propose, to the best of our knowledge, the first streaming end-to-end ITN model. Our approach achieves accuracy comparable to non-streaming end-to-end baselines, outperforms streaming hybrid methods, and eliminates the need for manual FST rules and IOB labeling. Our experiments are conducted on Vietnamese, a challenging language for ITN due to its complex morphological nuances and frequent phonetic borrowing. By relying strictly on standard text-to-text foundations rather than language-specific heuristics, our approach ensures cross-linguistic generalization and is compatible with any seq2seq architecture. Our main contributions include:

• A recipe for making pretrained seq2seq models truly streamable: We present a practical set of techniques: architectural changes, a streaming-aware multi-task objective, and a novel data augmentation technique. That together enable end-to-end ITN with strict left-to-right processing.

• An efficient Read–Tag–Write decoding policy with KV Caching and Copy–Paste Strategy: We introduce a simultaneous inference procedure that (i) passes through nonnormalized tokens with minimal delay and (ii) triggers generation only upon completing a normalization span, further accelerated by KV caching and a Copy–Paste strategy that keeps historical context and removes redundant computation.

• Comprehensive empirical evaluation: We rigorously evaluate our approach on a newly curated 5M-sentence Vietnamese dataset featuring automated alignments and boundary tags, verifying its highly efficient performance.

## 2. Streaming End-to-End Inverse Text Normalization

In this section, we explain the chosen pretrained text-to-text models and describe our proposed framework for streaming end-to-end ITN.

## 2.1. Pretrained Text-to-Text Models

Pretrained text-to-text models such as BART [21] and T5 [22] have advanced seq2seq learning by unifying diverse NLP tasks under a generative framework. Unlike encoder-only architectures, these models utilize a complete Transformer encoderdecoder structure where both inputs and outputs are represented as text strings. BART [21] is pretrained using a denoising objective including token masking, deletion, and sentence permutation to learn robust bidirectional encoder representations and autoregressive decoder capabilities. Similarly, T5 [22] employs a span-corruption objective on the Colossal Clean Crawled Corpus (C4), facilitating effective transfer learning for downstream tasks such as translation and summarization. Extensive empirical evidence indicates that leveraging these pretrained foundations significantly improves performance on downstream tasks compared to training from scratch, especially in low-resource or domain-specific scenarios [21, 22, 23].

For the Vietnamese language, EnViT5<sup>1</sup> [24] adapts the T5- base architecture to a bilingual setting. Following the standard configuration, the model features 12 layers in both the encoder and decoder, a hidden dimension of 768, 12 attention heads, and a gated-GELU feed-forward projection with an intermediate dimension of 2048, totaling approximately 275 million parameters. EnViT5 was pretrained on a massive bilingual corpus, including an 80GB filtered subset of CC100 [25], using a 15% span-corruption objective for both English and Vietnamese. It utilizes a shared SentencePiece tokenizer [26] with a 50,048- subword vocabulary specifically optimized for the morphological nuances of the Vietnamese language. EnViT5 achieves superior performance on Vietnamese-English translation tasks on the PhoMT benchmark [27], outperforming larger models such as mBART [24]. Given its superior semantic alignment and generative efficiency, we adopt EnViT5 as the foundation for our end-to-end ITN system, subsequently adapting the architecture to satisfy streaming requirements.

## 2.2. Model Architecture

The proposed streaming end-to-end ITN model, illustrated in Fig. 1, is built upon the pretrained EnViT5 architecture. It comprises a streaming-capable encoder and an autoregressive decoder, optimized for real-time inverse text normalization.

## 2.2.1. Streaming Encoder with Joint Tagger

To support streaming inference within the encoder, we employ the Dynamic Right Context (DRC) masking technique [28] during training. Specifically, we utilize an attention mask to incorporate variable chunk sizes and dynamic look-ahead while maintaining access to the full historical left context. By varying these parameters during training, the masking strategy effectively simulates the restricted future context inherent in realtime processing, ensuring robustness to diverse latency constraints during inference.

For an input sequence $X = \{ x _ { 0 } , \ldots , x _ { n - 1 } \}$ , the encoder generates hidden representations $H \in \mathbb { R } ^ { n \times d }$ through Transformer layers constrained by the DRC mask, where d denotes the hidden dimensionality of the model. A linear tagging layer (the Tagger module as illustrated in Fig. 1), is integrated atop these hidden representations to facilitate on-the-fly detection of normalization spans (e.g., punctuation, capitalization, or numerical expressions). Following [2], the tagger classifies each token x<sub>i</sub> using a {B, I, O} (Beginning, Inside, Outside) labeling scheme. The predicted tag $\hat { t } _ { i }$ is determined by:

$$
\hat { t } _ { i } = \operatorname { a r g m a x } ( H _ { i } W _ { t a g } ) , \quad W _ { t a g } \in \mathbb { R } ^ { d \times 3 }\tag{1}
$$

These predicted tags serve as the control signals for the Read–Tag–Write policy (described in Section 2.3). Unlike hybrid approaches [3, 12] that require fine-grained tags for various ITN categories to trigger specific FST rules, our tagger utilizes only 3 classes. This reduced label space simplifies classification and leads to more robust span detection compared to complex multi-class systems.

## 2.2.2. Autoregressive Decoder and Training Objective

During training, the decoder generates written-form tokens y<sub>t</sub> by cross-attending to the encoder representations $H ,$ conditioned on the ground-truth previous tokens $y _ { < t }$ . While the encoder is constrained for streaming, the decoder maintains its standard autoregressive structure. The probability distribution over the vocabulary at each step is computed as:

$$
P ( y _ { t } | y _ { < t } , H ) = \mathrm { s o f t m a x } ( e _ { t } W _ { f f n } ^ { \top } )\tag{2}
$$

where $e _ { t } \in \mathbb { R } ^ { d }$ is the decoder hidden state at step $t , W _ { f f n } \in$ $\mathbb { R } ^ { V \times d }$ is the vocabulary projection matrix, and $V$ denotes the vocabulary size.

The model is optimized using a multi-task objective. We define $\mathcal { L } _ { t a g }$ as the cross-entropy loss for span detection and $\mathcal { L } _ { g e n }$ as the sequence generation loss:

$$
\mathcal { L } _ { t a g } = - \frac { 1 } { \sum _ { b = 1 } ^ { B } N _ { b } } \sum _ { b = 1 } ^ { B } \sum _ { i = 1 } ^ { N _ { b } } \log P ( t _ { b , i } \mid x _ { b , i } )\tag{3}
$$

$$
\mathcal { L } _ { g e n } = - \frac { 1 } { \sum _ { b = 1 } ^ { B } T _ { b } } \sum _ { b = 1 } ^ { B } \sum _ { t = 1 } ^ { T _ { b } } \log P ( y _ { b , t } \mid y _ { b , < t } , H _ { b } )\tag{4}
$$

Here, B denotes the batch size, while $N _ { b }$ and $T _ { b }$ represent the source and target sequence lengths for the b-th sample, respectively. The terms $t _ { b , i }$ and $_ { y _ { b , t } }$ denote the ground-truth tag at step i and the target token at step t. The probability $P ( t _ { b , i } \mid x _ { b , i } )$ is computed by applying the softmax function to the tagger logits $H _ { b , i } W _ { t a g } .$

The overall objective is a combination of these components:

$$
\mathcal { L } = \mathcal { L } _ { t a g } + \mathcal { L } _ { g e n }\tag{5}
$$

## 2.3. Read–Tag–Write Policy

To enable efficient streaming inference, we propose the Read–Tag–Write (RTW) policy. Unlike SiMT policies that manage global structural reordering [13], RTW exploits the local, order-preserving nature of ITN to coordinate the interaction between the streaming encoder and the autoregressive decoder, ensuring that generative normalization occurs only when necessary. The policy operates in three distinct phases:

1. Read: The system receives incoming ASR tokens in discrete chunks. Let $C _ { k } = \{ x _ { i } , \ldots , x _ { i + c - 1 } \}$ be the k-th chunk of size c. The encoder, constrained by the DRC mask, processes $C _ { k }$ by leveraging the cached historical context and a limited look-ahead of r tokens to compute hidden representations $H _ { i : i + c - 1 } { \mathrm { , } }$ , which are appended to the global hidden state sequence $H _ { 0 : i + c - 1 } .$

2. Tag: A joint tagging layer performs token-level classification on the hidden states of the newly arrived chunk. For each token $x _ { j } ~ \in ~ C _ { k }$ , the model predicts a tag $\hat { t } _ { j }$ ∈ $\{ B , I , O \}$ as:

$$
\hat { t } _ { j } = \mathrm { a r g m a x } ( H _ { j } W _ { t a g } )\tag{6}
$$

The policy identifies transformation spans using $\{ B , I \}$ labels. A span (e.g., a single B or a B-I sequence) is considered complete only when the tagger encounters a tag $\hat { t } _ { j + 1 } \in \{ B , O \}$ or a sentence boundary. If ${ \hat { t } } _ { j + 1 } = I ,$ the token is treated as a continuation of the current span and the system continues to “Read” without triggering the $\mathrm { ^ { * } W i t e ^ { * } }$ phase.

3. Write: Unlike standard seq2seq models that decode the entire sequence, the RTW policy invokes the decoder selectively based on predicted tags:

• Verbatim Pass-through: For tokens tagged as O, the system directly emits the spoken-form token. This ensures minimal latency and preserves the integrity of verbatim segments by bypassing the generative decoder.

• Selective Decoding: When a $B \mathrm { ~ - ~ } I$ span (or a sequence of consecutive spans) is completed, the decoder is triggered. To optimize throughput, multiple consecutive spans within a chunk are concatenated and processed in a single decoder invocation. Let m be the index of the final token in the last completed span. The decoder generates the written-form segment $\hat { y } _ { l o c a l }$ by cross-attending to the encoder hidden states $H _ { 0 : m } \colon$

$$
P ( \hat { y } _ { l o c a l } \mid \hat { y } _ { p r e } , H _ { 0 : m } ) = \prod _ { u = 1 } ^ { U } P ( y _ { u } \mid [ \hat { y } _ { p r e } ; y _ { < u } ] , H _ { 0 : m } )\tag{7}
$$

where $[ \cdot ; \cdot ]$ denotes the concatenation operator, $\hat { y } _ { p r e }$ represents the written-form prefix generated in previous steps, and $y _ { u }$ is the u-th token of the local sequence of length U. Crucially, cross-attention is restricted to $H _ { 0 : m } .$ , ensuring look-ahead context does not influence the current transformation. This constraint prevents the decoder from overgenerating or emitting extraneous tokens beyond the target span. The decoder generates tokens until an <EOS> token is predicted, signaling the end of the local transformation. To prevent infinite decoding loops in edge cases where the model fails to emit an <EOS> token, we implement a forced hard-stop that truncates generation at a maximum of $L _ { \mathrm { m a x \_ l o c a l } }$ tokens per local span.

## 2.4. Inference Optimization via Copy–Paste Strategy and KV Caching

Encoder KV Caching: To maximize efficiency, we implement an incremental Key-Value (KV) cache for the encoder, enabling the reuse of computations from previous chunks. Similarly, we cache the encoder hidden states and the previously decoded written-form prefix. When the decoder is triggered, the entire cross-attention and self-attention KV states need to be updated to incorporate the newly computed encoder states. We perform an initial single batched forward pass through the decoder to populate the self-attention and cross-attention KV caches using the existing prefix, bypassing the need for autoregressive decoding during this initialization phase. Subsequent tokens are then produced via standard autoregressive generation. These optimizations significantly improve inference throughput and minimize latency.

Copy–Paste Strategy: In contrast to [2], which completely excludes historical context from the autoregressive decoder, we argue that such context is vital for accurate normalization. Since spoken and written forms are frequently identical in ITN, the O tag is the most prevalent. Our RTW policy leverages this by bypassing the decoder for these tokens and copying them directly to the output. Crucially, these skipped tokens are still incorporated into the historical context to guide the decoding of B − I tokens. This approach not only reduces computational overhead and mitigates hallucinations in verbatim segments but also allows the decoder to maintain the full historical context efficiently.

Decoder KV Caching: To mitigate the computational overhead inherent in the Copy–Paste mechanism and autoregressive decoding, where standard processing would redundantly recompute the entire history for both copied and emitted tokens, we implement the “fully streaming” version by enabling the decoder KV Caching for both self- and cross-attention. This allows the model to reuse states from previous invocations and perform updates only for new tokens. While this eliminates initial passes, it introduces a training-inference mismatch: training uses the full target prefix, whereas streaming inference updates states incrementally without re-attending to past tokens under new encoder states. Our experiments show this results in only a marginal performance fluctuation compared to the first strategy.

## 2.5. Prefix-based Training Augmentation

The RTW policy requires the decoder to emit an <EOS> token to terminate each local normalization span. Standard training, however, typically conditions <EOS> prediction on a global source-side boundary. This creates a train-test mismatch during streaming inference, where local spans must terminate without observing a global <EOS> or future context.

Algorithm 1 Prefix-based Training Augmentation   
Require: Training pair $\left( \mathbf { x } _ { 0 : N - 1 } , \mathbf { y } _ { 0 : M - 1 } \right)$ , probability p<sub>trunc</sub>   
1: $r \sim$ Uniform(0, 1)   
2: $\mathbf { i f } \ r < p _ { \mathrm { t r u n c } }$ then {Truncated case}   
3: Sample $k \sim \mathrm { U n i f o r m } \{ 1 , \dots , \hat { N } - 1 \}$   
4: $\mathbf { x } ^ { \prime }  x _ { 0 : k - 1 } \{ \mathrm { N o } $ source-side <EOS>}   
5: $\mathbf { y } ^ { \prime } \gets [ y _ { 0 : l - 1 } ; \langle \mathtt { E O S } > ] \left\{ y _ { 0 : l - 1 } \right.$ is the target aligned with   
${ x } _ { 0 : k - 1 } \}$   
6: else {Full-sequence case}   
7: $\mathbf { x } ^ { \prime } \gets [ \mathbf { x } _ { 0 : N - 1 } ; < \mathtt { E O S } > ]$   
8: $\mathbf { y } ^ { \prime } \gets [ \mathbf { y } _ { 0 : M - 1 } ; < \mathtt { E O S } > ]$   
9: end if   
10: return $( \mathbf { x } ^ { \prime } , \mathbf { y } ^ { \prime } )$

Simply appending an <EOS> token to source chunks during inference is unsuitable, as it would invalidate the incremental KV cache through self- and cross-attention and exacerbate distribution shifts.

To bridge this train-test discrepancy, we introduce Prefixbased Training Augmentation (Alg. 1), a computationally efficient method applied on-the-fly during training. We stochastically truncate training examples in an aligned manner with a probability $p _ { \mathrm { t r u n c } }$ . In these instances, the source-side <EOS> is omitted, while the target-side prefix is explicitly terminated with an <EOS> token. This strategy compels the model to learn that any valid completion of a normalization span, even when occurring mid-sentence, must trigger termination based solely on the current partial encoder hidden states. By simulating these artificial boundaries, the model becomes robust to the truncated inputs encountered during streaming inference. Consequently, the decoder can reliably terminate generation without compromising the incremental KV cache, ensuring the system remains responsive to subsequent incoming chunks.

## 2.6. Data Construction

We construct our dataset from the $b i n h \nu q ^ { 2 }$ news corpus, a large-scale collection containing approximately 6 billion words across diverse domains. Using the proposed data generation pipeline, we produce 5M training sentence pairs, 50k validation pairs, and 50k test pairs. In addition to spoken-written text pairs, the dataset provides deterministic phrase-level alignments between spoken-form inputs and written-form targets, as well as token-level normalization boundary annotations. For span detection, each token is labeled using the IOB tagging scheme to identify segments requiring normalization. These spans encompass the entire affected word. For instance, a trailing punctuation mark is grouped within the preceding word’s span. These annotations facilitate both prefix-based training augmentation and auxiliary tagging supervision.

Categories: We define a comprehensive set of ITN categories that reflect common phenomena in ASR outputs. These categories include punctuation restoration, capitalization, semiotic normalization, and phonetic normalization. For punctuation restoration, we consider four standard punctuation marks: comma, period, question mark, and exclamation mark [12]. Semiotic normalization covers entities such as numbers, dates, times, fractions, currencies, digit sequences, and electronic addresses, while phonetic normalization targets out-of-vocabulary (OOV) expressions, including foreign words, named entities, domain names, and letter sequences.

Pipeline: The dataset is generated through an automated data generation pipeline that builds upon the data preparation framework introduced in [2], with substantial extensions to support a broader range of ITN categories. Specifically, we define a comprehensive set of regular expressions (regex) to identify normalization regions in written-form text:

• Capitalization and punctuation restoration: Spoken-form inputs are generated by removing all punctuation marks and converting the text to lowercase.

• Semiotic normalization: Each class (e.g., cardinal/ordinal numbers, decimals, currencies, time expressions, and factions) is identified using dedicated regex patterns. Since these expressions are detected in written form, rule-based transformations generate corresponding spoken-form candidates. Recognizing that a single written entity may have multiple valid spoken realizations, the pipeline randomly selects one candidate during data generation to reflect natural spoken variability. To ensure high quality, we manually inspect representative samples from each semiotic class to verify the correctness and coverage of the transformation rules.

• Phonetic normalization: We define phonetic phrases as tokens that are neither semiotic expressions nor present in the ASR vocabulary. This allows us to extract written-form phonetic entities, such as foreign named entities and domainspecific terms. We then perform manual verification of an automated grapheme-to-phoneme output to construct a dedicated phonetic dictionary comprising approximately 125k common foreign and OOV terms. This dictionary is subsequently used to map written-form phonetic phrases back to their spoken-form counterparts during dataset generation.

Through this regex-driven transformation process, the pipeline deterministically constructs phrase-level spoken-written alignments and assigns specific category labels to each normalization segment, providing precise supervision for both prefix-based training augmentation and the auxiliary tagging task for identifying normalization boundaries.

## 3. Experimental Setup

## 3.1. Datasets

For training and evaluation, we utilize the dataset described in Section 2.6. For hyperparameter tuning, we use a small subset of 500k samples from the training set and evaluate performance on the validation set. To better assess the system’s real-world performance, we construct an additional test set by passing all 50k samples from the original test set through the Microsoft Edge TTS system<sup>3</sup> to synthesize speech. The generated audio is then processed by the ChunkFormer ASR model<sup>4</sup> [29] to produce ASR transcripts. The word error rate (WER) of this ASR output relative to the reference transcripts (spoken text) is 14.21%. This error rate reflects typical ASR artifacts, such as misrecognitions and phonetic ambiguities. These issues are especially pronounced because the dataset contains numerous OOV and foreign words, providing a challenging and realistic input stream. This ASR-generated text serves as the input for evaluating the robustness of our ITN model in a real-world deployment scenario, following the evaluation protocol in [3].

## 3.2. Implementation Details

Training: Models are trained for up to 20 epochs using 4 NVIDIA H100 (80GB) GPUs and are implemented via the Hugging Face Transformers framework<sup>5</sup>. We employ the AdamW optimizer [30] with a peak learning rate of $1 0 ^ { - 4 } ,$ a one-epoch linear warmup, and a cosine learning rate scheduler. Additional optimizer hyperparameters include $\beta _ { 1 } = 0 . 9$ $\beta _ { 2 } ~ = ~ 0 . 9 9 9$ , weight decay $\lambda ~ = ~ 0 . 0 1$ , and a global batch size of 512. Our streaming ITN models utilize dynamic context masking with variable chunk sizes $c \in \{ 1 , 2 , 3 , 4 , 5 \}$ and right-context sizes $r \in \{ 0 , 1 , 2 \}$ with $p _ { \mathrm { t r u n c } } = 0 . 8$ for prefixbased training augmentation. To benchmark against a streaming hybrid model, we retrain the architecture from [12] on our dataset and integrate it with our own weighted FST module. For a baseline streaming end-to-end approach, we train a multi-path wait-k model [15]. We also compare our approach against the AdapITN model [2], which is trained on our dataset with the biasing mechanism excluded to ensure a fair comparison.

Evaluation: During inference, we employ greedy decoding. Following the methodology in [1], we evaluate performance using ITN Word Error Rate (I-WER) to measure normalization quality and Non-ITN Word Error Rate (NI-WER) to quantify unintended perturbations to the source text. Standard WER, the composite error of the entire process of converting spoken text to written text, is used for comprehensive performance assessment. Tagging performance is evaluated using the $F _ { 1 }$ -score. To simulate real-time inference for benchmarking, inputs are processed in discrete chunks. The reported metrics represent the average WER across an ensemble of configurations, where chunk sizes are sampled from $\{ 3 , 4 , 5 \}$ tokens and right-context sizes are randomly selected from $\{ 1 , 2 \}$ tokens. For configurations utilizing the RTW policy, we set $L _ { \mathrm { m a x \mathrm { { \_ l o c a l } } } } = 6 4 $ . This evaluation strategy ensures that the model’s robustness is accurately reflected across various streaming latencies, reflecting real-world usage scenarios.

Statistical Evaluation: To ensure the reliability of our results, we employ a bootstrap method with 10,000 iterations to determine 95% confidence intervals (CIs) for the metric of interest, leveraging the toolkit provided in [31]. Results are presented in the format mean<sub>[LB,UB]</sub>, where mean represents the metric’s value on the original test set and the interval [LB, UB] denotes the lower and upper bounds of the 95% confidence interval.

## 4. Results and Analysis

## 4.1. ITN Performance

In this section, we comprehensively evaluate the ITN performance of various model configurations, categorized into three groups:

• Non-streaming end-to-end baselines (B): $B _ { 0 }$ is an En-ViT5 model trained from scratch, while $B _ { 1 }$ utilizes pretrained weights. Both models are trained in a seq2seq manner using the $\mathcal { L } _ { g e n }$ loss function. $B _ { 2 }$ further enhances $B _ { 1 }$ with an additional auxiliary tagging loss, $\mathcal { L } _ { t a g }$ . We also include AdapITN [2] as a comparative baseline.

• Hybrid streaming (H): We benchmark against the hybrid architecture from [12], denoted as $H _ { 1 }$ (with look-ahead $r =$ 0) and $H _ { 2 }$ (with r ∈ {1, 2}).

• Proposed end-to-end streaming (S): $S _ { C }$ is our causalmasked encoder model with non-streaming inference. $S _ { 0 } ,$ $S _ { 1 }$ and $S _ { 2 }$ are our streaming variants using the RTW policy, whereas $S _ { 3 }$ and $S _ { 4 }$ represent the “fully streaming” versions utilizing decoder KV caching (as described in Section $2 . 4 )$ Note that $S _ { 0 }$ is trained from scratch, while all other S variants leverage pretrained weights. As an additional end-to-end streaming baseline, we benchmark against a multi-path waitk model [15] decoded with $k = 9$

Table 1: Evaluation ofITNperformance with and without punctuation and capitalization $( P n C )$ . Parameters c and r denote chunk and right-context sizes during inference. All metrics are reported as percentages $( \% )$ . Model categories: B (Non-streaming baselines), H (Hybrid streaming models), and S (Proposed end-to-end streaming variants).
<table><tr><td rowspan="2">Model</td><td rowspan="2"></td><td rowspan="2">Pretrained</td><td rowspan="2">c</td><td rowspan="2">r</td><td rowspan="2"> $\mathbf { F } _ { 1 }$ </td><td colspan="3">With PnC</td><td colspan="3">Without PnC</td></tr><tr><td>I-WER</td><td>NI-WER</td><td>WER</td><td>I-WER</td><td>NI-WER</td><td>WER</td></tr><tr><td rowspan="5">Non-seam</td><td> $B _ { 0 }$ </td><td> $\times$ </td><td>∞</td><td>∞</td><td>一</td><td>14.20</td><td>1.51</td><td>4.61</td><td>8.28</td><td>0.54</td><td>1.11</td></tr><tr><td> $B _ { 1 }$ </td><td>EnViT5</td><td>∞</td><td>∞</td><td></td><td>11.69</td><td>1.30</td><td>3.84</td><td>5.17</td><td>0.39</td><td>0.74</td></tr><tr><td> $B _ { 2 }$ </td><td>EnViT5</td><td>8</td><td>∞</td><td>97.34</td><td>10.80</td><td>1.32</td><td>3.63</td><td>5.03</td><td>0.16</td><td>0.52</td></tr><tr><td> $\operatorname { A d a p I T N } \left[ 2 \right]$ </td><td>EnViBert</td><td>∞</td><td>∞</td><td>96.65</td><td>156.88</td><td>2.50</td><td>40.08</td><td>14.58</td><td>0.40</td><td>1.45</td></tr><tr><td> $S _ { C } \ ( \mathrm { o u r s } )$ </td><td>EnViT5</td><td>{3,4,5}</td><td>{0,1,2}</td><td>96.62</td><td>11.48</td><td>1.31</td><td>3.79</td><td>5.83</td><td>0.18</td><td>0.60</td></tr><tr><td rowspan="9">Stream</td><td> $H _ { 1 } \ [ 1 2 ]$ </td><td>PhoBert</td><td>{3,4,5}</td><td>0</td><td>86.30†</td><td>34.79</td><td>2.22</td><td>10.16</td><td>49.34</td><td>1.05</td><td>4.62</td></tr><tr><td>H2 [12]</td><td>PhoBert</td><td>{3,4,5}</td><td>{1,2}</td><td>90.21†</td><td>28.39</td><td>1.78</td><td>8.27</td><td>47.36</td><td>0.75</td><td>4.20</td></tr><tr><td>Wait-k [15]</td><td>EnViT5</td><td></td><td>with k = 9</td><td></td><td>19.40</td><td>3.00</td><td>6.99</td><td>16.79</td><td>2.68</td><td>3.72</td></tr><tr><td> $S _ { 0 }$  (ours)</td><td>×</td><td>{3,4,5}</td><td>0</td><td>95.09</td><td>21.39</td><td>1.56</td><td>6.39</td><td>10.54</td><td>1.03</td><td>1.73</td></tr><tr><td>(ours)  $S _ { 1 }$ </td><td>EnViT5</td><td>{3,4,5}</td><td>0</td><td>95.48</td><td>16.66</td><td>1.47</td><td>5.18</td><td>7.42</td><td>0.29</td><td>0.82</td></tr><tr><td> $S _ { 2 }$  (ours)</td><td>EnViT5</td><td>{3,4,5}</td><td>{1,2}</td><td>97.19</td><td>12.91</td><td>1.36</td><td>4.18</td><td>6.24</td><td>0.19</td><td>0.64</td></tr><tr><td> $S _ { 3 }$  (ours)</td><td>EnViT5</td><td>{3,4,5}</td><td>0</td><td>95.48</td><td>17.03</td><td>1.37</td><td>5.19</td><td>8.79</td><td>0.26</td><td>0.89</td></tr><tr><td> $S _ { 4 }$  (ours)</td><td>EnViT5</td><td>{3,4,5}</td><td>{1,2}</td><td>97.19</td><td>13.22</td><td>1.31</td><td>4.21</td><td>7.30</td><td>0.16</td><td>0.69</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

<sup>†</sup> denotes the hybrid model’s $F _ { 1 }$ score, which is based on fine-grained multi-class classification.

Table 1 presents ITN performance across two settings: With PnC (punctuation, capitalization, and semiotic/phonetic normalization) and Without PnC (semiotic and phonetic normalization only).

## 4.1.1. Non-Streaming Evaluation

Leveraging the pretrained EnViT5 $( B _ { 1 } )$ yields significant gains over training from scratch $( B _ { 0 } )$ , with relative WER improvements of $1 6 . 7 0 _ { [ 1 0 . 4 5 , 2 6 . 1 5 ] } \%$ and $3 3 . 3 3 _ { [ 5 . 3 3 , 6 3 . 1 6 ] } \%$ in the “With PnC” and “Without PnC” settings, respectively. Consistently lower I-WER and NI-WER scores highlight the practical benefits of pretrained text-to-text models for end-to-end ITN. By inheriting linguistic priors, the model better captures the global context and semantic alignment necessary to map informal speech to structured written forms. Incorporating the auxiliary tagging loss $( B _ { 2 } )$ further improves results, establishing a strong baseline with WERs of 3.63% and 0.52%.

Although AdapITN [2] is an end-to-end system, it is not fine-tuned from a pretrained text-to-text model. Specifically, AdapITN utilizes encoder-only pretrained EnViBert [32] for both the encoder and decoder, with a tagging layer to isolate normalization regions. However, its decoder only processes isolated text spans, excluding the surrounding sentential context and making capitalization and punctuation recovery extremely difficult. As a result, it struggles to generalize in the “With PnC” evaluation, yielding an I-WER of 156.88% and a high overall WER of 40.08%. Even in the “Without $\mathrm { P n C } ^ { \prime \prime }$ setting, which is consistent with its original evaluation, AdapITN’s 1.45% WER is $2 . 7 9 _ { [ 2 . 6 1 , 3 . 3 0 ] } \times$ higher than our $B _ { 2 }$ baseline. These results demonstrate the critical importance of incorporating surrounding context and leveraging a pretrained text-to-text model for complex ITN tasks.

## 4.1.2. Streaming Evaluation

Streaming performance follows a similar trend. $S _ { 1 }$ (pretrained) outperforms $S _ { 0 }$ (trained from scratch) with relative WER improvements of $1 8 . 9 4 _ { [ 1 0 . 9 2 , 2 7 . 8 4 ] } \%$ and $5 2 . 6 0 _ { [ 2 9 . 3 0 , 6 7 . 0 8 ] } \%$ . Our proposed $S _ { 2 }$ variant achieves WERs of 4.18% and 0.64%, significantly outperforming the Wait-k model [15] by factors of $1 . 6 7 _ { [ 1 . 6 0 , 1 . 7 5 ] } \times$ and $5 . 8 1 _ { [ 4 . 9 6 , 6 . 7 3 ] } \times .$ , respectively. The failures of Wait-k stem from forced generation: even after waiting k tokens, the source input may still lag behind the target output, since a single target token can correspond to multiple source tokens. This situation occurs during both training and inference. Additionally, $S _ { 2 }$ substantially surpasses the hybrid baseline $H _ { 2 }$ [12] by $1 . 9 8 _ { [ 1 . 9 4 , 2 . 0 1 ] } \times$ and $6 . 5 6 _ { [ 5 . 7 4 , 7 . 4 0 ] } \times$ . The high I-WERs observed for $H _ { 2 } ~ ( 2 \dot { 8 } . 3 9 \%$ and 47.36%) highlight the limitations of manually crafted FST rules for complex normalization, whereas our end-to-end approach learns these transformations directly from data.

Regarding the cost of streaming, the causal encoder with non-streaming inference $( S _ { C } )$ performs comparably to $B _ { 2 } .$ with marginal WER increases of $0 . 1 6 _ { [ 0 . 0 3 , 0 . 2 7 ] } \%$ and $0 . 0 8 _ { [ - 0 . 0 3 , 0 . 1 8 ] } \%$ . Under full streaming inference, $S _ { 2 }$ remains robust: it incurs only an $8 . 8 5 _ { [ - 4 . 1 2 , 1 6 . 7 7 ] } \%$ relative WER increase over $B _ { 1 }$ in the “With PnC” setting and improves performance by $1 3 . 5 1 _ { [ - 3 8 . 6 7 , 5 0 . 6 2 ] } \%$ in the “Without $\mathrm { \bar { P } n C ^ { \prime } }$ setting. This efficiency is attributed to our RTW policy and prefix-based training augmentation. Compared to $B _ { 2 } , \ S _ { 2 }$ shows relative WER increases of $1 5 . 1 5 _ { [ 1 0 . 2 9 , 1 7 . 1 7 ] } \%$ and $2 3 . 0 7 _ { [ 0 . 6 5 , 4 7 . 1 5 ] } \%$ quantifying the inherent trade-off when future context is unavailable to resolve global dependencies. In terms of tagging accuracy, $S _ { 2 }$ achieves a high $\bar { \boldsymbol { F } } _ { 1 }$ score of 97.19%, comparable to the $B _ { 2 }$ baseline, demonstrating its reliable boundary detection capability for the RTW policy.

## 4.2. Impact of KV Caching and Copy–Paste Strategy

We benchmark per-chunk processing latency across configurations and chunk sizes with a fixed look-ahead $r = 1$ . To simulate high-concurrency production, we use a batch size of 100 with sequences of 100 - 200 words on an NVIDIA H100 GPU.

![](images/c0a99781ef72d90324a1ff31d087e471202a88a36f94f431b75e05c00e02c802.jpg)  
Figure 2: Per-chunk latency breakdown (Encoder vs. Total)for four streaming configurations across chunk sizes of $3 , 4 ,$ and 5.

Average encoder and total latencies are evaluated for four cumulative configurations, as shown in Fig. 2:

• BL (Baseline): Standard streaming without optimizations, requiring full decoder processing for all tokens.

• CP (+Copy–Paste): Uses Copy–Paste strategy to bypass the decoder for O-tagged tokens, directly emit spoken-form text.

• EC (+Encoder Caching): Adds incremental encoder KV caching to CP (corresponding to $S _ { 1 }$ and $S _ { 2 }$ variants).

• DC (+Decoder Caching): The most optimized configuration, adding self- and cross-attention decoder KV caching to EC, representing the “fully streaming” S<sub>3</sub> and $S _ { 4 }$ variants.

As illustrated in Fig. 2, the Copy–Paste strategy significantly reduces total latency and mitigates hallucinations by eliminating redundant decoder invocations for verbatim tokens. Encoder KV caching (EC) dramatically reduces encoder latency from 54 ms to 10 ms, while decoder KV caching (DC) further minimizes overhead by calculating updates only for new tokens, bypassing the initial batched forward pass. Combined, these optimizations achieve a 3.43× speedup at $c = 5 \AA$ , where total latency decreases from 206 ms (BL) to 60 ms (DC). The total latency exhibits only marginal absolute increases across chunk sizes, offering flexible trade-offs between latency and context.

While the “fully streaming” DC approach introduces a minor training-inference mismatch, the performance impact is negligible. Table 1 shows that $S _ { 4 } ~ ( \mathrm { D C } )$ maintains nearly identical performance to $S _ { 2 } ~ ( \mathrm { E C } )$ , with absolute WER increases of only $0 . 0 3 _ { [ - 0 . 0 5 , 0 . 0 9 ] } \%$ and $0 . 0 5 _ { [ - 0 . 0 3 , 0 . 1 1 ] } \%$ across settings. However, DC improves total per-chunk latency by 31.82% relative to EC at $c = 5$ . We also evaluate the $9 5 ^ { t h }$ percentile (p<sub>95</sub>) latency to assess worst-case scenarios requiring heavy normalization. The p<sub>95</sub> latency for DC is 165.5 ms, compared to 249.0 ms for EC. Given that a chunk size of 5 typically represents 500 ms of audio, our model ensures real-time processing even in the $9 5 ^ { t h }$ percentile, demonstrating its capacity to serve 100 concurrent users in large-scale deployments.

## 4.3. Impact of Look-ahead

As illustrated in Table 1, incorporating look-ahead context (S<sub>2</sub>) yields consistent performance gains across all primary metrics. Specifically, the $S _ { 2 }$ model achieves relative WER reductions of $1 9 . 3 1 _ { [ 1 8 . 6 2 , 1 9 . 8 9 ] } \%$ and $2 1 . 9 5 _ { [ 1 7 . 9 5 , 2 5 . 0 4 ] } \%$ compared to the zero-look-ahead variant (S ) in the “With PnC” and “Without $\mathrm { P n C } ^ { \prime \prime }$ settings, respectively. Similarly, the $S _ { 4 }$ model achieves relative WER reductions of $1 8 . 8 8 _ { [ 1 8 . 5 0 , 1 9 . 2 9 ] } \%$ and 22.47<sub>[20.62,23</sub> $. 9 9 ] \%$ compared to its zero-look-ahead counterpart $( { \dot { S } } _ { 3 } )$ . A comparable trend is observed in the hybrid streaming architecture, where $H _ { 2 }$ outperforms $H _ { 1 }$ with WER reductions of $1 8 . 6 0 _ { [ 1 8 . 3 3 , 1 8 . 9 3 ] } \%$ and $9 . 0 9 _ { [ 8 . 7 9 , 9 . 4 2 ] } \%$ in the corresponding settings. These findings underscore the critical role of look-ahead context during both training and inference.

## 4.4. Real-world evaluation

Table 2: Real-world ITN performance on noisy ASR transcripts. All metrics are reported as percentages (%).
<table><tr><td></td><td>Model</td><td>I-WER</td><td>NI-WER</td><td>WER</td></tr><tr><td></td><td> $B _ { 0 }$ </td><td>37.30</td><td>2.44</td><td>10.94</td></tr><tr><td>No-aam</td><td> $B _ { 1 }$ </td><td>34.09</td><td>2.24</td><td>10.01</td></tr><tr><td></td><td> $B _ { 2 }$ </td><td>35.33</td><td>2.28</td><td>10.34</td></tr><tr><td></td><td> $S _ { C }$ </td><td>36.18</td><td>2.27</td><td>10.54</td></tr><tr><td rowspan="5">Siream</td><td> $H _ { 2 }$ </td><td>47.56</td><td>2.75</td><td>13.67</td></tr><tr><td> $\mathrm { W a i t } – k$ </td><td>41.16</td><td>2.48</td><td>11.91</td></tr><tr><td> $S _ { 0 }$ </td><td>46.83</td><td>2.51</td><td>13.30</td></tr><tr><td> $S _ { 2 }$ </td><td>37.55</td><td>2.29</td><td>10.89</td></tr><tr><td> $S _ { 4 }$ </td><td>37.08</td><td>2.24</td><td>10.74</td></tr></table>

To assess performance in a practical deployment context, we evaluate our models on a test set of noisy ASR transcripts. Unlike the original test set, these inputs contain inherent recognition errors and acoustic artifacts typical of end-to-end ASR systems. The experimental results are summarized in Table 2.

In the streaming scenario, our proposed $S _ { 4 }$ variant achieves the highest performance, significantly outperforming the hybrid streaming baseline $H _ { 2 }$ by $2 1 . 4 3 _ { [ 2 1 . 0 0 , 2 1 . 9 5 ] } \%$ and surpassing the Wait-k by $9 . 8 2 _ { [ 0 . 8 6 , 1 9 . 4 6 ] } \%$ in terms of relative WER reduction. When compared to the strongest offline baseline $( B _ { 1 } ) , S _ { 4 }$ maintains competitive performance, incurring a relative WER increase of only $7 . 2 9 _ { [ 6 . 8 5 , 7 . 6 9 ] } \%$ . Interestingly, on these noisy transcripts, the “fully streaming” model $S _ { 4 }$ slightly surpasses the $S _ { 2 }$ variant, with a relative improvement of $1 . 3 8 _ { [ 0 . 4 8 , 2 . 7 3 ] } \%$ This suggests that the inclusion of decoder KV caching does not negatively impact robustness to ASR errors. Rather, it may provide slight stability in noisy streaming contexts.

## 4.5. Impact of Truncation Probability

To assess our prefix-based training augmentation, we evaluate performance on a validation subset across two modes: Streaming Inference (RTW policy) and Offline Inference (causal encoder). All experiments use c=3 and $r { = } 1$ . Figure 3 illustrates the sensitivity of WER to the truncation probability $p _ { \mathrm { t r u n c } }$

For offline inference, the model remains robust across $0 . 0 \leq \ p _ { \mathrm { t r u n c } } \leq 0 . 9$ . While $p _ { \mathrm { t r u n c } } = 0$ yields the lowest WER by avoiding termination-related distribution shifts, performance stays stable as $p _ { \mathrm { t r u n c } }$ increases. A sharp degradation occurs only at $p _ { \mathrm { t r u n c } } = 1 . 0$ , where the lack of natural sentence endings during training impairs generalization to full sequences.

![](images/14011fc794d1acfbfbcdbc404c15ec65f094f8c627e5fed702f016f980b27b0b.jpg)  
Figure 3: Impact oftruncation probability $p _ { t r u n c }$ on the Word $E r \mathrm { - }$ ror Ratefor both offline and streaming inference ofour model.

In the streaming scenario, augmentation is essential. Without it $( p _ { \mathrm { t r u n c } } = 0 )$ , the WER exceeds 100% as the model fails to predict the local <EOS> tokens required by the RTW policy. Introducing prefix-based augmentation $( p _ { \mathrm { t r u n c } } > 0 )$ immediately stabilizes performance. As shown in Figure 3, increasing $p _ { \mathrm { t r u n c } }$ correlates with improved streaming accuracy, as the model learns to identify local boundaries from partial context.

The gap between offline and streaming modes narrows significantly for $p _ { \mathrm { t r u n c } } ~ \in ~ \{ 0 . 6 , 0 . 7 , 0 . 8 , 0 . 9 \}$ . The optimal balance is achieved at $p _ { \mathrm { t r u n c } } = 0 . 8 , \mathrm { y i e l d i n g }$ the lowest streaming WER while maintaining competitive offline performance. The $p _ { \mathrm { t r u n c } } ~ = ~ 1 . 0$ case is also detrimental to streaming, suggesting that complete removal of full-sequence supervision harms longrange dependency modeling. These results confirm that prefixbased augmentation effectively bridges the training-inference discrepancy for streaming ITN.

## 4.6. Generalizability Across Architectures and Languages

To demonstrate the versatility of our framework, we evaluate it across alternative text-to-text architectures, pretraining objectives, and languages. For Vietnamese, we adopt the pretrained BartPho model<sup>6</sup> [33] as the backbone, which was pretrained using a denoising autoencoder objective on Vietnamese text. For English, we leverage a pretrained T5 model<sup>7</sup> to assess crosslingual compatibility, with training and evaluation conducted on the English subset of the standard Google TN dataset [34].

We evaluate streaming models under the RTW policy and non-streaming models using standard decoding. Table 3 reports the average WER across all configurations $( c \in \{ 3 , 4 , 5 \}$ and $r ~ \in ~ \{ 0 , 1 , 2 \} ,$ ). The results show that our framework consistently narrows the performance gap between streaming and non-streaming settings across architectures and languages. While the larger EnViT5 backbone delivers the best absolute performance, the more compact BartPho model demonstrates comparable robustness under streaming constraints and noisy ASR inputs. A similar trend is observed in English, where the T5 backbone also achieves strong and stable performance. Together, these results suggest that the proposed method is architecture-agnostic and language-agnostic, maintaining effectiveness across different model scales and linguistic contexts.

Table 3: Generalizability of ITN performance across architectures and languages. Values represent average WER (%) across $c \in \{ 3 , 4 , 5 \}$ and $r \in \{ 0 , 1 , 2 \}$ . Vietnamese evaluation uses ASR-L (ground-truth spoken text) and ASR-O (noisy ASR transcripts); English evaluation uses With and Without PnC.
<table><tr><td></td><td>Lang. Backbone Params Stream</td><td></td><td></td><td colspan="2">WER (%)</td></tr><tr><td rowspan="4">Vi</td><td></td><td></td><td></td><td>ASR-L</td><td>ASR-O</td></tr><tr><td rowspan="2">BartPho</td><td rowspan="2">132M</td><td>X</td><td>4.61 6.31</td><td>11.07 12.84</td></tr><tr><td>√</td><td></td><td></td></tr><tr><td>EnViT5</td><td>275M</td><td>X √</td><td>3.84 4.54</td><td>10.01 11.02</td></tr><tr><td rowspan="4">En</td><td rowspan="4">T5</td><td rowspan="4">223M</td><td></td><td>W/PnC</td><td>W/o PnC</td></tr><tr><td></td><td>6.48</td><td></td></tr><tr><td>X √</td><td>8.17</td><td>1.33 1.84</td></tr><tr><td></td><td></td><td></td></tr></table>

## 4.7. Limitations and Future Work

Despite the efficiency gains of our streaming ITN framework, several limitations offer avenues for future research:

• Phonetic Normalization and Failure Modes: While semiotic classes are handled effectively, qualitative error analysis reveals that performance on OOV expressions and foreign entities remains challenging. A common failure mode occurs when the tagger successfully identifies a phonetic span, but the decoder hallucinates an orthographically plausible but factually incorrect spelling for a rare foreign entity (e.g., generating “Marid” instead of “Madrid” due to phonetic ambiguity in the ASR output). Future work will investigate biasing mechanisms to integrate phonetic dictionaries directly into the decoding process to improve I-WER without disrupting the incremental KV cache or increasing latency.

• Early Emission Strategy: The current RTW policy waits for a complete B −I span before emission, which may introduce perceived lag for long sequences. We aim to explore predictive decoding strategies that allow for partial output as soon as sufficient linguistic context is captured, rather than waiting for span completion.

• Edge Deployment Optimization: With 275M parameters, the model may limit deployment on resource-constrained devices. Future work will explore compression methods, such as knowledge distillation and quantization, to reduce memory footprint and improve throughput on low-power hardware.

## 5. Conclusion

This paper introduces an efficient end-to-end streaming ITN framework leveraging pretrained text-to-text models. By integrating a chunk-based causal encoder with an RTW decoding policy, we adapt non-streaming architectures for real-time inference. To bridge the train-test discrepancy, we propose a prefix-based training augmentation strategy that ensures reliable local termination. Combined with KV caching and a Copy–Paste mechanism, our system meets production-grade latency requirements without sacrificing accuracy. Experimental results demonstrate that our approach significantly outperforms traditional hybrid streaming methods, achieving performance comparable to offline baselines. Finally, evaluations across architectures and languages confirm that the framework is scalable, backbone-agnostic, and language-agnostic, offering a versatile solution for complex normalization tasks.

## 6. Acknowledgments

We thank UNEY for sponsoring this research and providing the computational resources essential for model training and evaluation.

## 7. Generative AI Use Disclosure

Generative AI tools were used solely for editing and language polishing of the manuscript. They were not used to generate scientific content, experimental designs, results, analyses, or conclusions. All authors are responsible for the full content of this paper and consent to its submission.

## 8. References

[1] M. Sunkara, C. Shivade, S. Bodapati, and K. Kirchhoff, “Neural inverse text normalization,” in ICASSP 2021. IEEE, 2021, pp. 7573–7577.

[2] T.-B. Nguyen, L. D. M. Nhat, Q. M. Nguyen, Q. T. Do, C. M. Luong, and A. Waibel, “Adapitn: A fast, reliable, and dynamic adaptive inverse text normalization,” in ICASSP 2023. IEEE, 2023, pp. 1–5.

[3] Y. Gaur, N. Kibre, J. Xue, K. Shu, Y. Wang, I. Alphanso, J. Li, and Y. Gong, “Streaming, fast and accurate on-device inverse text normalization for automatic speech recognition,” in IEEE SLT 2022. IEEE, 2023, pp. 237–244.

[4] M. Ihori, A. Takashima, and R. Masumura, “Large-context pointer-generator networks for spoken-to-written style conversion,” in ICASSP 2020. IEEE, 2020, pp. 8189–8193.

[5] S.-J. Chen, D. Paul, Y. Pang, P. Su, and X. Zhang, “Language agnostic data-driven inverse text normalization,” in Interspeech 2023, 2023, pp. 451–455.

[6] H. Choi, M. Choi, Y. Lim, M. Lee, S. Kim, S. Yun, D. Kim, and S. Kim, “Spoken-to-written text conversion with Large Language Model,” in Interspeech 2024, 2024, pp. 2410–2414.

[7] M. Shugrina, “Formatting time-aligned asr transcripts for readability,” in Human Language Technologies: The 2010 Annual Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics, 2010, pp. 198–206.

[8] Y. Zhang, E. Bakhturina, and B. Ginsburg, “Nemo inverse text normalization: From development to production,” in Interspeech 2021, 2021, pp. 4857–4859.

[9] Alexandra Antonova and Evelina Bakhturina and Boris Ginsburg, “Thutmose Tagger: Single-pass neural model for Inverse Text Normalization,” in Interspeech 2022, 2022, pp. 550–554.

[10] Debjyoti Paul and Yutong Pang and Szu-Jui Chen and Xuedong Zhang, “Improving Data Driven Inverse Text Normalization using Data Augmentation and Machine Translation,” in Interspeech 2022, 2022, pp. 5221–5222.

[11] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in neural information processing systems, vol. 30, 2017.

[12] L. Ho, K. Le, V. Pham, B. Nguyen, T. Tran, and D. Chau, “Dynamic Context-Aware Streaming Pretrained Language Model For Inverse Text Normalization,” in Interspeech 2025, 2025, pp. 4433–4437.

[13] M. Wang, T. Vu, Y. Wang, E. Shareghi, and G. Haffari, “Conversational simulmt: Efficient simultaneous translation with large language models,” in IWSLT 2025, 2025, pp. 93–105.

[14] M. Ma, L. Huang, H. Xiong, R. Zheng, K. Liu, B. Zheng, C. Zhang, Z. He, H. Liu, X. Li et al., “Stacl: Simultaneous translation with implicit anticipation and controllable latency using prefix-to-prefix framework,” in ACL 2019, 2019, pp. 3025–3036.

[15] M. Elbayad, L. Besacier, and J. Verbeek, “Efficient Wait-k Models for Simultaneous Machine Translation,” in Interspeech 2020, 2020, pp. 1461–1465.

[16] L. Zhao, K. Fan, W. Luo, W. Jing, S. Wang, Z. Zeng, and Z. Huang, “Adaptive policy with wait-k model for simultaneous translation,” in EMNLP 2023, 2023, pp. 4816–4832.

[17] N. Arivazhagan, C. Cherry, W. Macherey, C.-C. Chiu, S. Yavuz, R. Pang, W. Li, and C. Raffel, “Monotonic infinite lookback attention for simultaneous machine translation,” in ACL 2019, 2019, pp. 1313–1323.

[18] B. Zheng, R. Zheng, M. Ma, and L. Huang, “Simpler and faster learning of adaptive policies for simultaneous translation,” in EMNLP-IJCNLP 2019, 2019, pp. 1349–1354.

[19] S. Zhang and Y. Feng, “Hidden markov transformer for simultaneous machine translation,” in ICLR 2023, 2023.

[20] S. Guo, S. Zhang, Z. Ma, and Y. Feng, “Large language models are read/write policy-makers for simultaneous generation,” in AAAI 2025, vol. 39, no. 22, 2025, pp. 23 969–23 977.

[21] M. Lewis, Y. Liu, N. Goyal, M. Ghazvininejad, A. Mohamed, O. Levy, V. Stoyanov, and L. Zettlemoyer, “Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension,” in ACL 2020, 2020, pp. 7871–7880.

[22] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu, “Exploring the limits of transfer learning with a unified text-to-text transformer,” Journal of machine learning research, vol. 21, no. 140, pp. 1–67, 2020.

[23] L. Xue, N. Constant, A. Roberts, M. Kale, R. Al-Rfou, A. Siddhant, A. Barua, and C. Raffel, “mt5: A massively multilingual pre-trained text-to-text transformer,” in Proceedings of the 2021 conference of the North American chapter of the association for computational linguistics: Human language technologies, 2021, pp. 483–498.

[24] C. Ngo, T. H. Trinh, L. Phan, H. Tran, T. Dang, H. Nguyen, M. Nguyen, and M.-T. Luong, “Mtet: Multi-domain translation for english and vietnamese,” arXiv preprint arXiv:2210.05610, 2022.

[25] G. Wenzek, M.-A. Lachaux, A. Conneau, V. Chaudhary, F. Guzman, A. Joulin, and E. Grave, “Ccnet: Extracting high qual-´ ity monolingual datasets from web crawl data,” in LREC 2020, 2020, pp. 4003–4012.

[26] T. Kudo and J. Richardson, “Sentencepiece: A simple and language independent subword tokenizer and detokenizer for neural text processing,” in Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, 2018, pp. 66–71.

[27] L. Doan, N. L. Tran, T. Hoang, D. Q. Nguyen et al., “Phomt: A high-quality and large-scale benchmark dataset for vietnameseenglish machine translation,” in EMNLP 2021, 2021, pp. 4495– 4503.

[28] K. Le and D. Chau, “Improving Streaming Speech Recognition With Time-Shifted Contextual Attention And Dynamic Right Context Masking,” in Interspeech 2024, 2024, pp. 4478–4482.

[29] K. Le, T. V. Ho, D. Tran, and D. T. Chau, “Chunkformer: Masked chunking conformer for long-form speech transcription,” in ICASSP 2025. IEEE, 2025, pp. 1–5.

[30] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in ICLR 2019, 2019.

[31] L. Ferrer and P. Riera, “Confidence intervals for evaluation in machine learning.” [Online]. Available: https://github.com/ luferrer/ConfidenceIntervals

[32] T. B. Nguyen, Q. M. Nguyen, T. T. H. Nguyen, Q. T. Do, and C. M. Luong, “Improving Vietnamese Named Entity Recognition from Speech Using Word Capitalization and Punctuation Recovery Models,” in Proc. Interspeech 2020, 2020, pp. 4263–4267.

[33] N. L. Tran, D. Le, and D. Q. Nguyen, “BARTpho: Pre-trained Sequence-to-Sequence Models for Vietnamese,” in Interspeech 2022, 2022, pp. 1751–1755.

[34] R. Sproat and N. Jaitly, “An RNN Model of Text Normalization,” in Interspeech 2017, 2017, pp. 754–758.