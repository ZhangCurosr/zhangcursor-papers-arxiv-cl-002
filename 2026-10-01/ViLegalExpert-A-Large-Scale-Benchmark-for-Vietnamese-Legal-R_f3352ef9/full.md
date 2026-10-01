# ViLegalExpert: A Large-Scale Benchmark for Vietnamese Legal Retrieval and Question Answering from Real-World Consultations

Dat Tien Nguyen<sup>1,2</sup>, Nghia Hieu Nguyen<sup>1,2</sup>, Anh Thi-Hoang Nguyen<sup>1,2</sup>, Dung Ha Nguyen<sup>1,2</sup>, Kiet Van Nguyen<sup>1,2</sup>, and Ngan Luu-Thuy Nguyen<sup>1,2</sup>

<sup>1</sup> University of Information Technology

2 Vietnam National University, Ho Chi Minh city, Viet Nam 23520262@gm.uit.edu.vn, {nghiangh, anhnth, dungnh, kietnv, ngannlt}@uit.edu.vn

Abstract. Trustworthy Legal AI requires systems that can answer legal questions while grounding their responses in authoritative sources. However, existing Vietnamese legal benchmarks provide limited coverage of real-world legal consultations. We introduce ViLegalExpert, a large-scale benchmark constructed from authentic citizen–lawyer consultations, containing over 172K questions across 34 legal domains, together with professional answers and expert-verified legal evidence. ViLegalExpert supports legal information retrieval, extractive QA, and abstractive QA. Experiments with representative retrieval methods and language models reveal substantial challenges in evidence retrieval and grounded answer generation. While pretrained models perform strongly on QA, hybrid retrieval achieves the best retrieval performance. These results demonstrate the dificulty of mapping naturally expressed legal questions to authoritative provisions and establish ViLegalExpert as a challenging benchmark for reliable Vietnamese Legal AI.

Keywords: Legal AI · Benchmark · Question Answering · Document Retrieval

## 1 Introduction

Recent advances in Large Language Models (LLMs) have accelerated the development of Legal Artificial Intelligence (Legal AI), enabling applications such as legal assistants and automated legal consultation [40]. However, legal applications require not only fluent responses but also accurate answers grounded in authoritative legal sources [19, 34, 12, 32]. This requirement has become particularly important with Retrieval-Augmented Generation (RAG), where retrieval quality directly afects the reliability of generated answers [13]. Consequently, realistic benchmarks that jointly evaluate legal retrieval and question answering are essential for developing trustworthy Legal AI systems.

Several Vietnamese legal datasets have recently been introduced for information retrieval, question answering, and legal reasoning. Despite their contributions, existing resources are largely constructed for curated or task-specific settings, including shared tasks, manually annotated questions, and synthetic evaluation scenarios. They therefore provide limited coverage of the diverse information needs and linguistic characteristics of real-world legal consultations. Moreover, the limited scale or domain coverage of existing question-based resources poses challenges for evaluating modern retrieval and LLM-based systems.

To address these limitations, we introduce ViLegalExpert, a large-scale Vietnamese benchmark constructed from authentic online consultations between citizens and legal professionals. ViLegalExpert contains over 172K real-world legal questions spanning 34 legal domains, together with professional answers and expert-verified provision-level legal evidence. These annotations enable a unified evaluation of legal information retrieval, extractive QA, and abstractive QA, providing a realistic setting for assessing both evidence retrieval and answer generation.

We benchmark representative retrieval methods, pretrained language models, and instruction-tuned LLMs on ViLegalExpert. Our experiments reveal substantial challenges in mapping naturally expressed legal questions to authoritative provisions and generating correctly grounded answers, highlighting important limitations of current approaches.

Our main contributions are:

– We introduce ViLegalExpert, a large-scale Vietnamese legal benchmark containing over 172K authentic consultation questions across 34 legal domains, together with professional answers and expert-verified legal evidence. We establish a unified benchmark for legal retrieval, extractive QA, and abstractive QA, enabling evaluation of both evidence retrieval and answer generation.

We provide comprehensive baselines and analyses across representative retrieval and language models, revealing key challenges for reliable and evidencegrounded Vietnamese Legal AI.

## 2 Related Work

## 2.1 Legal NLP Benchmarks

Benchmark datasets have played a fundamental role in advancing Legal Natural Language Processing (Legal NLP) by providing standardized evaluation protocols for a wide range of legal tasks. Existing benchmarks cover legal document classification [3, 33], legal information retrieval [8, 16], legal summarization [1, 11], legal entailment, and legal question answering [39, 5]. More recently, the emergence of Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs) has further increased the importance of benchmarks that jointly evaluate document retrieval and evidence-grounded answer generation. These resources have substantially accelerated the development of Legal AI in highresource languages such as English and Chinese by enabling consistent comparison of retrieval models, pretrained language models, and LLM-based systems.

Table 1. Comparison of representative legal retrieval and question answering benchmarks. ViLegalExpert is the largest Vietnamese benchmark constructed from authentic citizen–lawyer consultation records and supports both legal information retrieval and question answering across 34 legal domains.
<table><tr><td>Dataset</td><td>Lang. Year</td><td></td><td>Source</td><td>Retrieval QA #Samples #Domains</td><td></td><td></td><td></td></tr><tr><td>COLIEE</td><td>EN</td><td>2014</td><td>Bar exam / Case law</td><td>√</td><td>√</td><td></td><td>1</td></tr><tr><td>JEC-QA</td><td>ZH</td><td>2020</td><td>Judicial examination</td><td>×</td><td>√</td><td>28,641</td><td>13</td></tr><tr><td>ALQAC</td><td>JA</td><td>2020</td><td>Legal documents</td><td>√</td><td>√</td><td>272</td><td>4</td></tr><tr><td>EQUALS</td><td>EN</td><td>2023</td><td>Legal documents</td><td>X</td><td>√</td><td>8,448</td><td></td></tr><tr><td>Zalo AI Challenge</td><td>VI</td><td>2021</td><td>Competition</td><td>√</td><td>X</td><td></td><td></td></tr><tr><td>ALQAC 2022</td><td>VI</td><td>2022</td><td>Legal documents</td><td>√</td><td>√</td><td>520</td><td>4</td></tr><tr><td>ALQAC 2023</td><td>VI</td><td>2023</td><td>Legal documents</td><td>√</td><td>√</td><td>1,200+</td><td>4</td></tr><tr><td>SoICT LegalIR</td><td>VI</td><td>2024</td><td>Competition</td><td>√</td><td>×</td><td>10,000</td><td></td></tr><tr><td>ViRHE4QA</td><td>VI</td><td>2024</td><td>Curated QA</td><td>√</td><td>√</td><td>9,758</td><td></td></tr><tr><td>VLSP LegalIR</td><td>VI</td><td>2024</td><td>Competition</td><td>√</td><td>×</td><td>2,000</td><td></td></tr><tr><td>VLegal-Bench</td><td>VI</td><td>2025</td><td>Curated benchmark</td><td>√</td><td>√</td><td>10,450</td><td></td></tr><tr><td>VLQA</td><td>VI</td><td>2025</td><td>Annotated legal QA</td><td>√</td><td>√</td><td>24,012</td><td>27</td></tr><tr><td>ViLegalExpert (Ours) VI</td><td></td><td></td><td>2026 Citizen-lawyer consultations</td><td>√</td><td>√</td><td>177,158</td><td>34</td></tr></table>

## 2.2 Vietnamese Legal NLP Resources

Vietnamese Legal NLP has witnessed rapid progress in recent years, driven by the release of several benchmark datasets and shared tasks. Early eforts focused on legal text retrieval and question answering through the ALQAC shared tasks, which introduced benchmark datasets for legal retrieval and answer extraction. Subsequent work expanded to larger retrieval corpora and more realistic retrieval settings, including the SoICT LegalIR Hackathon, ViRHE4QA, and the VLSP LegalIR shared task. Recent benchmarks have further broadened the scope of Vietnamese Legal AI by introducing datasets for legal natural language inference (ViLegalNLI), legal reasoning and Retrieval-Augmented Generation (VLegal-Bench), and multilingual legal question answering (VLQA) [6, 21].

Despite these advances, existing Vietnamese resources exhibit several limitations. First, many benchmarks are designed for specific shared tasks or individual downstream problems, such as retrieval, question answering, or natural language inference, making it dificult to evaluate Legal AI systems under a unified framework. Second, many datasets rely on manually curated evaluation sets or questions constructed from legal documents, which only partially reflect the diverse information needs encountered in real legal consultation. Finally, although some benchmarks provide large legal corpora, the number of authentic legal questions available for training and evaluation remains relatively limited.

## 2.3 Comparison with Existing Benchmarks

Table 1 compares ViLegalExpert with representative Vietnamese and multilingual legal benchmarks. Unlike previous datasets that primarily focus on individual tasks or curated benchmark settings, ViLegalExpert is constructed from authentic legal consultation forums, where citizens seek legal advice from legal professionals. Consequently, the dataset captures naturally occurring legal information needs expressed in everyday language, providing a more realistic evaluation setting for Legal AI.

ViLegalExpert contains over 177K authentic legal questions spanning 34 legal domains, together with corresponding answers and a structured retrieval corpus supporting both legal information retrieval and question answering. Compared with existing Vietnamese benchmarks, ViLegalExpert substantially expands the scale of authentic legal consultation questions while unifying retrieval and question answering under a single benchmark. We believe these characteristics make ViLegalExpert a valuable resource for developing, benchmarking, and analyzing modern Vietnamese Legal AI systems, particularly retrieval-based and LLMbased approaches.

## 3 The ViLegalExpert Dataset

## 3.1 Dataset Construction

The construction of ViLegalExpert consists of four main stages: (1) collecting real-world legal questions and answers, (2) constructing a corpus of authoritative Vietnamese legal documents, (3) linking questions to relevant legal documents through expert annotation, and (4) formatting and splitting the resulting data.

Collection of Legal Questions and Answers. We first collect question– answer pairs from publicly accessible online legal consultation platforms, where citizens can submit questions about legal issues and receive answers from legal professionals. These questions naturally arise from practical situations encountered in everyday life, rather than being manually written for benchmark construction. As a result, they exhibit substantial diversity in writing style, level of detail, and legal information needs. Each collected instance consists of a citizen’s question and its corresponding answer provided by a legal professional. The collected questions cover a broad range of legal topics, including civil, criminal, labor, land, marriage and family, business, and administrative law.

Collection of Legal Documents. In parallel, we construct a legal document corpus from oficial Vietnamese government sources. The corpus contains diferent types of normative legal documents, including the Constitution, laws, decrees, circulars, and other regulatory documents. These documents provide authoritative legal sources that can be used to support the answers in the collected consultation data. The resulting corpus also serves as the candidate document collection for the legal information retrieval task.

Expert Verification of Legal References. The answers collected from legal consultation platforms frequently contain explicit references to legal provisions, such as articles, clauses, and sections, cited by legal professionals to support their responses. Rather than treating these references as automatically correct, we employ legal experts to verify whether the cited provisions are valid and relevant to the corresponding questions and answers. For each instance, the experts examine the referenced article, clause, or section, assess whether it provides an appropriate legal basis for answering the question, and identify the corresponding authoritative legal document in the collected corpus. References that are incorrect or irrelevant are not retained as supporting evidence. This verification process produces expert-validated links between real-world legal questions and specific legal provisions, which subsequently serve as relevance annotations for the legal information retrieval task.

![](images/b69d9eacdbbd012f22d61f8cbca1406910f4c4966535863d523fa3b53103c01e.jpg)  
Fig. 1. Distribution of legal domains in the proposed dataset.

Data Formatting and Splitting. Finally, we normalize and organize the collected data into a unified representation that preserves the original questions and professional answers together with their expert-verified legal references. Each annotated instance therefore connects a naturally occurring legal question to its answer and the corresponding legal provision, including the relevant article, clause, or section and its source legal document. Based on these annotations, we construct the question answering and legal information retrieval subsets and partition the resulting data into training, development, and test sets for standardized evaluation.

## 3.2 Dataset Statistics

Table 2. Overall statistics of ViLegalExpert.
<table><tr><td>Statistic</td><td>Value</td></tr><tr><td>#IR Samples</td><td>87,695</td></tr><tr><td>#QA Samples</td><td>177,158</td></tr><tr><td>#Legal Domains</td><td>34</td></tr><tr><td>#Legal Documents</td><td>9,563</td></tr><tr><td>Average Question Length</td><td>21.08 tokens</td></tr><tr><td>Median Question Length</td><td>20 tokens</td></tr><tr><td>Average Answer Length</td><td>380.51 tokens</td></tr><tr><td>Median Answer Length</td><td>343 tokens</td></tr></table>

Table 2 summarizes ViLegalExpert, which contains 117,158 QA instances and 87,695 retrieval instances derived from over 172K authentic consultation questions across 34 legal domains. Figure 1 further shows that the dataset covers a broad range of legal topics with varying frequencies, reflecting diverse real-world legal information needs.

Figure 2 presents the distributions of question and answer lengths, showing substantial variation in both user queries and professional responses. This reflects the complexity of real-world legal consultations, ranging from concise inquiries to detailed questions and explanations. Figure 3 further illustrates the diversity of question types in ViLegalExpert, highlighting the heterogeneous legal information needs captured by the dataset.

## 4 Experimental Setup

We evaluate ViLegalExpert on its three supported tasks: legal information retrieval (IR), extractive question answering (Extractive QA), and abstractive question answering (Abstractive QA). The experiments are designed to assess the benchmark across diferent modeling paradigms, ranging from conventional retrieval and machine reading comprehension methods to pretrained language models (PLMs) and recent instruction-tuned large language models (LLMs).

![](images/1ae513fc13cccf02258497ff663e0fdb94eb65c803c0f2ac170092f54e3d47b1.jpg)

![](images/7dcb2e70bf0fc3947ed9e0af4016b4567b366035808d4113c1f930c0fe4635df.jpg)  
Fig. 2. Token-length distributions of questions and answers in ViLegalExpert.

## 4.1 Experimental Configuration

All experiments are implemented in PyTorch and conducted on a single NVIDIA A100 GPU with 40 GB of memory. Trainable models are optimized on the training set, with the best checkpoints selected based on development-set performance. We use publicly available checkpoints and their corresponding tokenizers, with model-specific hyperparameters following the recommended configurations.

For QA, models are provided with the question and its associated legal context to either extract an answer span or generate a free-form response. For IR, each question is used as a query over the legal corpus, with expert-verified legal provisions serving as relevance annotations. Instruction-tuned LLMs are evaluated using a consistent prompting template to ensure comparability across models.

## 4.2 Baselines and Evaluation Metrics

We evaluate representative baselines for all three tasks supported by ViLegalExpert: legal information retrieval, extractive question answering, and abstractive question answering. The selected methods cover conventional task-specific architectures, pretrained language models, retrieval-augmented approaches, and recent instruction-tuned large language models.

![](images/29a1ce5f5b3c62b03ecdd5054f06ef2da4275b9241b4fb9271534eff02f41cbe.jpg)  
Fig. 3. Distribution of legal domains in the proposed dataset.

Legal Information Retrieval. We evaluate several representative retrieval approaches, including MiniRAG [7], LightRAG [9], HippoRAG [10], and RAP-TOR [27]. These methods represent recent retrieval and retrieval-augmented architectures that exploit diferent strategies for organizing and retrieving information from large document collections. We additionally evaluate BM25 [25] under sparse, dense, and hybrid retrieval settings [18]. All methods retrieve from the same legal corpus and are evaluated against the expert-verified legal references in ViLegalExpert.

Extractive Question Answering. For extractive QA, we consider two groups of baselines. The first consists of established machine reading comprehension models, including QANet [37], DEEP-CASCADE [36], and TD-SAN [41]. The second consists of Transformer-based pretrained language models, including LEGAL-BERT [4], ViDeBERTa [30], multilingual BERT (mBERT) [24], and PhoBERT [20]. These models enable comparisons among task-specific MRC architectures, legal-domain pretrained representations, multilingual models, and Vietnamesespecific pretrained language models.

Abstractive Question Answering. For abstractive QA, we evaluate models from three representative families. First, we consider conventional MRC and neural generation approaches, including CPG [29], S-NET [28], LatentQA [2], DCMM+ [38], MultiStyle [22], GAQA [26], and CHIME [17]. Second, we evaluate pretrained sequence-to-sequence language models, including ViT5 [23], BARTpho [31], mT5 [35], and mBART [14]. Finally, we evaluate recent opensource instruction-tuned LLMs, including Qwen2.5-7B-Instruct, Llama3.1-8B-Instruct, Llama3-8B-Instruct, Qwen2.5-14B-Instruct, and Mistral-7B-Instructv0.3. This diverse set of baselines allows us to assess the progression from specialized neural architectures to pretrained sequence-to-sequence models and modern instruction-tuned LLMs on real-world Vietnamese legal questions.

Evaluation Metrics. For both extractive and abstractive QA, we report ROUGE-L, METEOR, and BERTScore to measure lexical overlap and semantic similarity between predicted and reference answers. For legal IR, we use F1, Precision, and Mean Reciprocal Rank (MRR) to evaluate retrieval accuracy and ranking quality. Higher values indicate better performance for all metrics.

## 5 Results

## 5.1 Question Answering Results

Table 3. Extractive QA results. R-L, MET, and BS denote ROUGE-L, METEOR, and BERTScore, respectively. Best results within each model group are in bold.
<table><tr><td>Method R-L</td><td>MET</td><td>BS</td></tr><tr><td>MRC Models</td><td></td><td></td></tr><tr><td>QANet [37] 35.06 DEEP-ČASCADE [36] 34.80</td><td>40.12 40.10</td><td>61.06 72.09</td></tr><tr><td>TD-SAN [41] 31.34 Pre-trained Language Models</td><td>36.63</td><td>70.37</td></tr><tr><td>LEGAL-BERT [4] 62.48</td><td></td><td>84.08</td></tr><tr><td>ViDeBERTa [30] 62.79</td><td>64.60 66.08</td><td>86.27</td></tr><tr><td>mBERT [24] 62.36</td><td>64.26</td><td>86.17</td></tr><tr><td>PhoBERT [20] 50.68</td><td>53.54</td><td>81.84</td></tr></table>

Extractive Question Answering. Table 3 shows a substantial performance gap between conventional MRC architectures and pretrained language models. Among the MRC models, QANet achieves the highest ROUGE-L (35.06) and METEOR (40.12), while DEEP-CASCADE obtains the highest BERTScore of 72.09. In contrast, all evaluated pretrained language models achieve considerably stronger results, demonstrating the efectiveness of contextualized pretrained representations for Vietnamese legal answer extraction.

Among the pretrained models, ViDeBERTa performs best across all three metrics, achieving 62.79 ROUGE-L, 66.08 METEOR, and 86.27 BERTScore. mBERT closely follows in ROUGE-L and BERTScore, whereas LEGAL-BERT also demonstrates competitive performance despite not being specifically pretrained for Vietnamese. Interestingly, PhoBERT performs below the other pretrained models, suggesting that Vietnamese-specific pretraining alone does not necessarily provide an advantage for legal answer extraction. Overall, the results indicate that pretrained contextual representations provide a substantial improvement over conventional MRC architectures, while leaving further room for domain- and task-specific adaptation.

Table 4. Abstractive QA results. R-L, MET, and BS denote ROUGE-L, METEOR, and BERTScore, respectively. Best results within each model group are in bold.
<table><tr><td>Method</td><td>R-L</td><td>MET</td><td>BS</td></tr><tr><td colspan="4">MRC Models</td></tr><tr><td>CPG [29]</td><td>29.85</td><td>26.15</td><td>73.22</td></tr><tr><td>S-NET [28]</td><td>26.39</td><td>24.02</td><td>70.42</td></tr><tr><td>LatentQA [2]</td><td>26.45</td><td>24.30</td><td>71.26</td></tr><tr><td>DCMM+ [38]</td><td>25.84</td><td>23.75</td><td>70.73</td></tr><tr><td>MultiStyle [22]</td><td>25.06</td><td>23.11</td><td>70.35</td></tr><tr><td>GAQA [26]</td><td>25.93</td><td>23.82</td><td>70.85</td></tr><tr><td>CHIME [17]</td><td>25.59</td><td>23.55</td><td>70.84</td></tr><tr><td colspan="4">Pre-trained Language Models</td></tr><tr><td>ViT5 [23] BARTpho [31]</td><td>40.59 44.36</td><td>44.74 49.40</td><td>76.37 78.52</td></tr><tr><td>mT5 [35]</td><td>30.55</td><td>30.17</td><td>72.82</td></tr><tr><td>mBART [15]</td><td>39.34</td><td>45.26</td><td>76.20</td></tr><tr><td colspan="4">Open-source Language Models</td></tr><tr><td>Qwen2.5-7B-Instruct</td><td>24.20</td><td>22.57</td><td>71.34</td></tr><tr><td>Llama3.1-8B-Instruct</td><td>25.15</td><td>23.20</td><td>72.10</td></tr><tr><td>Llama3-8B-Instruct</td><td>28.31</td><td>28.69</td><td>73.00</td></tr><tr><td>Qwen2.5-14B-Instruct</td><td>23.04</td><td>21.44</td><td>70.12</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>Mistral-7B-Instruct-v0.3</td><td>12.40</td><td>12.94</td><td>63.98</td></tr></table>

Abstractive Question Answering. Table 4 presents a diferent performance pattern. Among conventional MRC and generation models, CPG achieves the strongest overall results, with 29.85 ROUGE-L, 26.15 METEOR, and 73.22 BERTScore. Nevertheless, these models are substantially outperformed by pretrained sequence-to-sequence language models. BARTpho achieves the best performance across all evaluated approaches, reaching 44.36 ROUGE-L, 49.40 ME-TEOR, and 78.52 BERTScore. ViT5 also performs strongly, particularly in ME-TEOR and BERTScore.

Surprisingly, the evaluated instruction-tuned LLMs do not outperform the smaller pretrained sequence-to-sequence models. Llama3-8B-Instruct is the strongest model in this group, with 28.31 ROUGE-L, 28.69 METEOR, and 73.00 BERTScore, but remains substantially behind BARTpho. Increasing model size also does

not consistently improve performance: Qwen2.5-14B-Instruct performs below its 7B counterpart across all three metrics. These results suggest that general instruction-following capability and model scale alone are insuficient for Vietnamese legal question answering. In contrast, language-specific sequence-tosequence pretraining appears particularly efective when models are adapted to the target task.

## 5.2 Legal Information Retrieval Results

Table 5. Legal information retrieval results. P denotes Precision. Best overall results are in bold.
<table><tr><td>Method</td><td>F1</td><td>MRR</td><td>P</td></tr><tr><td>MiniRAG [7]</td><td>12.28</td><td>17.73</td><td>8.32</td></tr><tr><td>LightRAG [9]</td><td>15.96</td><td>20.53</td><td>9.58</td></tr><tr><td>HippoRAG [10]</td><td>15.64</td><td>12.63</td><td>6.40</td></tr><tr><td>RAPTOR [27]</td><td>14.80</td><td>21.44</td><td>10.03</td></tr><tr><td>BM25 (Sparse) [25]</td><td>16.03</td><td>23.23</td><td>10.86</td></tr><tr><td>BM25 (Dense) [18]</td><td>12.40</td><td>17.51</td><td>8.41</td></tr><tr><td>BM25 (Hybrid)</td><td>18.06</td><td>26.16</td><td>12.24</td></tr></table>

Table 5 shows that retrieving supporting legal evidence for real-world consultation questions remains challenging. Among the recent retrieval and RAG-based approaches, LightRAG achieves the highest F1 score (15.96), while RAPTOR obtains the highest MRR (21.44) and Precision (10.03). HippoRAG performs comparably in F1 but exhibits a considerably lower MRR and Precision, indicating diferences in the ability of these methods to rank relevant legal evidence near the top of the retrieved results.

The sparse BM25 setting provides a strong baseline, achieving 16.03 F1, 23.23 MRR, and 10.86 Precision and outperforming all evaluated RAG-based retrieval methods on these metrics. This result suggests that lexical matching remains particularly efective for Vietnamese legal retrieval. Legal texts contain highly standardized terminology and recurring lexical expressions, providing useful exact-match signals that can be dificult to preserve using semantic representations alone.

The hybrid retrieval setting achieves the best overall performance, with 18.06 F1, 26.16 MRR, and 12.24 Precision. Its consistent improvement over both sparse and dense settings indicates that lexical and semantic signals are complementary: sparse retrieval efectively captures terminology-level correspondence, whereas dense representations can identify semantically related evidence when exact lexical overlap is insuficient. Nevertheless, the relatively low absolute performance of all retrieval approaches highlights the dificulty of mapping naturally expressed citizen questions to the appropriate provisions in formal legal documents.

Overall, across the three tasks, the results reveal diferent strengths of current modeling paradigms. Pretrained encoder models substantially outperform conventional MRC architectures for extractive QA, while Vietnamese-oriented sequence-to-sequence models are particularly efective for abstractive QA and outperform the evaluated instruction-tuned LLMs. For legal information retrieval, lexical retrieval remains a strong baseline, with the hybrid setting providing further improvements by combining lexical and semantic signals. At the same time, none of the evaluated approaches achieves consistently strong performance across all tasks, demonstrating the challenges posed by real-world Vietnamese legal consultation data and leaving substantial room for future advances.

## 6 Conclusion

We introduce ViLegalExpert, a large-scale Vietnamese benchmark for legal retrieval and question answering constructed from authentic citizen–lawyer consultations with expert-verified legal evidence. Covering retrieval, extractive QA, and abstractive QA, our experiments reveal substantial challenges in mapping real-world legal questions to authoritative provisions and generating correctly grounded answers. We hope ViLegalExpert will support future research toward more reliable and evidence-grounded Vietnamese Legal AI.

## References

1. Anand, D., Wagh, R.: Efective deep learning approaches for summarization of legal texts. J. King Saud Univ. Comput. Inf. Sci. 34(5), 2141–2150 (May 2022). https://doi.org/10.1016/j.jksuci.2019.11.015, https://doi.org/10.1016/j.jksuci.2019.11.015

2. Bi, B., Wu, C., Yan, M., Wang, W., Xia, J., Li, C.: Generating well-formed answers by machine reading with stochastic selector networks. Proceedings of the AAAI Conference on Artificial Intelligence 34(05), 7424–7431 (Apr 2020). https://doi.org/10.1609/aaai.v34i05.6238, https://ojs.aaai.org/index.php/AAAI/article/view/6238

3. Chalkidis, I., Fergadiotis, E., Malakasiotis, P., Androutsopoulos, I.: Large-scale multi-label text classification on EU legislation. In: Korhonen, A., Traum, D., M\`arquez, L. (eds.) Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. pp. 6314–6322. Association for Computational Linguistics, Florence, Italy (Jul 2019). https://doi.org/10.18653/v1/P19- 1636, https://aclanthology.org/P19-1636/

4. Chalkidis, I., Fergadiotis, M., Malakasiotis, P., Aletras, N., Androutsopoulos, I.: Legal-bert: The muppets straight out of law school (2020), https://arxiv.org/abs/2010.02559

5. Chen, A., Yao, F., Zhao, X., Zhang, Y., Sun, C., Liu, Y., Shen, W.: Equals: A real-world dataset for legal question answering via reading chinese laws. pp. 71–80 (09 2023). https://doi.org/10.1145/3594536.3595159

6. Do, D.T., Luu, S.T., Pham, T., Vo, T., Chu, N.H., Chu, Q.H., Nguyen, C., Nguyen, M., Trieu, A., Nguyen, D., Tran, T., Nguyen, C., Nguyen, H., Nguyen, C., Le, N.K., Nguyen, D.H., Dang, B., Nguyen, P., Nguyen, H.T., Tran, V., Nguyen,

L.M.: A summary of the alqac 2024 competition. In: 2024 16th International Conference on Knowledge and System Engineering (KSE). pp. 422–427 (2024). https://doi.org/10.1109/KSE63888.2024.11063484

7. Fan, T., Wang, J., Ren, X., Huang, C.: Minirag: Towards extremely simple retrieval-augmented generation. arXiv preprint arXiv:2501.06713 (2025)

8. Goebel, R., Kano, Y., Kim, M.Y., Kwan, C., Satoh, K., Yamada, H., Yoshioka, M.: An overview of the coliee 2025 competition: Legal case law and statute law information retrieval and entailment. In: Proceedings of the Twentieth International Conference on Artificial Intelligence and Law. p. 506–515. ICAIL ’25, Association for Computing Machinery, New York, NY, USA (2026). https://doi.org/10.1145/3769126.3785016, https://doi.org/10.1145/3769126.3785016

9. Guo, Z., Xia, L., Yu, Y., Ao, T., Huang, C.: Lightrag: Simple and fast retrievalaugmented generation. arXiv preprint arXiv:2410.05779 2(3) (2024)

10. Guti´errez, B.J., Shu, Y., Gu, Y., Yasunaga, M., Su, Y.: Hipporag: Neurobiologically inspired long-term memory for large language models. Advances in neural information processing systems 37, 59532–59569 (2024)

11. Kanapala, A., Pal, S., Pamula, R.: Text summarization from legal documents: a survey. Artif. Intell. Rev. 51(3), 371–402 (Mar 2019). https://doi.org/10.1007/s10462- 017-9566-2, https://doi.org/10.1007/s10462-017-9566-2

12. Khazaeli, S., Punuru, J., Morris, C., Sharma, S., Staub, B., Cole, M., Chiu-Webster, S., Sakalley, D.: A free format legal question answering system. In: Aletras, N., Androutsopoulos, I., Barrett, L., Goanta, C., Preotiuc-Pietro, D. (eds.) Proceedings of the Natural Legal Language Processing Workshop 2021. pp. 107–113. Association for Computational Linguistics, Punta Cana, Dominican Republic (Nov 2021). https://doi.org/10.18653/v1/2021.nllp-1.11, https://aclanthology.org/2021.nllp-1.11/

13. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., K¨uttler, H., Lewis, M., Yih, W., Rockt¨aschel, T., Riedel, S., Kiela, D.: Retrieval-augmented generation for knowledge-intensive NLP tasks. CoRR abs/2005.11401 (2020), https://arxiv.org/abs/2005.11401

14. Liu, Y., Gu, J., Goyal, N., Li, X., Edunov, S., Ghazvininejad, M., Lewis, M., Zettlemoyer, L.: Multilingual denoising pre-training for neural machine translation (2020), https://arxiv.org/abs/2001.08210

15. Liu, Y., Gu, J., Goyal, N., Li, X., Edunov, S., Ghazvininejad, M., Lewis, M., Zettlemoyer, L.: Multilingual denoising pre-training for neural machine translation. Transactions of the Association for Computational Linguistics 8, 726–742 (2020), https://api.semanticscholar.org/CorpusID:210861178

16. Louis, A., Spanakis, G.: A statutory article retrieval dataset in French. In: Muresan, S., Nakov, P., Villavicencio, A. (eds.) Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 6789–6803. Association for Computational Linguistics, Dublin, Ireland (May 2022). https://doi.org/10.18653/v1/2022.acl-long.468, https://aclanthology.org/2022.acl-long.468/

17. Lu, J., Pergola, G., Gui, L., Li, B., He, Y.: CHIME: Cross-passage hierarchical memory network for generative review question answering. In: Scott, D., Bel, N., Zong, C. (eds.) Proceedings of the 28th International Conference on Computational Linguistics. pp. 2547–2560. International Committee on Computational Linguistics, Barcelona, Spain (Online) (Dec 2020). https://doi.org/10.18653/v1/2020.coling-main.229, https://aclanthology.org/2020.coling-main.229/

18. Ma, X., Sun, K., Pradeep, R., Lin, J.: A replication study of dense passage retriever (2021), https://arxiv.org/abs/2104.05740

19. Monroy, A., Calvo, H., Gelbukh, A.: NLP for Shallow Question Answering of Legal Documents Using Graphs, vol. 5449, pp. 498–508 (03 2009). https://doi.org/10.1007/978-3-642-00382-0˙40

20. Nguyen, D.Q., Nguyen, A.T.: Phobert: Pre-trained language models for vietnamese (2020), https://arxiv.org/abs/2003.00744

21. Nguyen, T.M., Nguyen, H.T., Dao, T.K., Phan, X.H., Nguyen, H.T., Vuong, T.H.Y.: Vlqa: The first comprehensive, large, and high-quality vietnamese dataset for legal question answering (2025), https://arxiv.org/abs/2507.19995

22. Nishida, K., Saito, I., Nishida, K., Shinoda, K., Otsuka, A., Asano, H., Tomita, J.: Multi-style generative reading comprehension (2019), https://arxiv.org/abs/1901.02262

23. Phan, L., Tran, H., Nguyen, H., Trinh, T.H.: ViT5: Pretrained text-to-text transformer for Vietnamese language generation. In: Ippolito, D., Li, L.H., Pacheco, M.L., Chen, D., Xue, N. (eds.) Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies: Student Research Workshop. pp. 136–142. Association for Computational Linguistics, Hybrid: Seattle, Washington + Online (Jul 2022). https://doi.org/10.18653/v1/2022.naacl-srw.18, https://aclanthology.org/2022.naacl-srw.18/

24. Pires, T., Schlinger, E., Garrette, D.: How multilingual is multilingual BERT? In: Korhonen, A., Traum, D., M\`arquez, L. (eds.) Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. pp. 4996– 5001. Association for Computational Linguistics, Florence, Italy (Jul 2019). https://doi.org/10.18653/v1/P19-1493, https://aclanthology.org/P19-1493/

25. Robertson, S., Zaragoza, H.: The probabilistic relevance framework: Bm25 and beyond. Found. Trends Inf. Retr. 3(4), 333–389 (Apr 2009). https://doi.org/10.1561/1500000019, https://doi.org/10.1561/1500000019

26. Roy, K., Balapanuru, V., Nayak, T., Goyal, P.: Investigating the generative approach for question answering in E-commerce. In: Malmasi, S., Rokhlenko, O., Uefing, N., Guy, I., Agichtein, E., Kallumadi, S. (eds.) Proceedings of the Fifth Workshop on e-Commerce and NLP (ECNLP 5). pp. 210–216. Association for Computational Linguistics, Dublin, Ireland (May 2022). https://doi.org/10.18653/v1/2022.ecnlp-1.24, https://aclanthology.org/2022.ecnlp-1.24/

27. Sarthi, P., Abdullah, S., Tuli, A., Khanna, S., Goldie, A., Manning, C.: Raptor: Recursive abstractive processing for tree-organized retrieval. In: International Conference on Learning Representations. vol. 2024, pp. 32628–32649 (2024)

28. Tan, C., Wei, F., Yang, N., Du, B., Lv, W., Zhou, M.: S-net: From answer extraction to answer generation for machine reading comprehension (2018), https://arxiv.org/abs/1706.04815

29. Tay, Y., Wang, S., Luu, A.T., Fu, J., Phan, M.C., Yuan, X., Rao, J., Hui, S.C., Zhang, A.: Simple and efective curriculum pointer-generator networks for reading comprehension over long narratives. In: Korhonen, A., Traum, D., M\`arquez, L. (eds.) Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. pp. 4922–4931. Association for Computational Linguistics, Florence, Italy (Jul 2019). https://doi.org/10.18653/v1/P19- 1486, https://aclanthology.org/P19-1486/

30. Tran, C.D., Pham, N.H., Nguyen, A.T., Hy, T.S., Vu, T.: ViDeBERTa: A powerful pre-trained language model for Vietnamese. In: Vlachos, A., Augenstein, I. (eds.) Findings of the Association for Computational Linguistics: EACL 2023. pp. 1071–1078. Association for Computational Linguistics, Dubrovnik, Croatia (May 2023). https://doi.org/10.18653/v1/2023.findingseacl.79, https://aclanthology.org/2023.findings-eacl.79/

31. Tran, N.L., Le, D.M., Nguyen, D.Q.: Bartpho: Pre-trained sequence-to-sequence models for vietnamese (2022), https://arxiv.org/abs/2109.09701

32. Trautmann, D., Ostapuk, N., Grail, Q., Pol, A., Bonifazi, G., Gao, S., Gajek, M.: Measuring the groundedness of legal question-answering systems. In: Aletras, N., Chalkidis, I., Barrett, L., Goant<sub>,</sub>˘a, C., Preot<sub>,</sub>iuc-Pietro, D., Spanakis, G. (eds.) Proceedings of the Natural Legal Language Processing Workshop 2024. pp. 176–186. Association for Computational Linguistics, Miami, FL, USA (Nov 2024). https://doi.org/10.18653/v1/2024.nllp-1.14, https://aclanthology.org/2024.nllp-1.14/

33. Tuggener, D., von D¨aniken, P., Peetz, T., Cieliebak, M.: LEDGAR: A large-scale multi-label corpus for text classification of legal provisions in contracts. In: Calzolari, N., B´echet, F., Blache, P., Choukri, K., Cieri, C., Declerck, T., Goggi, S., Isahara, H., Maegaard, B., Mariani, J., Mazo, H., Moreno, A., Odijk, J., Piperidis, S. (eds.) Proceedings of the Twelfth Language Resources and Evaluation Conference. pp. 1235–1241. European Language Resources Association, Marseille, France (May 2020), https://aclanthology.org/2020.lrec-1.155/

34. Vold, A., Conrad, J.G.: Using transformers to improve answer retrieval for legal questions. In: Proceedings of the eighteenth international conference on artificial intelligence and law. pp. 245–249 (2021)

35. Xue, L., Constant, N., Roberts, A., Kale, M., Al-Rfou, R., Siddhant, A., Barua, A., Rafel, C.: mt5: A massively multilingual pre-trained text-to-text transformer (2021), https://arxiv.org/abs/2010.11934

36. Yan, M., Xia, J., Wu, C., Bi, B., Zhao, Z., Zhang, J., Si, L., Wang, R., Wang, W., Chen, H.: A deep cascade model for multi-document reading comprehension. Proceedings of the AAAI Conference on Artificial Intelligence 33(01), 7354–7361 (2019). https://doi.org/10.1609/aaai.v33i01.33017354, http://dx.doi.org/10.1609/aaai.v33i01.33017354

37. Yu, A.W., Dohan, D., Luong, M.T., Zhao, R., Chen, K., Norouzi, M., Le, Q.V.: Qanet: Combining local convolution with global self-attention for reading comprehension. arXiv preprint arXiv:1804.09541 (2018)

38. Zhang, S., Zhao, H., Wu, Y., Zhang, Z., Zhou, X., Zhou, X.: Dcmn+: Dual co-matching network for multi-choice reading comprehension (2020), https://arxiv.org/abs/1908.11511

39. Zhong, H., Xiao, C., Tu, C., Zhang, T., Liu, Z., Sun, M.: Jec-qa: A legal-domain question answering dataset (2019), https://arxiv.org/abs/1911.12011

40. Zhong, H., Xiao, C., Tu, C., Zhang, T., Liu, Z., Sun, M.: How does NLP benefit legal system: A summary of legal artificial intelligence. In: Jurafsky, D., Chai, J., Schluter, N., Tetreault, J. (eds.) Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. pp. 5218–5230. Association for Computational Linguistics, Online (Jul 2020). https://doi.org/10.18653/v1/2020.aclmain.466, https://aclanthology.org/2020.acl-main.466/

41. Zhuang, Y., Wang, H.: Token-level dynamic self-attention network for multipassage reading comprehension. In: Proceedings of the 57th annual meeting of the association for computational linguistics. pp. 2252–2262 (2019)