# WHEN MODELS DON'T MANIPULATE MANIFOLDS: THE GEOMETRY OF A COMPARISON TASK

Sai Sumedh R. Hindupur1, Hadas Orgad2, Thomas Fel³ and Demba Ba1,2 School of Engineering and Applied Science, Harvard University¹, Kempner Institute, Harvard University², Goodfire AI³

## ABSTRACT

One of the current premises of mechanistic interpretability research is that detailed accounts of the geometry of neural network representations can tell us how models perform computations, and how to effectively intervene on them. While low dimensional manifolds have been observed for multiple concepts in the literature (e.g. numbers encoded on helices, days of the week on a circle, ...), with structure believed to reflect properties of data and tasks, the extent to which models rely on them for computation, and how they manipulate them, remains unclear. We characterize precisely the geometry of computation in a number-comparison task, as an abstraction of comparison for decision making, and how models utilize geometry in an elegant fashion to implement it. Specifically, we study the causal geometry of number comparison in Qwen2.5-7B-Instruct, a capable and widely studied open-weight model, and find Qwen largely uses linear representations of numbers despite the presence of curved geometry. To compare two numbers, the model first encodes each number along a vector and adds the two representations using attention and the residual connection, bringing them into a shared space in the residual stream. Then, the model uses MLP neurons to compare the pair of numbers on local regions in this shared space, which correspond to smaller intervals of input numbers, and combines these to obtain the position of the maximum. In fact, this reliance on linear representations for comparison also persists when the model compares three numbers. Our findings demonstrate that the manifold hypothesis can co-exist with linear representations: while concepts that are ordered may have manifold structure in representations, the model may use an underlying linear structure of the concept in certain computations. 1

## 1 INTRODUCTION

How do language models represent concepts internally, and how do they use these representations to perform computations? The linear representation hypothesis (Park et al., 2023; Elhage et al., 2021) posited that models represent ordered concepts as one-dimensional subspaces, which found evidence across diverse applications (Zhu et al., 2024; Marks & Tegmark, 2024; Voynov & Babenko, 2020; Tigges et al., 2023; Lee et al., 2024). Recent work has challenged this perspective and found that these concepts live on low dimensional manifolds – a phenomenon known as the manifold hypothesis: examples include number helices (Kantamneni & Tegmark, 2025), curved manifolds for dates/years (Modell et al., 2025), character count manifolds (Gurnee et al., 2026), as well as age and temperature manifolds (Bhalla et al., 2026). Where does this manifold structure come from? Theoretical analyses suggest that the manifold structure arises from symmetries in data (Karkada et al., 2026) or task symmetries (Hwang & Park, 2026). These insights have also revealed how steering along manifolds can control model behavior (Wurgaft et al., 2026). However, given a specific computation, which aspects of the representation structure the model uses to perform the computation remains an open question. Does the model manipulate nonlinear concept manifolds, as Gurnee et al. (2026) show using character count manifolds on a line-breaking task, or does the model use simpler linear representations for certain tasks despite the presence of manifold structure?

![](images/80640c06d806f87a469412c38d71ab665cbaba0e1bb1d25136b41376b5d630c0.jpg)  
Figure 1: Linear number representations are used by language model Qwen for comparison. When asked to compare numbers, the model first represents them on a nonlinear manifold, but uses structure along a single linear direction in computation. It combines information about the two numbers by adding the corresponding linear representations using a single attention head and a residual connection, creating a shared space which enables comparison. On this shared space, the model identifies which number is larger using MLP neurons across two layers, and uses the answer position to read out the answer to the max task.

Elucidating how models manipulate internal representations to perform computations has become an active area of research. Framed as such, this question amounts to investigating the algorithmic level in Marr's levels of analysis (Marr, 2010), which claim that a system can be understood at the computational (abstract, behavioral), algorithmic (variables and how they are manipulated) and implementation levels (low-level implementation of the algorithm). Mechanistic interpretability research, despite having largely focused on finding circuits of computation, namely the implementation level, has also become interested in the algorithmic level (Geiger et al., 2021) which operates at a higher level of abstraction. Multiple examples of algorithmic understanding of models exist, including trigonometry on circular/ helical structures for addition (Kantamneni & Tegmark, 2025), an addition mechanism using Fourier features for multiple concepts such as week days, months, etc (Feucht et al., 2026), and explaining attention heads with python programs (Hayes et al., 2026). Our work contributes to this burgeoning and exciting line of work on algorithmic interpretability.

Motivated by the above lines of inquiry, we ask how language models implement comparison using representation geometry. Comparison is a widely useful operation for intelligent systems, and language models in particular. It is an integral component of weighing options and making decisions especially using numerical or ordered concepts (which can be represented on a line; see (Gärdenfors, 2000)). For example, queries like "Which of these apartments is closest to my office?", "Which quarter had the worst sales in this five-year report?" require performing comparison of the specified objects using the specified attributes. Despite reports of nonlinear, helical representations of numbers (Kantamneni & Tegmark, 2025), the aspects of number representations models use for the purpose of comparison, and how do they manipulate them to implement the operation remains poorly understood.

Previous works have studied how models perform comparison. (Hanna et al., 2023) characterize in great detail a comparison circuit in GPT-2. However, they do not establish causal role of representation geometry in comparison by acknowledging that "GPT2's structured number representations may be relevant to its greater-than ability. However, our experiments struggle to prove this causally" (quoted from Hanna et al. (2023)). (El-Shangiti et al., 2025) found a linear subspace which causally affects model outputs, but their analysis does not concern the mechanism by which the model compares values represented in this subspace. (Yuchi et al., 2026) study mixed notation number comparison and compare behavioral accuracy with classifier performance. Taken together, these works have focused on circuit discovery, model performance, or probing accuracy: they however do not study causal representation geometry and how it is used by the model to perform comparison. We provide a detailed geometric account of the algorithm language models employ to compare numbers (Fig. 1). In addition to being descriptive, our account is causal: we can predict how causal interventions will affect model behavior.

Concretely, we make the following contributions:

• Reconciling linear representations and manifolds: We demonstrate how linear representations of numbers are causally involved in the model's number comparison, despite the presence of underlying curved manifold structure.

• Algorithm for number comparison using linear representations: We further show how the model manipulates number representations to perform comparison: by additive mixing of individual number directions, followed by local comparisons (for specific intervals of input numbers) which are then combined to give the global comparison answer.

Algorithm 1 Pairwise Number Comparisons   
Require: numbers $y _ { 1 } , y _ { 2 }$ at times $t = 1 , 2$ resp.   
if $\mathbf { \dot { \tau } } t = 1$ then   
store $y _ { 1 }$ as ${ \pmb x } _ { l } ^ { 1 } = { \bf u } f _ { 1 } ^ { \prime } ( y _ { 1 } )$ Fig. 2 (a, b)   
else if $t = 2$ then   
store $y _ { 2 }$ as ${ \pmb x } _ { l } ^ { 2 } = { \pmb v } _ { 2 } f _ { 2 } ( y _ { 2 } )$ Fig. 2 (c)   
copy $\mathbf { \Delta } \mathbf { x } _ { l } ^ { 1 }$ , transform as $v _ { 1 } f _ { 1 } ( y _ { 1 } ) ,$ Done by Attention Head(s), Fig. 2(b, d)   
add to residual $\pmb { x } _ { l + 1 } ^ { 2 } = \pmb { x } _ { l } ^ { 2 } + \pmb { v } _ { 1 } f _ { 1 } ( y _ { 1 } )$ Residual Stream, Fig. 3 (a–c)   
divide span $( v _ { 1 } , v _ { 2 } )$ into regions $\{ \mathcal { R } _ { i } \}$ ▷ MLP neurons, Fig. 4(a, b)   
compare $y _ { 1 } , y _ { 2 }$ in $\{ \mathcal { R } _ { i } \}$ as $\mathsf { \bar { C } } _ { i } = \bar { \mathbb { I } } ( y _ { 1 } > y _ { 2 } ) . \mathbb { I } ( \pmb { x } _ { l + 1 } ^ { 2 } \in \mathcal { R } _ { i } )$ ▶ MLP neuron outputs, Fig. 4(b)   
combine local comparisons $\{ C _ { i } \}$ to get $a n s = \mathbb { I } ( y _ { 1 } > y _ { 2 } )$ ▶ later MLP neurons, Fig. 4(c, d)   
end if

• Extension of the comparison algorithm to longer sequences of numbers: We show how the algorithm using linear number representations extends to three-length sequences.

## 2 PAIRWISE COMPARISON OF NUMBERS

In this section, we first describe the pairwise comparison mechanism in an LLM. We state the algorithm explicitly, and discuss the main steps involved. Furthermore, we provide evidence describing how the model implements this algorithm in subsequent subsections. We extend the algorithm to three number comparisons and include evidence in Section 3.

Notation. Computationally relevant subspaces within model activations are denoted by $\boldsymbol { x } _ { l } ^ { t } .$ , where l denotes the layer index within the model and t denotes time (token position). Numbers present in the input prompt are denoted by $\{ y _ { t } \} . ~ \{ v _ { i } \}$ are directions in model activations, which belong to the same space (same layer and token position). These directions encode numbers, with ${ \mathbf { } } v _ { i }$ encoding yi as ${ { v } _ { i } } f _ { i } ( y _ { i } )$ , where $\dot { f _ { i } }$ may be a nonlinear function of $y _ { i }$

## 2.1 ALGORITHM FOR PAIRWISE COMPARISONS

In line with the known distinction between availability and utility of features (Garg et al., 2026), our claims in the following sections are about linear features for numbers being used by the model for a specific task: comparison. Other tasks and concepts, like addition/ periodic concepts (Feucht et al., 2026; Wurgaft et al., 2026)) may use more intricate manifold structure.

The model encodes each number's magnitude $y _ { i }$ as a nonlinear function $f _ { i } ( y _ { i } )$ along a single direction ${ \boldsymbol { v } } _ { i }$ which is different for each number position. Since the two numbers y1, y2 are provided as inputs at different times (distinct token positions), the model then creates a shared representation from the two numbers by (1) copying information about $y _ { 1 }$ into $y _ { 2 } \cdot \mathrm { s }$ position, and (2) adding together the single number representations. The shared representation is expressed as:

$$
{ \pmb x } = { \pmb v } _ { 1 } f _ { 1 } ( y _ { 1 } ) + { \pmb v } _ { 2 } f _ { 2 } ( y _ { 2 } )\tag{1}
$$

This representation spans a two-dimensional plane span $( v _ { 1 } , v _ { 2 } )$ . Moving along certain directions in this plane (e.g., along $\alpha { \pmb v } _ { 1 } - \beta { \pmb v } _ { 2 }$ for any $\alpha > 0 , \beta \geq 0 )$ changes the probability of the model answering $y _ { 1 }$ as the greater number. The comparison $y _ { 1 } > y _ { 2 }$ can be linearly decoded on this plane. However, the model implements comparison in two stages, as described below.

From this shared representation, MLP neurons first perform local comparisons, identifying and comparing the two numbers $y _ { 1 } , y _ { 2 }$ for specific ranges of individual numbers or their combinations. This occurs because of the gating-based nonlinearity of MLP neurons (SwiGLU for Qwen2.5-7B), whose sigmoid gate and overall expression leads to local regions of activation on intersection with span $_ { 1 } ( \pmb { v } _ { 1 } , \pmb { v } _ { 2 } )$ . The outputs of the neurons are nonlinear on local regions, a consequence of approximate quadratic behavior of the SwiGLU nonlinearity on active regions.

(a)  
![](images/0a9bd237648de829296e3370d6911bdd2f67d4f52ee536c2f583ed8f6c42797f.jpg)

(d)  
![](images/9f3787618cf61c49f54ff4cb3d2ef3f458f644762edef9e5afb14154bfffe583.jpg)

(b)  
![](images/ae79ff6c57db6c29793b00297d0d4a1deb03b3a9104b4e7d94b0146c96a9a4c8.jpg)  
(e)

(c)  
![](images/d5f57463fb420bf22431da44fd9757cc6dcfdb6ff4bb501c3120ea0f65f4814a.jpg)

![](images/a3214d8afe7955e28ba459f9b56ccbc91442a8f437773fbb866324eb5fbd75c6.jpg)

![](images/ede32014bacce816d19e5092864933d326524f8a530497605f3c45746e65eaca.jpg)  
Figure 2: A single causal direction for numbers controls model comparison. (a) A causal direction u found in Layer $1 3 \mathrm { { ^ , s } }$ residual stream causally affects model behavior, despite the presence of curved geometry in the activations, as observed in principal components 1, 3. (b, c) Projection of activations onto the obtained causal directions ${ \bf u } , { \pmb v } _ { 1 } , { \pmb v } _ { 2 }$ encode the number magnitude for a wide range of values. While u is in layer 13 residual stream at the first number $y _ { 1 } \mathbf { \ ' } _ { \mathbf { s } }$ position, ${ \boldsymbol { v } } _ { 1 } , { \boldsymbol { v } } _ { 2 }$ are causal directions encoding $y _ { 1 }$ , Y2 resp. at $y _ { 2 } \cdot \mathrm { s }$ position. ${ \pmb v } _ { 1 }$ is obtained using DAS at an attention head H14's outputs. (d) Attention head 14 in layer 14 attends to the position of the first number $y _ { 1 }$ , irrespective of the number value, serving as a copy head. (e) Among all the attention heads in Layer 14, head H14 is causally involved in the model's computation, showing significantly higher position recovery than others. (f) Patching along u, ${ \boldsymbol { v } } _ { 1 } , { \boldsymbol { v } } _ { 2 }$ has significant causal effects on model behavior, as shown by interchange intervention accuracy (IIA).

Subsequent MLP neurons then combine these local comparisons to create global comparator neurons, which nearly perfectly capture $\mathbb { I } ( y _ { 1 } > y _ { 2 } )$ . These neurons then construct a single direction in the residual stream which encodes the comparison's answer, and causally affects the model's outputs.

The algorithm is stated in Alg. 1, visualized in App. Fig. 9, along with evidence demonstrating each step.

Evidence for the pairwise comparison algorithm. We perform experiments using the open-weight model Qwen2.5-7B-Instruct (Yang et al., 2024). We ask the model to compare pairs of numbers, and provide the model with a one-shot example for output format. The prompt is "Answer in the following format with a single answer. The maximum of 12 and 4 is $^ { l 2 . }$ The maximum of $_ y I$ and $y 2 i s \prime \prime$ . Our causal analyses involve activation patching using interchange interventions, where we patch specific component activations from a model running on a 'clean' prompt to when the model is processing another 'corrupt' prompt ((Meng et al., 2022)). For example, suppose the 'clean' prompt has inputs (60, 10). The corrupt prompt then uses (3, 10) as inputs, and patching activations from clean to corrupt changes the model's outputs on the 'corrupt’ prompt. In this example, since the first number is changed between the clean and corrupt prompts, we refer to this as $\cdot _ { y _ { 1 } }$ perturbed’ in subsequent figures Fig. 2, 3, 4, 5. Further insights into our experimental setup is included in App. A.

The model computes the argmax position and uses that to produce the answer. First, we observe that when patching is successful, patching model activations from a 'clean' run to a 'corrupted' run leads to changing the model's answer position, instead of the value (see App. C.1). For instance, patching activations from (60, 10) into a model processing (3, 10) will make the model answer $^ { 3 , }$ and not 60. Therefore, internal model activations compute the arg max position and use that to produce the answer. We restrict our analysis to the model's computation of the arg max position.

(a)  
![](images/0324342753b2d3f138dc452a42b2e7058088c95d97be5436560e5741e9adc0cd.jpg)  
(b)

![](images/69a5615991868f4427edecc50bc06f412786623b14a2af9eb9f24493d09a69c0.jpg)

![](images/d45d7771d9ef89711d5f9a7fd6553465131096c682d1cd6f22fa2b93a888bfb4.jpg)

(c)  
![](images/786140606478523dd741c24ac510ffd74386462010d9f429992c5aea027063b0.jpg)  
(f)

![](images/a62dfbd966ce919c2a7f044bc27ba5119dd357712b1b68f2e184b4b8cfadfb07.jpg)

![](images/a7035c4d93efe01bc67fe4f621e3e2e6b50d93e7ede3c68b1bc6e86de7e47e4b.jpg)  
Figure 3: The model uses causal number directions ${ \boldsymbol { v } } _ { 1 } , { \boldsymbol { v } } _ { 2 }$ to construct a shared two-dimensional representation encoding the two inputs $y _ { 1 } , y _ { 2 }$ . (a)–(c) The linear span of ${ \pmb v } _ { 1 }$ and ${ \pmb v } _ { 2 }$ shows each number is encoded along its own direction, and the answer to comparison is linearly separable in this shared representation space. (d) By copying $y _ { 1 }$ from u to ${ \pmb v } _ { 1 }$ , the model reduces the alignment between the two number representations: $| \cos ( \pmb { v } _ { 1 } , \pmb { v } _ { 2 } ) | < < | \cos ( \mathbf { u } , \pmb { v } _ { 2 } ) |$ . (e) While the residual stream at layer 13, denoted $\stackrel { \cdot } { z } _ { l = 1 3 } ^ { 2 } .$ encodes $y _ { 2 } , v _ { 1 }$ brings in causally useful information into the $y _ { 2 }$ position. The shared representation is additive: Patching $\mathbf { \sigma } _ { v _ { 1 } \& v _ { 2 } }$ together (two rank-one patches) nearly matches the IIA of the rank-two patch onto the ${ \boldsymbol { v } } _ { 1 } , { \boldsymbol { v } } _ { 2 }$ plane $( \mathrm { i . e . , \ s p a n } ( \pmb { v } _ { 1 } , \pmb { v } _ { 2 } ) )$ . The plane itself captures as much information as the next layer 14 residual stream. (f) Moving around the ${ \pmb v } _ { 1 }$ , v2 plane, which is a two-dimensional plane in the 3,584-dimensional residual stream, is sufficient to change model behavior predictably.

We demonstrate the pairwise comparison algorithm by analyzing individual number representations (Fig. 2), which combine to form the shared representation (Fig. 3). MLP neurons then perform the comparison in two stages, first locally and then globally (Fig. 4).

## 2.2 INDIVIDUAL NUMBER REPRESENTATIONS

Setup. Using 2000 ordered pairs of two-digit numbers, which include 1000 unique randomly chosen pairs $( a , b )$ and their reflections $( b , a )$ , we collect model activations at all layers and all token positions while processing the input prompt. The pairs a, b are chosen to have distinct leading digits, so that the effect of patching is visible at the model logits since tokens are individual digits (two digit numbers having different first digits are a large fraction of all possible pairs, $\sim 9 0 \% )$ . We use both PCA and Distributed Alignment Search (DAS) (Geiger et al., 2024) to find causally relevant directions encoding each number $y _ { 1 } , y _ { 2 } \colon$ we employ DAS whenever the principal components are not causally relevant.

Observations. Fig. 2a shows that despite nonlinear manifold structure of number representations (shown in PC1-PC3 projection), there exists a direction u in layer $1 3 \mathrm { { ^ , s } }$ residual stream which causally affects the model's answer, as measured by Interchange Intervention Accuracy (IIA, (Geiger et al., 2021)) (Fig. 2f). Note that we compute IIA using the immediate next token generated by the model. However, the IIA scores are very similar for patching along directions and subspaces of interest even when computed using the entire number generated by the model (see App. C.7). A single attention head H14 of layer 14 copies information about $y _ { 1 }$ from layer 13 in token position $y _ { 1 }$ to y2. It consistently attends to the y1 position (Fig. 2 d), and patching this head's output has the highest effect on the model's logits, as measured using position recovery (a modified version of recovery (Meng et al., 2022) that patching changes the model's answer position instead of value).

![](images/fc70d93e9da0a0627b06bdd6b16048b23a3fe84e5ee98f40f301e5f0a32d831b.jpg)

(a)  
![](images/dcafd746f7186a25c931bdba5354396564b1e49f6705fabd2f913f4ab4742656.jpg)  
(b)  
(c)

![](images/57b85885d75e1596d67b3ededca22a2a70d4b7dac9c8622fd9d193804cd6a7cd.jpg)

![](images/5ddaed08676b9963822c7149febd12cdfd506ff91f909fc156b37f93ca1f3f2e.jpg)

(d)  
![](images/5d5a0a3af71565f7f9c00bbf05487f25a6e6c2c9c1e216fc6f7ef1796eb598c1.jpg)

![](images/3d79e4182c8f0595990d3642f2a45d414e6ff695057730d9eeec69749095409b.jpg)  
Figure 4: The model compares numbers $y _ { 1 } , y _ { 2 }$ by combining local comparisons on the shared $v _ { 1 } , v _ { 2 }$ plane. (a) Individual neuron weights in layer 14 MLP are specific directions in the $v _ { 1 } , v _ { 2 }$ plane. The y-axis is $\pmb { v } _ { 2 } ^ { \perp }$ , the component of $\mathbf { \boldsymbol { v } } _ { 2 }$ orthogonal to ${ \pmb v } _ { 1 }$ (since ${ \boldsymbol { v } } _ { 1 } , { \boldsymbol { v } } _ { 2 }$ are not exactly orthogonal) (b) Neurons in layer 14 MLP (ranked by attribution scores) localize specific regions of the inputs $y _ { 1 } , y _ { 2 }$ (like neuron #6150), or perform comparisons in localized regions (neuron #9459). (c) Select neurons in layer 15 MLP, which combine the outputs of layer 14 MLP neurons, are global comparators: they respond positively when $y _ { 2 } > y _ { 1 }$ and negatively otherwise. (d) A single direction in the layer 15 residual stream (after layer 15 MLP), which is formed by inputs from global comparator neurons from layer 15 MLP, encodes the position of the answer arg max $( y _ { 1 } , y _ { 2 } )$ . (e) There are 12 comparator neurons (6 each in layer 14, 15 MLPs) which perform comparison: freezing these neurons significantly degrades the IIA achieved by patching in the $( v _ { 1 } , v _ { 2 } )$ plane . The one-dimensional comparison direction in layer 15 (panel (d)) controls model behavior.

$$
P R = \frac { L D _ { p a t c h } - L D _ { c o r r u p t } } { L D _ { c l e a n } - L D _ { c o r r u p t } }\tag{2}
$$

where LD is the logit difference between the pair $( \operatorname* { m i n } ( y _ { 1 } , y _ { 2 } ) , \operatorname* { m a x } ( y _ { 1 } , y _ { 2 } ) )$ where $y _ { 1 } , y _ { 2 }$ are inputs on the corrupt prompt, since upon patching from clean to corrupt prompts, models output $\operatorname* { m i n } ( y _ { 1 } , y _ { 2 } )$ as the answer to the corrupt max-prompt. Note that we call this position recovery to observe how well the model reorganizes its logits to patching and answers with the patched position. The denominator is only meant to provide a rough scale of logit difference. ${ \pmb v } _ { 1 }$ is then obtained as the DAS direction at the output of H14. ${ \pmb v } _ { 2 }$ is obtained as the top principal component from the layer 13 residual stream. Therefore, obtained causal directions u, ${ \boldsymbol { v } } _ { 1 } , { \boldsymbol { v } } _ { 2 }$ encode the values of $y _ { 1 } , y _ { 1 } , y _ { 2 }$ respectively, as shown in panels (b, c). These directions are causal, as shown by their IIA scores in (Fig. 2f).

## 2.3 SHARED REPRESENTATION ENABLES COMPARISON

Using the causal directions of individual numbers ${ \boldsymbol { v } } _ { 1 } , { \boldsymbol { v } } _ { 2 }$ , the model creates a shared representation of both numbers, which is a two-dimensional plane in the pre-MLP residual stream of layer 14. We visualize the projection of model activations onto the linear span of these directions (Fig. 3). This shared representation lives in the pre-MLP residual stream of layer 14. Fig. 3 (a)–(c) show activations in this space encoding each number along its own direction, while allowing linear separation of the two cases $y _ { 1 } > y _ { 2 } , y _ { 1 } < y _ { 2 }$ . The probability contours of the model's answer are approximately parallel to the decision boundary (Fig. 3 f). The directions ${ \boldsymbol { v } } _ { 1 } , { \boldsymbol { v } } _ { 2 }$ are nearly orthogonal (Fig. 3 d) and less aligned than their counterparts which lived at different token positions $( \mathbf { u } , \pmb { v } _ { 2 } )$ . Patching these directions using two rank-one patches (i.e., by projecting along ${ \pmb v } _ { 1 }$ and v2) shows a similar causal effect (IIA) as patching the rank-two shared representation space (by projecting onto span $( \pmb { v } _ { 1 } , \pmb { v } _ { 2 } ) )$ (Fig. 3 e), indicating linear combination of individual number representations create this shared space. This observation is nontrivial because $\mathbf { \boldsymbol { v } } _ { 2 }$ is obtained in layer 13's residual stream and ${ \pmb v } _ { 1 }$ is from head H14's output (which itself uses the layer 13 residual stream in computation), making nonlinear interactions possible.

Algorithm 2 Three-Number Comparisons   
Require: numbers $y _ { 1 } , y _ { 2 } , y _ { 3 }$ at times $t = 1 , 2 , 3$ resp., $\tilde { T }$ is the last token.   
if t = 1 then   
store $y _ { 1 }$ as ${ \pmb x } _ { l } ^ { 1 } = { \bf u } _ { 1 } f _ { 1 } ( y _ { 1 } )$   
else if $t = 2$ then   
store $y _ { 2 }$ as ${ \pmb x } _ { l } ^ { 2 } = { \bf u } _ { 2 } f _ { 2 } ( y _ { 2 } )$   
compute $\alpha _ { 2 } \doteq \mathbb { I } ( y _ { 2 } = \operatorname* { m a x } ( y _ { 1 } , y _ { 2 } ) )$ and store along $\mathbf { d } _ { 2 } \triangleright$ Use Alg. 1; Fig. 5 (d), App. Fig. 26   
else if $t = 3$ then   
store $y _ { 3 }$ as ${ \pmb x } _ { l } ( 3 ) = { \pmb v } _ { 3 } f _ { 3 } ( y _ { 3 } )$ Fig. 5(a)   
copy $\mathbf { \Delta } x _ { l } ^ { 1 } , \mathbf { \Delta } x _ { l } ^ { 2 } .$ transform as $v _ { 1 } f _ { 1 } ( y _ { 1 } ) , v _ { 2 } f _ { 2 } ( y _ { 2 } ) \qquad v$ Done by Attention Heads, App. Fig. 26   
add to residual $\pmb { x } _ { l + 1 } ( 3 ) = \pmb { x } _ { l } ( 3 ) + \pmb { v } _ { 1 } f _ { 1 } ( y _ { 1 } ) + \pmb { v } _ { 2 } f _ { 2 } ( y _ { 2 } )$ Residual Stream, App. Fig. 26   
divide span $( \pmb { v } _ { 1 } , \pmb { v } _ { 2 } , \pmb { v } _ { 3 } )$ into regions $\mathcal { R } _ { i }$ Fig. 5(c)   
compare in $\mathcal { R } _ { i }$ as $\overset { \cdot } { C } _ { i } = \mathbb { I } ( y _ { 3 } = \operatorname* { m a x } ( y _ { 1 } , y _ { 2 } , y _ { 3 } ) ) . \mathbb { I } ( \pmb { x } _ { l + 1 } ( 3 ) \in \mathcal { R } _ { i } ) \ \triangleright \mathbf { M L P }$ neurons, Fig. 5(c)   
combine $\left\{ C _ { i } \right\}$ , get $\alpha _ { 3 } = \mathbb { I } ( y _ { 3 } = \operatorname* { m a x } ( y _ { 1 } , y _ { 2 } , y _ { 3 } ) )$ , store on ${ \bf d } _ { 3 }$ Later MLP neurons, Fig. 5(d)   
else if $t = \tilde { T }$ then   
combine $\alpha _ { 2 } \mathbf { d } _ { 2 } , \alpha _ { 3 } \mathbf { d } _ { 3 }$ to get $\begin{array} { r } { \mathbf { x } _ { \ell ^ { \prime } } ( \tilde { T } ) = \sum _ { t } w _ { t } \alpha _ { t } \mathbf { d } _ { t } } \end{array}$ Attention Head, Fig. 5(e), App. Fig. 26   
Read arg max $\left( y _ { 1 } , y _ { 2 } , y _ { 3 } \right)$ from ${ \pmb x } _ { \ell ^ { \prime } } ( { \tilde { T } } )$   
end if

## 2.4 GLOBAL COMPARISON IS A COMBINATION OF LOCAL COMPARISONS

Setup. We first identify which layer MLPs are involved in the maximum computation by performing freezing experiments. Here, we patch the span(v1, v2) space, but freeze the downstream MLP outputs to remain the same as the no-patch case. This allows us to check the contribution of individual layer MLPs in the subsequent computation: we expect that freezing important MLPs will result in a significant drop in the patching effectiveness. For relevant MLPs, we then identify neurons of interest using first-order attribution scores from the model logits (App. B.5, App. C.5). The obtained neurons are tested for causal relevance by further patch-and-freeze experiments (see App. B for details).

Observations. MLP neurons in layers 14 and 15 operate on the shared representation space span $( \pmb { v } _ { 1 } , \pmb { v } _ { 2 } )$ (Fig. 4(a)). We find a set of 12 neurons (6 neurons in MLP of layer 14 and 6 in layer 15's MLP) which together contribute to the comparison computation, and freezing these neurons significantly reduces the IIA from patching the $( v _ { 1 } , v _ { 2 } )$ space (see Fig. 4 (e) and App. C.5). They perform the comparison in two stages: first, neurons in the MLP of layer 14 perform localized comparisons: they respond to specific ranges of values of $y _ { 1 }$ or $y _ { 2 } ,$ or a combination thereof $( \mathrm { e . g . }$ $\left| y _ { 1 } - y _ { 2 } \right| < \eta )$ (see Fig. 4 (b) which shows the receptive fields of neurons, i.e., their activations as a heatmap in the $( y _ { 1 } , y _ { 2 } )$ space). While the geometry of the span $( v _ { 1 } , v _ { 2 } )$ space makes comparison possible by linear readouts (Fig. 3c), MLP neurons seem to use their linear transform followed by nonlinearity to perform local comparisons. Some neurons perform the comparison $y _ { 1 } < y _ { 2 }$ on these bounded ranges (like neuron $\# 9 4 5 9$ in Fig.4(b)) while others serve to identify the ranges (like neuron #6150 in Fig. 4(b)). These neurons are then combined in the second stage (MLP of layer 15), into neurons which serve as global comparators, encoding the answer position over the entire range of values of both numbers (Fig. 4(c)). We find a single direction (using DAS) in layer 15's residual stream (after MLP of layer 15), which is written to by several global comparator neurons, and encodes the answer to the comparison as a binary value (Fig.4(d)). The importance of these twelve MLP neurons, as well as the causal relevance of the DAS direction obtained in layer 15, are shown using IA in panel (Fig. 4(e)).

(a)  
![](images/8ecdce401fb04cdfafe9dff4dc9ebabdf2dfd967519fc33e2b6cdfd3b52bc2dc.jpg)  
(d)

(b)  
![](images/f3e29b1ff2d90780d3c96cbb5ea474e5a326bcfd2d34b0ccaeed985f58452f30.jpg)

(c)  
![](images/ed52913dc0cec432cc645248fcbe1ce2f8773ba1dd754de02a8636b5aa2b7792.jpg)

![](images/8c4efe06e8fb11ae44d986e7309d8df08d14b8b65e3e14be991d73bde1afd288.jpg)

![](images/46c5f510e385801426e124db97b9d9a953cca73ad2a5d2f50e746990a57e51bd.jpg)

(e)  
![](images/694676c273721d906dd85a6abc301897532c39c48766fd2570dcf78bd657cc48.jpg)

(f)  
![](images/74a7c6c18c6a9a122c6272d23f4e89eebdcd94db519c38a87bbdc9e0136d92ce.jpg)  
Figure 5: The model continues to use linear number representations for three-number comparisons. (a) We observe that the causal directions ${ \pmb v } _ { 1 } , { \pmb v } _ { 2 } , { \pmb v } _ { 3 }$ (obtained at the third number $y _ { 3 } \mathrm { ^ { * } s }$ position) encode the numbers $y _ { 1 } , y _ { 2 } , y _ { 3 }$ , through nearly monotonic components $f _ { i } ( y _ { i } )$ . (b) The three directions ${ \pmb v } _ { 1 } , { \pmb v } _ { 2 } , { \pmb v } _ { 3 }$ have causal effects on the model outputs (logits, measured using position recovery, PR). The effect on $y _ { 2 }$ is lower at this position $( y _ { 3 }$ position). (c) Receptive fields from layer 14, layer 15 MLP neurons show local comparisons now being performed in the $y _ { 1 } , y _ { 2 } .$ y3 space (whose three two-dimensional views are shown). (d) The model represents its answer $f l a g$ along a single direction in the layer 15 residual stream at both the $y _ { 2 }$ and the $y _ { 3 }$ position. The flag identifies if $y _ { t }$ is the largest upto time $t .$ Note that $y _ { 2 }$ max refers to $y _ { 2 } > y _ { 1 }$ here. (e) The arg max position is represented in the top two principal components of layer $2 0$ residual stream, at the last token position (after the third number $y _ { 3 } .$ ,and right before the model answers). (f). The flags at $y _ { 2 }$ and $y _ { 3 }$ positions, as well as the answer position from (e), are causal and affect model logits.

## 3 MULTIPLE COMPARISONS: THE CASE WITH THREE NUMBERS

Having stated and established evidence for the pairwise comparisons algorithm in Qwen, we now extend it to multiple number comparisons: does the model continue to use linear representations of numbers for multiple comparisons? The model is very good at the multiple comparison task: comparing long sequences of numbers in a single forward pass (see App. C.1). Using three numbers as a case study, we state the model's algorithm and describe evidence for this algorithm in model activations.

## 3.1 ALGORITHM FOR MULTIPLE COMPARISONS

To compare longer sequences of numbers, language models employ an interesting strategy: at every time t, they compute a binary flag which answers the following question: is $y _ { t }$ the largest number seen so far? To achieve this, they construct an additive sum of linear representations of all numbers up to and including $y _ { t }$ . Individual neurons then perform local comparisons on this shared representation, which is combined to give a global comparator that identifies whether $y _ { t }$ is the largest entry seen so far. This information is stored along a single direction $\mathbf { d } _ { t }$ , as a binary variable $\alpha _ { t } = \mathbb { I } ( y _ { t } = \operatorname* { m a x } ( y _ { 1 } , . . . , y _ { t } ) )$

$$
\begin{array} { r } { \pmb { x } = \pmb { v } _ { 1 } f _ { 1 } ( y _ { 1 } ) + \pmb { v } _ { 2 } f _ { 2 } ( y _ { 2 } ) + \pmb { v } _ { 3 } f _ { 3 } ( y _ { 3 } ) , } \\ { \mathrm { a t } t , \quad \alpha _ { t } = \mathbb { I } ( y _ { t } = \operatorname* { m a x } ( y _ { 1 } , \dots , y _ { t } ) ) . } \end{array}\tag{3}
$$

At this stage of computation, if $y _ { t }$ is not the maximum upto time t, i.e., the flag $\alpha _ { t } = 0 ;$ , the position of the maximum is not otherwise stored (i.e., the model simply knows that the maximum is not at t, but not if the maximum is at an intermediate position). Information about whether the first number $y _ { 1 }$ (which is a special case since there are no numbers before it) is the maximum is also computed at the $y _ { 3 }$ position. At the final token $\tilde { T }$ before generating the answer, the model combines these directions using a single attention head which attends to the answer position $( \mathrm { i } . \mathrm { e } . , w _ { t }$ is high at arg $\operatorname* { m a x } ( y _ { 1 } , \dots , y _ { T } ) )$ to obtain $\begin{array} { r } { \pmb { x } _ { \ell ^ { \prime } } ( \tilde { T } ) = \sum _ { t = 1 } ^ { T } w _ { t } \alpha _ { t } \mathbf { d } _ { t } \left( \mathrm { A p p } \right. } \end{array}$ . Fig. 26). This final representation encodes the answer position which is then read off by the model. While our results describe how individual number representations are used to create causally relevant flags, we empirically find the above attention head. We leave an investigation into the complete mechanism of how the head works and how the model decodes the answer position to future work.

We hypothesize that the model may read this information by exploiting the order of numbers, checking sequentially (in decreasing t) if $\alpha _ { t } = 1$ and identifying the first time this occurs. We state the algorithm in Alg. 2.

## 3.2 EVIDENCE FOR THE THREE NUMBER COMPARISON ALGORITHM

Setup. We extend our analysis to studying comparison of three numbers. We choose 1500 triples and 400 held-out examples. In this case, IIA does not show enough signal for analysis, so we employ position recovery $( \operatorname { E q } . { \dot { 2 } } )$ to test for causal effects of activation patching on the model's logits.

Observations. First, we note that when given three numbers, the model seems to perform two parallel computations (see App. C.6). Fig. $5 \mathrm { { s } }$ first row shows the computation occurring at the third number position. Using three directions ${ \pmb v } _ { 1 } , { \pmb v } _ { 2 } , { \pmb v } _ { 3 }$ which encode the magnitudes of the three numbers $y _ { 1 } , y _ { 2 } , y _ { 3 }$ (panel a), the model creates a three-dimensional space span $( \pmb { v } _ { 1 } , \pmb { v } _ { 2 } , \pmb { v } _ { 3 } )$ Fig. 5(b) shows that patching each direction affects the model outputs when the corresponding number is perturbed, and that patching the span affects multiple numbers. The effect of perturbing $y _ { 2 }$ is minimal here, indicating that the model uses different computation for determining $y _ { 2 }$ is the largest. Representative MLP neurons in layer 14 MLP perform local comparisons in the $( y _ { 1 } , y _ { 2 } , y _ { 3 } )$ space (Fig. 5(c)). These neurons construct a single direction in layer 15's residual stream space which encodes the answer $\mathbb { I } ( y _ { 3 } = \operatorname* { m a x } ( y _ { 1 } , y _ { 2 } , y _ { 3 } )$ ) (Fig. 5(d), bottom). At the second number $y _ { 2 } \cdot \mathrm { s }$ position, the model compares $y _ { 2 }$ and $y _ { 1 }$ (using Alg. 1), and creates a single direction from DAS which encodes the answer position in layer $1 5 \mathrm { { : } }$ residual stream (Fig. 5(d), top). At the last token position, layer 20's residual stream combines the flag information (using an attention head, see App. Fig. 26) encodes the position of the answer, as visible on a two dimensional PCA Fig. 5(e). Position recovery scores in Fig. 5(f) show that the flags at the $y _ { 2 }$ and $y _ { 3 }$ positions, as well as the two-dimensional subspace encoding the overall answer in layer $2 0$ (from Fig. 5(e)) are causal. The two principal components in layer 20 have the same position recovery as the entire residual stream at that position (Fig. 5(f)).

## 4 DISCUSSION AND LIMITATIONS

Our work is a concrete illustration of how a language model (Qwen2.5-7B-Instruct in our case) uses linear representations and additive mixing for specific model computations, number comparison in our case. The model uses linear representations for each number position despite the presence of a nonlinear manifold in its activations. We demonstrate how the model methodically combines information from both numbers by adding the number representations together, constructing a twodimensional plane in its residual stream. The model then employs MLP neurons to focus on specific local regions within this space, which correspond to smaller intervals of the input numbers, and performs comparison on these patches. Finally, the model combines these local patches into a global comparison. In addition, we show how this algorithm extends to three number comparisons. Our findings challenge a central hypothesis in current mechanistic interpretability research, namely that detailed accounts of the representation geometry of individual concepts necessarily make it easier to characterize how models perform meaningful computations using these concepts. As we demonstrate, even though one can precisely characterize the geometry of number manifolds in model activations, simpler linear representations of numbers may be enough to causally affect the model's computation. Our findings complement and add nuance to our understanding of what representation geometry of neural networks truly teaches us: while the intricacies of their geometric structure may reflect the properties of the underlying data distribution, simpler structures within this geometry may be sufficient computationally and useful to the model.

Limitations. The limitations and assumptions of our work are stated below:

• Our analysis is restricted to a specific kind of input – numbers – and a specific operation – computing the maximum. Our claims about linear representations being useful despite the presence of manifold geometry hold for concepts that are ordered, i.e., those that can be mapped to the number line (without periodicity)

• We rely on DAS (Geiger et al., 2024) to obtain causal directions in model activations, whenever principal components have low causal effects. Therefore, our analysis may partially inherit the same challenges as DAS, such as finding shortcuts (Wu et al., 2024). However, we include additional evidence grounded in the model, such as neurons, attention heads, etc which may partially alleviate concerns of shortcuts

• Much of our analysis shows sufficiency: we identify directions and subspaces which can modify the model's outputs when patched. We don't claim necessity: removing these components may not affect the model's outputs – the model can still use other paths of information processing within its layers to solve the same task (like the hydra effect (McGrath et al., 2023)).

## ACKNOWLEDGMENTS

This work has been made possible in part by a gift from the Chan Zuckerberg Initiative Foundation to establish the Kempner Institute at Harvard University. SSRH and DB thank the Kempner Institute for access to compute resources. SSRH thanks members of the CRISP lab at Harvard SEAS for useful discussions and feedback on the manuscript. SSRH further thanks Atticus Geiger and Ekdeep Singh Lubana for insightful discussions about the project.

## AI USE STATEMENT

In this work, we used generative AI tools for writing and editing code, literature search, drafting portions of the appendix, and feedback on research ideas and experimental methodology. We have reviewed and verified all AI-assisted work. We take responsibility for the final content of this work including text, claims, code, or artifacts produced with the aid of generative AI.

## REPRODUCIBILITY STATEMENT

Appendix A provides the experimental setup, prompts, datasets and sampling procedures, while Appendix B provides details of the intervention, DAS, attribution, and evaluation methods. Sample sizes and hyperparameters are summarized in Table 1.

## REFERENCES

Guillaume Alain and Yoshua Bengio. Understanding intermediate layers using linear classifier probes. arXiv preprint arXiv:1610.01644, 2016.

Yonatan Belinkov. Probing classifiers: Promises, shortcomings, and advances. Computational Linguistics, 48(1):207–219, 2022. doi: 10.1162/coli\_a\_00422.

Usha Bhalla, Thomas Fel, Can Rager, Sheridan Feucht, Tal Haklay, Daniel Wurgaft, Siddharth Boppana, Matthew Kowal, Vasudev Shyam, Jack Merullo, Atticus Geiger, and Ekdeep Singh Lubana. Do sparse autoencoders capture concept manifolds?, 2026. URL https : //arxiv. org/abs/2604.28119.

Lawrence D. Brown, T. Tony Cai, and Anirban DasGupta. Interval estimation for a binomial proportion. Statistical Science, 16(2):101–133, 2001. doi: 10.1214/ss/1009213286.

Lawrence Chan, Adrià Garriga-Alonso, Nicholas Goldowsky-Dill, Ryan Greenblatt, Jenny Nitishinskaya, Ansh Radhakrishnan, Buck Shlegeris, and Nate Thomas. Causal scrubbing: A method for rigorously testing interpretability hypotheses. AI Alignment Forum, https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/ causal-scrubbinq-a-method-for-rigorously-testinq,2022.

Ahmed Oumar El-Shangiti, Tatsuya Hiraoka, Hilal AlQuabeh, Benjamin Heinzerling, and Kentaro Inui. The geometry of numerical reasoning: Language models compare numeric properties in linear subspaces. In Luis Chiruzzo, Alan Ritter, and Lu Wang (eds.), Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers), pp. 550–561, Albuquerque, New Mexico, April 2025. Association for Computational Linguistics. ISBN 979-8-89176-190-2. doi: 10.18653/v1/ 2025.naacl-short.47.URL https://aclanthology.org/2025.naacl-short.47/.

Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, et al. A mathematical framework for transformer circuits. Transformer Circuits Thread, 2021. URL https://transformer-circuits. pub/2021/framework/index.html.

Sheridan Feucht, Tal Haklay, Usha Bhalla, Daniel Wurgaft, Can Rager, Raphaël Sarfati, Jack Merullo, Thomas McGrath, Owen Lewis, Ekdeep Singh Lubana, et al. Arithmetic in the wild: Llama uses base-10 addition to reason about cyclic concepts. arXiv preprint arXiv:2605.01148, 2026.

Peter Gärdenfors. Conceptual spaces, volume 3. MIT press Cambridge, MA, 2000.

Nikhil Garg, Jon Kleinberg, and Kenny Peng. How many features can a language model store under the linear representation hypothesis?, 2026. URL https://arxiv.org/abs/2602.11246.

Atticus Geiger, Hanson Lu, Thomas Icard, and Christopher Potts. Causal abstractions of neural networks. In Advances in Neural Information Processing Systems (NeurIPS), volume 34, pp. 9574–9586, 2021. arXiv:2106.02997.

Atticus Geiger, Zhengxuan Wu, Christopher Potts, Thomas Icard, and Noah D. Goodman. Finding alignments between interpretable causal variables and distributed neural representations. In Proceedings of the Third Conference on Causal Learning and Reasoning (CLeaR), volume 236 of Proceedings of Machine Learning Research, pp. 160–187. PMLR, 2024. arXiv:2303.02536.

Nicholas Goldowsky-Dill, Chris MacLeod, Lucas Sato, and Aryaman Arora. Localizing model behavior with path patching. arXiv preprint arXiv:2304.05969, 2023.

Wes Gurnee and Max Tegmark. Language models represent space and time. In International Conference on Learning Representations (ICLR), 2024. arXiv:2310.02207.

Wes Gurnee, Emmanuel Ameisen, Isaac Kauvar, Julius Tarng, Adam Pearce, Chris Olah, and Joshua Batson. When models manipulate manifolds: The geometry of a counting task, 2026. URL https://arxiv.org/abs/2601.04480.

Michael Hanna, Ollie Liu, and Alexandre Variengien. How does gpt-2 compute greater-than?: Interpreting mathematical abilities in a pre-trained language model. Advances in Neural Information Processing Systems, 36:76033–76060, 2023.

Amiri Hayes, Belinda Z Li, and Jacob Andreas. Explaining attention with program synthesis, 2026. URLhttps://arxiv.org/abs/2606.19317.

John Hewitt and Percy Liang. Designing and interpreting probes with control tasks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 2733– 2743, 2019. doi: 10.18653/v1/D19-1275.

Hyeonbin Hwang and Yeachan Park. Intrinsic task symmetry drives generalization in algorithmic tasks. arXiv preprint arXiv:2603.01968, 2026.

Subhash Kantamneni and Max Tegmark. Language models use trigonometry to do addition, 2025. URLhttps://arxiv.org/abs/2502.00873.

Dhruva Karkada, Daniel J Korchinski, Andres Nava, Matthieu Wyart, and Yasaman Bahri. Symmetry in language statistics shapes the geometry of model representations. arXiv preprint arXiv:2602.15029, 2026.

János Kramár, Tom Lieberum, Rohin Shah, and Neel Nanda. AtP\*: An efficient and scalable method for localizing LLM behaviour to components. arXiv preprint arXiv:2403.00745, 2024.

Andrew Lee, Xiaoyan Bai, Itamar Pres, Martin Wattenberg, Jonathan K. Kummerfeld, and Rada Mihalcea. A mechanistic understanding of alignment algorithms: A case study on dpo and toxicity, 2024.URLhttps://arxiv.org/abs/2401.01967.

Samuel Marks and Max Tegmark. The geometry of truth: Emergent linear structure in large language model representations of true/false datasets, 2024. URL https : //arxiv.org/abs/2310. 06824.

David Marr. Vision: A computational investigation into the human representation and processing of visual information. MIT press, 2010.

Thomas McGrath, Matthew Rahtz, Janos Kramar, Vladimir Mikulik, and Shane Legg. The hydra effect: Emergent self-repair in language model computations, 2023. URL https : //arxiv. org/abs/2307.15771.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in GPT. In Advances in Neural Information Processing Systems (NeurIPS), volume 35, pp. 17359–17372, 2022. arXiv:2202.05262.

Alexander Modell, Patrick Rubin-Delanchy, and Nick Whiteley. The origins of representation manifolds in large language models, 2025. URL https://arxiv.org/abs/2505.18235.

Neel Nanda. Attribution patching: Activation patching at industrial scale. https : //www. neelnanda.io/mechanistic-interpretability/attribution-patching, 2023.

Kiho Park, Yo Joong Choe, and Victor Veitch. The linear representation hypothesis and the geometry of large language models. arXiv preprint arXiv:2311.03658, 2023.

Aaquib Syed, Can Rager, and Arthur Conmy. Attribution patching outperforms automated circuit discovery. In NeurIPS 2023 Workshop on Attributing Model Behavior at Scale (ATTRIB), 2023. arXiv:2310.10348.

Curt Tigges, Oskar John Hollinsworth, Atticus Geiger, and Neel Nanda. Linear representations of sentiment in large language models, 2023. URL https://arxiv.org/abs/2310.15154.

Jesse Vig, Sebastian Gehrmann, Yonatan Belinkov, Sharon Qian, Daniel Nevo, Yaron Singer, and Stuart Shieber. Investigating gender bias in language models using causal mediation analysis. In Advances in Neural Information Processing Systems (NeurIPS), volume 33, pp. 12388–12401, 2020.

Andrey Voynov and Artem Babenko. Unsupervised discovery of interpretable directions in the gan latent space, 2020.URL https://arxiv.org/abs/2002.03754.

Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. Interpretability in the wild: A circuit for indirect object identification in GPT-2 small. In International Conference on Learning Representations (ICLR), 2023. arXiv:2211.00593.

Zhengxuan Wu, Atticus Geiger, Thomas Icard, Christopher Potts, and Noah D. Goodman. Interpretability at scale: Identifying causal mechanisms in alpaca, 2024. URL https : //arxiv. org/abs/2305.08809.

Daniel Wurgaft, Can Rager, Matthew Kowal, Vasudev Shyam, Sheridan Feucht, Usha Bhalla, Tal Haklay, Eric Bigelow, Raphael Sarfati, Thomas McGrath, Owen Lewis, Jack Merullo, Noah Goodman, Thomas Fel, Atticus Geiger, and Ekdeep Singh Lubana. Manifold steering reveals the shared geometry of neural network representation and behavior, 2026. URL https : //arxiv. org/abs/2605.05115.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.

Fengting Yuchi, Li Du, and Jason Eisner. Llms know more about numbers than they can say, 2026. URLhttps://arxiv.org/abs/2602.07812.

Wentao Zhu, Zhining Zhang, and Yizhou Wang. Language models represent beliefs of self and others, 2024.URLhttps://arxiv.org/abs/2402.18496.

## A EXPERIMENTAL SETUP

## A.1 MODEL, TASK AND PROMPTS

We use Qwen2.5-7B-Instruct (Yang et al., 2024) (28 layers, $d _ { \mathrm { m o d e l } } = 3 5 8 4$ , 28 heads of dimension 128, SwiGLU MLPs of width $d _ { \mathrm { f f } } = 1 8 9 4 4 )$ in $\pm 1 0 \mathsf { a t } 1 6$ with all parameters frozen, on a single NVIDIA A100 40GB GPU. The task max $\left( y _ { 1 } , \dots , y _ { K } \right)$ is posed as a raw completion (no chat template) with a one-shot exemplar that fixes the answer format:

Answer in the following format with a single answer. The   
maximum of 12 and 4 is 12. The maximum of {y₁} and {y₂} is

for $K = 2$ in both digit regimes, with the exemplar “The maximum $\cot \ 1 2 , \quad 4 3 7$ and 5 $\mathrm { i } s \textrm { } 4 3 7 . \mathrm { \textdegree }$ and a question“The maximum $\cot \ \mathsf { \bar { \Omega } } \{ y _ { 1 } \} , \quad \{ y _ { 2 } \}$ and $\{ y _ { 3 } \} \ i \ s ^ { , , }$ for $K \geq 3 ;$ each prompt ends in a single space.

Qwen2.5 tokenizes numbers digit by digit, so an operand's first token is its leading digit. We read every residual-stream quantity at a number's last token, its position, where the whole number is first visible under causal attention; all operands are first visible at the position of $y _ { 2 } \left( K = 2 \right)$ or y3 $( K = 3 )$ . Positions are found from the operands’ character spans after the last occurrence of $" \mathrm { T h e }$ maximum $\bigcirc \pm \mathrm { ~ \mathfrak ~ { ~ \mathfrak ~ { ~ n ~ } ~ } ~ }$ via the tokenizer's offset mapping. Within an experiment all operands have the same number of digits, so every prompt has the same token layout $( T = 3 6$ with $y _ { 1 }$ , Y2 at tokens 29, 33 for two 2-digit operands; $\dot { T } = 3 8$ with tokens 30, 35 for two 3-digit operands; $T = 4 6$ with tokens 35, 39, 43 for three 2-digit operands). All sampled operands have pairwise-distinct leading digits, so that each readout token identifies one operand.

Two-digit operands lie in [10, 100) with a minimum gap $g = 1 0$ between the values of a sampled tuple, and three-digit operands in [100, 1000) with $g = 1 0 0$ ; the three-digit regime is used only to replicate the number-representation analysis.

## A.2 COUNTERFACTUAL PAIRS

For $K = 2$ we sample triples $( a , b , r )$ which satisfy $a > b + g > r + 2 g ( g$ is the gap). The clean prompt $( \boldsymbol { \mathrm { e } } . \boldsymbol { \mathrm { g } } . , ( a , b ) )$ has a at the winning operand and b at the other (answer is a), and the corrupted prompt replaces a in place by r (so corrupted prompt has $( r , b ) )$ (answer b), so that only which operand wins changes. The same triples are used in both arrangements (only the order of values is changed), $y _ { 1 } > y _ { 2 }$ and $y _ { 2 } > y _ { 1 }$ . Patching a clean state into the corrupted run makes the model name r (smaller value in its input pair) rather than a (Section C.1): the patch changes which operand the model treats as the maximum, not the value it outputs. We therefore score interventions on $t ( r )$ against t(b), where $t ( \cdot )$ is a number's first token; unlike $t ( a ) , t ( r )$ cannot be raised by a patch that merely re-inserts the clean value.

For $K = 3$ we sample quadruples $a > b > c > r$ with consecutive gaps larger than g, place $a , b , c$ in a chosen value order in the clean prompt, and replace a in place by r in the corrupted one, so that the corrupted winner is always the clean runner-up. The head groups $y _ { 1 } > y _ { 2 } > y _ { 3 }$ (patched at the position of y2), $y _ { 1 } > y _ { 3 } > y _ { 2 }$ and $y _ { 2 } > y _ { 3 } > y _ { 1 }$ (both patched at the position of $y _ { 3 } )$ vary the runner-up's position and the maximum's position one at a time and are used to identify heads. All other analyses use three perturbation cases, patched at the position of $y _ { 3 }$ and named by the number the corruption changes: case y1 $( y _ { 1 } > y _ { 3 } > y _ { 2 } )$ , case y2 $( y _ { 2 } > y _ { 3 } > y _ { 1 } )$ and case $y _ { 3 }$ (y3 $> y _ { 2 } > y _ { 1 } )$ 1

## A.3 POPULATIONS AND SPLITS

Intervention scores are computed on 400 evaluation examples, and directions and neuron rankings are fitted on 128 disjoint fitting examples, both taken from one deduplicated pool drawn by a deterministic sampler, so that every analysis uses the same examples. For $K = 3$ we draw 1,400 quadruples, keep the 1,341 whose clean and corrupted prompts are solved (next-token argmax $t ( a )$ and $t ( b ) )$ in all four value orders, following Meng et al. (2022), and use the first 528. For $K = 2$ no filter is applied: in the two-digit regime the model answers every ordered pair of distinct two-digit numbers correctly, while the three-digit populations include one example per arrangement with a non-positive clean-minus-corrupted gap. Geometry and probes use separate clouds of clean prompts, with the two-number clouds containing each unordered pair in both orders.

Behavioural accuracy is measured by greedy decoding of one token more than the operands' digit count, comparing the first integer of the output with the true maximum, on all 8,010 ordered pairs of distinct two-digit numbers and on 4,000 three-digit pairs drawn with replacement. The operand-count sweep uses 200 prompts of distinct two-digit operands per length in the three-operand prompt format with Wilson score intervals (Brown et al., 2001). Table 1 lists all sample sizes and hyperparameters.

Table 1: Sample sizes and hyperparameters.
<table><tr><td></td><td>quantity</td><td>value</td></tr><tr><td rowspan="4">populations</td><td>evaluation / fitting examples</td><td>400 /128 2000 / 3000 / 1500</td></tr><tr><td>clouds (L13 two-number / L14 / three-number)</td><td></td></tr><tr><td>operand counts k in the accuracy sweeps, prompts per k</td><td>2, 3, 4, 5, 10, 20; 200</td></tr><tr><td>base random seed</td><td>52</td></tr><tr><td rowspan="4">learned directions</td><td>Adam steps, learning rate</td><td>100,0.05</td></tr><tr><td>minibatch (gradients accumulated)</td><td>16-32</td></tr><tr><td>initializations in the stability check (u)</td><td>2</td></tr><tr><td>minibatch, half-precision loss scale</td><td>16,100</td></tr><tr><td rowspan="3">attribution</td><td>candidate neurons (MLP14 ∪ MLP15)</td><td>37,888</td></tr><tr><td>freeze-sweep depths k</td><td> $1 , 3 , 1 0 , 3 0 , 1 0 ^ { 2 } , 3 { \cdot } 1 0 ^ { 2 } , 1 0 ^ { 3 } , 3 { \cdot } 1 0 ^ { 3 }$ </td></tr><tr><td></td><td></td></tr><tr><td rowspan="3">sweeps and fields</td><td>dose-response grid values, window prompts per dose-response cell (K=2)</td><td>15, 25, . . . , 95; ±2 64</td></tr><tr><td>receptive-field grid</td><td>stride 4 over y ∈ [11, 99]</td></tr><tr><td></td><td></td></tr><tr><td rowspan="2">probes</td><td>ridge penalty grid (leave-one-out)</td><td>13 log-spaced values in  $[ 1 0 ^ { - 2 } , 1 0 ^ { 4 } ]$ </td></tr><tr><td>held-out fraction</td><td>0.2</td></tr></table>

## B METHODS

## B.1 INTERVENTIONS AND SITES

Causal measurements are interchange interventions (Geiger et al., 2021; 2024): we run the corrupted prompt and replace a component of the activation at one site by its value on the clean run. With x the corrupted activation, $\pmb { x } ^ { \mathrm { c l e a } }$ the clean one and $P \in \mathbb { R } ^ { d \times k }$ an orthonormal basis,

$$
{ \pmb x } ^ { \prime } = { \pmb x } + P P ^ { \top } \big ( { \pmb x } ^ { \mathrm { c l e a n } } - { \pmb x } \big ) ,\tag{4}
$$

computed in single precision. $P = I$ is the full-rank restoration of Meng et al. (2022). Each rankrestricted patch is shown next to a full-rank patch at the same site as a reference rather than an upper bound: rank-restricted patches can exceed it.

The sites are: (i) the residual stream leaving a decoder block, at one position; (ii) one attention head's contribution $z _ { h } ( W _ { O } ^ { h } ) ^ { \top }$ , obtained from the head's 128-dimensional slice of the output-projection input, where a direction P is patched by reading the coefficient $( z _ { h } ^ { \mathrm { c l e a n } } - z _ { h } ) ( W _ { O } ^ { h } ) ^ { \top } P$ from the head's contribution and adding it along $\bar { P }$ to the residual stream (P is not constrained to the head's output space; for ${ \pmb v } _ { 1 }$ in L14.H14, 81% of its norm lies inside it); (iii) the pre-MLP residual ${ \pmb x } _ { \ell - 1 } +$ $\mathrm { A t t n } _ { \ell } ( { \pmb x } _ { \ell - 1 } )$ , patched through the attention output, which gives the same result as an end-of-layer patch at full rank; and (iv) MLP neurons, the coordinates of the post-SwiGLU vector that feeds the down-projection.

## B.2 METRICS

With $\mathrm { P L D } = \log \mathrm { i t } [ t ( r ) ] - \log \mathrm { i t } [ t ( b ) ]$ computed at the last token, using the same token pair on the clean, corrupted and patched runs, position recovery is

$$
\mathrm { P R } = \frac { \mathrm { P L D } _ { \mathrm { p a t c h e d } } - \mathrm { P L D } _ { \mathrm { c o r r } } } { \mathrm { P L D } _ { \mathrm { c l e a n } } - \mathrm { P L D } _ { \mathrm { c o r r } } } ,\tag{5}
$$

the normalized restoration of Meng et al. (2022) applied to the logit difference of Wang et al. (2023), computed per example and averaged. The numerator is the indirect effect of the patched component on the PLD (Vig et al., 2020), and the denominator is a per-example logit scale of 10–13 logits on average, which makes values comparable across sites and cases but is not meant to make a full effect equal 1: since $t ( r )$ is not an answer on the clean prompt, $\mathrm { P L D } _ { \mathrm { c l e a n } }$ can be negative. Where IIA is near zero, as in three-number cases $y _ { 1 }$ and $y _ { 2 }$ , PR measures how far the PLD moves rather than a change of answer. PR need not lie in [0, 1].

Interchange intervention accuracy (Geiger et al., 2021) is $\mathrm { I I A } = \mathbb { I } \left[ \mathrm { a r g } \operatorname* { m a x } _ { v \in \mathcal { V } } \mathrm { l o g i t } [ v ] = t ( r ) \right]$ over the full vocabulary at the last token, which scores the first generated token, the leading digit of r, so it also counts the mixed answers produced by a full-rank patch at the last token of $y _ { 2 }$ (Section C.1). For the ablations we use full-number accuracy: the answer is decoded greedily for one token more than the operands’ digit count and counted correct when its leading characters match the true maximum.

## B.3 DIRECTIONS

Learned directions are rank-k subspaces found by distributed alignment search for a single causal variable (Geiger et al., 2024): the basis $Q = \operatorname { q r } ( R )$ (using QR factorization) of a free matrix R is inserted into Equation 4 at the chosen site and trained on the fitting examples with a cross-entropy toward $t ( r )$ restricted to $\{ t ( b ) , t ( r ) \}$ , with gradients accumulated so that each step uses the full-batch mean. With 16 fitting examples a fit reached a training IIA of 1.000 against a held-out 0.688, hence the 128 used here. Retraining u from a second initialization gives $\cos | = 0 . 9 9$ and the same held-out IIA (0.94) for two-digit operands, and $| \cos | = 0 . 8 2$ with IIAs 0.86 and 0.80 for three-digit operands; other directions use a single initialization. Where a site may carry one variable shared across cases, a subspace fitted on each case is evaluated on every case. Directions are signed so that their component rises with the number they carry. Table 2 lists the directions.

Table 2: Directions, where they live and how they are obtained. DAS directions are rank one and fitted in the listed case; PC denotes the top principal component(s) of the corresponding clean cloud.
<table><tr><td> $K$ </td><td>direction</td><td>position</td><td>site</td><td>obtained by</td></tr><tr><td>2</td><td>u</td><td>y1</td><td>L13 residual</td><td>DAS, y1 perturbed</td></tr><tr><td>2</td><td> ${ \pmb v } _ { 1 }$ </td><td>y2</td><td>output of L14.H14</td><td>DAS, y1 perturbed</td></tr><tr><td>2</td><td> $\mathbf { v } _ { 2 }$ </td><td>y2</td><td>L13 residual</td><td>PC</td></tr><tr><td>2</td><td>comparison flag</td><td>y2</td><td>L15 residual</td><td>DAS, each case</td></tr><tr><td>3</td><td> $\mathbf { \delta u } _ { 1 } , \mathbf { \delta u } _ { 2 }$ </td><td>Y1, Y2</td><td>L13 residual</td><td>DAS, cases  $y _ { 1 } , y _ { 2 }$ </td></tr><tr><td>3</td><td> $v _ { 1 } , v _ { 2 }$ </td><td>y3</td><td>outputs of L14.H14, L14.H18</td><td>DAS, cases y1, y2</td></tr><tr><td>3</td><td> $\mathbf { \boldsymbol { v } } _ { 3 }$ </td><td>Y3</td><td>L13 residual</td><td>PC</td></tr><tr><td>3</td><td> $w _ { 1 } , w _ { 2 }$ </td><td>y2</td><td>L13 residual</td><td>DAS, cases y1, y2  $( w _ { 2 } = { \pmb u } _ { 2 } )$ </td></tr><tr><td>3</td><td>comparison flags</td><td> $y _ { 2 } , y _ { 3 }$ </td><td>L15 residual</td><td>DAS, each case</td></tr><tr><td>3</td><td>answer subspace</td><td>last token</td><td>L20 residual</td><td>top two PCs</td></tr></table>

${ \pmb v } _ { 2 }$ is a principal component because a direction learned at that site partly encodes the comparison outcome instead of $y _ { 2 }$ (Section C.3), and heads L14.H14 and L14.H18 come from our earlier localization of the three-number task. Multi-dimensional patches use the span of the $\boldsymbol { v } _ { j }$ , orthogonalized by Gram-Schmidt with ${ \pmb v } _ { 1 }$ first; the directions are nearly orthogonal to begin with (cos $( { \pmb v } _ { 1 } , { \pmb v } _ { 2 } ) = - 0 . 0 8$ for two numbers), and because they are obtained independently, the span's score is a lower bound for a rank-k patch at that site.

Principal components are fitted on mean-centred, unscaled activations (Gurnee & Tegmark, 2024). Linear decodability is measured with ridge regression (Alain & Bengio, 2016; Belinkov, 2022) on the logarithm of each number, and the comparison $y _ { 1 } > y _ { 2 }$ with a ridge classifier, reporting held-out $R ^ { 2 }$ or accuracy; probes for operands that a position cannot depend on serve as floors, in the spirit of the control tasks of Hewitt & Liang (2019). Log-linear trends are summarized by the least-squares fit p log y + q.

## B.4 LOCALIZATION

The causal trace (Meng et al., 2022) restores the clean residual stream leaving one layer at one token, for every layer and every token from the first token of $y _ { 1 }$ to the last token of the prompt, and reports position recovery averaged within each arrangement. Heads are then interchanged at site (ii) of layer 14 (Wang et al., 2023; Goldowsky-Dill et al., 2023), each together with a restoration of the layer-13 residual at the same position (in full for three numbers, along ${ \pmb v } _ { 2 }$ for two numbers), since a single head written into a corrupted residual stream has almost no effect; the co-patch alone is shown as the baseline, and for two numbers the sweep is also shown without it. At the last token, full-rank patches of the residual leaving layers 18 to 22 over the head-group orders locate the read-out at layer 20, where mean position recovery increases most (0.09 to 0.55). At that layer we patch each head's contribution without a co-patch, with the whole attention sublayer as a reference, and compare a patch of the top two principal components of the residual with a full-rank patch.

## B.5 NEURONS

We rank the neurons of MLP14 and MLP15 by attribution patching (Nanda, 2023; Syed et al., 2023; Kramár et al., 2024), the first-order estimate of the effect on the PLD of freezing neuron i while the shared representation is patched, $s _ { i } \approx ( a _ { i } ^ { \mathrm { c o r r } } - a _ { i } ^ { \mathrm { p a t c h e d } } ) \partial \mathrm { P L D } / \partial a _ { i }$ , with the gradient taken at the patched state through a zero-valued differentiable input at the down-projection. Rankings are computed on the fitting examples and verified by freezing the top k neurons on the evaluation examples, against two random draws of k neurons. The shared set is the intersection of the top-20 rankings of the two two-number cases, 12 neurons (six in each MLP), which the three-number analysis reuses unchanged.

To freeze a sub-block or a set of neurons, we pin its output (for neurons, their post-SwiGLU values) at the target position to its value on the unpatched corrupted run, the severed-path variant of causal tracing (Meng et al., 2022; Vig et al., 2020), for the attention and MLP sub-blocks of layers 14 to 16 individually and for MLP14 and MLP15 jointly. For injection we write only the clean outputs of the selected MLPs into the corrupted run at the target position. For ablation we set the shared neurons’ post-SwiGLU activations to zero, at every position or only at the last token of the final operand, throughout greedy decoding, and compare with two random sets of 12 neurons zeroed at every position, scoring full-number accuracy.

## B.6 DOSE-RESPONSE SWEEPS, RECEPTIVE FIELDS AND CONNECTIVITY

For the dose-response sweep (Geiger et al., 2021; Chan et al., 2022) we set the two coordinates of the $( v _ { 1 } , v _ { 2 } )$ frame in the pre-MLP layer-14 residual at the position of $y _ { 2 }$ to the mean coordinate of a chosen number (over $\mathbf { a } \pm 2$ window in the cloud), on a $9 \times 9$ grid with one value per leading digit, and score the fraction of argmax answers that name the first operand among those naming either operand, masking the diagonal. The pre-MLP site is used because a per-number mean explains $R ^ { 2 } = \bar { 0 } . 8 7$ of the ${ \pmb v } _ { 1 }$ coordinate there, against 0.35 at the end of the layer.

A neuron's receptive field is its post-SwiGLU activation at the target position over a grid of prompts, one forward pass per cell and without the distinct-leading-digit constraint. Three-number fields are shown as pairwise marginals, each averaged over the third number and divided by its own peak, and an operand that a position cannot depend on is held fixed $( y _ { 3 } = 5 5$ at the position of $y _ { 2 } )$

The causal edge from an MLP14 neuron to an MLP15 neuron is the shift in the MLP15 neuron's activation when only the MLP14 neuron is set to its clean value, in units of the MLP15 neuron's standard deviation over a 1,500-prompt two-number cloud. The virtual weight (Elhage et al., 2021) through the gate projection is the cosine between the upstream write column and the downstream gate read row with the RMSNorm scale folded in, and the alignment of a write direction ${ \pmb w } _ { j }$ with a learned subspace $Q$ is $\rho _ { j } = \| Q ^ { \top } \pmb { w } _ { j } \| / \| \pmb { w } _ { j } \|$ , read against the distribution over all 18,944 neurons of the layer, since write directions are not isotropic (the analytic level for a random unit vector and a k-dimensional subspace is about $\sqrt { k / d _ { \mathrm { m o d e l } } } )$ . As a specificity control, each three-number direction is re-evaluated, without refitting, in the cases where the number it carries does not change, where it should have no effect.

## C FURTHER EXPERIMENTAL RESULTS

The figures below follow the order of the main-text algorithm. Unless a caption says otherwise, scores are means over the 400 held-out counterfactuals of Section A.3, error bars are one standard error, and PR (position recovery) and IIA (interchange intervention accuracy) are as defined in Section B.2.

## C.1 BEHAVIOUR AND WHAT AN INTERVENTION CHANGES

We first report task accuracy and the answers the model generates under two interventions.

![](images/089d78416a0e1b69299c20139a87b7a2ffe2c04e12e9ca64c1e48f03ad04e85b.jpg)  
Figure 6: Accuracy on max $( y _ { 1 } , \ldots , y _ { k } )$ against the number of operands $k ,$ under greedy decoding, for 200 prompts of distinct two-digit operands per length. Error bars are Wilson score intervals.

![](images/020eac393548a3c0c56f6ed4579c4a7d784811721827b340de54aa74ad9bbf04.jpg)  
Figure 7: Answers generated by greedy decoding on the 400 held-out counterfactuals, as fractions per category: r, b, a, the leading digit of r followed by the last digit of $^ { a , }$ and other. Rows: the corrupted prompt with no patch and with the full layer-13 residual patched at the last token of $y _ { 2 }$ (both with $y _ { 2 }$ perturbed), and with u patched at the position of $y _ { 1 } \left( y _ { 1 } \right.$ perturbed). An answer equal to r is counted as r.

Table 3: Examples of generated answers on held-out counterfactuals: the clean and corrupted operands, their maxima, and the answer generated under each intervention.
<table><tr><td>intervention</td><td>clean  $( y _ { 1 } , y _ { 2 } )$ </td><td>max</td><td>corrupted  $( y _ { 1 } , y _ { 2 } )$ </td><td>max</td><td>patched answer</td></tr><tr><td>u at y1</td><td>(99, 61)</td><td>99</td><td>(12, 61)</td><td>61</td><td>12</td></tr><tr><td>u at  $y _ { 1 }$ </td><td>(70, 55)</td><td>70</td><td>(33,55)</td><td>55</td><td>33</td></tr><tr><td>full at  $y _ { 2 }$ </td><td>(61, 99)</td><td>99</td><td>(61, 12)</td><td>61</td><td>12</td></tr><tr><td>full at  $y _ { 2 }$ </td><td>(56, 97)</td><td>97</td><td>(56,27)</td><td>56</td><td>27</td></tr><tr><td>full at y2</td><td>(55, 70)</td><td>70</td><td>(55, 33)</td><td>55</td><td>30</td></tr><tr><td>full at  $y _ { 2 }$ </td><td>(71, 89)</td><td>89</td><td>(71, 52)</td><td>71</td><td>59</td></tr></table>

## C.2 WHERE THE COMPUTATION HAPPENS

Causal traces and linear probes over every layer and question token (Section B.4) locate the sites analysed in the rest of the appendix.

(a)  
![](images/182a12b4086e8583e085e99cae385883bff2462145b4267cf72b6378522b2c17.jpg)

(b)  
![](images/96f4fe7951246b92f052cded47959c8cb846579678c0764c5936f0fedc685753.jpg)  
Figure 8: Position recovery when the clean residual stream leaving one layer (rows) is restored at one token (columns), from the first token of $y _ { 1 }$ to the last token of the prompt, with (a) $y _ { 1 }$ and (b) y2 perturbed. Each operand spans two tokens and the tick marks its last one. The panels share a colour scale.

![](images/7a39ea4536bb364093989b8af756cbea7ff9fa081780289ee8af6c2c148e7997.jpg)  
Figure 9: Illustration of the key components implementing algorithm underlying the pairwise number comparisons algorithm Alg. 1

![](images/1c94b1f4c595965f26d2dea8c8abedfec4fcd3e12499b3f7f3c0e30a1c1a0841.jpg)

![](images/c0f9107d66310642255ff68f52f1a915b89fd0d0eacf6b24adee4066e5ae7b00.jpg)

![](images/93662a423f995fbe73b5d2689a91a4591f350ebca80bc27e46c2e57f40c3d67e.jpg)  
Figure 10: Ridge probes fitted at every layer (rows) and token (columns) on 2,000 clean two-number prompts and scored on a 20% held-out split. (a,b) Held-out $R ^ { 2 }$ for log $y _ { 1 }$ and log y2. (c) Held-out accuracy of a ridge classifier for $y _ { 1 } > y _ { 2 }$ . The axes are those of Figure 8.

(a)  
![](images/08165e37d31cdb033c8da7e798822e223c7544ce567aed4f4eda5cb39a6368c5.jpg)

![](images/dad250ff775fb1681e4ba2ee5d0f039dae6ccb82f82550d132b0c5a0666953e4.jpg)  
(c)

![](images/61f15643f2cf948e70f2040855065bcabe0506a9857b1e467f73fc8c2a539f4d.jpg)  
Figure 11: The causal trace of Figure 8 for three numbers, with (a) y1, (b) y2 and (c) $y _ { 3 }$ perturbed, on the 400 held-out quadruples. The panels share a colour scale.

## C.3 INDIVIDUAL NUMBER REPRESENTATIONS

For the direction u at the position of $y _ { 1 }$ , we show its place in the layer-13 geometry, a comparison with the top principal component, the same analysis at earlier layers and with three-digit operands. We then show the learned alternative to $\mathbf { v } _ { 2 }$ and the output of the transport head L14.H14.

![](images/6b0c260534ad59a74a1604e165ccd6d381ec92ba2bebf1357232681b5d373580.jpg)

(b)  
![](images/601d9b8670f14a07cc1df74326c40c84835d4b669e59a282dc0d749b3b86e17d.jpg)  
Figure 12: The layer-13 residual at the position of $y _ { 1 }$ , one point per value of $y _ { 1 }$ . (a) Projection on the first and third principal components, coloured by $y _ { 1 }$ , with the smoothed mean position along $y _ { 1 }$ (line) and the direction of u's projection onto the plane (arrow). (b) Component along u against $y _ { 1 }$ , with the least-squares fit p log $y _ { 1 } + q$ (dotted).

(a)  
![](images/e960ef5dffb4d7adf38ab5c8512775fefc6da2559793669a00c8915ac7e3e5c4.jpg)

(b)  
![](images/e7204d761e063f64fb12f80281a35408ca03bf1398b18d25d2ac19b1b6a8db84.jpg)

(c)  
![](images/ddd149dff21feabd989c13bc77191ef067ab5d1ac58419769e47c23ed7418222.jpg)

(d)  
![](images/7c4d70f8141e5fa41a7c7a59a1e679c0804ae6b912651e7d9f2bb4dac7a9e59a.jpg)  
Figure 13: The layer-13 residual at the position of $y _ { 1 }$ , with $y _ { 1 }$ perturbed. (a) IIA and (b) position recovery of a full-rank patch, a rank-one patch of u and a rank-one patch of the top principal component (PC1). (c) Standardized components along u and PC1 against $y _ { 1 }$ , as per-value means. (d) Variance explained by each of the first 20 principal components (bars) and by u (dashed line).

(a)  
![](images/cc36519c7937b4acf7aee43b1b8e952baf57bc43bf98821c28f4eea0a25da7bd.jpg)

![](images/5b31fa044ffd9bc4580c4261534939e29a4a57e830973585ce73ac9c0551c54b.jpg)

(c)  
![](images/20b6c18cb7b3e8f01b36e43475ae4d26f7349f2358cca60c4122501b0e099791.jpg)

(d)  
![](images/4b44fe85b78e7982d52f155b9f4d582a6390b7f21cf3028c2173c9d51b43e5c0.jpg)

(e)  
![](images/e865ea3f033b2259127e0717a5d46907776a8250a94a075212564281a604af6f.jpg)  
Figure 14: The number-representation analysis at layers 1 to 13, with u refitted at each layer. (a) IIA at the position of $y _ { 1 }$ , with y1 perturbed, for a full-rank patch, u and that layer's top principal component. (b) IIA at the position of $y _ { 2 } .$ with $y _ { 2 }$ perturbed, for a full-rank patch and $\mathbf { v } _ { 2 } ,$ taken as that layer's top principal component. (c-e) Component along u against $y _ { 1 }$ at layers 3, 7 and 13, as per-value means, with the fit p log $y _ { 1 } + q$ (dotted).

(a)  
![](images/ce887ad89715f150230b08ad6b8d59c43bc9d9f45d147175682f568559e2eeb0.jpg)

(b)  
![](images/2669ffe32826c6872d16794a110dcdb0b7ed113410ba7b9929998a5c35df0d3b.jpg)

(c)  
![](images/53b66b6081fae795c1fa094e3d0ac09eaa4aafc7a50475f0a7cabd36aaf91643.jpg)  
Figure 15: The number representations with three-digit operands. (a) Component along u at the position of $y _ { 1 }$ against $y _ { 1 }$ , and (b) component along $\mathbf { v } _ { 2 }$ at the position of y2 against $y _ { 2 } ,$ both averaged in bins of 25, with the fit p log $y + q$ (dotted). (c) IIA of a full-rank patch, u and PC1 at the position of $y _ { 1 }$ with $y _ { 1 }$ perturbed, and of a full-rank patch and $\mathbf { v } _ { 2 }$ at the position of $y _ { 2 }$ with $y _ { 2 }$ perturbed.

(a)  
![](images/964bfc0c9a2d958fddfe40a143694a0a854a2a124e9e01e9e4030ae41473ac7b.jpg)

(b)  
![](images/f057ebba3ecb381c9b436a6b385f565368ebcd95c1207fe5505bb58892f10cae.jpg)

(c)  
![](images/e3e43a651ce12c58f7d802e8333ef6b447b378ecf1392819167a29569145ab75.jpg)  
Figure 16: The layer-13 residual at the position of $y _ { 2 }$ on the 2,000-prompt cloud, coloured by which number is larger. (a) Component along a rank-one direction fitted with $y _ { 2 }$ perturbed, against $y _ { 2 } .$ (b) Component along $\mathbf { v } _ { 2 }$ , the top principal component, against $y _ { 2 }$ . (c) IIA with $y _ { 2 }$ perturbed for a full-rank patch, the fitted direction (DAS) and $\mathbf { v } _ { 2 }$

(a)  
![](images/120e46f8a0ffa253c057e299cf70727968601f3a4e8ff318da0a521c453cca58.jpg)

(b)  
![](images/6e0f8b9b98debb50be8a18b04015090050046984f0bc068c7ed51a2700df3301.jpg)

(c)  
![](images/eefb234f7c19e51fadf56eac0bdb9a37ed08aa1ec3036c679b1d8a05706c9061.jpg)  
Figure 17: The output of head L14.H14 at the position of $y _ { 2 } .$ (a) Its top two principal components, coloured by $y _ { 1 }$ , with the smoothed mean position along $y _ { 1 }$ (line) and the direction of $\mathbf { v } _ { 1 } \mathbf { \ ' } _ { \mathbf { s } }$ projection onto the plane (arrow). (b) The head's attention from the position of $y _ { 2 }$ to $y _ { 1 }$ , against $y _ { 1 }$ , as per-value means. (c) Component along $\mathbf { v } _ { 1 }$ against $y _ { 1 }$ , as per-value means.

## C.4 THE SHARED REPRESENTATION

Probes and position recovery complement the main-text IIA results for the shared representation at the position of $y _ { 2 }$

(a)  
![](images/55781622c33cda8ede556e28fa97443766bfc60d8fa8e5ffc47b685d889c7c76.jpg)  
(b)

![](images/d5c01a82161ff66b31b475ebba246c4282d0cc4644209564173afdde996cbb1f.jpg)

(c)  
![](images/3101c033c6ca1be9be4070268b8b3729b90c5f29601ed5e008b5220f38b1e007.jpg)  
(d)

![](images/2c29cdb77fb1ca26c81589c2cd81adc764ae3609c9f8c10be3d34b0c676c15d8.jpg)  
Figure 18: Components of the pre-MLP layer-14 residual at the position of $y _ { 2 }$ along ridge probes for log y1 $( \mathbf { p } _ { 1 }$ , top row) and log y2 $( \mathbf { p } _ { 2 } .$ , bottom row), against $y _ { 1 }$ (left) and $y _ { 2 }$ (right), on 600 held-out prompts.

![](images/57f156dafe7c3a07992c238892dc63b9572af3d69d5250128dcb54e607b0a4dc.jpg)  
Figure 19: Position recovery at the position of $y _ { 2 }$ for the patches of the main-text shared-representation figure, with $y _ { 1 }$ or $y _ { 2 }$ perturbed: the full layer-13 residual, $\mathbf { v } _ { 2 } , \mathbf { v } _ { 1 } , \mathbf { v } _ { 1 }$ and $\mathbf { v } _ { 2 }$ as two separate rank-one patches, the rank-two $\left( \mathbf { v } _ { 1 } , \mathbf { v } _ { 2 } \right)$ plane in the pre-MLP layer-14 residual, and the full pre-MLP layer-14 residual. Hatched bars are the two conditions that patch both directions.

## C.5 THE COMPARATOR NEURONS

Freezing, attribution, receptive fields and connectivity for MLP14 and MLP15 appear in the order of the selection procedure of Sections B.5–B.6.

(a)  
![](images/f4e138af75d4a6f8fbabc371c35777f5660e7fb66904c46e9b14ff05d43a8def.jpg)

(b)  
![](images/4181693d9e235fa16cbfffef21fa79154e854d164dc1b9dea24507ed05a24725.jpg)  
Figure 20: (a) IIA of the rank-two $\left( \mathbf { v } _ { 1 } , \mathbf { v } _ { 2 } \right)$ plane patch when one sub-block's output at the position of $y _ { 2 }$ is frozen to its corrupted value, with y1 or y2 perturbed. The layer-14 attention freeze is omitted, because the pre-MLP patch is written through that output (Section B.1). (b) IIA when only the clean outputs of the listed MLPs are injected at the position of $y _ { 2 }$

![](images/f2063f51d08029138c133821711d885222c2ec01b5789ef85ae04249b7a7f59b.jpg)

(b)  
![](images/d966515ea76a73e259ce25c5ba16f6410eecd4ae39f0059ff74ff54abf505e26.jpg)

(c)  
![](images/331be7b9ad85233d872110ee359ce48e045271e037f48782ec609820f20d0844.jpg)  
Figure 21: (a) Attribution scores of the 37,888 neurons of MLP14 and MLP15, with $y _ { 1 }$ perturbed (x-axis) and $y _ { 2 }$ perturbed (y-axis), on symmetric-log axes. The 12 neurons in the top 20 of both cases are highlighted. (b) IIA of the plane patch when the top k neurons by attribution (solid) or k random neurons (dashed) are frozen. (c) IIA with no neurons frozen, with the 12 shared neurons frozen, with each case's own 20 top-ranked neurons frozen, and with 12 random neurons frozen.

![](images/f47e0a5fa0c64f68f9a4065054facc069b9ac2c0d9de669e2076abbae843541a.jpg)

![](images/958f6f60d3b18e78bc82e42f3393d74eb1bdbaf6e5ead329237e57716c056eac.jpg)

![](images/ebfb669186977b13144d5de8857d6280b526bf0d5c296b92b330f47d18774279.jpg)

![](images/0e2ab201a372f54c24a1395fb34b1c7f11c1b17f487526e3984c04fed6696946.jpg)

![](images/52bcf71f1e192b6ea375a0da30d495a9c75d04e0a88cfc47264b2a3bf816188b.jpg)

![](images/a3b5c09f42a6a57cbc474c446b214b356c50f989626db20fb82b309913f3c8b0.jpg)  
Figure 22: Receptive fields of the 12 shared neurons, six in MLP14 (top) and six in MLP15 (bottom), ordered by worst-case attribution rank. Each field is the post-SwiGLU activation at the position of $y _ { 2 }$ over a grid of $( y _ { 1 } , y _ { 2 } )$ prompts with stride 4, divided by its peak. The dashed line is $y _ { 1 } = y _ { 2 }$

![](images/2cc33ff2f3a5529ad7f67c2fa3f1dc90840a0489d3371ba5e8dd6aa3edc36ce6.jpg)  
Figure 23: Receptive fields, drawn as in Figure 22, of the six highest-ranked neurons that are in the top 20 of one perturbation case only: y1 perturbed (top) and y2 perturbed (bottom).

(a)  
![](images/e1e89710707e49829f1044fcd3a5e17360769ea1d0c0bd6f5c7a200d08a9c921.jpg)

![](images/30b05bac1e3ac6a43fbc0e62110e1e6a13d97bf9b693240e22c7684730b7a043.jpg)

![](images/e26b36a50a2be51a80cc920ab990227185053ce59aeee2e2ea53adab3d5e8685.jpg)  
Figure 24: (a,b) Causal edges from each shared MLP14 neuron (rows) to each shared MLP15 neuron (columns): the shift in the MLP15 neuron's activation, in units of its standard deviation, when only the MLP14 neuron is set to its clean value, with (a) $y _ { 1 }$ or (b) y2 perturbed. (c) Virtual weights through the gate projection, computed as the cosine between each MLP14 neuron's write direction and each MLP15 neuron's normalization-folded gate read direction (Section B.6).

![](images/866af2c9af63088c75845b3e639ab569fb34b26e3ee80381adc8c9ac75b262a0.jpg)

![](images/5a84bf7b57cdd8ec0da595717ea2d6e474c47e245f5567afd563d44ac86c115b.jpg)

(c)  
![](images/76738c14efc39f32cecc8148d2335ff6cc563abe501cb6b55b3b3afdf01dabe1.jpg)  
Figure 25: (a) IIA of a rank-one direction in the layer-15 residual at the position of $y _ { 2 } ,$ fitted in the case it is evaluated on (fitted, same case) or in the other case (fitted, other case), and of the full-rank layer-15 patch, for each perturbation case. (b,c) Histograms of the alignment $\rho _ { j }$ of the write direction of every MLP15 neuron with the direction fitted with (b) y1 or (c) y2 perturbed, on a log count axis. Vertical lines mark the six shared MLP15 neurons.

## C.6 THREE NUMBERS

A summary figure covers the positions of $y _ { 3 }$ and $y _ { 2 }$ and the last token, and the figures after it expand each part.

![](images/1c4bb16326f8b8f7f7137cf61ba6e38b22088411763f84b71eb05c53e07c4664.jpg)

![](images/e52be92a34f8724bc15f2b005af20432d5746587caeea68c86089530ff4c1d44.jpg)

(c)  
![](images/0294e9c7b6fd4bc9a4bf3ed96f9d46f93241350c134d6c0fd924c60ed2ec58f3.jpg)

![](images/e1561793d6ad4f18ba4cee8b6c80296bedf604ddab57c8714c9d13ff06f83f63.jpg)

![](images/f124a3734b441d4d59f3a115e349dc3473bbe72f4ab4a1b8fc0ef48350ba7e39.jpg)

![](images/ea4ac7aa43040c7ea40101269e4f7f3cc3278f6eda4efcde0fbe0a0c11cf7704.jpg)

(g)  
![](images/367cade24ae2bc226969f84afc3da3c2f36f04598d4dfb57035d7ce72eceacad.jpg)

(h)  
![](images/77a796acd733fe11c314c49a4ed314ef8890b034b629c7086bbc6e91b627bdaf.jpg)

![](images/fd33e3e9bb7c397e17b19f9808e4d0dc4582d38b5ae661e84b237f91b10e6fca.jpg)

![](images/bbf3c4ef94bc0ca20c9a678df14a5b8057abbace3ee6aabf23721fbc59661224.jpg)

![](images/37168a9905f001232b83c75ed0719fd8fb3693e8a5f5b7b9ac6dae5a88e81ee9.jpg)

![](images/a0ae76cbc2e88cefb0619a50e64d7640647fb13a8d71923f5ba7e90163af73d4.jpg)  
Figure 26: The three-number analysis at the position of $y _ { 3 }$ (top row), the position of $y _ { 2 }$ (middle row) and the last token (bottom row). (a) Cosine similarities between $\mathbf { u } _ { 1 } .$ u2, $\mathbf { v } _ { 1 } .$ $\mathbf { v } _ { 2 }$ and ${ \bf v } _ { 3 } ;$ the pairs among $\mathbf { v } _ { 1 } , \mathbf { v } _ { 2 } , \mathbf { v } _ { 3 }$ are boxed. (b) Position recovery at the position of $y _ { 3 } .$ , by perturbed number, for each direction alone, the three as separate rank-one patches, their span in the pre-MLP layer-14 residual, and the full layer-14 residual. (c) Receptive fields of $\mathrm { L } 1 4 \# 5 0 7 6$ (top) and L15#12784 (bottom) at the position of $y _ { 3 }$ over a $( y _ { 1 } , y _ { 2 } , y _ { 3 } )$ grid with stride 4, shown as the three pairwise marginals, each averaged over the third number and divided by its own peak. (d) The 1,500-prompt cloud projected on a rank-one direction in the layer-15 residual fitted with y3 perturbed, split by whether $y _ { 3 }$ is the maximum; the inset shows the position recovery of this direction and of the full layer-15 residual. (e-h) The same at the position of $y _ { 2 } .$ where $\mathbf { w } _ { 1 }$ and $\mathbf { w } _ { 2 }$ are rank-one directions in the layer-13 residual fitted with $y _ { 1 }$ and $y _ { 2 }$ perturbed $( \mathbf { w } _ { 2 }$ and $\mathbf { u } _ { 2 }$ are the same fit). The fields in (g) are over $( y _ { 1 } , y _ { 2 } )$ with $y _ { 3 } = 5 5 . \ \mathrm { ( i ) }$ Top two principal components of the residual leaving layer 20 at the last token, coloured by which operand holds the maximum. (j) Attention of head L20.H27 from the last token to the question tokens, averaged by the position of the maximum. (k) Position recovery of each layer-20 head's own output at the last token, for the order $y _ { 1 } > y _ { 2 } > y _ { 3 }$ . (1) Position recovery of a full-rank patch and of a patch of the top two principal components of the layer-20 residual at the last token, for three value orders; the principal components are fitted on a separate 1,500-prompt cloud.

![](images/8b517a2f739ea9a8ee11f62df06ba0b3885fad34a336d1057ee61403bbbc70b8.jpg)

![](images/493d7762d60673bc948faf9f445aff3f65753090241f02551ad97d0e297494ec.jpg)

![](images/896782c132d9d990845f99ed779015347ced45695cc937dab54244ca8c128995.jpg)

![](images/e2b455cb1426e26417872f74fda7ade1a41f14b19f234e9f4f14756d3ee624ce.jpg)

![](images/36d507b67507898d532861e42bb7e1caae9917c1f9efc9c46f8981a8b9f01e09.jpg)  
Figure 27: Component along each direction of the three-number analysis against the number it carries, as per-value means on the 1,500-prompt cloud. (a,b) $\mathbf { u } _ { 1 }$ and $\mathbf { u } _ { 2 }$ in the layer-13 residual at the positions of $y _ { 1 }$ and $y _ { 2 }$ . (c,d) $\mathbf { v } _ { 1 }$ and $\mathbf { v } _ { 2 }$ in the outputs of heads L14.H14 and L14.H18 at the position of $y _ { 3 }$ . (e) $\mathbf { v } _ { 3 }$ in the layer-13 residual at the position of $y _ { 3 }$

![](images/cc3bb0124b7f35d1b25e4d3fea7a83768221d0a4118262f19d0fb63c2187231f.jpg)

![](images/601f90bb3cceda1e11316700fc5a6bc127208f4afd94a67c7705d8c8e9463551.jpg)

![](images/0ab8e1ef42c979e80ee447aae03455dfc5a9bbf8f1f2b84649c0b7582063011b.jpg)  
Figure 28: The spaces at the position of $y _ { 3 }$ that contain the three-number directions, each in its top two principal components, coloured by the number its direction carries, with the smoothed mean position along that number (line) and the direction of the corresponding projection onto the plane (arrow). (a) The output of head L14.H14 with $\mathbf { v } _ { 1 }$ . (b) The output of head L14.H18 with $\mathbf { v } _ { 2 } . \mathbf { \rho } ( \mathbf { c } )$ The layer-13 residual with $\mathbf { v } _ { 3 }$

![](images/b73949b826c54aab950936bcdded54b0c2faf4b21905c921c7b8556e5dcd0cde.jpg)  
Figure 29: Position recovery of a rank-one patch of each direction (rows) in each perturbation case (columns). Colours are clipped at 1; printed values are not.

![](images/0118ffec423fc8ce8e42f119291ce0b59edc5980dc32aaf514d0bf3f36f6f4a5.jpg)  
Figure 30: Position recovery of each layer-14 head's own output, interchanged at the runner-up's position together with the full layer-13 residual there, for three value orders and the eight heads with the largest mean absolute effect. Dashed lines show the layer-13 patch alone for each order.

![](images/186a5bf8a47072233454fe1f91988d1c729834cb216610714077ef3593d335a6.jpg)

(b)  
![](images/0f00c37d1c94a4506d046849672af8045b1078bd3f9091d95fba5873a97d6243.jpg)

(c)  
![](images/c5a00081dcc60603814aa670ddbfac2aaa7233f0c8b49c791aaef5a71635bc06.jpg)

Figure 31: Position recovery of the patch of the span of $\mathbf { v } _ { 1 } , \mathbf { v } _ { 2 } , \mathbf { v } _ { 3 }$ in the pre-MLP layer-14 residual at the position of $y _ { 3 } ,$ by perturbed number, while parts of the network are frozen. (a) One sub-block's output at the position of $y _ { 3 }$ frozen to its corrupted value. (b) The top k neurons by this task's attribution ranking (solid) or k random neurons (dashed) frozen. (c) No neurons frozen, the 12 shared neurons of the two-number task frozen, and 12 random neurons frozen.  
(a)  
![](images/e83c679adbed2faef3ed19ab87a09d5cb60259d4234462a4f8d5e2b21772e4a1.jpg)

(b)  
![](images/caafa19c48d98cd7980ae9b1e70fce716c1e3db664f738d9a7a9ff4997bb175a.jpg)

Figure 32: Position recovery of a rank-one direction in the layer-15 residual fitted in one case (rows) and evaluated in each case (columns), with the full-rank layer-15 patch in the last row, (a) at the position of $y _ { 3 }$ and (b) at the position of $y _ { 2 } .$ Colours are clipped at 2; printed values are not.  
![](images/984f221c25d4287c43be83355b59274bab33c25ca5c40577ba2f365841d950de.jpg)

![](images/effa7400a7763cbdff1f27bd88eec25bff7050fcf8656312776607782e4638c5.jpg)

![](images/e3475cc246906c3b3b75685e0ebe8d75ed9c2dfba48e04b26505288c410d48a6.jpg)

![](images/4cc447e9c03a5a6b6e54418f197441bae3325091210dfa6eb00a5840378bbfee.jpg)

(e)  
![](images/60f53d248fbb5f2d0d940b5c1bf9965c44953abb1dd0e1e2915aa67541d61620.jpg)  
Figure 33: Top two principal components of the residual stream leaving layers 18 to $^ { 2 2 }$ at the last token, on 1,500 clean three-number prompts, coloured by which operand holds the maximum.

(a)  
![](images/ba72090ae62b3cada3d06f958ffc365fcb6f531c1e5b1c9e02e488ff61a6ba98.jpg)

![](images/a11548f1ff1048f8b8e0bc9dd59367b997033316251c8c21a402bbcb408be733.jpg)  
Figure 34: (a) Position recovery of a full-rank patch of the residual leaving each of layers 18 to 22 at the last token, for three value orders. (b) Position recovery of each layer-20 head's own output at the last token, for the eight heads with the largest effect and the same three orders. Dashed lines show a patch of the whole layer-20 attention sublayer for each order (Section B.4).

![](images/4505b3527228b33a0edcc674073c2f25b162951be5ae50312a4aa5bc2790c313.jpg)  
Figure 35: Accuracy on $\operatorname* { m a x } ( y _ { 1 } , \dots , y _ { k } )$ for 200 two-digit prompts per length, scored on the full decoded number (Section B.2): the intact model, the 12 shared neurons zeroed at every token, the same neurons zeroed only at the last token of the final operand (“last token" in the legend), and 12 random neurons zeroed at every token (two random sets, pooled).

## C.7 IIA ON THE WHOLE GENERATED NUMBER

We rescore the main-text interventions on the whole number the model generates.

![](images/cadd9ac950866b41dcda997cdab8fe7d1cd65a975d6507082c0d8f0fbe7b72e1.jpg)

![](images/cd6f5f47f435f00e4316f65c9f9894c9fe813e0537f4955e098ec45a8690513e.jpg)  
Figure 36: IIA of the number-representation patches, scored on (a) the first generated token, the leading digit of r (Section B.2), and (b) the whole generated number: the answer decoded greedily for one token more than the operands’ digit count, counted correct when its first integer equals $^ { r } \cdot$ With $y _ { 1 }$ perturbed: the full layer-13 residual and u at the position of $y _ { 1 }$ , and $\mathbf { v } _ { 1 }$ at the position of y2, patched inside the output of L14.H14 as in the main text or in the pre-MLP layer-14 residual (hatched). With $y _ { 2 }$ perturbed: the full layer-13 residual and $\mathbf { v } _ { 2 }$ at the position of $y _ { 2 } .$ The rank-one patches score the same on both, while the full-rank patches mostly generate a number other than r that starts with its leading digit.

![](images/c97bb1eafb1836c0c8102672e9e7eb3e3f210a1bc00450f77046b3ad86bc182d.jpg)

![](images/18b9df717d9f7eb1854937b4aeba361be199892f09bcace868dc7201d2dfaeb8.jpg)  
Figure 37: IIA of the patches of the main-text shared-representation figure at the position of $y _ { 2 } .$ with y1 or y2 perturbed, scored on (a) the first generated token and (b) the whole generated number, as in Figure 36: the full layer-13 residual, v2, v1, $\mathbf { v } _ { 1 }$ and $\mathbf { v } _ { 2 }$ as two separate rank-one patches, the rank-two $\left( \mathbf { v } _ { 1 } , \mathbf { v } _ { 2 } \right)$ plane in the pre-MLP layer-14 residual, and the full residual leaving layer 14. Hatched bars are the two conditions that patch both directions. Only the full-rank patches with $y _ { 2 }$ perturbed lose IIA on the whole number.

![](images/0839c36fde9026cb9e98d4d222af75509aa6bf2c5962bb7114f8d713f4ac8f0c.jpg)  
Figure 38: IIA of the rank-one direction in the layer-15 residual at the position of $y _ { 2 } ,$ fitted with $y _ { 1 }$ perturbed and evaluated in both cases, scored on the first generated token and on the whole generated number, as in Figure 36.

Table 4: IIA on the first generated token and on the whole generated number for the patches of Figures 36–38. Dashes mark cases in which a patch is not evaluated.
<table><tr><td></td><td colspan="2">y1 perturbed</td><td colspan="2">Y2 perturbed</td></tr><tr><td>patch</td><td>first digit</td><td>whole number</td><td>first digit</td><td>whole number</td></tr><tr><td>L13 full at  $y _ { 1 }$ </td><td>0.072</td><td>0.003</td><td>一</td><td>一</td></tr><tr><td>u</td><td>0.940</td><td>0.940</td><td></td><td></td></tr><tr><td> $\mathbf { v } _ { 1 }$ </td><td>0.460</td><td>0.460</td><td>0.000</td><td>0.000</td></tr><tr><td> $\mathbf { v } _ { 1 } , \mathrm { p r e - M L P }$ </td><td>0.710</td><td>0.710</td><td></td><td></td></tr><tr><td>L13 full at  $y _ { 2 }$ </td><td>0.000</td><td>0.000</td><td>0.980</td><td>0.115</td></tr><tr><td> $\mathbf { v } _ { 2 }$ </td><td>0.000</td><td>0.000</td><td>0.790</td><td>0.785</td></tr><tr><td> $\mathbf { v } _ { 1 } \ \& \ \mathbf { v } _ { 2 }$ </td><td>0.675</td><td>0.675</td><td>0.885</td><td>0.880</td></tr><tr><td> $\left( \mathbf { v } _ { 1 } , \mathbf { v } _ { 2 } \right) { \mathrm { p l a n e } }$ </td><td>0.810</td><td>0.810</td><td>0.948</td><td>0.940</td></tr><tr><td>L14 full</td><td>0.825</td><td>0.825</td><td>0.985</td><td>0.115</td></tr><tr><td>L15 DAS</td><td>1.000</td><td>1.000</td><td>0.655</td><td>0.650</td></tr></table>

Table 5: Generated answers under the patches of Figure 36, five per patch, chosen to include correct answers and mistakes: the perturbed number, the clean and corrupted operands, r, and the generated number. A patch succeeds when the model generates r.
<table><tr><td>patch</td><td>perturbed</td><td>clean  $( y _ { 1 } , y _ { 2 } )$ </td><td>corrupted  $( y _ { 1 } , y _ { 2 } )$ </td><td>r</td><td>generated</td></tr><tr><td>L13 full at y1</td><td>y1</td><td>(93, 41)</td><td>(13, 41)</td><td>13</td><td>13</td></tr><tr><td></td><td>y1</td><td>(96, 46)</td><td>(20, 46)</td><td>20</td><td>26</td></tr><tr><td></td><td>y1</td><td>(69, 42)</td><td>(13, 42)</td><td>13</td><td>12</td></tr><tr><td></td><td>y1</td><td>(60,38)</td><td>(15, 38)</td><td>15</td><td>12</td></tr><tr><td></td><td>y1</td><td>(99, 61)</td><td>(12, 61)</td><td>12</td><td>61</td></tr><tr><td>u</td><td>y1</td><td>(99, 61)</td><td>(12, 61)</td><td>12</td><td>12</td></tr><tr><td></td><td>y1</td><td>(70, 55)</td><td>(33,55)</td><td>33</td><td>33</td></tr><tr><td></td><td>y1</td><td>(89, 71)</td><td>(52, 71)</td><td>52</td><td>52</td></tr><tr><td></td><td>y1</td><td>(95, 64)</td><td>(27, 64)</td><td>27</td><td>27</td></tr><tr><td></td><td>y1</td><td>(52, 33)</td><td>(19, 33)</td><td>19</td><td>33</td></tr><tr><td> $\mathbf { v } _ { 1 }$ </td><td>y1</td><td>(99, 61)</td><td>(12, 61)</td><td>12</td><td>12</td></tr><tr><td></td><td>y1</td><td>(95, 64)</td><td>(27, 64)</td><td>27</td><td>27</td></tr><tr><td></td><td>y1</td><td>(75, 49)</td><td>(24, 49)</td><td>24</td><td>24</td></tr><tr><td></td><td>y1</td><td>(96, 46)</td><td>(20, 46)</td><td>20</td><td>20</td></tr><tr><td></td><td>y1</td><td>(70,55)</td><td>(33,55)</td><td>33</td><td>55</td></tr><tr><td>v1, pre-MLP</td><td>y1</td><td>(99, 61)</td><td>(12, 61)</td><td>12</td><td>12</td></tr><tr><td></td><td>y1</td><td>(70, 55)</td><td>(33,55)</td><td>33</td><td>33</td></tr><tr><td></td><td>y1</td><td>(95, 64)</td><td>(27, 64)</td><td>27</td><td>27</td></tr><tr><td></td><td>y1</td><td>(87, 60)</td><td>(34, 60)</td><td>34</td><td>34</td></tr><tr><td></td><td>y1</td><td>(89, 71)</td><td>(52, 71)</td><td>52</td><td>71</td></tr><tr><td>L13 full at  $y _ { 2 }$ </td><td>y2</td><td>(61,99)</td><td>(61, 12)</td><td>12</td><td>12</td></tr><tr><td></td><td>y2</td><td>(56,97)</td><td>(56, 27)</td><td>27</td><td>27</td></tr><tr><td></td><td>y2</td><td>(55, 70)</td><td>(55, 33)</td><td>33</td><td>30</td></tr><tr><td></td><td>y2</td><td>(71, 89)</td><td>(71, 52)</td><td>52</td><td>59</td></tr><tr><td></td><td>y2</td><td>(33, 52)</td><td>(33, 19)</td><td>19</td><td>52</td></tr><tr><td> $\mathbf { v } _ { 2 }$ </td><td>y2</td><td>(61,99)</td><td>(61, 12)</td><td>12</td><td>12</td></tr><tr><td></td><td>y2</td><td>(71, 89)</td><td>(71, 52)</td><td>52</td><td>52</td></tr><tr><td></td><td>y2</td><td>(44, 90)</td><td>(44, 11)</td><td>11</td><td>111</td></tr><tr><td></td><td>y2</td><td>(45, 79)</td><td>(45, 11)</td><td>11</td><td>111</td></tr><tr><td></td><td>y2</td><td>(55,70)</td><td>(55, 33)</td><td>33</td><td>55</td></tr></table>

Table 6: Generated answers, as in Table 5, under the remaining patches of Figures 37 and 38.
<table><tr><td>patch</td><td>perturbed</td><td>clean  $( y _ { 1 } , y _ { 2 } )$ </td><td>corrupted  $( y _ { 1 } , y _ { 2 } )$ </td><td>r</td><td>generated</td></tr><tr><td rowspan="5"> $\mathbf { v } _ { 1 } \ \& \ \mathbf { v } _ { 2 }$ </td><td>y1</td><td>(99, 61)</td><td>(12, 61)</td><td>12</td><td>12</td></tr><tr><td>y2</td><td>(61, 99)</td><td>(61, 12)</td><td>12</td><td>12</td></tr><tr><td>y2</td><td>(44, 90)</td><td>(44, 11)</td><td>11</td><td>111</td></tr><tr><td>y2</td><td>(45, 79)</td><td>(45, 11)</td><td>11</td><td>111</td></tr><tr><td>y2</td><td>(55,70)</td><td>(55, 33)</td><td>33</td><td>55</td></tr><tr><td rowspan="5"> $\left( \mathbf { v } _ { 1 } , \mathbf { v } _ { 2 } \right)$  plane</td><td>y1</td><td>(99, 61)</td><td>(12, 61)</td><td>12</td><td>12</td></tr><tr><td>y2</td><td>(61, 99)</td><td>(61, 12)</td><td>12</td><td>12</td></tr><tr><td>Y2</td><td>(62, 94)</td><td>(62, 11)</td><td>11</td><td>111</td></tr><tr><td>y2</td><td>(56,88)</td><td>(56, 11)</td><td>11</td><td>111</td></tr><tr><td>y1</td><td>(89, 71)</td><td>(52, 71)</td><td>52</td><td>71</td></tr><tr><td rowspan="5">L14 full</td><td>y2</td><td>(61,99)</td><td>(61, 12)</td><td>12</td><td>12</td></tr><tr><td>y1</td><td>(70,55)</td><td>(33,55)</td><td>33</td><td>33</td></tr><tr><td>Y2</td><td>(55, 70)</td><td>(55, 33)</td><td>33</td><td>30</td></tr><tr><td>Y2</td><td>(71, 89)</td><td>(71, 52)</td><td>52</td><td>59</td></tr><tr><td>y1</td><td>(99, 61)</td><td>(12, 61)</td><td>12</td><td>61</td></tr><tr><td rowspan="5">L15 DAS</td><td>y1</td><td>(99, 61)</td><td>(12, 61)</td><td>12</td><td>12</td></tr><tr><td>y1</td><td>(70,55)</td><td>(33,55)</td><td>33</td><td>33</td></tr><tr><td>y2</td><td>(62, 75)</td><td>(62, 10)</td><td>10</td><td>102</td></tr><tr><td>y2</td><td>(55, 83)</td><td>(55, 10)</td><td>10</td><td>105</td></tr><tr><td>y2</td><td>(61, 99)</td><td>(61, 12)</td><td>12</td><td>61</td></tr></table>