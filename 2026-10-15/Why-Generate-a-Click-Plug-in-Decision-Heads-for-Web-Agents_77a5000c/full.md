# Why Generate a Click? Plug-in Decision Heads for Web Agents

Chengguang Gan<sup>1</sup>, Yunhao Liang<sup>2</sup>, Hanjun Wei<sup>2</sup>,

Qinghao Zhang<sup>3</sup>, Zhixi Cai<sup>4</sup>, Shiwen Ni<sup>5</sup>

<sup>1</sup>Independent Researcher, <sup>2</sup>University of Chinese Academy of Sciences,

<sup>3</sup>Pusan National University, <sup>4</sup>Monash University,

<sup>5</sup>Shenzhen University of Advanced Technology

Correspondence: chengguangg1024@gmail.com

## Abstract

Web agents built on vision language models decide every action by generating text such as click(12), although most of that text encodes a choice among a few action types and a few dozen marked elements. We read these choices directly from the hidden states that the generator computes while it reads the goal and the screenshot. Plug-in decision heads on a frozen generator predict the action type, pool the visual tokens inside each numbered box to pick the target, and point at a listed text option or at a WRITE slot that leaves the writing of new text to the generator. A confidence cascade executes the heads’ action when they are sure and otherwise lets the generator decode from the KV cache of the same forward pass. With Qwen3-VL 4B and 8B generators trained on 213k web steps, a strict threshold keeps the cascade between 0.3 points below and 1.4 points above the generator on seven offline test sets while the heads decide 12 to 64% of the steps. A looser threshold raises step success on unseen Mind2Web websites by 3.1 and 1.8 points, costs up to 2 points on GUIAct and MiniWoB++, and lifts WebArena task success from 12.5 to 13.8% (4B) and from 13.6 to 14.6% (8B) with less than half of the generated tokens. Training the heads jointly with the generator lowers the generator’s own step success by up to 4.9 points; heads on a frozen generator avoid this cost.

## 1 Introduction

A web agent built on a vision language model acts by writing. At each step the model reads the goal and a screenshot whose elements carry numbered boxes, then generates a string such as click(6) or type(5, "chair") that a harness parses and executes (Zheng et al., 2024; Koh et al., 2024a; Xu et al., 2024; Qin et al., 2025). Most of this string records a choice from a short list: one of eleven action types and one of a few dozen boxes. Only the text of a typing or answering step is open ended.

![](images/832aa892fb4067f57f65a2a99faa18c4ee18715dd74cd55c69450b239063dae8.jpg)  
Figure 1: WebArena task 513 as solved by our 4B cascade $( \tau = 0 . 5 ,$ seed 0). On the page of step 2 the generator would spell click(6) as five generated tokens, while the heads read the type and the box from the generator’s prefill and act without generating a token. Over the episode the heads act alone twice, twice choose the action while the generator writes only the text, and hand one uncertain step to the generator (more episodes in Appendix I).

Writing a choice out token by token costs one sequential decoding pass per token, and it scores only the one string written, with no explicit distribution over whole alternative actions.

By the time the generator writes its first token, it has read the goal, the history and every box on the screen, and that first token already commits to an action type. Like other properties that probes recover from intermediate layers (Alain and Bengio, 2016), and like the knowledge that probing and editing studies locate inside large language and multimodal models (Jiang et al., 2026b, 2025a; Qi et al., 2025), the decision should therefore be readable from the hidden states of the prefill (Figure 1). This suggests a cheaper way to act: read the decision from those states with small heads, and generate only what the heads cannot express. Two questions determine whether this is worth doing. On which steps do heads read the decision as reliably as the generator writes it? And can an agent use the heads on those steps and generation on the rest without paying for both?

![](images/f0bf2a43ec3c1e89c49a6d9bfb3d68702a43cb15bd71c6318a18ac0cc8f83f33.jpg)  
Figure 2: Overview. The frozen generator reads the goal, the history, the marked screenshot and the listed text options in one prefill. Decision heads read a learned mix of five of its layers and return a distribution over action types, boxes and options (with a WRITE slot). When the confidence c<sub>t</sub> reaches τ the agent acts at once or, for WRITE, lets the generator write only the text; otherwise the generator decodes the whole action from the same KV cache.

Our answer is a set of plug-in decision heads (Figure 2). A type head pools the whole prefill, a target head pools the visual tokens inside each numbered box and scores the box against a learned query, and an argument head points at the text options listed in the prompt or at a WRITE slot that returns the writing of new text to the generator. The heads read a learned mix of five layers of a frozen generator and add about 0.1% to its parameters. A confidence cascade executes the heads’ action when the product of their probabilities reaches a threshold τ . Otherwise the generator decodes from the KV cache of the forward pass the heads already read.

We train Qwen3-VL generators of 4B and 8B parameters on 213k web steps from MiniWoB++, Mind2Web, GUIAct, UGround and WebLINX, and then train heads on each frozen generator. At τ = 0.9 the cascade matches its generator on all seven offline test sets, from 0.3 points below to 1.4 above, while the heads decide 12 to 64% of the steps. At τ = 0.5 step success on Mind2Web websites unseen in training rises by 3.1 (4B) and 1.8 (8B) points, and the four GUIAct test sets lose 1.0 to 1.9 points. On WebArena the cascade completes 13.8% and 14.6% of 174 test tasks, against 12.5% and 13.6% for the generators, and generates

3.1 and 3.3 tokens per step instead of 7.4 and 7.5; the gain comes from tasks whose intent template also occurs in training. Training the heads together with the generator in one adapter lowers the generator’s own step success by 0.5 to 4.9 points, and on Mind2Web and WebArena this cost outweighs the stronger heads that joint training yields.

Decision heads give an existing web agent a second and cheaper way to act without retraining it. The generator stays untouched, so every step the heads hand back is decided exactly as before, and every step they decide comes with an explicit distribution over types, boxes and options that a harness can threshold, log or search over (Koh et al., 2024b). Decoding is then reserved for uncertain steps and for text that the option list does not contain.

## 2 Related Work

Web agents that act by generation. Mini-WoB++ (Shi et al., 2017; Liu et al., 2018), Mind2Web (Deng et al., 2023), WebArena (Zhou et al., 2024) and VisualWebArena (Koh et al., 2024a) evaluate agents that map a page and a goal to an action. Current vision language agents write this action as text, either as element ids on Set-of-Mark screenshots (Yang et al., 2023; Zheng et al., 2024) or as screen coordinates after large-scale grounding training (Cheng et al., 2024; Hong et al., 2024; Gou et al., 2025; Wu et al., 2025; Xu et al., 2024; Qin et al., 2025), with parsers such as Omni-Parser (Lu et al., 2024) supplying the elements. Earlier agents used dedicated output layers: CC-Net (Humphreys et al., 2022) predicts factored actions with policy heads, MindAct (Deng et al., 2023) ranks candidate elements before a language model chooses among them, and GUI-Actor (Wu et al., 2026) replaces coordinate generation with an attention head over image patches, which can also be trained on a frozen backbone.

Spending less computation per decision. Early exit methods stop at an intermediate layer once a classifier is confident (Teerapittayanon et al., 2016; Xin et al., 2020; Schuster et al., 2022), selective classifiers abstain below a confidence threshold (Geifman and El-Yaniv, 2017), cascades send easy queries to cheaper models (Chen et al., 2023), speculative decoding and Medusa draft tokens that the full model verifies (Leviathan et al., 2023; Cai et al., 2024), and visual token compression shortens the multimodal prefill itself (Peng et al., 2025). Swift-Sage pairs a fast small agent with a slow large one (Lin et al., 2023). These methods save computation within text generation or across models, and the agent’s decision is still written out as text. Appendix H discusses further related work.

What is new. Earlier action heads either replace the language model, as in CC-Net, or replace generation for one part of the action, as in MindAct and GUI-Actor. Our heads read the whole action, its type, target box and text option, from the prefill of a frozen generator agent and return control to that generator through a WRITE slot or a confidence threshold, so heads and generation become two readouts of one forward pass. We also measure what joint training costs the agent’s own generation. Decision heads thus become an add-on for existing generator agents.

## 3 Plug-in Decision Heads

## 3.1 Observation and Action Space

An episode comes with a goal g. At step t the agent sees a screenshot $I _ { t }$ on which each of $N$ candidate elements is drawn as a numbered box, the history $h _ { t }$ of its earlier actions, one line each in Mind2Web’s text form with the element’s role and visible text (for example [button] Search -> CLICK), and a list $O _ { t }$ of text options, one per line: strings taken from the goal and the page and the choices of select elements, listed without the element they belong to (Appendix A). The prompt x<sub>t</sub> places $g$ and $h _ { t }$ before the image, so that under causal attention every visual token is computed with the task already in view, and puts $O _ { t }$ and the action grammar after it (Figure 2, left).

An action is a triple $a = ( y , m , z )$ of a type $y \in \mathcal { V }$ , a target box $m \in \{ 1 , \ldots , N \}$ and an argument z. The eleven types are click, type, select, hover, clear, enter, scroll down, scroll up, go back, answer and no operation; the last five take no box. Typing needs a text, selecting needs one of the element’s choices, and answering needs the answer text. The generator $G _ { \theta }$ , a vision language model with a LoRA adapter (Hu et al., 2021), writes a as a string w such as type $( 5 , \because \mathsf { c h a i r ^ { \prime \prime } } )$ and learns from demonstrations with

$$
\mathcal { L } _ { \mathrm { g e n } } ( \theta ) = - \sum _ { k = 1 } ^ { | w | } \log p _ { \theta } \left( w _ { k } \mid w _ { < k } , x _ { t } \right) .\tag{1}
$$

## 3.2 Heads on the Generator’s Prefill

Before it writes $w _ { 1 }$ , the generator runs one forward pass over $x _ { t }$ . This prefill yields hidden states $H ^ { ( \ell ) } \in \mathbb { R } ^ { T \times d }$ at every layer ℓ, the KV cache that decoding continues from, and the logits of the first token. The heads read a learned mix of five of the 36 layers, $\mathcal { S } = \{ 1 2 , 1 8 , 2 4 , 3 0 , 3 6 \}$ , in the manner of a scalar mix (Peters et al., 2018), followed by layer normalization (LN):

$$
\tilde { H } = \mathrm { L N } \Big ( \sum _ { \ell \in \cal S } \beta _ { \ell } H ^ { ( \ell ) } \Big ) , \qquad \beta = \mathrm { s o f t m a x } ( \alpha ) .\tag{2}
$$

Three learned queries $q _ { c } ~ \in ~ \mathbb { R } ^ { d _ { h } }$ , one for each head $c \in \{ \mathrm { t y p e , t g t , a r g } \}$ , summarize the whole sequence by attention pooling:

$$
u _ { c } = \sum _ { i = 1 } ^ { T } \mathrm { s o f t m a x } _ { i } \bigg ( \frac { q _ { c } ^ { \top } W _ { K } \tilde { h } _ { i } } { \sqrt { d _ { h } } } \bigg ) W _ { V } \tilde { h } _ { i } .\tag{3}
$$

Type head. $p ( y \mid x _ { t } ) = \mathrm { s o f t m a x } \big ( \mathrm { M L P } ( u _ { \mathrm { t y p e } } ) \big )$ restricted to the types the page allows: typing needs a text field among the boxes, selecting an element with choices, and hovering a menu item. The harness reads these element roles from the page.

Target head. Qwen3-VL (Bai et al., 2025) resizes the screenshot to multiples of 32 pixels and encodes each 32 × 32 cell as one visual token. We map every box into the resized image, and $P _ { m }$ holds the positions of the cells that box m touches; a box smaller than a cell receives the cell that contains it. The head averages these tokens and scores the box against the target summary:

$$
v _ { m } = \frac { 1 } { | P _ { m } | } \sum _ { i \in P _ { m } } { \tilde { h } } _ { i } , \quad s _ { m } = \frac { ( W _ { B } v _ { m } ) ^ { \top } W _ { Q } u _ { \mathrm { t g t } } } { \sqrt { d _ { h } } }\tag{4}
$$

and $p ( m \mid y , x _ { t } )$ is the softmax of $s _ { m }$ over the boxes that can take type y (every box for a click, text fields for typing). Each box is represented by what the generator computed for its pixels, so the target head needs the box coordinates and no element text.

Argument head. Let $r _ { k }$ be the hidden state at the last token of option line $k .$ In the manner of a pointer network (Vinyals et al., 2015), the head scores every option and a learned key $r _ { \mathrm { W } }$ for WRITE:

$$
o _ { k } = \frac { ( W _ { A } r _ { k } ) ^ { \top } W _ { Q ^ { \prime } } u _ { \mathrm { a r g } } } { \sqrt { d _ { h } } } , \quad o _ { \mathrm { W } } = \frac { r _ { \mathrm { W } } ^ { \top } W _ { Q ^ { \prime } } u _ { \mathrm { a r g } } } { \sqrt { d _ { h } } }\tag{5}
$$

For typing, $p ( z \mid y , m , x _ { t } )$ is the softmax over the text options and WRITE; for selecting, over the choices of box m. WRITE states that the text to type is not among the options, so the generator has to write it. Both scores condition on $y$ and m through the set of boxes or options that each softmax ranges over.

With $d _ { h } = 5 1 2$ the heads hold 6.0M parameters on the 4B generator and 9.2M on the 8B one, about 0.1% of the backbone.

## 3.3 Training on a Frozen Generator

We first train the generator alone with Eq. 1. We then freeze θ and train only the heads’ parameters $\phi$ on the same demonstrations:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { h e a d s } } ( \phi ) = - \log p ( y ^ { \star } ) - \log p ( m ^ { \star } \mid y ^ { \star } ) } \\ & { \phantom { \mathcal { L } } - \log p ( z ^ { \star } \mid y ^ { \star } , m ^ { \star } ) + 0 . 3 \mathcal { L } _ { \mathrm { d o n e } } , } \end{array}\tag{6}
$$

where $( y ^ { \star } , m ^ { \star } , z ^ { \star } )$ is the demonstrated action, $z ^ { \star }$ is WRITE whenever the demonstrated text is not a listed option, the target and argument terms vanish for types without them, and ${ \mathcal { L } } _ { \mathrm { d o n e } }$ is an auxiliary binary cross entropy, from a linear output on $u _ { \mathrm { t y p e } } ,$ on whether the step completes the task, used only as a training signal. Since θ stays fixed, the generator’s own actions do not change.

3.4 Confidence Cascade with a Shared Prefill The heads decide greedily,

$$
\begin{array} { r l } & { \boldsymbol { \hat { y } = \arg \operatorname* { m a x } _ { \boldsymbol { y } } p ( \boldsymbol { y } \mid x _ { t } ) , } } \\ & { \boldsymbol { \hat { m } = \arg \operatorname* { m a x } _ { \boldsymbol { m } } p ( \boldsymbol { m } \mid \hat { \boldsymbol { y } } , x _ { t } ) , } } \\ & { \boldsymbol { \hat { z } = \arg \operatorname* { m a x } _ { \boldsymbol { z } } p ( \boldsymbol { z } \mid \hat { \boldsymbol { y } } , \hat { \boldsymbol { m } } , x _ { t } ) , } } \end{array}\tag{7}
$$

and report the action confidence

$$
c _ { t } = p ( \hat { y } \mid x _ { t } ) p ( \hat { m } \mid \hat { y } , x _ { t } ) p ( \hat { z } \mid \hat { y } , \hat { m } , x _ { t } ) ,\tag{8}
$$

where the factor of an absent part is 1. With a threshold $\tau$ and W denoting the WRITE slot, the agent takes

$$
a _ { t } = \left\{ \begin{array} { l l } { ( \hat { y } , \hat { m } , \hat { z } ) } & { c _ { t } \geq \tau , \hat { z } \neq \mathrm { W } , } \\ { ( \hat { y } , \hat { m } , G _ { \theta } ( \cdot  { | } x _ { t } , \pi _ { t } ) ) } & { c _ { t } \geq \tau , \hat { z } = \mathrm { W } , } \\ { G _ { \theta } ( \cdot  { | } x _ { t } ) } & { c _ { t } < \tau , } \end{array} \right.\tag{9}
$$

where $\pi _ { t }$ is the start of the action string, such as type $( 5 , ~ "$ . We say the heads decide a step when $c _ { t } \geq \tau$ and act alone when, in addition, $\hat { z } \neq \mathrm { W }$

A step handed back starts decoding from the cached prefill of Section 3.2. For WRITE, the harness appends $\pi _ { t }$ to the cached prompt in one forward pass, the generator decodes at most 24 tokens, and the text up to the first quote becomes the argument. An answer always takes this path with $\pi _ { t } =$ answer $( ^ { n }$ , and its confidence is p(answer $\mid x _ { t } )$ . A handed back step therefore costs what the generator alone would spend on it plus a few matrix products, and a step on which the heads act alone costs one prefill and no decoding. A threshold above 1 recovers the generator exactly.

## 4 Experimental Setup

## 4.1 Data and Backbones

The generators are Qwen3-VL-4B-Instruct and Qwen3-VL-8B-Instruct (Bai et al., 2025) with a rank 16 LoRA adapter on every attention and MLP projection of the language model. They train for one epoch on 213k steps that we convert into the format of Section 3.1: successful MiniWoB++ episodes on training seeds (9.2k steps, repeated three times), the Multimodal-Mind2Web training split (Deng et al., 2023; Zheng et al., 2024) (7.3k), GUIAct web single and multi step tasks (Chen et al., 2025) with scroll and answer steps (94k), web grounding data from UGround (Gou et al., 2025) (65k) and WebLINX demonstrations (Lù et al., 2024) (38k). The heads train for one epoch on the same steps with the generator frozen. Appendix A describes the conversion and Appendix B the optimization.

<table><tr><td colspan="2"></td><td colspan="3">M2W</td><td colspan="4">GUIAct web</td></tr><tr><td colspan="2">System</td><td>Task</td><td>Website</td><td>Domain</td><td>Single</td><td>Multi</td><td>Single (ext.)</td><td>Multi (ext.)</td></tr><tr><td rowspan="6">4B</td><td>Generator</td><td>46.2</td><td>40.9</td><td>44.3</td><td>84.1</td><td>56.1</td><td>78.9</td><td>63.5</td></tr><tr><td>Heads only</td><td>39.7</td><td>37.9</td><td>39.1</td><td>81.0</td><td>54.7</td><td>75.8</td><td>60.3</td></tr><tr><td>Cascade, τ = 0.9</td><td>46.6</td><td>42.3</td><td>44.4</td><td>84.0</td><td>55.9</td><td>78.8</td><td>63.7</td></tr><tr><td> $\mathbf { C a s c a d e } , \tau = 0 . 5$ </td><td>47.6</td><td>44.0</td><td>44.6</td><td>82.9</td><td>54.2</td><td>77.9</td><td>61.8</td></tr><tr><td>Joint, generation</td><td>42.0</td><td>36.0</td><td>42.4</td><td>83.6</td><td>52.1</td><td>78.1</td><td>60.5</td></tr><tr><td> $\mathrm { J o i n t , c a s c a d e } \tau = 0 . 5$ </td><td>45.5</td><td>40.8</td><td>44.5</td><td>83.2</td><td>54.3</td><td>77.7</td><td>61.9</td></tr><tr><td rowspan="6">8B</td><td>Generator</td><td>50.4</td><td>44.5</td><td>48.1</td><td>84.4</td><td>58.8</td><td>79.3</td><td>64.1</td></tr><tr><td>Heads only</td><td>41.7</td><td>41.2</td><td>41.7</td><td>81.3</td><td>54.9</td><td>76.1</td><td>60.3</td></tr><tr><td>Cascade, τ = 0.9</td><td>50.6</td><td>45.9</td><td>48.1</td><td>84.1</td><td>58.7</td><td>79.0</td><td>64.0</td></tr><tr><td>Cascade, τ = 0.5</td><td>50.2</td><td>46.3</td><td>48.2</td><td>82.8</td><td>56.9</td><td>78.0</td><td>62.6</td></tr><tr><td>Joint, generation</td><td>46.1</td><td>42.1</td><td>46.2</td><td>83.6</td><td>54.9</td><td>78.6</td><td>61.4</td></tr><tr><td>Joint, cascade  $\tau = 0 . 5$ </td><td>48.7</td><td>46.5</td><td>47.4</td><td>82.9</td><td>56.4</td><td>77.8</td><td>61.8</td></tr></table>

Table 1: Offline step success (%). Our cascade (shaded) uses heads trained on the frozen generator of the first row. The heads decide 12 to 18% of the M2W steps and 27 to 64% of the GUIAct steps at $\tau = 0 . 9 .$ and 37 to 46% and 66 to 89% at $\tau = 0 . 5$

For WebArena, the generator and then its heads continue for three epochs on steps that we replay in the live sites from successful expert trajectories on 115 training tasks disjoint from the test tasks (Appendix C).

## 4.2 Benchmarks and Metrics

Offline. We use the three Multimodal-Mind2Web (M2W) test splits, cross-task, cross-website and cross-domain, whose tasks, websites or domains are unseen in training, and the GUIAct web test sets for single and multi step tasks together with their extended versions (ext.) that add scroll and answer steps. For each M2W step we cut a 1280 × 1000 window at a random height that keeps the gold element visible and draw up to 64 candidates: the gold element, the other positive candidates and random negatives. GUIAct steps likewise show the target and up to 63 other elements. This gives 1,253, 973 and 3,820 M2W steps and 1,089, 421, 1,410 and 854 GUIAct steps; Section 5.4 also counts, as failures, the 372 M2W steps whose gold element cannot be shown. A step succeeds when the chosen box is a gold box and the operation, including any text, matches.

Online. MiniWoB++ runs all 119 tasks with five seeds each through BrowserGym (Chezelles et al., 2024), at most 10 steps per episode; 12 tasks are held out from training. WebArena (Zhou et al., 2024) runs 174 test tasks on six sites with at most 25 steps per task. Success follows the WebArena checkers, with GPT-5.5 judging answers that need fuzzy matching, and every run starts from freshly reset sites.

Every success rate we report is the mean over three training seeds (per-seed results in Appendix D).

## 4.3 Compared Systems

Generator: the LoRA generator alone, decoding greedily without masks; an output that does not parse is an invalid action. Heads only: our heads act on every step; a WRITE falls back to the best listed text and an answer is left empty. Cascade: Eq. 9 with the frozen generator, at $\tau = 0 . 9$ and $\tau = 0 . 5$ . We fixed both thresholds in early Mini-WoB++ experiments: 0.9 as a strict setting and 0.5, at which the heads decide only when their action holds at least half of the probability. Joint: one adapter trained on ${ \mathcal { L } } _ { \mathrm { g e n } } + { \mathcal { L } } _ { \mathrm { h e a d s } }$ with equal weights, the same data and the same schedule, used for generation alone or in the same cascade. All systems share the prompt and the boxes. Section 5.4 compares with published M2W results, and Section 6 adds two-model cascades, a patch attention target head and a generator that writes a thought before each action.

## 5 Results

## 5.1 Offline Benchmarks

Table 1 compares the systems on the seven offline test sets. $\mathbf { A } \mathbf { t } { \boldsymbol { \tau } } = 0 . 9$ the cascade tracks its generator on every set while the heads decide between 12% (M2W cross-website) and 64% (GUIAct single, ext.) of the steps. An agent can therefore run

<table><tr><td rowspan="2">System</td><td colspan="2">MiniWoB++</td><td colspan="2">WebArena</td><td colspan="2">Tokens/step</td></tr><tr><td>4B</td><td>8B</td><td>4B</td><td>8B</td><td>4B</td><td>8B</td></tr><tr><td>Generator</td><td>53.3</td><td>55.5</td><td>12.5</td><td>13.6</td><td>7.4</td><td>7.5</td></tr><tr><td>Cascade,  $\tau = 0 . 5$ </td><td>52.5</td><td>53.6</td><td>13.8</td><td>14.6 3.1</td><td></td><td>3.3</td></tr><tr><td>Joint, generation</td><td>53.3</td><td>55.0</td><td>11.5</td><td>11.1</td><td>7.3</td><td>7.4</td></tr><tr><td>Joint, cascade</td><td>53.2</td><td>54.2</td><td>11.9</td><td>12.6</td><td>3.0</td><td>2.9</td></tr><tr><td>Joint, heads only</td><td>45.4</td><td>45.3</td><td>3.1</td><td>4.4</td><td>0</td><td>0</td></tr></table>

Table 2: Task success (%) online and tokens generated per WebArena step, counting every token the generator decodes, excluding the forced prefix of a WRITE step. Cascades use $\tau = 0 . 5$
<table><tr><td>System</td><td>Seen template (47) New template (127)</td></tr><tr><td>Generator 4B</td><td>34.8 4.2</td></tr><tr><td>Cascade 4B</td><td>41.8 3.4</td></tr><tr><td>Generator 8B</td><td>33.3 6.3</td></tr><tr><td>Cascade 8B</td><td>36.9 6.3</td></tr></table>

Table 3: WebArena task success (%) on the 47 test tasks whose intent template occurs among the 115 replayed training tasks, and on the other 127.

at $\tau = 0 . 9$ by default.

The looser threshold $\tau = 0 . 5$ moves accuracy in opposite directions on the two benchmarks. On M2W it improves every split for the 4B model, by up to 3.1 points on unseen websites, and the crosswebsite split for the 8B model by 1.8. On GUIAct it costs 1.0 to 1.9 points at both scales. Section 6.2 locates the steps responsible.

Heads only trails the generator by 1.4 to 8.7 points; the threshold confines the heads to the steps they read well. The joint rows start from a weaker generator, which Section 6.1 examines.

## 5.2 Online Agents on MiniWoB++ and WebArena

Table 2 reports the online agents. On WebArena the cascade is the strongest system at both scales. Its heads act alone on 52% (4B) and 51% (8B) of the steps and decide a WRITE on another 17% and 14%, which brings the generated tokens below half, and episodes get shorter, from 15.7 to 14.0 steps per task for the 4B agent and from 16.8 to 16.2 for the 8B one.

Table 3 splits the test tasks by intent template. A test task counts as seen when its intent matches the template of a replayed training task, as task 513 of Figure 1 matches training task 514 through “Add a {product} to my wish list”. The whole gain comes from the 47 seen tasks, where the 4B cascade completes 41.8% against 34.8% and the 8B cascade 36.9% against 33.3%. On the 127 new templates both systems stay below 7%, and the cascade trails the 4B generator by 0.8 points and ties the 8B one. On WebArena the heads therefore pay off where the replayed demonstrations cover a task’s template.

<table><tr><td>System</td><td>s/step</td><td>Speedup</td><td>Act alone</td></tr><tr><td>Generator</td><td>0.234</td><td>1.00×</td><td>0%</td></tr><tr><td>Cascade,  $\tau = 0 . 9$  Cascade,  $\tau = 0 . 5$ </td><td>0.207 0.168</td><td>1.13× 1.39×</td><td>29% 67%</td></tr><tr><td>Joint, generation Joint, cascade</td><td>0.238 0.155</td><td>1.00× 1.54×</td><td>0% 67%</td></tr></table>

Table 4: Model time per step on MiniWoB++ (4B, one agent per GPU) and share of steps on which the heads act alone. Speedups are relative to the generation that hosts the heads: the generator for our cascades, joint generation for the joint cascade.

![](images/af872672708b51968219133aa9fb9638c4b9e29f807a37a5905470409d2756f0.jpg)  
Figure 3: Step success against the share of steps the heads decide, for thresholds from 0.99 (left of each curve) to 0.5 (right). The leftmost point is the generator and the hollow marker at 100% is heads only; dashed lines give the expected success when the same share of steps goes to the heads at random. Appendix E lists all test sets.

On MiniWoB++ the cascade at $\tau = 0 . 9$ reaches 53.1% (4B) and 55.4% (8B), within 0.2 points of the generator, with the heads acting alone on 29% and 33% of the steps; at $\tau = 0 . 5$ it gives up 0.8 and 1.9 points while they act alone on 67% and 69%.

## 5.3 Speed and the Confidence Threshold

Table 4 measures the model time per step on Mini-WoB++ (protocol in Appendix F). The 4B cascade is 1.13 times faster than its generator at $\tau = 0 . 9$ and 1.39 times at $\tau = 0 . 5 ;$ success in these timing runs stays within 0.2 points of Section 5.2. The saving grows with the length of what the generator writes, which Section 6.4 measures for a generator that writes a thought before each action.

Figure 3 sweeps the threshold. On M2W crosswebsite the cascade climbs far above random allocation and peaks between $\tau = 0 . 5$ and $\tau = 0 . 7 .$ where the heads decide about a third of the steps; at $\tau = 0 . 5$ the 4B cascade reaches 44.0% where random allocation of the same share would give 39.8%. On GUIAct multi (ext.) the curve stays flat down to $\tau = 0 . 8$ and then falls toward random allocation. How low the threshold can go therefore depends on the test distribution: on M2W the cascade gains down to $\tau = 0 . 5$ , while on GUIAct it should stay at 0.8 or above.

<table><tr><td rowspan="2">Test set</td><td colspan="2">Generation</td><td colspan="2">Heads only</td></tr><tr><td>4B</td><td>8B</td><td>4B</td><td>8B</td></tr><tr><td>M2W task</td><td>-4.2</td><td>-4.3</td><td>+1.1</td><td> $+ 1 . 0$ </td></tr><tr><td>M2W website</td><td>-4.9</td><td>-2.4</td><td> $+ 1 . 3 \quad + 1 . 1$ </td><td></td></tr><tr><td>M2W domain</td><td>-1.9</td><td>-1.9</td><td> $+ 1 . 2 \quad + 1 . 0 \quad$ </td><td></td></tr><tr><td>GUIAct single</td><td>-0.5</td><td>-0.8</td><td> $+ 1 . 2 \quad + 1 . 0 \quad$ </td><td></td></tr><tr><td>GUIAct multi</td><td>-4.0</td><td>-3.9</td><td> $+ 1 . 6$ </td><td>+1.7</td></tr><tr><td>GUIAct single, ext.</td><td>-0.8</td><td>-0.7</td><td> $+ 0 . 7$ </td><td>+0.3</td></tr><tr><td>GUIAct multi, ext.</td><td>-3.0</td><td>-2.7</td><td> $+ 1 . 9$ </td><td>+0.7</td></tr><tr><td>MiniWoB++</td><td>0.0</td><td>-0.5</td><td> $+ 0 . 9$ </td><td> $+ 0 . 6$ </td></tr><tr><td>WebArena</td><td>-1.0</td><td>-2.5</td><td></td><td></td></tr></table>

Table 5: Effect of joint training, in points of success: generation of the joint adapter minus that of the separately trained generator, and joint heads minus heads trained on the frozen generator.

## 5.4 Comparison with Published Systems

Table 17 in Appendix G reports M2W under the standard protocol, where every test step counts and the 86, 46 and 240 steps outside our windows count as failures. The ordering of our systems matches Table 1: the 4B cascade improves on its generator on all three splits and the 8B cascade on cross-website and cross-domain, and the two-model cascades of Section 6.1 reach up to 3.8 points above the generator (48.8 against 45.0 on 8B cross-website). Published systems choose among their own candidate sets or the full page, and their rows serve as reference points. Aguvis 7B is ahead of all our systems on the three splits.

## 6 Analysis

## 6.1 Joint Training Costs Generation

Table 5 shows what joint training costs. The joint adapter generates worse on all seven offline sets at both scales, by up to 4.9 points, and its generation loss on the validation steps rises from 0.214 to 0.235 (4B) and from 0.197 to 0.218 (8B), both for seed 0. The drop in step success is largest on the M2W cross-task and cross-website splits and on multi step GUIAct, and it carries over to

<table><tr><td rowspan="2">System</td><td colspan="2">M2W</td><td colspan="2">GUIAct</td></tr><tr><td>Task Web.</td><td>Dom.</td><td>Single Multi</td><td></td></tr><tr><td>Cascade 4B,  $\tau = 0 . 5$  Two models 4B,  $\tau = 0 . 5$ </td><td>47.6 44.0 47.9 44.3</td><td>44.6 46.1</td><td>82.9 83.6</td><td>54.2 56.0</td></tr><tr><td>Two models 4B,  $\tau = 0 . 9$ </td><td>46.7 41.9</td><td>44.5</td><td>84.3</td><td>56.0</td></tr><tr><td>4B heads, 8B gen.</td><td>47.8 45.6</td><td>47.2</td><td>82.9</td><td>56.5</td></tr><tr><td>Cascade 8B,  $\tau = 0 . 5$ </td><td>50.2 46.3</td><td>48.2</td><td>82.8</td><td>56.9</td></tr><tr><td>Two models  $8 \mathbf { B } , \tau = 0 . 5$ </td><td>51.4 48.5</td><td>49.1</td><td>83.7</td><td>57.7</td></tr><tr><td>Two models  $8 \mathbf { B } , \tau = 0 . 9$ </td><td>50.7 45.5</td><td>48.3</td><td>84.3</td><td>58.5</td></tr></table>

Table 6: Offline step success (%) of cascades whose heads come from the joint adapter and whose generator is the separately trained one. The fourth row pairs 4B joint heads with the generation of the 8B joint adapter at $\tau = 0 . 5$

WebArena.

Joint training gives stronger heads in exchange, and heads only gains 0.3 to 1.9 points. On Mini-$\mathrm { W o B + + }$ , where joint training leaves generation intact, the joint cascade reaches 53.2% and 54.2% against 52.5% and 53.6% for our cascade, and on GUIAct the two cascades at $\tau = 0 . 5$ stay within 0.8 points of each other. On WebArena and on five of the six M2W columns our cascade is ahead, because it starts from the stronger generator.

A two-model cascade combines both strengths: the joint heads decide, and uncertain steps go to the separately trained generator. Table 6 shows that this gives the best M2W results at both scales, 51.4, 48.5 and 49.1 at 8B, at the cost of a second adapter in memory and a second prefill on every handed back step. On MiniWoB++ the 4B two-model cascade reaches 53.1% at $\tau = 0 . 5$ and 52.8% at $\tau = 0 . 9$ , and pairing 4B joint heads with the 8B joint adapter reaches 54.2%, between the two generators.

## 6.2 Where Heads Beat Generation

Table 7 splits each test set into the steps the heads decide at $\tau \ : = \ : 0 . 5$ and the rest. On M2W the heads solve their steps more often than the generator would, by 8.3 points on unseen websites, and the generator solves only 29 to 33% of the steps they leave. On GUIAct the generator is ahead on the heads’ steps by 1.4 and 2.9 points; weighted by the 85% and 66% of steps the heads decide, this gives the 1.2 and 1.9 point loss of Table 1.

Typing improves on every M2W split, by 9.1, 9.1 and 6.3 points (Appendix E): the head points at goal text when it is listed and otherwise passes only the text to the generator. Selecting improves most on unseen websites, from 14.8% to 39.8% of

<table><tr><td colspan="2"></td><td colspan="2">Decided steps</td><td>Rest</td></tr><tr><td>Test set</td><td>Share</td><td>Heads</td><td>Gen.</td><td>Gen.</td></tr><tr><td>M2W task</td><td>38%</td><td>72.2</td><td>68.4</td><td>32.7</td></tr><tr><td>M2W website</td><td>37%</td><td>68.6</td><td>60.3</td><td>29.4</td></tr><tr><td>M2W domain</td><td>40%</td><td>65.1</td><td>64.3</td><td>31.1</td></tr><tr><td>GUIAct single</td><td>85%</td><td>87.4</td><td>88.8</td><td>56.5</td></tr><tr><td>GUIAct multi</td><td>66%</td><td>67.3</td><td>70.2</td><td>28.5</td></tr><tr><td colspan="3">M2W website by operation 33%</td><td>Gen. → Cascade</td><td></td></tr><tr><td colspan="3">Click (781 steps)</td><td> $4 5 . 8  4 6 . 0$ </td><td></td></tr><tr><td colspan="3">Type (120 steps) 51% 56%</td><td> $2 4 . 2  3 3 . 3$ </td><td> $1 4 . 8  3 9 . 8 $ </td></tr></table>

Table 7: The 4B cascade at $\tau = 0 . 5 .$ Top: step success on the steps the heads decide, by the heads and by the generator, and on the remaining steps by the generator. Bottom: step success by operation on M2W cross-website.
<table><tr><td colspan="3">M2W</td><td colspan="2">GUIAct</td></tr><tr><td>Target head</td><td>Task Web.</td><td>Dom.</td><td>Single</td><td>Multi</td></tr><tr><td>Patch attention</td><td>35.3</td><td>34.7</td><td>37.8</td><td>54.4 11.7</td></tr><tr><td>Box pooling</td><td>37.3 35.2</td><td>39.2</td><td>54.2</td><td>13.5</td></tr></table>

Table 8: Heads only step success (%) with two target heads, both trained as joint heads on MiniWoB++ and M2W; GUIAct is held out from this training.

72 steps, and by 4.0 points on unseen tasks. The generator has to pair each select element with its choices in the option list and copy the text, while the argument head scores only the choices of the box it selected. Clicks change by +0.1, +0.2 and −0.8 points while the heads decide about a third of them.

## 6.3 Target Head and Layer Weights

Box pooling against patch attention. We replace our target head with the patch attention head of GUI-Actor (Wu et al., 2026): the target query attends over every visual token covered by a candidate, training raises the attention mass on the tokens of the gold box, and at test time the top token selects the smallest candidate that contains it. We train both variants as joint heads on the Mini-WoB++ and M2W portion of the data. Heads only with box pooling is ahead on four of five test sets (Table 8), for instance 37.3 against 35.3 on M2W cross-task. Box pooling gives every candidate its own score. Scoring each box by the attention mass inside it instead fails on M2W, where 82% of the test steps contain a larger candidate that encloses the gold box and always collects at least as much mass; the smallest-candidate rule of GUI-Actor is needed to avoid this.

<table><tr><td>System</td><td>Success</td><td>s/step</td><td>Act alone</td></tr><tr><td>Generator, no thought</td><td>53.9</td><td>0.24</td><td>0%</td></tr><tr><td>Reasoning generator</td><td>49.9</td><td>1.65</td><td>0%</td></tr><tr><td>Joint, generation</td><td>51.0</td><td>1.56</td><td>0%</td></tr><tr><td>Joint, cascade  $\tau = 0 . 5$ </td><td>51.5</td><td>0.70</td><td>66%</td></tr></table>

Table 9: MiniWoB++ success (%) of 4B generators that write a thought before the action. All rows are trained on MiniWoB++ only; the joint rows train the reasoning generator and the heads in one adapter.

Layer weights. The learned mix (Eq. 2) gives the largest weight to layer 24 (0.34) and the smallest to the last layer (0.07) on the 4B generator, with the same profile on the 8B one. Under joint training the largest weight moves to layer 30 (0.38).

## 6.4 Heads on a Reasoning Generator

A generator can also write a short thought before its action, in the style of ReAct (Yao et al., 2022). We train such a 4B generator on MiniWoB++ only. Its thoughts come from sampling four thought and action pairs per step from the untuned 8B model and keeping a thought whose action matches the demonstration, in the spirit of STaR (Zelikman et al., 2022); this covers 57% of the 7,675 deduplicated steps, and the others receive a thought written with the demonstrated action in view. Table 9 shows that thoughts do not help here: the reasoning generator trails the generator without thoughts by 4.0 points and is seven times slower.

The heads save the most on such a generator, because every step they take alone skips a whole thought. Trained in the same adapter, the joint heads act alone on two thirds of the steps, raise success from 51.0 to 51.5% over the adapter’s own generation, and cut the time per step from 1.56 to 0.70 seconds. With thoughts, the joint adapter also generates better than the separately trained reasoning generator (51.0 against 49.9).

## 7 Conclusion

A web agent can read many of its decisions from the prefill it already computes. Heads on a frozen generator decide the steps they read well and pass the rest, together with any new text, to the same generator. At a strict threshold the agent keeps the generator’s accuracy, and at a looser one it gains on unseen websites and on WebArena while generating less than half of the tokens.

## Limitations

Our heads choose among numbered boxes, so agents that act through raw screen coordinates would need a different target head. The heads are tied to one generator; after fine-tuning the generator for WebArena we trained them again. All experiments use Qwen3-VL backbones and a single action grammar, and the threshold is fixed per experiment.

## References

Guillaume Alain and Yoshua Bengio. 2016. Understanding intermediate layers using linear classifier probes. arXiv preprint arXiv:1610.01644.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. 2025. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631.

Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D Lee, Deming Chen, and Tri Dao. 2024. Medusa: Simple LLM inference acceleration framework with multiple decoding heads. arXiv preprint arXiv:2401.10774.

Lingjiao Chen, Matei Zaharia, and James Zou. 2023. FrugalGPT: How to use large language models while reducing cost and improving performance. arXiv preprint arXiv:2305.05176.

Wentong Chen, Junbo Cui, Jinyi Hu, Yujia Qin, Junjie Fang, Yue Zhao, Chongyi Wang, Jun Liu, Guirong Chen, Yupeng Huo, et al. 2025. GUICourse: From general vision language model to versatile GUI agent. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 21936–21959.

Kanzhi Cheng, Qiushi Sun, Yougang Chu, Fangzhi Xu, Li YanTao, Jianbing Zhang, and Zhiyong Wu. 2024. SeeClick: Harnessing GUI grounding for advanced visual GUI agents. In Proceedings of the 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 9313– 9332.

De Chezelles, Thibault Le Sellier, Sahar Omidi Shayegan, Lawrence Keunho Jang, Xing Han Lù, Ori Yoran, Dehan Kong, Frank F Xu, Siva Reddy, Quentin Cappart, et al. 2024. The BrowserGym ecosystem for web agent research. arXiv preprint arXiv:2412.05467.

Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Sam Stevens, Boshi Wang, Huan Sun, and Yu Su. 2023. Mind2Web: Towards a generalist agent for the web. Advances in Neural Information Processing Systems, 36:28091–28114.

Baochen Fu, Wenzhi Deng, Baihao Jin, Yang Li, Zihan Nie, Kailin Jiang, Yuntao Du, and Weiye Song. 2026. Can multimodal large language models understand OCT? arXiv preprint arXiv:2607.16609.

Yonatan Geifman and Ran El-Yaniv. 2017. Selective classification for deep neural networks. Advances in neural information processing systems, 30.

Boyu Gou, Demi Ruohan Wang, Boyuan Zheng, Yanan Xie, Cheng Chang, Yiheng Shu, Huan Sun, and Yu Su. 2025. Navigating the digital world as humans do: Universal visual grounding for GUI agents. In International Conference on Learning Representations, volume 2025, pages 30851–30883.

Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, et al. 2024. CogAgent: A visual language model for GUI agents. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14281–14290. IEEE.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. LoRA: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685.

Peter C Humphreys, David Raposo, Tobias Pohlen, Gregory Thornton, Rachita Chhaparia, Alistair Muldal, Josh Abramson, Petko Georgiev, Adam Santoro, and Timothy Lillicrap. 2022. A data-driven approach for learning to control computers. In International Conference on Machine Learning, pages 9466–9482. PMLR.

Yifan Jia, Yuntao Du, Kailin Jiang, Yuyang Liang, Qihan Ren, Yi Xin, Rui Yang, Fenze Feng, MingCai Chen, Hengyang Lu, et al. 2026. Benchmarking multimodal knowledge conflict for large multimodal models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 22283– 22291.

Kailin Jiang, Yuntao Du, Yukai Ding, Yuchen Ren, Ning Jiang, Zhi Gao, Zilong Zheng, Lei Liu, Bin Li, and Qing Li. 2026a. When large multimodal models confront evolving knowledge: Challenges and explorations. In The Fourteenth International Conference on Learning Representations.

Kailin Jiang, Zhi Gao, Chenrui Shi, Zilong Zheng, Siyuan Qi, Qing Li, et al. 2025a. MMKE-Bench: A multimodal editing benchmark for diverse visual knowledge. In International Conference on Learning Representations, volume 2025, pages 526–555.

Kailin Jiang, Hongbo Jiang, Ning Jiang, Zhi Gao, Jinhe Bi, Yuchen Ren, Bin Li, Yuntao Du, Lei Liu, and Qing Li. 2025b. KORE: Enhancing knowledge injection for large multimodal models via knowledge-oriented augmentations and constraints. arXiv preprint arXiv:2510.19316.

Kailin Jiang, Ning Jiang, Yuntao Du, Yuchen Ren, Yuchen Li, Yifan Gao, Jinhe Bi, Yunpu Ma, Bin Li, Lei Liu, et al. 2026b. MINED: Probing and updating with multimodal time-sensitive knowledge for large multimodal models. In Findings ofthe Association for Computational Linguistics: ACL 2026, pages 13766–13795.

Kailin Jiang, Lei Liu, Jian Xi, Yangqi Chen, Hui Xu, Hongwei Zhao, Bin Li, Yu Lu, and Haibo Shi. 2026c. AdaTutoRank: Learning to rerank document sets via adaptive tutoring optimization for RAG and deep research. arXiv preprint arXiv:2609.32472.

Kailin Jiang, Lei Liu, Jian Xi, Hui Xu, Junlin Liu, Baochen Fu, Bin Li, Yu Lu, Haibo Shi, et al. 2026d. Beyond relevance-centric retrieval: Rubric-oriented document set selection and ranking. arXiv preprint arXiv:2607.19747.

Jing Yu Koh, Robert Lo, Lawrence Jang, Vikram Duvvur, Ming Lim, Po-Yu Huang, Graham Neubig, Shuyan Zhou, Russ Salakhutdinov, and Daniel Fried. 2024a. VisualWebArena: Evaluating multimodal agents on realistic visual web tasks. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 881–905.

Jing Yu Koh, Stephen McAleer, Daniel Fried, and Ruslan Salakhutdinov. 2024b. Tree search for language model agents. arXiv preprint arXiv:2407.01476.

Yaniv Leviathan, Matan Kalman, and Yossi Matias. 2023. Fast inference from transformers via speculative decoding. In International conference on machine learning, pages 19274–19286. PMLR.

Ruihan Li, Jiyang Tan, Kailin Jiang, Huining Li, Hengyang Lu, Yu Huang, Qian Li, and Yuntao Du. 2026. KnowHal: A knowledge-driven benchmark for comprehensive multimodal hallucination evaluation. arXiv preprint arXiv:2608.03782.

Bill Yuchen Lin, Yicheng Fu, Karina Yang, Faeze Brahman, Shiyu Huang, Chandra Bhagavatula, Prithviraj Ammanabrolu, Yejin Choi, and Xiang Ren. 2023. SwiftSage: A generative agent with fast and slow thinking for complex interactive tasks. Advances in Neural Information Processing Systems, 36:23813– 23825.

Evan Zheran Liu, Kelvin Guu, Panupong Pasupat, Tianlin Shi, and Percy Liang. 2018. Reinforcement learning on web interfaces using workflow-guided exploration. arXiv preprint arXiv:1802.08802.

Junlin Liu, Jiangwang Chen, Zixin Song, Shuaiyu Zhou, Chunji Lv, Hank Wu, Kailin Jiang, Jinyang Wu, Bohan Yu, and Chenxi Zhou. 2026. From proprietary to open-source: Bridging the distribution gap via multiagent protocol distillation in agentic search. arXiv preprint arXiv:2607.24280.

Ilya Loshchilov and Frank Hutter. 2017. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101.

Xing Han Lù, Zdenek Kasner, and Siva Reddy. 2024.ˇ WebLINX: Real-world website navigation with multiturn dialogue. arXiv preprint arXiv:2402.05930.

Yadong Lu, Jianwei Yang, Yelong Shen, and Ahmed Awadallah. 2024. OmniParser for pure vision based GUI agent. arXiv preprint arXiv:2408.00203.

Tianfan Peng, Yuntao Du, Pengzhou Ji, Shijie Dong, Kailin Jiang, Mingchuan Ma, Yijun Tian, Jinhe Bi, Qian Li, Wei Du, et al. 2025. Can visual input be compressed? a visual token compression benchmark for large multimodal models. arXiv preprint arXiv:2511.02650.

Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. 2018. Deep contextualized word representations. In Proceedings ofthe 2018 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 2227–2237, New Orleans, Louisiana. Association for Computational Linguistics.

Siyuan Qi, Bangcheng Yang, Kailin Jiang, Xiaobo Wang, Jiaqi Li, Yifan Zhong, Yaodong Yang, and Zilong Zheng. 2025. In-context editing: Learning knowledge from self-induced distributions. In International Conference on Learning Representations, volume 2025, pages 77563–77585.

Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, et al. 2025. UI-TARS: Pioneering automated GUI interaction with native agents. arXiv preprint arXiv:2501.12326.

Tal Schuster, Adam Fisch, Jai Gupta, Mostafa Dehghani, Dara Bahri, Vinh Tran, Yi Tay, and Donald Metzler. 2022. Confident adaptive language modeling. Advances in Neural Information Processing Systems, 35:17456–17472.

Tianlin Shi, Andrej Karpathy, Linxi Fan, Jonathan Hernandez, and Percy Liang. 2017. World of Bits: An open-domain platform for web-based agents. In International Conference on Machine Learning, pages 3135–3144. PMLR.

Surat Teerapittayanon, Bradley McDanel, and Hsiang-Tsung Kung. 2016. BranchyNet: Fast inference via early exiting from deep neural networks. In 2016 23rd international conference on pattern recognition (ICPR), pages 2464–2469. IEEE.

Oriol Vinyals, Meire Fortunato, and Navdeep Jaitly. 2015. Pointer networks. Advances in neural information processing systems, 28.

Qianhui Wu, Kanzhi Cheng, Rui Yang, Chaoyun Zhang, Jianwei Yang, Huiqiang Jiang, Jian Mu, Baolin Peng, Bo Qiao, Reuben Tan, et al. 2026. GUI-Actor: Coordinate-free visual grounding for GUI agents. Advances in Neural Information Processing Systems, 38:15101–15128.

Zhiyong Wu, Zhenyu Wu, Fangzhi Xu, Yian Wang, Qiushi Sun, Chengyou Jia, Kanzhi Cheng, Zichen Ding, Liheng Chen, Paul Pu Liang, et al. 2025. OS-ATLAS: Foundation action model for generalist GUI agents. In International Conference on Learning Representations, volume 2025, pages 5090–5108.

Ji Xin, Raphael Tang, Jaejun Lee, Yaoliang Yu, and Jimmy Lin. 2020. DeeBERT: Dynamic early exiting for accelerating BERT inference. In Proceedings of the 58th annual meeting of the association for computational linguistics, pages 2246–2251.

Yiheng Xu, Zekun Wang, Junli Wang, Dunjie Lu, Tianbao Xie, Amrita Saha, Doyen Sahoo, Tao Yu, and Caiming Xiong. 2024. Aguvis: Unified pure vision agents for autonomous GUI interaction. arXiv preprint arXiv:2412.04454.

Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. 2023. Set-of-Mark prompting unleashes extraordinary visual grounding in GPT-4V. arXiv preprint arXiv:2310.11441.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2022. ReAct: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629.

Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Goodman. 2022. STaR: Bootstrapping reasoning with reasoning. Advances in Neural Information Processing Systems, 35:15476–15488.

Boyuan Zheng, Boyu Gou, Jihyung Kil, Huan Sun, and Yu Su. 2024. GPT-4V(ision) is a generalist web agent, if grounded. arXiv preprint arXiv:2401.01614.

Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, et al. 2024. WebArena: A realistic web environment for building autonomous agents. In International Conference on Learning Representations, volume 2024, pages 15585–15606.

## A Data Conversion

<table><tr><td>Source</td><td>Steps</td><td>Main operations</td></tr><tr><td>MiniWoB++</td><td>9,171</td><td>click, type, no op</td></tr><tr><td>Mind2Web train</td><td>7,320</td><td>click, type, select</td></tr><tr><td>GUIAct single</td><td>66,089</td><td>click, scroll, answer</td></tr><tr><td>GUIAct multi</td><td>27,700</td><td>click, scroll, answer</td></tr><tr><td>UGround</td><td>65,066</td><td>click</td></tr><tr><td>WebLINX</td><td>38,171</td><td>click, type, scroll</td></tr><tr><td>Total</td><td>213,517</td><td></td></tr><tr><td>WebArena replay</td><td>1,298</td><td>click, type, answer</td></tr></table>

Table 10: Converted training steps per source, before deduplication. MiniWoB++ excludes the held-out tasks; the WebArena replay is used only for the WebArena fine-tuning.

Every source is converted into the same step record: a screenshot, the candidate boxes in reading order, the element list with roles and names, the goal, the history and the demonstrated action in our grammar (Table 10). Boxes are drawn at load time with a colored frame and the number inside the top left corner, so that the label never covers a neighboring element. Web pages are rendered 960 pixels wide. After removing 4,472 duplicate steps and 426 steps whose action cannot be parsed, 208,619 steps remain, 7,675 of them from MiniWoB++. We hold out 10,430 steps for validation and repeat the remaining MiniWoB++ steps three times, which gives 212,761 training examples.

MiniWoB++. Episodes come from successful rollouts of earlier text based policies and scripted demonstrations on training seeds, with a screenshot saved at every step. The task area is cropped and enlarged by a factor of two. Twelve tasks from ten families (for example bisect-angle, enter-password, use-colorwheel) are removed from training and kept for evaluation.

Mind2Web. The window of Section 4.2 is scaled by 0.75. Negative containers that cover more than a quarter of the window and boxes smaller than 6 pixels are dropped. Hover and enter count as click, as in the original benchmark. The history uses the dataset’s own text form of previous actions, and all positive candidates inside the window are gold boxes.

GUIAct. Each screenshot ships with its element list, which gives the candidates. Click, hover, input and select become our actions, and the extended versions keep scroll and answer as well. Multi step rows are grouped into episodes by record.

UGround and WebLINX. UGround screenshots carry many referring expressions with boxes; every box becomes a candidate, and two expressions per screenshot (32,533 screenshots) become goals of the form “Click the element: . . . ”. WebLINX demonstrations give clicks, text input and selections on real websites; the goal is the instructor’s last two utterances, and candidates are the visible elements of the page’s box file.

Text options. The options listed after the image are, in order: each quoted span of the goal in its original, upper, lower and capitalized form; when the goal has no quotes, each capitalized word after the first and each number in the goal, together with the names (at most three words and 25 characters) of page elements whose role is cell, text, label or list item, up to 16 strings; then the choices of every select element. The list is cut at 24 lines, each written as “- text .”. In 7,641 training steps the demonstrated text is not among the options.

## B Training Details

Table 11 lists the settings. We optimize with AdamW (Loshchilov and Hutter, 2017). The generator and the joint adapter train with data parallelism on two GPUs; the 4B joint adapter takes about eleven hours. Heads on a frozen generator train with the backbone in inference mode, so only the heads keep optimizer state. On the validation steps of seed 0, the frozen 4B heads predict the type correctly 92.0% of the time, the target box 74.6% and the argument 86.1%; the 8B heads reach 92.2%, 75.2% and 87.2%. The 8B heads weight layers 12 to 36 by 0.24, 0.25, 0.30, 0.14 and 0.08, and the joint 4B heads by 0.14, 0.13, 0.21, 0.38 and 0.14.

<table><tr><td>Setting</td><td>Value</td></tr><tr><td>LoRA rank, scale, dropout LoRA targets</td><td>16, 32, 0.05 attention and MLP</td></tr><tr><td>LoRA parameters (4B / 8B)</td><td>33.0M / 43.6M</td></tr><tr><td></td><td></td></tr><tr><td>Head parameters (4B / 8B)</td><td>6.0M / 9.2M</td></tr><tr><td>Head width  $d _ { h }$ </td><td>512</td></tr><tr><td>Layers read</td><td>12, 18, 24, 30, 36</td></tr><tr><td>Optimizer</td><td>AdamW, weight decay 0</td></tr><tr><td>LR (LoRA / heads)</td><td> $1 0 ^ { - 4 } / 1 0 ^ { - 3 }$ </td></tr><tr><td>Schedule</td><td>5% warmup, cosine</td></tr><tr><td>Gradient clipping</td><td>1.0</td></tr><tr><td>Examples per update</td><td>16</td></tr><tr><td>Epochs</td><td>1</td></tr><tr><td>Train / val. examples</td><td>212,761 /10,430</td></tr><tr><td>WebArena tuning</td><td>3 epochs, LR of the above</td></tr><tr><td>Decoding</td><td>greedy, at most 24 tokens</td></tr><tr><td>GPU</td><td>RTX PRO 6000, 96 GB</td></tr></table>

Table 11: Training settings shared by all runs. Generators that write a thought (Section 6.4) train for three epochs on MiniWoB++ only and decode at most 160 tokens.

## C WebArena Harness and Expert Replay

Split and replay. We use 174 test tasks of WebArena (gitlab 44, shopping admin 39, shopping 34, reddit 29, map 24, wikipedia 4) and a disjoint set of training tasks, split at the task level. Expert trajectories come from a GPT-5.5 Set-of-Mark agent on the training tasks; we keep its 234 successful trajectories on 139 tasks. Because our boxes differ from the expert’s, we replay each trajectory in the live sites, record at every step what our agent would see (the viewport scaled from 1440 × 900 to 960 × 600, our candidate boxes and the history), and write the expert action in our grammar. A step whose target is not among our boxes is executed but not kept, and a trajectory is dropped when the page diverges or the replay is not scored successful again. This yields 191 trajectories on 115 tasks with 1,298 steps; after removing 369 duplicates and 13 unparsable steps, 871 steps train and 45 validate.

Fine-tuning and evaluation. The joint adapter is fine-tuned on ${ \mathcal { L } } _ { \mathrm { g e n } } + { \mathcal { L } } _ { \mathrm { h e a d s } }$ with the same schedule. Each run resets the six sites, logs in again, and runs one worker per site copy with three workers per GPU. Actions run on the live page: type replaces a field’s content, enter presses Enter on a field, answer ends the episode with the text. The generator decodes at most 24 tokens per action, and an action that does not parse is an invalid step; on the WRITE path the text is cut at the first quote and at 60 characters. Tasks that hit an environment error count as failures for every system.

<table><tr><td>System</td><td>Shop</td><td>Admin</td><td>Reddit</td><td>GitLab</td><td>Map</td><td>Wiki</td></tr><tr><td>Generator 4B</td><td>11.8</td><td>8.5</td><td>20.7</td><td>12.1</td><td>12.5</td><td>0.0</td></tr><tr><td>Cascade 4B</td><td>17.6</td><td>4.3</td><td>23.0</td><td>15.2</td><td>12.5</td><td>0.0</td></tr><tr><td>Generator 8B</td><td>10.8</td><td>14.5</td><td>17.2</td><td>16.7</td><td>8.3</td><td>0.0</td></tr><tr><td>Cascade 8B</td><td>18.6</td><td>11.1</td><td>19.5</td><td>14.4</td><td>11.1</td><td>0.0</td></tr></table>

Table 12: WebArena task success (%) per site.

Table 12 breaks the results down by site. The cascade gains most on the shopping site at both scales, by 5.8 (4B) and 7.8 (8B) points, and loses on the shopping admin site. $\mathbf { A t } \boldsymbol { \tau } = 0 . 9$ the joint 4B cascade completes 11.9% of the tasks, the same as at $\tau = 0 . 5$ , while its heads act alone on 12% of the steps, against 51% at $\tau = 0 . 5$

<table><tr><td>System</td><td>M2W task</td><td>M2W website</td><td>M2W domain</td><td>GUIAct single</td><td>GUIAct multi</td><td>Single, ext.</td><td>Multi, ext.</td></tr><tr><td rowspan="4">Generator 4B</td><td>45.8/46.3/46.4</td><td></td><td>40.2/40.7/41.7</td><td>44.1/44.4/44.5 84.8/84.0/83.4</td><td>54.4/56.8/57.0</td><td>79.4/79.0/78.4</td><td>62.8/64.1/63.8</td></tr><tr><td>Cascade 0.9 Cascade 0.5</td><td>46.1/46.3/47.2</td><td>41.6/42.2/43.144.3/44.5/44.5</td><td></td><td>84.8/83.8/83.454.4/56.8/56.5</td><td>79.4/78.8/78.2</td><td>63.0/64.3/63.7</td></tr><tr><td></td><td>46.9/47.9/48.0 39.3/40.0/39.7</td><td>43.9/44.2/43.8</td><td>44.5/44.9/44.5</td><td>84.0/82.3/82.453.4/54.4/54.6</td><td>78.7/77.4/77.4</td><td>61.6/62.5/61.4</td></tr><tr><td>Heads only</td><td></td><td>38.1/37.8/37.7</td><td>39.0/39.3/39.0 82.6/80.0/80.4</td><td>52.7/56.5/54.9</td><td>77.0/75.0/75.5</td><td>60.0/61.7/59.4</td></tr><tr><td rowspan="4">8B Heads only</td><td>Generator</td><td>49.6/51.4/50.1</td><td>45.1/44.7/43.7</td><td>47.8/48.1/48.3 84.7/84.8/83.7</td><td>58.7/59.6/58.2</td><td>79.1/79.7/78.9</td><td>63.5/64.2/64.6</td></tr><tr><td>Cascade 0.9</td><td></td><td>50.0/51.4/50.446.5/46.2/44.9</td><td>47.7/48.3/48.3 84.7/84.5/83.3</td><td></td><td>58.7/59.6/57.779.3/79.3/78.5</td><td>63.6/64.2/64.3</td></tr><tr><td>Cascade 0.549.6/50.8/50.146.1/46.8/46.047.7/48.5/48.4</td><td></td><td></td><td></td><td>82.9/83.5/82.156.8/57.7/56.378.1/78.5/77.461.9/62.8/63.2</td><td></td><td></td></tr><tr><td></td><td></td><td>42.3/41.6/41.340.3/42.1/41.141.8/41.6/41.781.5/81.6/80.755.3/55.3/54.276.4/76.5/75.560.0/60.3/60.7</td><td></td><td></td><td></td></tr></table>

Table 13: Offline step success (%) for seeds 0 / 1 / 2.
<table><tr><td>System</td><td>4B 8B</td></tr><tr><td>MiniWoB++ Generator 54.1 / 53.3 / 52.456.1 / 55.5 / 55.0 Cascade 0.954.1 / 53.1 / 52.155.6 / 55.6 / 54.8 Cascade 0.552.9 / 53.4 / 51.353.1 / 54.3 / 53.4 Heads only 44.9 / 44.9 / 43.944.4 / 44.4 / 45.4</td></tr><tr><td>WebArena Generator 12.1 / 11.5 / 13.813.8 / 14.9 / 12.1 Cascade 0.513.2 / 13.8 / 14.414.9 / 15.5 / 13.2</td></tr></table>

Table 14: Online task success (%) for seeds 0 / 1 / 2.

## D Per-Seed Results

Tables 13 and 14 list every seed of the main systems. A seed fixes the training run: the generator of seed s hosts the heads of seed s, and both online benchmarks use the same environment seeds and tasks for all systems.

## E Threshold Curves and Operations

<table><tr><td></td><td>Test set</td><td>τ = 0.5</td><td>0.6</td><td>0.7</td><td>0.8</td><td>0.9</td><td>0.95</td><td>0.99</td><td>Generator</td></tr><tr><td rowspan="7">4B</td><td>M2W task</td><td>47.6 (38)</td><td>47.6 (30)</td><td>47.2 (23)</td><td>47.0 (18)</td><td>46.6 (13)</td><td>46.4 (9)</td><td>46.2 (5)</td><td>46.2</td></tr><tr><td>M2W website</td><td>44.0 (37)</td><td>43.9 (29)</td><td>43.4 (23)</td><td>42.8 (18)</td><td>42.3 (12)</td><td>42.0 (8)</td><td>41.3 (3)</td><td>40.9</td></tr><tr><td>M2W domain</td><td>44.6 (40)</td><td>44.7 (32)</td><td>44.6 (25)</td><td>44.5 (19)</td><td>44.4 (14)</td><td>44.3 (10)</td><td>44.3 (5)</td><td>44.3</td></tr><tr><td>GUIAct single</td><td>82.9 (85)</td><td>83.2 (79)</td><td>83.6 (73)</td><td>83.8 (66)</td><td>84.0 (56)</td><td>84.1 (47)</td><td>84.1 (25)</td><td>84.1</td></tr><tr><td>GUIAct multi</td><td>54.2 (66)</td><td>54.7 (57)</td><td>55.3 (48)</td><td>55.8 (39)</td><td>55.9 (27)</td><td>55.9 (19)</td><td>56.1 (8)</td><td>56.1</td></tr><tr><td>Single, ext.</td><td>77.9 (87)</td><td>78.2 (82)</td><td>78.5 (75)</td><td>78.7 (69)</td><td>78.8 (60)</td><td>78.9 (51)</td><td>78.9 (33)</td><td>78.9</td></tr><tr><td>Multi, ext.</td><td>61.8 (75)</td><td>62.5 (66)</td><td>63.2 (57)</td><td>63.7 (47)</td><td>63.7 (33)</td><td>63.5 (23)</td><td>63.5 (9)</td><td>63.5</td></tr><tr><td rowspan="7">8B</td><td>M2W task</td><td>50.2 (46)</td><td>50.7 (38)</td><td>50.7 (30)</td><td>50.8 (24)</td><td>50.6 (17)</td><td>50.6 (13)</td><td>50.4 (7)</td><td>50.4</td></tr><tr><td>M2W website</td><td>46.3 (44)</td><td>46.6 (35)</td><td>46.6 (28)</td><td>46.4 (22)</td><td>45.9 (15)</td><td>45.6 (11)</td><td>44.9 (5)</td><td>44.5</td></tr><tr><td>M2W domain</td><td>48.2 (46)</td><td>48.6 (38)</td><td>48.5 (31)</td><td>48.4 (24)</td><td>48.1 (18)</td><td>48.0 (13)</td><td>48.0 (7)</td><td>48.1</td></tr><tr><td>GUIAct single</td><td>82.8 (88)</td><td>83.3 (82)</td><td>83.7 (77)</td><td>84.0 (70)</td><td>84.1 (61)</td><td>84.3 (53)</td><td>84.4 (34)</td><td>84.4</td></tr><tr><td>GUIAct multi</td><td>56.9 (68)</td><td>57.6 (60)</td><td>58.0 (51)</td><td>58.4 (43)</td><td>58.7 (33)</td><td>58.9 (26)</td><td>58.8 (12)</td><td>58.8</td></tr><tr><td>Single, ext.</td><td>78.0 (89)</td><td>78.3 (84)</td><td>78.7 (79)</td><td>78.9 (72)</td><td>79.0 (64)</td><td>79.1 (57)</td><td>79.2 (41)</td><td>79.3</td></tr><tr><td>Multi, ext.</td><td>62.6 (77)</td><td>63.1 (68)</td><td>63.5 (60)</td><td>63.8 (51)</td><td>64.0 (38)</td><td>64.1 (29)</td><td>64.1 (14)</td><td>64.1</td></tr></table>

Table 15: Step success (%) of the cascade across thresholds, with the share of steps the heads decide in parentheses. For every test step we store the generator’s action and, when $c _ { t } \geq 0 . 5$ , the heads’ action (with the generator’s text for WRITE), so one run yields every $\tau \geq 0 . 5$

Table 15 gives the curves of Figure 3 for every test set at both scales, and Table 16 the operation breakdown on all M2W splits. Heads only is weak on typing because of its WRITE fallback (Section 4.3). On GUIAct, hover steps are one source of the cascade’s loss, since the type mask allows hover only on menu items, which GUIAct elements never are; the cascade solves 5.4% of the 49 hover steps of the multi step test against 10.9% for the generator.

<table><tr><td>Split</td><td>Op.</td><td>Steps</td><td>Gen.</td><td>Heads</td><td>Cascade</td></tr><tr><td>Task</td><td>Click Type Select</td><td>1,037 157 59</td><td>48.5 36.1 32.2</td><td>43.4 19.5 28.2</td><td>48.6 45.2 36.2</td></tr><tr><td>Website</td><td>Click Type Select</td><td>781 120 72</td><td>45.8 24.2 14.8</td><td>40.8 17.5 40.3</td><td>46.0 33.3 39.8</td></tr><tr><td>Domain</td><td>Click Type Select</td><td>3,172 553 95</td><td>48.5 22.6 33.7</td><td>43.3 16.0 32.3</td><td>47.7 28.9 33.0</td></tr></table>

Table 16: Step success (%) by operation on M2W for the 4B generator, heads only, and the cascade at $\tau = 0 . 5$

## F Latency Measurement

Timing runs execute one agent process per GPU with batch size one in bfloat16, with the LoRA merged into the backbone, and time each step with Hugging Face generate from building the prompt, including image preprocessing and the vision encoder, to the parsed action; we average over every step of the 595 MiniWoB++ episodes. The heads add attention pooling and a few projections after the prefill. In WebArena, three agents share a GPU with the browser, and the model time per step falls from 0.414 to 0.364 seconds with the 4B cascade and from 0.369 to 0.240 with the 8B one. Table 2 reports generated tokens as the WebArena cost, since they do not depend on GPU sharing.

## G Mind2Web Standard Protocol

<table><tr><td rowspan="2">System</td><td colspan="3">Cross-task</td><td colspan="3">Cross-website</td><td colspan="3">Cross-domain</td></tr><tr><td>Ele.Acc</td><td>Op.F1</td><td>Step SR</td><td>Ele.Acc</td><td>Op.F1</td><td>Step SR</td><td>Ele.Acc</td><td>Op.F1</td><td>Step SR</td></tr><tr><td>MindAct, Flan-T5 XL (HTML)</td><td></td><td></td><td>52.0</td><td></td><td></td><td>38.9</td><td></td><td></td><td>39.6</td></tr><tr><td>SeeClick 9.6B</td><td></td><td></td><td>25.5</td><td></td><td></td><td>16.4</td><td></td><td></td><td>20.8</td></tr><tr><td>SeeAct, GPT-4V</td><td></td><td></td><td>40.2</td><td></td><td></td><td>32.4</td><td></td><td></td><td>36.8</td></tr><tr><td>OmniParser + GPT-4V</td><td></td><td></td><td>39.4</td><td></td><td></td><td>36.5</td><td></td><td></td><td>42.0</td></tr><tr><td>Aguvis 7B</td><td></td><td></td><td>60.4</td><td></td><td></td><td>54.6</td><td></td><td></td><td>56.6</td></tr><tr><td>Generator 4B</td><td>49.4</td><td>84.6</td><td>45.5</td><td>44.6</td><td>84.0</td><td>41.1</td><td>48.5</td><td>81.1</td><td>42.1</td></tr><tr><td>Cascade 4B,  $\tau = 0 . 5$ </td><td>51.0</td><td>85.0</td><td>46.6</td><td>47.7</td><td>84.8</td><td>43.1</td><td>49.0</td><td>81.4</td><td>42.4</td></tr><tr><td>Two models  $4 \mathrm { B } , \tau = 0 . 5$ </td><td>51.5</td><td>84.9</td><td>46.8</td><td>48.5</td><td>85.1</td><td>43.9</td><td>49.9</td><td>81.8</td><td>43.7</td></tr><tr><td>Generator 8B</td><td>54.3</td><td>84.8</td><td>49.3</td><td>48.6</td><td>84.3</td><td>45.0</td><td>51.8</td><td>81.2</td><td>45.6</td></tr><tr><td>Cascade 8B,  $\tau = 0 . 5$ </td><td>54.7</td><td>85.0</td><td>49.2</td><td>50.9</td><td>85.0</td><td>46.3</td><td>51.9</td><td>81.4</td><td>45.7</td></tr><tr><td>Two models 8B,  $\tau = 0 . 5$ </td><td>55.6</td><td>85.0</td><td>50.1</td><td>53.5</td><td>86.2</td><td>48.8</td><td>52.3</td><td>81.7</td><td>46.4</td></tr></table>

Table 17: Multimodal-Mind2Web under the standard protocol: element accuracy, operation F1 and step success rate, averaged within tasks and then over tasks, with every test step counted and steps whose gold element lies outside our window counted as failures. Published numbers are copied from MindAct (Deng et al., 2023), SeeClick (Cheng et al., 2024), SeeAct (Zheng et al., 2024), OmniParser (Lu et al., 2024) and Aguvis (Xu et al., 2024), which use their own candidate sets or full pages; our rows use the windows and candidate lists of Section 4.2.

## H Further Related Work

Knowledge inside large multimodal models. Our heads read a decision that a large multimodal model has already formed in its hidden states. A neighboring line of work studies the knowledge these models hold and how it can be measured and changed. MINED probes and updates time-sensitive multimodal knowledge (Jiang et al., 2026b), Jiang et al. (2026a) examine how such models cope with evolving knowledge, and KORE injects new knowledge through knowledge-oriented augmentations and constraints (Jiang et al., 2025b). MMKE-Bench evaluates the editing of diverse visual knowledge (Jiang et al., 2025a), in-context editing learns knowledge from self-induced distributions (Qi et al., 2025), and Jia et al. (2026) benchmark conflicts between multimodal knowledge sources. KnowHal measures knowledge-driven hallucination (Li et al., 2026), and Fu et al. (2026) test whether multimodal language models understand OCT images. These studies treat what the model knows as the object to probe or edit; our heads treat what the model has decided as the quantity to read out.

Ranking, distillation and efficient inference. Choosing a box among candidates is a ranking problem, and the same holds for choosing documents in retrieval augmented generation and deep research, where rubric-oriented selection (Jiang et al., 2026d) and adaptive tutoring (Jiang et al., 2026c) rank whole document sets. Our WebArena training data distill the trajectories of a proprietary agent into an open model, close in spirit to multi-agent protocol distillation for agentic search (Liu et al., 2026). Visual token compression (Peng et al., 2025) lowers the cost of the multimodal prefill that the cascade reuses, so its saving adds to the decoding that the heads skip.

## I Case Studies

The three episodes below come from WebArena test tasks whose intent template does not occur among the replayed training tasks.

Reddit, task 402. For “Change my reddit bio to ‘Freelance Web Developer’,” the 4B heads decide all six steps of seed 0: three navigation clicks with confidence above 0.95, typing the bio, where the argument head points at the quoted text that the goal provides (0.9998), a final click (0.86), and the answer. The episode generates only the final answer.

Shopping admin, task 14. For “Tell me the number of reviews that our store received by far that mention term ‘not useful’,” the 4B cascade of seed 1 hands the first click to the generator (0.49), lets the heads take the next click (0.985) and type the quoted term (0.873), hands another click to the generator (0.34), and answers 0 through WRITE.

Shopping admin, task 773. For “Delete all pending negative reviews” the 8B heads of seed 0 act alone on all fourteen clicks, with confidence between 0.66 and 0.98: three navigation clicks and then a three-click sequence on boxes 11 (9 the first time), 12 and 44 that repeats for each review, before they answer through WRITE.