# VLX-Go: Robust Embodied Tracking through Multi-Stage Learning and On-Policy Refinement

Kyusong Lee<sup>1</sup>, Heting Ying<sup>1</sup>, Tiancheng Zhao<sup>1,2</sup>

<sup>1</sup>Om AI Research, <sup>2</sup>Binjiang Institute of Zhejiang University tianchez@zju-bj.com

Abstract: Embodied visual tracking requires a robot to identify a designated target from visual observations and control its own motion to follow that target through an environment. Following an isolated person is a comparatively simple demonstration of embodied visual tracking, but following the intended person among distractors remains challenging. Existing methods exhibit a substantial drop in success from single-target to distractor tracking, motivating a focus on the shared spaces that person-following robots must navigate. We present VLX-Go, a vision–language waypoint policy built around the hypothesis that visual and language representations should be aligned before tracking-specific motion is learned. The policy combines complementary frozen visual encoders, multiscale temporal context, a transformer backbone, and a lightweight head for robot-centric waypoint prediction. Its training progresses through vision–language alignment, multimodal reasoning mid-training, and tracking specialization on 19,532 oracle episodes. Compared with 41.54% distractor-tracking success for our earlier OmTrackVLA model, the complete multistage recipe achieves 55.80%. We then explore on-policy refinement using filtered oracle labels for states visited by the policy. This preliminary study adds 2.42 percentage points of distractor-tracking success, reaching 58.22%, while reducing collision rate by 1.99 pts. The final policy records the highest distractor- and ambiguous-tracking success rates among the compared methods. These results support staged multimodal learning as a promising direction for tracking among distractors and motivate further study of on-policy refinement for person following in shared spaces.

Keywords: embodied navigation, vision–language models, short-horizon waypoint prediction, embodied visual tracking, imitation learning

## 1 Introduction

Embodied navigation requires a robot to translate visual observations and language into motion through its surroundings. Predicting a short sequence of robot-centric waypoints offers a direct interface between multimodal perception and local control. We study this interface in embodied visual tracking (EVT), where a robot identifies a language-specified person or object and follows it through a changing scene. The robot must preserve target identity through occlusion and distractors while maintaining a useful following distance. This setting is relevant to service robots, guidance, and human–robot collaboration, and provides the empirical focus of this report.

Following an isolated person is a comparatively simple setting: the robot has fewer competing candidates to distinguish. A successful single-person demonstration therefore leaves an important deployment challenge unresolved. In shared spaces, other people can enter the camera view, cross the target’s path, or resemble the target. Distractor tracking (DT) tests this challenge directly. Benchmark results reported by TrackVLA [1] show a substantial drop from single-target tracking (STT) to DT across methods: for example, Uni-NaVid falls from 25.70% to 11.30% success, while TrackVLA falls from 85.10% to 57.60%. The size of this gap varies, but it motivates our emphasis on improving DT as a practical requirement for person-following robots.

Traditional EVT systems typically decompose the problem into detection, association, tracking, and planning. Such pipelines are interpretable, but errors compound: a target-identity mistake corrupts the tracking state, which in turn misleads the planner. Recent vision–language–action (VLA) approaches instead learn a shared representation for visual recognition, language grounding, temporal reasoning, and action prediction. This direction builds on embodied language grounding [2, 3], direct vision– language action prediction [4], and open generalist robot policies [5, 6]. Uni-NaVid [7] studies a unified video-conditioned navigation formulation, while TrackVLA [1] specializes VLA learning for EVT and demonstrates strong performance on EVT-Bench. TrackVLA++ [8] further introduces explicit spatial reasoning and long-horizon target memory.

A short-horizon waypoint representation makes the connection between multimodal understanding and local control explicit. Waypoints express the intended motion in the robot’s coordinate frame, while a low-level controller handles execution. For target following, the prediction must combine the language-specified identity with both the current spatial configuration and recent motion. This motivates a policy that brings visual history and language into a shared representation before decoding a local trajectory.

VLX-Go formulates local motion generation as vision–language-conditioned short-horizon waypoint prediction. It uses complementary frozen DINOv3 [9] and SigLIP [10] encoders, a multiscale history representation, a pretrained transformer backbone [11], and a waypoint regression head. This design pairs a transformer backbone with an explicit spatial output: eight future robot-centric waypoints that can be passed to a low-level controller.

Our working hypothesis is that effective target following among distractors benefits from aligning visual features with language before learning tracking-specific motion. This follows the staged learning principle used by OmChat [12] and LLaVA [13], which aligns visual features with a pretrained language backbone before visual instruction tuning. We adopt this principle for embodied tracking through multimodal alignment, multimodal reasoning mid-training, and tracking specialization. Stage 1 trains only the vision–language adapter. Stage 2 jointly trains the transformer backbone and adapter using both the Stage 1 alignment data and additional VQA data. Stage 3 then starts from the Stage 2 base checkpoint and applies SFT on tracking data. Finally, we fine-tune on on-policy data aggregated with DAgger [14], filtering privileged oracle labels using target framing, proximity, and recent tracking history. The main benchmark results evaluate the resulting tracking model across all three tasks.

Starting from a baseline model [15] performance 41.54% DT success; the multi-stage policy improves the success rate to 55.80%. This substantial observed gain motivates the staged recipe, while leaving the contribution of each stage to be isolated. We then investigate on-policy refinement with DAgger as a preliminary extension: oracle supervision on policy-visited states raises DT success by another 2.42 percentage points to 58.22%. Its combination of higher DT success and fewer collisions is promising for further study of tracking in shared spaces.

In summary, this report contributes:

• A vision–language waypoint policy. VLX-Go combines frozen DINOv3 and SigLIP features, multiscale temporal context, a transformer backbone, and a lightweight decoder for short-horizon robot-centric waypoints.

• Three-stage training followed by DAgger. Adapter alignment, multimodal reasoning mid-training, and tracking SFT connect general multimodal learning to waypoint prediction. We describe the 19,532-episode oracle collection and its conversion into training examples, and a DAgger stage that filters privileged oracle labels.

• Evaluation across tracking conditions. VLX-Go reaches success rates of 58.22% and 60.93% on DT and AT, the highest listed SR on these tasks. Results emphasize episode success and

collision avoidance, with comparisons of the three-stage and DAgger policies across all three tasks.   
Interaction diagnostics provide additional context for further development.

## 2 Related Work

EVT differs from passive video tracking because the tracking agent controls the camera and therefore influences future observations. Earlier approaches include active visual tracking with reinforcement learning [16], modular visual-servoing systems, and methods combining visual foundation models with learned policies. Zhong et al. [17] combine visual foundation models with offline reinforcement learning to improve sample efficiency and generalization in embodied tracking. PoliFormer [18] scales transformer-based on-policy reinforcement learning for embodied navigation and can be adapted to tracking-oriented tasks.

EVT-Bench, introduced with TrackVLA [1], evaluates language-conditioned embodied tracking in realistic indoor environments. Its STT, DT, and AT settings progressively stress persistent tracking, distractor rejection, and semantic target disambiguation. This task family exposes the tight coupling between target identification and control: uncertainty about whom to follow directly changes the robot trajectory.

Language-conditioned robot learning has progressed from using language models as high-level planners to directly predicting embodied actions. SayCan grounds language-model proposals in executable robot affordances [2], while PaLM-E [3] injects continuous perceptual observations into an embodied multimodal language model. RT-2 [4] further established the vision–language–action formulation by co-fine-tuning vision–language models to emit robot actions.

Staged multimodal learning provides a foundation for adapting visual understanding to embodied tasks. OmChat [12] combines projector alignment, progressive multimodal pretraining, and visual instruction tuning to support long-context image and video understanding. Its separation of alignment from subsequent multimodal learning motivates our alignment–mid-training–specialization sequence, in which general multimodal representations are ultimately optimized for tracking-specific waypoint prediction.

For embodied navigation, NaVid [19] and Uni-NaVid [7] use video-conditioned language models to predict navigation actions across tasks. TrackVLA [1] specializes the VLA formulation for EVT, jointly learning target recognition and action generation, while TrackVLA++ [8] adds explicit spatial reasoning and target-identification memory for occlusion and distractors. MiniCPM-V [20] explores multimodal models for visual understanding, providing complementary context for the use of smaller backbones. Our earlier OmTrackVLA [15] provides an open-source policy that maps monocular video and language to short-horizon waypoints, together with model weights and a training pipeline. It serves as the direct predecessor and baseline for VLX-Go, which investigates staged multimodal training followed by oracle-supervised on-policy data aggregation with DAgger [14] for embodied tracking.

Fine-grained visual grounding provides complementary context for language-conditioned target identification. VLM-FO1 [21] combines complementary visual encoders with token-based region references, using region–language alignment followed by perception instruction tuning. VLX-Seek 1.5 [22] continues this region-reference direction with broader embodied-scene training and explicit rejection of absent targets. These studies relate to our use of complementary visual represen tations and staged multimodal learning, while addressing region-level perception. VLX-Go instead connects visual history and language to robot-centric waypoint prediction and is evaluated through closed-loop tracking.

Embodied policies also benefit from advances in visual representation learning. Strong frozen visual representations are increasingly used to reduce the amount of task-specific perception training required by robot policies. CLIP [23] established scalable language-aligned visual pretraining, while DINOv2 [24] demonstrated strong task-agnostic dense features learned without supervision. SigLIP [10] improves language–image alignment with a sigmoid objective, and DINOv3 [9] extends self-supervised visual representation learning at scale. VLX-Go combines the latter two families: DINOv3 provides dense spatial structure, while SigLIP supplies language-aligned semantics.

## 3 Task Formulation

At time t, the robot receives a natural-language target description q and a temporal window of monocular RGB observations,

$$
\mathcal { T } _ { t } = \{ \mathbf { I } _ { t - H + 1 } , \dots , \mathbf { I } _ { t } \} ,\tag{1}
$$

where H denotes the number of frames in the model history. The policy predicts K future robotcentric waypoints,

$$
\hat { \tau } _ { t } = \Big \{ ( \hat { x } _ { t , k } , \hat { y } _ { t , k } , \hat { \theta } _ { t , k } ) \Big \} _ { k = 1 } ^ { K } .\tag{2}
$$

A low-level controller converts the predicted local trajectory into robot motion.

Following EVT-Bench [1], we report success rate (SR) and collision rate (CR); higher SR and lower CR are better. Tracking rate (TR) describes adherence to the benchmark tracking conditions, but does not by itself establish overall model quality: useful obstacle detours, yielding, and target recovery can temporarily reduce it (Section 8). Because collision statistics depend strongly on implementation and episode dynamics, we additionally retain per-episode collision timing, human–robot distance, and recent forward velocity for diagnostic analysis.

## 4 VLX-Go

## 4.1 Architecture Overview

Figure 1 summarizes the architecture, offline training stages, and DAgger loop of VLX-Go. Historical and current RGB frames are encoded with frozen DINOv3 and SigLIP visual towers. The two feature maps are spatially aligned and concatenated. Historical frames are aggressively pooled into a small coarse grid, whereas the current frame retains a denser grid. These visual tokens are projected into the hidden space of a transformer backbone and concatenated with language tokens and a learnable action token. The hidden state of the action token is decoded into short-horizon robot-centric waypoints.

## 4.2 Complementary Frozen Visual Encoders

For each frame I , we extract DINOv3 and SigLIP features,

$$
\mathbf { f } _ { t } ^ { D } = E _ { D } ( \mathbf { I } _ { t } ) ,\tag{3}
$$

$$
\mathbf { f } _ { t } ^ { S } = E _ { S } ( \mathbf { I } _ { t } ) .\tag{4}
$$

DINOv3 provides strong dense visual representations [9], whereas SigLIP provides image–languagealigned semantics [10]. After spatial alignment, we concatenate the channels,

$$
\mathbf { f } _ { t } = \mathrm { C o n c a t } \left[ \mathbf { f } _ { t } ^ { D } , \mathrm { A l i g n } ( \mathbf { f } _ { t } ^ { S } ) \right] .\tag{5}
$$

In our default configuration, the DINOv3 branch contributes 384 dimensions and the SigLIP branch 1152, yielding a 1536-D concatenated feature before projection. Both visual towers are frozen. We precompute their outputs in FP16, decoupling expensive image encoding from policy optimization and enabling rapid controlled experiments.

## 4.3 Multiscale Temporal Tokenization

Dense visual tokens from every historical frame would make sequence length scale linearly with both history duration and spatial resolution. We therefore use asymmetric grid pooling. Historical frames are summarized with a coarse $G _ { c } \times G _ { c }$ grid, while the current frame uses a finer $G _ { f } \times G _ { f }$ grid:

$$
\mathcal { Z } _ { t } ^ { v } = \left[ \mathcal { P } _ { G _ { c } } ( \mathbf { f } _ { t - H + 1 } ) , \ldots , \mathcal { P } _ { G _ { c } } ( \mathbf { f } _ { t - 1 } ) , \mathcal { P } _ { G _ { f } } ( \mathbf { f } _ { t } ) \right] .\tag{6}
$$

The default experiment uses $H = 3 2 , G _ { c } = 2 ,$ and $G _ { f } = 8 .$ . This preserves long temporal context at low token cost while allocating spatial detail to the observation that most directly determines the next control decision.

A Vision-language waypoint policy  
![](images/56ab20edc2d6866624da7a6353326a394eec3f952a0f674b9adceb4c162950cb.jpg)

![](images/54a150894ea23186298a8fd2a7e136076f5cae6ec626edb8903d896628dd1d96.jpg)  
Figure 1: VLX-Go architecture and training workflow. (A) Frozen DINOv3 and SigLIP features are concatenated and pooled into 188 visual tokens: $3 1 \times ( 2 \times 2 )$ for history and $8 \times 8$ for the current frame. A vision–language adapter projects these features into a transformer backbone; temporal/view indicators, instruction tokens, and a learnable action query complete the input. A three-layer MLP decodes the action state into eight robot-centric waypoints. (B) Three offline stages train the adapter, then the backbone and adapter, and finally the backbone, adapter, and waypoint head. (C) DAgger collects policy rollouts, labels visited states with the oracle, filters the labels, and fine-tunes on aggregated rollouts and demonstrations. The updated policy supplies the next round of rollouts. Visual encoders remain frozen throughout. Scene and robot images are illustrative.

## 4.4 Projection, Temporal Indicators, and Action Query

A shared projector maps each 1536-D visual token to the transformer backbone’s hidden dimension:

$$
\begin{array} { r } { \tilde { \mathbf { z } } ^ { v } = W _ { 2 } \sigma ( W _ { 1 } \mathrm { L N } ( \mathbf { z } ^ { v } ) + b _ { 1 } ) + b _ { 2 } , } \end{array}\tag{7}
$$

where $\sigma$ is GELU. Temporal/view indicators are added so that the transformer can distinguish frame age and coarse-history versus fine-current streams.

The natural-language instruction is tokenized using the backbone’s tokenizer. We form the multimodal sequence

$$
\mathcal { Z } _ { t } = \left[ \mathcal { Z } ^ { \mathrm { t e x t } } , \mathcal { Z } ^ { \mathrm { h i s t o r y } } , \mathcal { Z } ^ { \mathrm { c u r r e n t } } , \mathbf { z } ^ { \mathrm { A C T } } \right] .\tag{8}
$$

The transformer backbone processes the sequence, and we read the final hidden state at the learnable action position,

$$
\begin{array} { r } { \mathbf { h } _ { t } ^ { \mathrm { A C T } } = \mathrm { T r a n s f o r m e r } ( \mathcal { Z } _ { t } ) [ \mathrm { A C T } ] . } \end{array}\tag{9}
$$

## 4.5 Waypoint Planner

A small MLP maps $\mathbf { h } _ { t } ^ { \mathrm { A C T } }$ to 3K values and reshapes them into K robot-centric waypoints:

$$
\boldsymbol { \hat { \tau } } _ { t } = \mathrm { r e s h a p e } \left( f _ { \mathrm { w p } } ( \mathbf { h } _ { t } ^ { \mathrm { A C T } } ) , K , 3 \right) .\tag{10}
$$

The default planner uses three layers. Given a validity mask $m _ { t , k }$ , the tracking objective is masked waypoint regression,

$$
\mathcal { L } _ { \mathrm { w p } } = \frac { \sum _ { t , k } m _ { t , k } \left\| \hat { \pmb { \tau } } _ { t , k } - \pmb { \tau } _ { t , k } \right\| _ { 2 } ^ { 2 } } { \sum _ { t , k } m _ { t , k } + \epsilon } .\tag{11}
$$

## 4.6 Three-Stage Training and DAgger

The model is trained sequentially through vision–language alignment, multimodal reasoning midtraining, and tracking SFT, and is then fine-tuned with on-policy data aggregation (DAgger). Table 1 summarizes the role and data of each step. Stage 2 continues from the aligned model, and its resulting base checkpoint initializes Stage 3.

Table 1: Training workflow: three offline stages followed by on-policy DAgger fine-tuning (Section 6.3). Visual encoders remain frozen throughout; backbone updates in Stages 2 and 3 and DAgger use full-parameter fine-tuning.
<table><tr><td>Stage</td><td>Objective</td><td>Training data</td><td>Parameter updates</td></tr><tr><td>1</td><td>Vision-language alignment</td><td>blip_laion_cc_sbu_558k alignment collection + human recognition dataset</td><td>Adapter only</td></tr><tr><td>2</td><td>Multimodal reasoning mid-training</td><td>Stage 1 alignment data + general-purpose VQA data</td><td>backbone + adapter</td></tr><tr><td>3</td><td>Tracking SFT from Stage 2 base</td><td>Oracle tracking collection</td><td>backbone + adapter + waypoint head</td></tr><tr><td>DAgger</td><td>On-policy fine-tuning from Stage 3</td><td>Filtered DAgger rollouts (20%) + oracle demonstrations (80%)</td><td>backbone + adapter + waypoint head</td></tr></table>

Stage 1: adapter-only vision–language alignment. Only the vision–language adapter is optimized on the blip\_laion\_cc\_sbu\_558k image–text alignment collection and human recognition dataset; the visual encoders and transformer backbone remain frozen. This stage establishes the mapping between visual features and the language representation before updating the transformer backbone.

Stage 2: multimodal reasoning mid-training. Starting from Stage 1, we jointly train the transformer backbone and adapter using a mixture of the Stage 1 alignment data and general-purpose VQA data. This stage uses supervised text generation to develop multimodal representations before tracking specialization. All backbone parameters are updated. The output of this stage is the multimodal base checkpoint used for tracking specialization.

Stage 3: tracking supervised fine-tuning. We initialize from the Stage 2 base checkpoint and jointly train the backbone, adapter, and waypoint head on the oracle tracking collection described in Section 5, without adding other tracking datasets. The backbone is again updated through fullparameter SFT. We evaluate the resulting model on STT, DT, and AT. The results therefore measure the complete alignment–VQA–tracking training sequence. Both visual encoders remain frozen throughout training.

On-policy fine-tuning with DAgger. The three stages are trained offline on fixed datasets. We then fine-tune the Stage 3 model on a mixture of oracle demonstrations and filtered rollouts of the policy itself, labeled by the oracle (Section 6.3). Unlike Stages 1–3, this step requires closed-loop rollouts in the simulator. We call the Stage 3 model VLX-Go (three-stage) and the fine-tuned model VLX-Go (three-stage + DAgger).

Training objectives. Stages 1 and 2 use the standard autoregressive cross-entropy objective for supervised text generation. For a target sequence $y _ { 1 : N }$ and multimodal context c,

$$
\mathcal { L } _ { \mathrm { t e x t } } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log p _ { \theta } ( y _ { i } \mid c , y _ { < i } ) .\tag{12}
$$

The trainable parameters θ are restricted to the adapter in Stage 1 and include the adapter and full transformer backbone in Stage 2. Stage 3 and DAgger fine-tuning use the masked waypoint mean-squared error $\mathcal { L } _ { \mathrm { w p } }$ defined above to optimize the tracking policy.

Table 2: Oracle episode collection used for tracking supervision in Stage 3.
<table><tr><td>Task</td><td>Episodes</td><td>Scenes</td></tr><tr><td>STT</td><td>6,906</td><td>675</td></tr><tr><td>DT</td><td>6,229</td><td>681</td></tr><tr><td>AT</td><td>6,397</td><td>684</td></tr><tr><td>Total</td><td>19,532</td><td>一</td></tr></table>

## 5 Oracle Data Collection and Supervision

Privileged oracle and rollout generation. We collect training trajectories in Habitat-Sim 0.3.1 using a scripted oracle controller for monocular person following. The oracle uses ground-truth robot and human positions to generate motion commands for following the designated target. It operates in the three EVT-Bench settings introduced with TrackVLA [1]: single-target tracking (STT), distractor tracking (DT), and ambiguous tracking (AT). These trajectories provide expert behavior-cloning supervision generated from simulator state.

At each environment step, we execute the oracle command and record the egocentric RGB observation, natural-language target description, robot and target positions, and expert action. The action is

$$
\mathbf { v } _ { t } = [ v _ { t } ^ { \mathrm { f w d } } , v _ { t } ^ { \mathrm { l a t } } , \omega _ { t } ] ,\tag{13}
$$

where the components denote forward velocity, lateral velocity, and yaw rate in the robot body frame. Observations and state logs are aligned at a 10 Hz control rate. The learned policy receives RGB history and the target description; privileged simulator state is used to construct supervision.

Training-data scale. The oracle collection contains 19,532 episodes and is the sole tracking-data source for Stage 3. Table 2 summarizes its task coverage. Scenes are drawn from HM3D [25] and Matterport3D [26]. Scene identifiers can appear across task splits, so scene counts are reported separately for each task. Training and benchmark evaluation episodes are disjoint, following the EVT-Bench episode split. The alignment and VQA data used in Stages 1 and 2 are separate from this collection.

Trajectory supervision. We convert the recorded robot trajectories into future waypoints in the robot’s coordinate frame. Each training window pairs the target description with 31 past RGB observations and the current frame (32 frames total), and provides an eight-step waypoint target with $\Delta t = 0 . 1 \mathrm { s }$ . The episode count in Table 2 describes the source collection rather than the number of training windows produced by preprocessing. Stage 3 optimizes the masked waypoint regression objective in Section 4. The same oracle also labels policy-visited states during DAgger; rollout collection, label filtering, and the mixture with oracle demonstrations are described in Section 6.3.

## 6 Experiments

## 6.1 Setup

We evaluate on the STT, DT, and AT subsets of EVT-Bench [1], with 1,405 episodes per task. STT measures continuous following of a single target. DT introduces distractors and stresses target identity preservation. AT emphasizes semantic ambiguity among visually similar candidates.

Unless otherwise specified, VLX-Go uses both frozen visual towers, a 32-frame history, 2 × 2 coarse history pooling, 8 × 8 current-frame pooling, a transformer backbone, and a three-layer waypoint planner. We evaluate both the three-stage model and its DAgger-fine-tuned variant (Section 4.6) on benchmark episodes that are disjoint from training episodes.

Stage 3 uses full-parameter SFT for two epochs; Stages 1 and 2 are summarized by their data, objectives, and trainable modules in Section 4.6.

Table 3: Comparison on the distractor and ambiguous tracking tasks of EVT-Bench. Scores for IBVS, PoliFormer, EVT, Uni-NaVid, and TrackVLA are taken from TrackVLA [1]. Both VLX-Go models are evaluated on 1,405 episodes per task.
<table><tr><td></td><td colspan="2">DT</td><td colspan="2">AT</td></tr><tr><td>Method</td><td>SR↑</td><td>CR↓</td><td>SR↑</td><td>CR↓</td></tr><tr><td>IBVS† [28]</td><td>10.60</td><td>6.14</td><td>15.20</td><td>4.90</td></tr><tr><td>PoliFormer† [18]</td><td>2.62</td><td>44.50</td><td>3.04</td><td>41.50</td></tr><tr><td>EVT [17]</td><td>3.23</td><td>47.90</td><td>17.40</td><td>45.60</td></tr><tr><td>EVT‡[17]</td><td>15.70</td><td>53.30</td><td>18.30</td><td>44.90</td></tr><tr><td>Uni-NaVid (Vicuna-7B) [7]</td><td>11.30</td><td>43.50</td><td>8.26</td><td>43.70</td></tr><tr><td>TrackVLA (Vicuna-7B) [1]</td><td>57.60</td><td>5.80</td><td>50.20</td><td>17.10</td></tr><tr><td>MiniCPM-RobotTrack [27]</td><td>53.20</td><td>13.60</td><td>58.00</td><td>9.00</td></tr><tr><td>Baseline [15]</td><td>41.54</td><td>11.31</td><td>60.04</td><td>7.60</td></tr><tr><td>VLX-Go (three-stage)</td><td>55.80</td><td>12.10</td><td>60.64</td><td>9.40</td></tr><tr><td>VLX-Go (three-stage + DAgger)</td><td>58.22</td><td>10.11</td><td>60.93</td><td>10.39</td></tr></table>

DAgger fine-tuning initializes from the Stage 3 checkpoint with a fresh optimizer and uses 390,000 samples for one epoch: 20% filtered DAgger data and 80% oracle demonstrations, each balanced across tasks (Section 6.3). We update the backbone, adapter, and waypoint head with AdamW (weight decay 0.01), a global batch size of 16 on two NVIDIA TITAN RTX GPUs, and bfloat16 mixed precision. The learning rate warms up for 50 steps to $1 0 ^ { - 5 }$ , then decays with a cosine schedule to $1 0 ^ { - 6 }$ . Training takes approximately 9.5 hours; we evaluate the final checkpoint.

## 6.2 Comparison with Prior Methods

Table 3 compares both VLX-Go variants with prior EVT and navigation baselines and the official MiniCPM-RobotTrack model released in OpenBMB’s MiniCPM-Robot repository [27]. We emphasize SR and CR on DT and AT because successful following among competing candidates and collision avoidance are central to deployment. Higher SR and lower CR are better. STT results for our three-stage and DAgger policies are reported in Table 4. Percentage-point differences below are computed from the displayed two-decimal values.

The following comparisons first consider the three-stage model. On DT, it reaches 55.80 SR and 12.10 CR, compared with TrackVLA’s 57.60 SR and 5.80 CR. Relative to official MiniCPM-RobotTrack, DT success improves by 2.60 percentage points and collision rate decreases by 1.50 points.

On AT, the three-stage model achieves 60.64 SR and 9.40 CR. Relative to TrackVLA, success increases by 10.44 percentage points and collision rate decreases by 7.70 points. Its success also exceeds MiniCPM-RobotTrack by 2.64 points, with similar CR (9.40 versus 9.00). IBVS has the lowest AT CR at 4.90, illustrating that episode success and collision avoidance capture complementary aspects of following.

Adding DAgger raises DT success to 58.22 and lowers CR to 10.11. The resulting policy has the highest DT and AT success rates in Table 3, at 58.22 and 60.93, respectively. Its DT success is 0.62 points above TrackVLA and 5.02 points above MiniCPM-RobotTrack. These comparisons are descriptive; small differences should be interpreted cautiously without repeated-run uncertainty estimates.

Compared with the baseline model (OmTrackVLA) [15], the three-stage policy increases the success of DT from 41.54% to 55.80% (+14.26 percentage points), while the DT CR increases from 11.31% to 12.10%. Subsequent DAgger refinement reaches 58.22% success and 10.11% CR, improving both metrics over OmTrackVLA (+16.68 SR and −1.20 CR points). On AT, the final policy has higher success (60.93% versus 60.04%) but also higher CR (10.39% versus 7.60%). These results characterize the complete training recipe rather than isolating the effect of alignment alone.

Together, these results position VLX-Go as a waypoint-prediction baseline with a particularly encouraging profile on ambiguous tracking. End-to-end efficiency also depends on the visual encoders, temporal processing, and deployment hardware, and remains to be measured. The benchmark study characterizes the complete training recipe and architecture.

## 6.3 On-Policy Data Aggregation

Behavior cloning trains the policy only on states visited by the expert. At test time, small errors take the robot into states that are underrepresented in the demonstrations: the target drifts out of frame, the robot falls behind, or it closes in too fast. The demonstrations may provide insufficient supervision for recovery from these states. We address this covariate shift with DAgger [14], adapted to a privileged oracle and a learner conditioned on RGB history and language.

Rollouts and labels. The current policy $\pi _ { i }$ drives the robot on training episodes of all three tasks (STT, DT, AT), at the 10 Hz control rate used for the demonstrations. At every step, the scripted oracle that generated the original demonstrations computes the command it would issue from the same state. It uses ground-truth positions of the robot and the humans. The policy’s command is executed $( \beta = 0$ , where $\beta$ is the probability of executing the oracle action), and the oracle’s command is stored only as the supervision label. Frames, instructions and labels are converted with the same pipeline as the demonstrations, so DAgger samples and demonstrations share one format.

Filtering privileged labels. The oracle is privileged and the learner is not. When the learner has lost the target, the oracle still issues a confident pursuit command toward a person who is not in the image. Training on such labels teaches the policy to accelerate without visual evidence, and in our experiments it raised collisions sharply. We therefore keep a DAgger sample only if one of these holds:

1. Framed: the target is within the success distance and well framed in the image (facing = 1).

2. Close range: the robot is closer than 1.2 m to the target, where the oracle backs off. These labels provide supervision for increasing separation.

3. Recovery: the learner had framed the target for at least 1 s and then lost it. We keep the steps from the loss until the target is framed again, for at most 2 s.

In recovery states the target is still in the policy’s observation history, providing visual context that may help the policy learn corrective behavior. These samples capture departures from sustained tracking during the policy’s own rollouts.

Aggregation and task balance. Following DAgger, each round adds its filtered rollouts to the data of earlier rounds. We fine-tune from the Stage 3 checkpoint on a fixed budget of T = 390,000 samples. DAgger samples make up 20% of T, with an equal number per task, and samples from the latest policy are used first. The remaining 80% are demonstrations, again with an equal number per task. We train for one epoch with learning rate $1 0 ^ { - 5 }$ , cosine decay, and batch size 16.

Evaluation protocol. We evaluate on the full validation sets of EVT-Bench (1,405 episodes per task) and report success rate (SR) and collision rate (CR). The public evaluation harness initializes the human avatars and the detector’s target identity only on the first episode of each process. Later episodes can therefore show the wrong people or track the wrong target. We refresh both on every episode, clear spawn poses left over from the previous episode, and keep the benchmark’s rule that STT episodes contain no distractors. In a repeat evaluation of the same model, approximately 23% of episodes changed outcome. This observation indicates evaluation variability but does not by itself establish an uncertainty interval for aggregate SR. We therefore interpret small SR differences cautiously.

Table 4: Effect of DAgger, full DT and AT validation sets of EVT-Bench (1,405 episodes per task).
<table><tr><td></td><td colspan="2">DT</td><td colspan="2">AT</td></tr><tr><td>Model</td><td>SR↑</td><td>CR↓</td><td>SR↑</td><td>CR↓</td></tr><tr><td>Oracle (privileged)</td><td>86.74</td><td>5.17</td><td>84.92</td><td>5.00</td></tr><tr><td>VLX-Go (three-stage)</td><td>55.80</td><td>12.10</td><td>60.64</td><td>9.40</td></tr><tr><td>VLX-Go (three-stage + DAgger)</td><td>58.22</td><td>10.11</td><td>60.93</td><td>10.39</td></tr></table>

Table 5: Changes from VLX-Go (three-stage) to VLX-Go (three-stage + DAgger). Metric differences are percentage points computed from the displayed scores in Table 4; count differences are computed from the episode logs. Positive SR and negative CR indicate improvement.
<table><tr><td>Task</td><td>∆SR</td><td>∆CR</td><td>∆Successes</td><td>∆Collisions</td></tr><tr><td>DT</td><td>+2.42</td><td>-1.99</td><td>+34</td><td>-28</td></tr><tr><td>AT</td><td>+0.29</td><td>+0.99</td><td>+4</td><td>+14</td></tr></table>

Results. DAgger improves both success and collision outcomes on DT (Table 4). Relative to the Stage 3 checkpoint, SR rises from 55.80% to 58.22% (+2.42 percentage points), while CR falls from 12.10% to 10.11% (−1.99 points). The collision reduction is concentrated in contacts during forward motion (Section 7). On AT, SR increases slightly from 60.64% to 60.93% (+0.29 points), while CR increases from 9.40% to 10.39% (+0.99 points). The observed benefit is therefore concentrated in DT; the small AT success gain does not establish a reliable improvement without repeated-run uncertainty estimates. The privileged oracle achieves 86.74% DT success and 84.92% AT success, leaving gaps of 28.52 and 23.99 percentage points, respectively, to the DAgger-refined policy.

## 7 Collision Analysis

Collision avoidance is a major concern for person-following robots in shared spaces. We therefore examine whether the DT success gain after on-policy refinement accompanies fewer contacts. Table 6 summarizes collisions from one evaluation per model on the same 1,405 episode IDs per task. We classify a collision as occurring during forward motion when mean forward velocity over the final 15 logged steps (or all available steps for shorter episodes) is at least 0.15 m/s. The remaining category includes slow, stopped, and reverse motion; these categories do not establish responsibility for contact.

DT success improves alongside fewer forward-motion collisions. On DT, success rises from 55.80% to 58.22% while collisions decrease from 170 to 142 (CR 12.10% to 10.11%). Forwardmotion collisions fall from 88 to 56, including a reduction from 62 to 34 in contacts after step 30. Below-threshold collisions change from 82 to 86. Thus, the observed collision benefit is concentrated in forward motion during distractor tracking. Better use of visual and language context to guide motion is one possible explanation, alongside improved close-range control learned from oracle labels. This comparison does not isolate multimodal understanding as the cause: both variants share the multi-stage recipe, and the logs do not identify which person was hit. Runs are matched by episode ID and instruction, without verified identical simulator random states.

## 8 Discussion

Multimodal learning for tracking among distractors. The gap between single-target and distractor tracking motivates learning to connect a target description with visual context before optimizing motion. Inspired by staged visual–language learning, VLX-Go first aligns modalities, then develops multimodal representations, and finally learns waypoint prediction. The increase from 41.54% DT success with OmTrackVLA [15] to 55.80% with the complete recipe supports this direction. It does not isolate the effect of alignment or individual stages, but is consistent with our hypothesis that multimodal preparation benefits tracking among competing candidates. Temporal context and robot-centric waypoints connect this representation to local control.

Table 6: Collision counts before and after on-policy refinement. Each entry reports three-stage → three-stage + DAgger. Forward: mean forward velocity $\geq 0 . 1 5$ m/s before contact; below threshold: all remaining collisions.
<table><tr><td>Task</td><td></td><td>All collisions Forward motion </td><td>Below threshold</td></tr><tr><td>DT</td><td> $\mathbf { 1 7 0 }  \mathbf { 1 4 2 }$ </td><td> $\mathbf { 8 8 }  \mathbf { 5 6 }$ </td><td> $8 2  8 6$ </td></tr><tr><td>AT</td><td> $1 3 2  1 4 6$ </td><td> $6 3 \to 6 8$ </td><td> $6 9  7 8$ </td></tr><tr><td>Overall</td><td> $3 7 1  3 6 2$ </td><td> $1 6 3  1 3 5$ </td><td> $2 0 8  2 2 7$ </td></tr></table>

Tracking rate and practical following quality. Our qualitative real-world observations suggest that high TR alone does not identify the most useful following policy. Effective following also requires remembering where the target moved and using that history to recover after losing sight of it. A robot may need to detour around obstacles, slow down near other people or corners, and temporarily change its following distance. Such behavior can reduce TR while helping the robot continue following without contact. In our deployments, the policy did not always maintain a constant following distance, and lower TR did not necessarily indicate a worse user experience. These observations motivate emphasizing episode success and collision rate in the reported comparisons. They do not establish that lower TR is inherently preferable.

Deployment experience motivates prioritizing DT. In our real-world deployments, we qualitatively observed that models with higher DT success provided a better user experience when following a person among distractors. Shared spaces frequently contain other people, making robustness to competing candidates important even when a model already performs well in isolated-person demonstrations. This observation reinforces our emphasis on DT when assessing practical personfollowing behavior. It is a qualitative deployment observation rather than a controlled user study; the relationship between benchmark DT scores and user experience remains to be quantified.

Promising preliminary on-policy refinement. Our DAgger study finds a promising benefit for EVT, particularly DT: supervision on policy-visited states adds 2.42 percentage points of success and reduces collision rate by 1.99 points. The collision analysis associates this gain with fewer contacts during forward motion, suggesting that refinement can improve how the policy acts in distractor-rich situations. Because AT collision rate increases despite a small success-rate gain, the result supports targeted refinement for DT rather than a general improvement across both tasks. The central finding is the potential of combining staged multimodal learning with supervision drawn from the policy’s own interactions.

## 9 Conclusion

We presented VLX-Go, a vision–language policy for short-horizon waypoint prediction in embodied navigation. Frozen DINOv3 and SigLIP features, multiscale temporal context, a transformer backbone, and a lightweight waypoint head connect RGB history and language to robot-centric local trajectories. A three-stage recipe brings together vision–language alignment, multimodal reasoning mid-training, and tracking-specific waypoint supervision, followed by on-policy data aggregation with filtered oracle labels.

Evaluation on EVT-Bench emphasizes distractor and ambiguous tracking, with success rates of 58.22% and 60.93% and collision rates of 10.11% and 10.39%, respectively. The DT results highlight multi-stage multimodal learning as a promising direction for following a designated person among distractors. Preliminary on-policy refinement further improves DT success and collision outcomes, motivating continued investigation for deployment in shared spaces.

Future work will further investigate DAgger for improving DT, including additional rounds of policydriven collection, the balance between demonstrations and aggregated rollouts, and the contribution of each label-filtering rule. We will examine whether these refinements improve target retention as well as collision avoidance, and evaluate their transfer to person following in real shared spaces.

## References

[1] S. Wang, J. Zhang, M. Li, J. Liu, A. Li, K. Wu, F. Zhong, J. Yu, Z. Zhang, and H. Wang. TrackVLA: Embodied visual tracking in the wild. In Proceedings of the 9th Conference on Robot Learning, volume 305 of Proceedings ofMachine Learning Research, pages 4139–4164. PMLR, 2025. URL https://proceedings.mlr.press/v305/wang25f.html.

[2] B. Ichter, A. Brohan, Y. Chebotar, C. Finn, K. Hausman, A. Herzog, D. Ho, J. Ibarz, A. Irpan, E. Jang, R. Julian, D. Kalashnikov, S. Levine, Y. Lu, C. Parada, K. Rao, P. Sermanet, A. T. Toshev, V. Vanhoucke, F. Xia, T. Xiao, P. Xu, M. Yan, et al. Do as i can, not as i say: Grounding language in robotic affordances. In Conference on Robot Learning, volume 205 of Proceedings of Machine Learning Research, pages 287–318. PMLR, 2022. URL https: //arxiv.org/abs/2204.01691.

[3] D. Driess, F. Xia, M. S. M. Sajjadi, C. Lynch, A. Chowdhery, B. Ichter, A. Wahid, J. Tompson, Q. Vuong, T. Yu, W. Huang, Y. Chebotar, P. Sermanet, D. Duckworth, S. Levine, V. Vanhoucke, K. Hausman, M. Toussaint, K. Greff, A. Zeng, I. Mordatch, and P. Florence. PaLM-E: An embodied multimodal language model. In Proceedings ofthe 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pages 8469– 8488. PMLR, 2023. URL https://proceedings.mlr.press/v202/driess23a.html.

[4] B. Zitkovich, T. Yu, S. Xu, P. Xu, T. Xiao, F. Xia, J. Wu, P. Wohlhart, S. Welker, A. Wahid, Q. Vuong, V. Vanhoucke, H. Tran, R. Soricut, A. Singh, et al. RT-2: Vision-language-action models transfer web knowledge to robotic control. In Proceedings of the 7th Conference on Robot Learning, volume 229 of Proceedings of Machine Learning Research, pages 2165–2183. PMLR, 2023. URL https://proceedings.mlr.press/v229/zitkovich23a.html.

[5] D. Ghosh, H. R. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, T. Kreiman, C. Xu, J. Luo, Y. L. Tan, L. Y. Chen, Q. Vuong, T. Xiao, P. R. Sanketi, D. Sadigh, C. Finn, and S. Levine. Octo: An open-source generalist robot policy. In Proceedings ofRobotics: Science and Systems, 2024. doi:10.15607/RSS.2024.XX.090. URL https://www.roboticsproceedings.org/ rss20/p090.html.

[6] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. P. Foster, P. R. Sanketi, Q. Vuong, T. Kollar, B. Burchfiel, R. Tedrake, D. Sadigh, S. Levine, P. Liang, and C. Finn. OpenVLA: An open-source vision-language-action model. In Proceedings ofthe 8th Conference on Robot Learning, volume 270 of Proceedings ofMachine Learning Research. PMLR, 2025. URL https://proceedings.mlr.press/v270/kim25c.html.

[7] J. Zhang, K. Wang, S. Wang, M. Li, H. Liu, S. Wei, Z. Wang, Z. Zhang, and H. Wang. Uni-NaVid: A video-based vision-language-action model for unifying embodied navigation tasks. In Robotics: Science and Systems, 2025. URL https://arxiv.org/abs/2412.06224.

[8] J. Liu, Y. Qi, J. Zhang, M. Li, S. Wang, K. Wu, H. Ye, H. Zhang, Z. Chen, F. Zhong, Z. Zhang, and H. Wang. TrackVLA++: Unleashing reasoning and memory capabilities in VLA models for embodied visual tracking. arXiv preprint arXiv:2510.07134, 2025. URL https://arxiv. org/abs/2510.07134.

[9] O. Siméoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, C. Jose, V. Khalidov, M. Szafraniec, S. Yi, M. Ramamonjisoa, F. Massa, D. Haziza, L. Wehrstedt, J. Wang, T. Darcet, T. Moutakanni, L. Sentana, C. Roberts, A. Vedaldi, J. Tolan, J. Brandt, C. Couprie, J. Mairal, H. Jégou, P. Labatut, and P. Bojanowski. DINOv3. arXiv preprint arXiv:2508.10104, 2025. URL https://arxiv.org/abs/2508.10104.

[10] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer. Sigmoid loss for language image pre-training. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 11975– 11986, 2023. URL https://openaccess.thecvf.com/content/ICCV2023/html/Zhai\_ Sigmoid\_Loss\_for\_Language\_Image\_Pre-Training\_ICCV\_2023\_paper.html.

[11] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, C. Zheng, D. Liu, F. Zhou, F. Huang, F. Hu, H. Ge, H. Wei, H. Lin, J. Tang, J. Yang, J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Zhou, J. Lin, K. Dang, K. Bao, K. Yang, L. Yu, L. Deng, M. Li, M. Xue, M. Li, P. Zhang, P. Wang, Q. Zhu, R. Men, R. Gao, S. Liu, S. Luo, T. Li, T. Tang, W. Yin, X. Ren, X. Wang, X. Zhang, X. Ren, Y. Fan, Y. Su, Y. Zhang, Y. Zhang, Y. Wan, Y. Liu, Z. Wang, Z. Cui, Z. Zhang, Z. Zhou, and Z. Qiu. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. URL https://arxiv.org/abs/2505.09388.

[12] T. Zhao, Q. Zhang, K. Lee, P. Liu, L. Zhang, C. Fang, J. Liao, K. Jiang, Y. Ma, and R. Xu. OmChat: A recipe to train multimodal language models with strong long context and video understanding. arXiv preprint arXiv:2407.04923, 2024. URL https://arxiv.org/abs/ 2407.04923.

[13] H. Liu, C. Li, Q. Wu, and Y. J. Lee. Visual instruction tuning. In Advances in Neural Information Processing Systems, volume 36, 2023. URL https://proceedings.neurips.cc/paper\_files/paper/2023/hash/ 6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html.

[14] S. Ross, G. Gordon, and D. Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, volume 15 of Proceedings of Machine Learning Research, pages 627–635. PMLR, 2011. URL https://proceedings.mlr.press/v15/ross11a. html.

[15] H. Y. Kyusong Lee and T. Zhao. Omtrackvla: Open-source visual language action model for visual navigation and following. https://github.com/om-ai-lab/OmTrackVLA, 2025.

[16] W. Luo, P. Sun, F. Zhong, W. Liu, T. Zhang, and Y. Wang. End-to-end active object tracking via reinforcement learning. In Proceedings ofthe 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research. PMLR, 2018. URL https://proceedings.mlr.press/v80/luo18a.html.

[17] F. Zhong, K. Wu, H. Ci, C.-r. Wang, and H. Chen. Empowering embodied visual tracking with visual foundation models and offline RL. In European Conference on Computer Vision. Springer, 2024. URL https://www.ecva.net/papers/eccv\_2024/papers\_ECCV/html/ 9241\_ECCV\_2024\_paper.php.

[18] K.-H. Zeng, Z. Zhang, K. Ehsani, R. Hendrix, J. Salvador, A. Herrasti, R. Girshick, A. Kembhavi, and L. Weihs. PoliFormer: Scaling on-policy RL with transformers results in masterful navigators. In Proceedings ofthe 8th Conference on Robot Learning, volume 270 of Proceedings ofMachine Learning Research, pages 408–432. PMLR, 2025. URL https://proceedings. mlr.press/v270/zeng25a.html.

[19] J. Zhang, K. Wang, R. Xu, G. Zhou, Y. Hong, X. Fang, Q. Wu, Z. Zhang, and H. Wang. NaVid: Video-based VLM plans the next step for vision-and-language navigation. In Robotics: Science and Systems, 2024. URL https://arxiv.org/abs/2402.15852.

[20] Y. Yao, T. Yu, A. Zhang, C. Wang, J. Cui, H. Zhu, T. Cai, H. Li, W. Zhao, Z. He, Q. Chen, H. Zhou, Z. Zou, H. Zhang, S. Hu, Z. Zheng, J. Zhou, J. Cai, X. Han, G. Zeng, D. Li, Z. Liu, and M. Sun. MiniCPM-V: A GPT-4V-level multimodal large language model on your phone. Nature Communications, 16:5509, 2025. URL https://arxiv.org/abs/2408.01800.

[21] P. Liu, H. Shen, C. Fang, Z. Sun, J. Liao, and T. Zhao. VLM-FO1: Bridging the gap between high-level reasoning and fine-grained perception in VLMs. arXiv preprint arXiv:2509.25916, 2025. URL https://arxiv.org/abs/2509.25916.

[22] OM AI Lab. VLX-Seek 1.5: Enhanced fine-grained perception for embodied scenarios. https://github.com/om-ai-lab/VLX-Seek, July 2026.

[23] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever. Learning transferable visual models from natural language supervision. In Proceedings ofthe 38th International Conference on Machine Learning, volume 139 of Proceedings ofMachine Learning Research, pages 8748–8763. PMLR, 2021. URL https://proceedings.mlr.press/v139/radford21a.html.

[24] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, M. Assran, N. Ballas, W. Galuba, R. Howes, P.-Y. Huang, S.-W. Li, I. Misra, M. Rabbat, V. Sharma, G. Synnaeve, H. Xu, H. Jégou, J. Mairal, P. Labatut, A. Joulin, and P. Bojanowski. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024. URL https://arxiv.org/abs/2304.07193.

[25] S. K. Ramakrishnan, A. Gokaslan, E. Wijmans, O. Maksymets, A. Clegg, J. Turner, E. Undersander, W. Galuba, A. Westbury, A. X. Chang, M. Savva, Y. Zhao, and D. Batra. Habitatmatterport 3d dataset (HM3D): 1000 large-scale 3d environments for embodied AI. In NeurIPS Datasets and Benchmarks Track, 2021.

[26] A. X. Chang, A. Dai, T. Funkhouser, M. Halber, M. Nießner, M. Savva, S. Song, A. Zeng, and Y. Zhang. Matterport3D: Learning from RGB-D data in indoor environments. In International Conference on 3D Vision, pages 667–676, 2017. doi:10.1109/3DV.2017.00081.

[27] OpenBMB. MiniCPM-Robot: Official Repository. GitHub repository, 2026. URL https: //github.com/OpenBMB/MiniCPM-Robot. Official release of MiniCPM-RobotTrack.

[28] M. Gupta, S. Kumar, L. Behera, and V. K. Subramanian. A novel vision-based tracking algorithm for a human-following mobile robot. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 47(7):1415–1427, 2016.