# TreeFlex: Agents Navigating Their Own Search Trees

Dongzhuo Li, Fangzhou Cheng, Weiwei Li, Kevin Han, Haixu Ma, Zhongchang Sun, Ruiyi Li, Serena Li, Xiangjun Fan, Lizhu Zhang, Yuandong Tian<sup>†</sup>

Meta AI

<sup>†</sup>Work done while at Meta

A ReAct-style agent accumulates observations, intermediate results, and failed attempts in a single context. As this context grows, stale and of-path information consumes its limited capacity and obscures relevant evidence, making focused reasoning increasingly dificult. We introduce TreeFlex, a framework in which the agent’s own policy decides where to search. The agent records each intermediate state, with a short summary, as a node in a search tree maintained by the harness. The policy is a single language model that acts in two roles. In the Reflection role, it has a global view of the tree through these summaries and picks a node to resume, together with an instruction. In the Execution role, the harness restores the path that led to that node, and the model carries out the instruction with the usual task tools until it saves a new node or proposes a solution, which Reflection then approves or sends back to another node. Through these choices, the policy decides whether to deepen the current line, branch from or revisit an earlier state, verify a proposal, or stop, without a fixed traversal order or a score for each node. We train navigation, task actions, checkpointing, and proposal review jointly with a tree-aware adaptation of GRPO, using task-level outcome rewards augmented by process rewards. After small-scale RL of a 4B model, with the same active-context limit for all methods, TreeFlex reaches 48.8% accuracy on a challenging deep-research benchmark, versus 41.5% for flat ReAct and 42.8% for FoldAgent, and its Pass@8 is about ten points higher than either. On SWE-bench Verified, with all three systems trained with RL from the same 9B model, TreeFlex resolves 54.9% of tasks, versus 43.3% for FoldAgent and 45.2% for its step-wise variant.

Correspondence: Dongzhuo Li at lidongzh@meta.com

Meta

## 1 Introduction

On long-horizon tasks, an agent rarely reaches a solution in one straight path. Several candidate solutions may build on the same partial progress, a failed attempt may still reach a state worth resuming from, and later evidence may make an abandoned direction worth revisiting. In deep research, an agent may check several candidate answers against the same evidence and return to an earlier one that it had dismissed when a later document changes the picture. In agentic software engineering (SWE), it may try several fixes for the same reproduced bug and return to an earlier repository state when a later test implicates code it had set aside. Yet a standard ReAct agent (Yao et al., 2023b) keeps all of this in one linear context. Every later model call then carries abandoned attempts and long of-path observations along with the evidence it needs, and undoing an attempt may also require undoing its efects on the environment. One remedy is to compress the history. Learned summarization and memory-consolidation methods select what past information remains available (Zhou et al., 2026; Li et al., 2026), but whatever they discard is gone for the rest of the episode. We instead keep several intermediate states that the agent can return to, and show the model only the context of the path it is currently working on.

Tree-search methods for language models make alternative states explicit (Yao et al., 2023a; Hao et al., 2023; Zhou et al., 2024), but having a tree does not tell the agent which branch to work on next. A hand-designed DFS, BFS, or MCTS controller imposes its own selection rule. A learned score for each node is more adaptive, but it evaluates each node in isolation, while its usefulness changes as other branches reveal new evidence and the search approaches its limits. A score also leaves open how to continue from the selected state. In other words, the useful decision is often a node paired with a plan for what to do there, not a node alone. We therefore ask whether the model that does the task can also make this choice, based on what the search has found so far.

![](images/e1b206f48542c06721f7f82e3db01d301c81bb39e32b6af9ac08a2ea4a7538d4.jpg)  
Figure 1 One TreeFlex search. Left: the search tree, numbered in creation order. Top: Reflection sees the task and a short card for every node. Although $B , B _ { 1 } ,$ , and $A _ { 2 }$ are newer, it calls navigate to resume $A _ { 1 }$ with a new instruction. Bottom: the harness restores only the path $r \to A \to A _ { 1 }$ and its saved state for Execution, so the other branches stay out of its context. Execution proposes $P ,$ and Reflection then approves it. Both roles use the same model. In deep research, the restored path consists of cards and the saved state is empty. In SWE, the restored path is the raw context along the path, and the saved state is a repository snapshot.

A second question is how to train this behavior end to end. Standard agentic RL treats each episode as one growing token sequence with one final reward. A tree search breaks this assumption, because the model makes decisions from diferent saved states and therefore sees a diferent history each time. Concatenating all branches into one sequence would show the model histories it never saw during the search and bring back the long-context problem.

We introduce TreeFlex, a framework that saves the agent’s intermediate states as nodes of a search tree, each summarized by the model itself. The model works in two roles, Reflection and Execution (we later use each role’s name for the model acting in that role). In Reflection, it receives the original task and a short summary of every node, which we call its card, and selects one node together with a focused instruction. The harness then restores the path to that node for Execution, where the model uses the usual task tools to carry out the instruction and either saves a new checkpoint or proposes a final solution. Reflection then either approves the proposal, which ends the search, or rejects it by selecting one node to continue from. Continuing from the current node deepens a line, while returning to an earlier node or the root starts a new branch from that state. In classical search terms, Reflection manages the frontier, which is the set of states from which search can continue. For example, best-first search keeps this frontier as an open list ordered by priority. However, unlike such a list, Reflection can return to states it has already expanded, and it pairs each choice with an instruction. Figure 1 shows one such search.

This design lets the search grow without any context having to hold every branch. Reflection’s record grows by only one short card per node, and each Execution segment only sees one path. A node saves the environment state, such as a repository snapshot, as well as the raw prefix or a summary card for the segment, and returning to an earlier node restores the workspace. Therefore, the agent can use more compute to deepen a line, start a sibling, revisit earlier work, or verify a proposal without adding every explored branch to later Execution contexts.

Although the tree branches, the model’s calls still happen one after another. We use this property to adapt GRPO (Shao et al., 2024). Each complete search counts as one rollout with one task reward, and each model output is trained on the exact context it was generated from, so no tree-shaped attention mask is needed.

We study TreeFlex on deep research with BrowseComp-Plus (Chen et al., 2026) and on SWE with SWE-bench Verified (Jimenez et al., 2024; OpenAI, 2024). On the BrowseComp-Plus test split used by FoldAgent (Sun et al., 2026), we compare it with flat ReAct, which keeps the whole history in one context, and FoldAgent under the same model and resource limits. Before RL, TreeFlex has the highest accuracy at both 4B and 30B. After RL training of Qwen3-4B-Instruct-2507 (Qwen Team, 2025) with a 32k active-context limit, TreeFlex reaches 48.8% accuracy, versus 41.5% for flat ReAct and 42.8% for FoldAgent, and 66.0% Pass@8, versus 56.0% and 55.3%. On SWE, we train TreeFlex and FoldAgent from Qwen3.5-9B with RL on SWE-Gym tasks (Pan et al., 2025) and select each run’s training step on separate SWE-Gym validation tasks. TreeFlex resolves 54.9% of tasks, versus 43.3% for FoldAgent and 45.2% for a FoldAgent variant with a step-wise delegation prompt that we wrote.

Our contributions are:

• Policy-directed tree search. One model decides when to continue, branch, backtrack, verify, or stop, while the harness only saves and restores contexts and environment states.

• Tree-aware policy optimization. We adapt GRPO to treat a complete tree search as one sampled rollout, train each model output on the exact context it was generated from, and combine the task reward with process penalties on the actions that caused them.

• Evaluation across two agent domains. On deep research, we compare three ways of organizing an agent’s history, a single linear context (ReAct), context folding (FoldAgent), and a search tree (TreeFlex), under the same model and context limit. On SWE, we compare the complete FoldAgent and TreeFlex systems. We report task success and resource use.

## 2 TreeFlex

## 2.1 Agent Design

Search tree. Let x be a task and D its ordinary domain tools. TreeFlex maintains a rooted tree $\tau$ whose root r stores the task prompt and the initial environment. Every other node v is saved at the end of an Execution segment, either by checkpoint or, for a proposed final solution, by commit. The segment starts from an existing node $p _ { v } ,$ its parent, and carries out instruction $I _ { v }$ . A node thus corresponds to a whole segment of work, which usually spans many turns. It stores everything needed to resume from it:

$$
\boldsymbol { v } = ( p _ { v } , \ I _ { v } , \ s _ { v } , \ m _ { v } , \ e _ { v } ) ,\tag{1}
$$

where $s _ { v }$ is a model-authored card, $m _ { v }$ the context state, and $e _ { v }$ the environment state, such as a repository snapshot in SWE.

Roles and tree tools. The language model acts in two roles that difer only in what they see and which tools they can call (Table 1). In Reflection, it sees the original task and a global record with one entry per node, which gives the node’s identifier, parent, and card but not the raw context of any segment. In Execution, it sees the path view: the context restored along the path from the root to the selected node, the selected node’s environment state $e _ { v } ,$ and the new instruction, but nothing from other branches. In deep research, retrieved evidence can be cited in a card and fetched again, so the restored context consists only of the cards along the path. In SWE, error traces and difs are hard to summarize faithfully, so the restored context is the raw context along the path. When it no longer fits, the harness stores a compact resumable context containing the path cards and a truncated repository dif (Appendix A.2).

This division of work follows from what each role can see. Execution has the detailed evidence of one path, so it does the work and proposes solutions. Reflection sees every branch through short cards, so it chooses where to work next and checks proposed solutions against evidence from all branches. It cannot edit a proposal, because it lacks the raw evidence.

Table 1 Tree tools of Reflection and Execution. Deep research calls commit with an answer and an explanation. SWE exposes it as OpenHands’ finish(message), which proposes the current repository state.
<table><tr><td>Tool</td><td>Role</td><td>What it does</td></tr><tr><td>navigate(node, instruction)</td><td>Reflection</td><td>Choose a node to resume and write an instruction for the next segment. Creates no node.</td></tr><tr><td>Domain tools D</td><td>Execution</td><td>Work on the task, such as retrieving documents or editing and testing code.</td></tr><tr><td>checkpoint(summary)</td><td>Execution</td><td>Save the current state as a new child of the node where the segment started, with summary as its card, and return the card to Reflection.</td></tr><tr><td>commit(solution)</td><td>Execution</td><td>Save the current state as a new child of the node where the segment started, holding a proposed final solution for Reflection to review.</td></tr><tr><td>approve_commit(node)</td><td>Reflection</td><td>Accept the proposed solution at node unchanged, which ends the search.</td></tr></table>

Search procedure. A search alternates between the two roles. Reflection’s first navigate call must choose the root. Each Execution segment ends with checkpoint or commit. If a segment runs out of context, the harness requires a checkpoint. The resulting card or proposal returns as the result of the pending navigate call. Reflection thus works like a ReAct loop over segment results (Appendix A.2) and reconsiders where to search after every segment.

Reflection’s choices shape the tree. Resuming the latest node deepens the current line, resuming an earlier node starts a sibling branch, and resuming the root starts a new line. The search can therefore behave like depth-first, breadth-first, or best-first search, and the model learns when to do which, without a fixed rule. Segments run one at a time, without parallel model calls.

The search ends when Reflection accepts a proposed solution or when the rollout reaches its turn limit. If Reflection navigates instead, the search continues from the chosen node, while the proposal remains available for later approval. The harness validates calls and manages saved state, but does not choose where to navigate. Appendices C.2 and A.1 give the settings and pseudocode.

## 2.2 RL Training

Rollouts. In standard agentic RL, a rollout is one growing token sequence. In contrast, a TreeFlex search or rollout is split across several sequences, because Reflection keeps its own context and each Execution segment starts from the saved context of its selected node. Training on such a search raises two questions: what counts as one sample and which context each output is trained on. For the first, we treat each complete search as one rollout and one member of its GRPO group, since only the complete search receives a task reward. The task reward therefore applies to every output in the search, whereas a process penalty applies only to the turn it concerns. For the second, each output is trained on exactly the context it was generated from. Its log-probability is computed given that context, and the restored context is used only as input, so the loss applies only to tokens the model newly generated.

Causal blocks and loss mask. A causal block is a maximal sequence over which context grows only by appending tokens. It therefore preserves the exact prefix from which each generated token was sampled. Reflection’s context grows throughout the search, so Reflection forms a single block. An Execution segment ends with checkpoint or commit, and the next segment starts from a restored context, so each segment forms its own block. A segment’s tokens are therefore trained only in its own block, even when several branches build on it (Figure 2). For rollout i, block b, and token position t, let $z _ { i , b , t }$ be the token, $h _ { i , b , t }$ the context that it was generated from, and $m _ { i , b , t } \in \{ 0 , 1 \}$ its loss mask. The mask is one for tokens that the model generated in block b, and zero for everything else, namely the task prompt, restored context, harness messages, and tool observations. Instructions or summary cards that are copied into the other role’s context are also masked. In other words, an instruction is trained once, in the Reflection turn that wrote it, and not again when Execution reads it. Likewise, a card is trained in the Execution segment that wrote it, not when Reflection reads it.

![](images/9627ab9940774f83b0cb6e051b720e7ee7dd7ce47c341632cd610bc1abebec87.jpg)  
Figure 2 Tree-aware GRPO on one search. (a) The search from Figure 1. In Reflection, the policy picks each node to resume. In Execution, it saves one checkpoint per segment $E _ { j }$ and, in $E _ { 6 } ,$ proposes P. The proposal P receives the terminal score $R _ { i }$ . (b) Reflection forms one causal block (R), and each Execution segment forms its own $\left( E _ { j } \right)$ . Colored spans are the model’s own outputs, which are trained, and gray spans are masked context. Instruction $I _ { j }$ is trained as a Reflection output in R and appears only as context in $E _ { j }$ (dotted). Card $S _ { j }$ is trained as an Execution output in $E _ { j }$ and appears only as context when it returns to R. The blocks follow the deep-research setting, in which Execution sees the cards along its path and only its own instruction. (c) Each of the K searches for a task receives one score. Its group-relative advantage $\hat { A } _ { i }$ applies to every generated token of search i, and a process penalty Q applies only to the turn it concerns. The objective uses a token, block, or rollout mean and can mix in a role mean with weight λ.

Rewards and advantages. For task $x ,$ the old policy samples K complete searches. Search i ends with a final output $y _ { i } ,$ the submitted answer in deep research or the graded repository state in SWE, and receives a binary terminal score $R _ { i } = R ( x , y _ { i } )$ . We use only non-positive process rewards, which we call process penalties. In successful searches, some generated turns receive a process penalty $Q _ { i , b , t }$ , which applies only to the tokens of that turn. The training signal has three components, and the ablations study a fourth.

• Terminal task reward. In deep research, a search scores one if the precision-first judge accepts its submitted answer. In SWE, it scores one if the oficial tests mark the task as resolved. Every other outcome scores zero.

• Failure penalty. Following FoldAgent (Sun et al., 2026), a turn that issues a malformed or rejected tool call receives Q = −1.

• Out-of-scope (OOS) penalty. In deep research, we also penalize work outside the assigned instruction. A judge sees the original task, the selected node’s card, the instruction, and brief metadata for each retrieval action. If the judge labels an action out of scope, the turn containing it receives Q = −0.2. Checkpoint and proposal turns are never penalized, and uncertain, malformed, or missing judge outputs give no penalty.

• Reflection finalization penalty (ablations only). In our design, it is Execution that decides when to propose a solution, so Reflection should not use its instruction to request one. An instruction that directs Execution to produce the solution rather than new evidence, such as “Propose Candidate A as the answer”, receives Q = −1. A pattern rule or the OOS judge detects such instructions (Appendix B.3).

There is no separate reward for checkpointing, selecting a particular node, or favoring any tree shape. Search structure is therefore not prescribed by any tree-specific reward.

We use the terminal rewards and process penalties to compute group-relative advantages. The group mean $\bar { R } _ { x }$ and standard deviation $s _ { x }$ are computed from the K terminal scores alone, so a penalty on one turn does not move the baseline of any other turn. Following FoldAgent (Sun et al., 2026), we divide by the standard deviation:

$$
\widehat { A } _ { i , b , t } = \frac { \mathrm { c l i p } ( R _ { i } + Q _ { i , b , t } , 0 , 1 ) - \bar { R } _ { x } } { s _ { x } } .\tag{2}
$$

When all K searches have the same score, we divide by one instead, so a group in which every search succeeds keeps its penalties at their original scale. Without penalties, every generated token of a search receives the same advantage.

Loss aggregation. Let $\rho _ { u } ( \theta )$ be the importance ratio of a generated token $u = ( i , b , t )$ under the context $h _ { i , b , t }$ it was generated from. The objective is a weighted sum of PPO’s clipped surrogate (Schulman et al., 2017) over the generated tokens U of one PPO minibatch:

$$
J _ { \mathrm { T R E E F L E X } } ( \theta ) = \mathbb { E } \Big [ \sum _ { u \in \mathcal { U } } w _ { u } \ \mathrm { m i n } \big ( \rho _ { u } ( \theta ) \widehat { A } _ { u } , \ \mathrm { c l i p } ( \rho _ { u } ( \theta ) , 1 - \epsilon _ { \mathrm { l o w } } , 1 + \epsilon _ { \mathrm { h i g h } } ) \widehat { A } _ { u } \big ) \Big ] ,\tag{3}
$$

where $w _ { u } \ge 0$ and $\textstyle \sum _ { u \in { \mathcal { U } } } w _ { u } = 1$ , and the clipping range is decoupled as in DAPO (Yu et al., 2025). For negative advantages, the per-token term is also bounded below by $c \widehat { A } _ { u }$ (dual clipping with $c = 3 ; \mathrm { A }$ ppendix C.5). A rollout spans several blocks, and both blocks and rollouts vary widely in length. The choice of $w _ { u }$ decides how much each part of a rollout counts. Write $k ( u ) , i ( u )$ , and $r ( u )$ for the block, rollout, and role of token $u ,$ and $u _ { k } , u _ { i } .$ , and $\mathcal { U } _ { r }$ for the generated tokens of block $k ,$ rollout i, and role r. We consider three averages, which difer in the unit that receives equal weight.

• Token mean. Set $w _ { u } = 1 / \vert \boldsymbol { { u } } \vert$ , which averages all tokens together. Every token counts the same, so a rollout that generates ten times as many tokens counts ten times as much.

• Block mean. Set $w _ { u } = 1 / ( | K | | \mathcal { U } _ { k ( u ) } | )$ , with K the blocks in the minibatch. It first averages the tokens within each block and then averages over blocks. Every block counts the same. This is the per-sequence average of common RL trainers, applied to our blocks. A rollout’s weight grows with its number of blocks, and a short Execution segment counts as much as a long one. A block too long for one training row is split, and each part then counts as a block (Appendix B).

• Rollout mean. Set $w _ { u } = 1 / ( \left| \mathcal { T } \right| \left| \mathcal { U } _ { i ( u ) } \right| )$ , with I the rollouts in the minibatch. It first averages the tokens within each rollout and then averages over rollouts. This extends the per-sample averaging of the original GRPO objective (Shao et al., 2024) to rollouts containing multiple blocks, giving each rollout equal total weight regardless of its length or number of blocks.

A separate imbalance arises between roles. Execution produces many more tokens than Reflection, so Reflection’s decisions about where to search and whether to accept a proposal receive little weight. The role mean, $w _ { u } ^ { \mathrm { r o l e } } = 1 / ( | \mathcal { R } _ { + } | \left| \mathcal { U } _ { r ( u ) } \right| )$ with $\mathcal { R } _ { + }$ the roles present, first averages within each role and then across roles, giving each role equal weight. A weight $\lambda \in [ 0 , 1 ]$ mixes the role mean with one of the three averages above, $w _ { u } = ( 1 - \lambda ) w _ { u } ^ { \mathrm { b a s e } } + \lambda w _ { u } ^ { \mathrm { r o l e } }$ . We train TreeFlex with the rollout mean and $\lambda = 0$ in both domains. Section 4.3 compares it with the other averages and with $\lambda = 0 . 3$ . Appendices B and C.3 give the remaining details.

## 3 Experimental Setup

## 3.1 Tasks and Datasets

Deep Research. We evaluate deep research with BrowseComp-Plus, whose compositional questions require finding and synthesizing evidence across multiple documents in a frozen corpus (Chen et al., 2026). We use the train/test split of Sun et al. (2026), with 680 training and 150 test questions, and further hold out 64 of the training questions as a fixed validation split, leaving 616 for training.

Agentic SWE. We draw 600 executable Python repository tasks from SWE-Gym (Pan et al., 2025), spanning ten repositories, and split them into 500 training tasks and 100 validation tasks. Each task pairs a natural-language issue with a repository snapshot and hidden executable tests. We train each method on tasks on which earlier training rollouts had mixed outcomes. FoldAgent trains on the 217 tasks on which its own rollouts had mixed outcomes, and TreeFlex on the 250 tasks on which the rollouts of either method had mixed outcomes. Evaluation uses all 500 SWE-bench Verified tasks and the oficial grader (Jimenez et al., 2024; OpenAI, 2024). Appendix D.1 describes how the training tasks were selected.

## 3.2 Comparison Protocol

Across both domains, comparisons between FoldAgent (Sun et al., 2026) and TreeFlex use a 32,768-token active-context limit. FoldAgent may launch at most ten branches, and TreeFlex may make at most ten navigations. Within these limits, each method keeps its own context layout, prompts, state management, and tool interfaces.

BrowseComp-Plus comparison. On BrowseComp-Plus, we compare flat ReAct, FoldAgent, and TreeFlex with the same base model, retriever, and retrieval limits. We evaluate the released Qwen3-4B-Instruct-2507 and Qwen3-30B-A3B-Instruct-2507 models (Qwen Team, 2025) without task-specific RL, and train all three methods from the 4B model.

Flat ReAct uses the same 32,768-token limit, except in one diagnostic run with a 254k-token limit. Appendix C.2 gives the exact accounting.

Following Sun et al. (2026), we train all three methods on the same data for 50 steps, using 32 questions per step, eight rollouts per question, and a learning rate of 10<sup>−6</sup>. The methods difer in their training objectives: TreeFlex uses tree-aware GRPO, FoldAgent uses FoldGRPO, and flat ReAct uses GRPO. We make four changes to the original setup. We use a 4B rather than a 36B model, select each run’s training checkpoint on the validation split, and sample eight rollouts per test question rather than one greedy rollout. We also apply the 32k active-context limit to the context after the initial prompt and use a larger model window so that the next call still fits. The answer judge of Appendix C.3 scores every rollout, and Appendix C.5 gives the full configuration.

SWE-bench Verified comparison. On SWE-bench Verified, we compare the complete FoldAgent and TreeFlex systems. Both start from the same Qwen3.5-9B model with thinking enabled (Qwen Team, 2026). On their respective task subsets drawn from the pool described above, we train TreeFlex with tree-aware GRPO and FoldAgent with FoldGRPO. Both use 32 tasks per step, eight rollouts per task, and a learning rate of 10<sup>−6</sup>. Each run trains for 25 steps, about three to four epochs over its tasks, and we select the checkpoint with the highest validation Mean@3 among steps 5, 10, 15, 20, and 25. For a controlled comparison, we also match the repository images, evaluation turn and time limits, sampling parameters, and oficial grader. On SWE, TreeFlex also allows at most two children per node. We sample three rollouts per test task and grade each on one repository state, whether or not the agent called finish. For FoldAgent, this is its final workspace. For TreeFlex, it is the approved proposal if there is one, otherwise the latest proposal, and otherwise the final workspace. Appendix D.3 gives the full alignment, including the remaining diferences in training rewards, and Appendix B.3 specifies all rewards.

## 4 Results

All comparisons are paired over the same questions or tasks. We call a diference significant when its 95% paired-bootstrap confidence interval excludes zero (Appendix E.1).

## 4.1 Deep Research

Main results. TreeFlex gives the highest mean accuracy for every model under the same context limit (Table 2).

After RL, TreeFlex reaches 48.8% Mean@8 and 66.0% Pass@8, significantly outperforming flat ReAct by 7.3 and 10.0 points and FoldAgent by 6.0 and 10.7 points, respectively. An RL-trained flat ReAct diagnostic with a 254k-token limit reaches only 42.7% Mean@8 and 54.7% Pass@8. The run does use the extra room, with 12% of its late-training rollouts exceeding 64k context tokens and a 90th percentile of 87.1k, versus 33.3k for the matched 32k run, yet its Mean@8 improves by only 1.2 points over the matched 32k run (Appendix C.2).

Table 2 TreeFlex gives the highest mean accuracy under matched context limits. Each cell aggregates eight samples for each of 150 questions. Mean@8 is the accuracy averaged over the eight samples, and Pass@8 counts a question as solved if any sample is correct. Usage values are means per rollout. Generated tokens exclude retrieved observations, and reads count get\_document and open\_page calls. Ctx. is the active-context limit after the initial prompt, and Br./nav. is the maximum number of branches (FoldAgent) or navigations (TreeFlex). The flat ReAct row with a 254k limit is a diagnostic outside the matched comparison.
<table><tr><td colspan="4"></td><td colspan="2">Quality (%)</td><td colspan="3">Usage per rollout</td></tr><tr><td>Model</td><td>Method</td><td>Ctx.</td><td>Br./nav.</td><td>Mean@8</td><td>Pass@8</td><td>Gen. tokens</td><td>Searches</td><td>Reads</td></tr><tr><td rowspan="3">Qwen3-30B (base)</td><td>Flat REACT</td><td>32k</td><td>0</td><td>42.6</td><td>55.3</td><td>584</td><td>3.1</td><td>3.7</td></tr><tr><td>FOLDAGENT</td><td>32k</td><td>10</td><td>46.1</td><td>66.7</td><td>3,212</td><td>8.7</td><td>11.5</td></tr><tr><td>TREEFLEX</td><td>32k</td><td>10</td><td>48.3</td><td>62.7</td><td>4,833</td><td>14.3</td><td>20.6</td></tr><tr><td rowspan="3">Qwen3-4B (base)</td><td>Flat REACT</td><td>32k</td><td>0</td><td>36.2</td><td>56.0</td><td>1,082</td><td>2.4</td><td>2.3</td></tr><tr><td>FOLDAGENT</td><td>32k</td><td>10</td><td>33.1</td><td>54.0</td><td>2,553</td><td>5.7</td><td>12.3</td></tr><tr><td>TREEFLEX</td><td>32k</td><td>10</td><td>43.6</td><td>65.3</td><td>4,493</td><td>8.8</td><td>15.6</td></tr><tr><td rowspan="4">Qwen3-4B (RL)</td><td>Flat REACT</td><td>32k</td><td>0</td><td>41.5</td><td>56.0</td><td>829</td><td>3.4</td><td>2.0</td></tr><tr><td>FOLDAGENT</td><td>32k</td><td>10</td><td>42.8</td><td>55.3</td><td>1,990</td><td>9.3</td><td>3.5</td></tr><tr><td>TREEFLEX</td><td>32k</td><td>10</td><td>48.8</td><td>66.0</td><td>5,677</td><td>20.7</td><td>9.7</td></tr><tr><td>Flat REACT</td><td>254k</td><td>0</td><td>42.7</td><td>54.7</td><td>1,501</td><td>5.5</td><td>2.5</td></tr></table>

Without task-specific RL, TreeFlex gives the highest Mean@8 at both 4B (43.6%) and 30B (48.3%), although FoldAgent has the highest 30B Pass@8 (66.7%). On validation, TreeFlex leads all baseline runs at every evaluated training step (Appendix C.6).

Repeated sampling and voting. These improvements require more inference. TreeFlex uses 5.7k generated tokens and 30.4 retrieval calls per rollout, compared with 2.0k and 12.8 for FoldAgent, and 0.8k and 5.4 for flat ReAct.

Figure 3 asks whether more samples let a baseline catch up. With the same number of samples, they do not. At k = 8, TreeFlex reaches 66.0% Pass@8, versus 56.0% for flat ReAct and 55.3% for FoldAgent. The paired outcomes are strongly asymmetric: TreeFlex solves 20 questions missed by all eight flat ReAct samples, while flat ReAct solves only 5 missed by all eight TreeFlex samples. The corresponding counts against FoldAgent are 20 and 4. Repeated flat sampling mostly revisits the same evidence. Across its eight samples, flat ReAct opens only 5.3 distinct documents per question and reads 34% of the annotated evidence documents, as much as a single TreeFlex rollout (34%) (Appendix E.3). Majority voting likewise improves only slightly with repeated flat sampling. From one to eight samples, flat ReAct improves by only 3.3 points, reaching 44.8%, whereas TreeFlex improves to 53.9%, 9.1 points above eight-sample flat voting.

With a similar number of retrieval calls, a single TreeFlex rollout is more accurate than repeated sampling of flat ReAct. At 30.4 calls it reaches 48.8%, 4.8 points above five-sample flat voting (44.0% at 27.0 calls).

Accuracy by question dificulty. We next ask whether branching helps most on questions that a flat agent rarely solves, rather than uniformly improving easy questions. Following Sun et al. (2026), an independent flat ReAct run (Appendix C.2) defines 58 easy questions (at least seven of eight samples correct), 67 hard questions (none correct), and 25 medium questions.

The improvements over flat ReAct are concentrated on harder questions (Figure 4). On easy questions, TreeFlex reaches 92.7%, close to flat ReAct (90.5%) and above FoldAgent (85.1%). On medium questions, TreeFlex reaches 42.0%, versus 28.5% for flat ReAct and 37.0% for FoldAgent. On hard questions, it reaches 13.4%, versus 3.9% and 8.4%. TreeFlex also searches more on harder questions. From easy to hard, searches rise from 8.9 to 29.9, reads from 7.6 to 11.8, and tree size from 5.8 to 9.1 nodes, whereas flat ReAct stays below four searches and about two reads in every bin.

![](images/222c48c89dc6f599e13ec5c0c3bb0132deeb3f22d1e5eb1b59b358b9eec5108e.jpg)

![](images/971a6b1c8c2156e8df24ed0e7d21fb89155f9c440b165f7ad291012cac4c5b1f.jpg)  
Figure 3 Eight samples of a baseline do not close the Pass@k gap. Curves use the 32k Qwen3-4B RL models and eight samples for each of 150 questions. (a) Pass@k. (b) Majority-vote accuracy over normalized answers against retrieval calls (searches plus reads), averaged over all k-sample subsets with ties broken at random. Voting never uses correctness labels. We omit k = 2 because it gives the same result as k = 1.

## 4.2 Agentic SWE

Table 3 compares the two systems before and after RL. It also includes a FoldAgent variant with a step-wise delegation prompt, described below. Without task-specific RL, TreeFlex reaches 51.1% Mean@3 and 64.4% Pass@3, versus 31.7% and 44.0% for FoldAgent. RL raises FoldAgent by 11.7 points, mostly by teaching it to finish within the context limit. The share of its rollouts that end with finish rises from 4% to 17%, and rollouts that finish and resolve the task account for 9.7 of the 11.7 points. RL adds a smaller 3.8 points for TreeFlex, whose base policy already ends 86% of rollouts with an approved proposal, so training has less to fix in how a search ends. After RL, TreeFlex resolves 54.9% of tasks, significantly more than FoldAgent (11.6 points) and its step-wise variant (9.7 points).

With the released FoldAgent prompt, the base Qwen3.5-9B model almost never launches a branch. Only 2 of its 1,500 rollouts on SWE-bench Verified do, so nearly every rollout keeps all its work in the main context, and 96% of rollouts end at the context limit. We therefore wrote a step-wise delegation prompt that asks the main agent to split the repair into a sequence of small branches, each returning one checkable result such as one edit or one focused test run, and to choose each next branch from the report the previous one returns (Appendix F.7). This raises FoldAgent to 49.5% Mean@3, close to TreeFlex’s 51.1%, although only 42% of its rollouts end with finish (Appendix E.2). RL on top of this prompt lowers its resolve rate to 45.2%. During training, this variant receives a positive task reward only if the main agent calls finish, whereas test grading evaluates the final workspace whether or not finish is called. RL increases the share of rollouts that both finish and resolve the task from 31.5% to 40.8%, but reduces the share that resolve without finish from 18.1% to 4.4%.

TreeFlex uses more turns and time per rollout. After RL, it takes 113 turns and 29.3 minutes, versus 49 and 6.4 for FoldAgent. Before RL, the step-wise variant takes more turns than TreeFlex (132 versus 98) and a similar time (21.9 versus 24.8 minutes), and it is still less accurate.

## 4.3 Training Objective Ablations

Deep research. On BrowseComp-Plus, we vary three settings of the training objective: the loss aggregation of Equation (3) (token, block, or rollout mean), the role weight λ, which gives Reflection’s few tokens more weight, and a finalization penalty, which discourages Reflection instructions that merely ask Execution to

![](images/6d7492fb744d3fb88ad7138abade8edb33dd9eb3ca486dee401662ab899d9f2b.jpg)  
Figure 4 TreeFlex leads in every difficulty bin. Bars show mean accuracy over eight samples per question; error bars are 95% question-bootstrap intervals. Dificulty follows the independent-reference rule of Sun et al. (2026).

Table 3 TreeFlex resolves the most SWE-bench Verified tasks, before and after RL. Each row uses Qwen3.5-9B with a 32k active-context limit and at most ten branches or navigations, and aggregates three rollouts for each of 500 tasks. Mean@3 is the resolve rate averaged over the three rollouts, and Pass@3 counts a task as resolved if any rollout resolves it. RL models are selected among steps 5, 10, 15, 20, and 25 by Mean@3 on 100 SWE-Gym validation tasks. Turns and minutes are means per rollout.

<table><tr><td rowspan="2">System</td><td rowspan="2">RL</td><td colspan="2">Resolved (%)</td><td colspan="2">Usage per rollout</td></tr><tr><td>Mean@3</td><td>Pass@3</td><td>Turns</td><td>Minutes</td></tr><tr><td rowspan="2">FOLDAGENT</td><td>none</td><td>31.7</td><td>44.0</td><td>34</td><td>5.0</td></tr><tr><td>FoldGRPO</td><td>43.3</td><td>57.8</td><td>49</td><td>6.4</td></tr><tr><td>FoLDAGENT, step-wise</td><td>none</td><td>49.5</td><td>63.0</td><td>132</td><td>21.9</td></tr><tr><td>delegation prompt</td><td>FoldGRPO</td><td>45.2</td><td>58.2</td><td>96</td><td>10.0</td></tr><tr><td rowspan="2">TREEFLEX</td><td>none</td><td>51.1</td><td>64.4</td><td>98</td><td>24.8</td></tr><tr><td>tree-aware GRPO</td><td>54.9</td><td>68.0</td><td>113</td><td>29.3</td></tr></table>

finish. Table 4 lists seven variants. All other settings match the main configuration (Appendix C.5), including step selection on validation.

None of these settings is essential. All seven variants outperform flat ReAct (41.5%) and FoldAgent (42.8%). The most accurate variant is the main configuration, which uses the rollout mean with λ = 0 (48.8% Mean@8). Each variant is a single training run, so we read the diferences below as observations about these runs rather than as efects of the settings.

The loss aggregation mainly changes the shape of the search. With λ = 0, the rollout mean leaves the search much as it was at the start of training. The token mean makes the policy branch far more often and return to the root more often, and the block mean makes it fork more often below the root (Table 4). Yet the three variants end within 2.1 points of one another, so how often and where the policy branches does not predict its accuracy.

The role weight and the finalization penalty show no consistent pattern. Raising λ to 0.3 lowers accuracy with the rollout mean but raises it slightly with the token mean, and its efect on branching reverses once the penalty is added. The penalty lowers the accuracy of the token mean at both values of λ.

Agentic SWE. On SWE, RL improves the resolve rate over the base model only with the rollout mean (Table 5). With the same data, λ = 0, and checkpoint-selection procedure, the token mean reaches 52.1%, close to the base model’s 51.1%, whereas the rollout mean reaches 54.9%, significantly above both.

Table 4 Branching varies widely across variants while test accuracy stays within a few points. Loss is the aggregation of Equation (3), λ the role weight, and Pen. the finalization penalty. Val. is Mean@3 on the 64 validation questions at the step selected for each run, and Mean@8 and Pass@8 are test results at that step. Tree behavior compares the first and last ten training steps, using 160 inspected rollouts from each. Branched is the share of rollouts in which some node has more than one child. Root and Ancestor nav. are the shares of navigations that target the root or an earlier ancestor. Internal fork is the share of rollouts in which a node other than the root has more than one child.
<table><tr><td></td><td></td><td></td><td colspan="3">Accuracy (%)</td><td colspan="5">Tree behavior during training (%)</td></tr><tr><td>Loss</td><td>λ</td><td>Pen.</td><td>Val.</td><td>Mean@8</td><td>Pass@8</td><td>Branched</td><td></td><td></td><td>Root nav. Ancestor nav. Internal fork</td><td></td></tr><tr><td>rollout (main)</td><td>0</td><td>no</td><td>38.0</td><td>48.8</td><td>66.0</td><td>53 → 59</td><td></td><td> $1 9  2 3$ </td><td> $5  2$ </td><td> $1 9 \to 1 8$ </td></tr><tr><td>block</td><td>0</td><td>no</td><td>34.9</td><td>48.4</td><td>65.3</td><td>38 → 66</td><td></td><td> $1 7  1 9$ </td><td> $3  1 0$ </td><td> $1 2  3 2$ </td></tr><tr><td>rollout</td><td>0.3</td><td>no</td><td>35.4</td><td>45.3</td><td>66.0</td><td>40 → 51</td><td></td><td> $1 3  9$ </td><td> $5  5$ </td><td> $1 6  2 6$ </td></tr><tr><td>token</td><td>0.3</td><td>no</td><td>40.1</td><td>47.7</td><td>66.0</td><td>50 → 63</td><td></td><td> $2 0  1 9$ </td><td> $4  0$ </td><td> $1 9  1 2$ </td></tr><tr><td>token</td><td>0</td><td>no</td><td>33.9</td><td>46.7</td><td>65.3</td><td> $4 9  8 3$ </td><td></td><td> $1 7  3 3$ </td><td> $5  1$ </td><td> $1 8  2 4$ </td></tr><tr><td>token</td><td>0</td><td>yes</td><td>35.4</td><td>46.4</td><td>66.0</td><td> $5 0  4 4$ </td><td></td><td> $1 2  6$ </td><td> $8  1 1$ </td><td> $2 5  3 1$ </td></tr><tr><td>token</td><td>0.3</td><td>yes</td><td>36.5</td><td>44.2</td><td>62.0</td><td> $4 7  7 1$ </td><td></td><td> $1 9  2 2$ </td><td> $3  4$ </td><td> $1 4  3 0$ </td></tr></table>

Table 5 On SWE, the rollout mean improves over the base model and the token mean does not. Both RL runs train on the 250-task union and use the step among 5, 10, 15, 20, and 25 with the highest validation Mean@3 (Val.). Test values aggregate three rollouts for each of 500 SWE-bench Verified tasks; $\Delta$ is the change from the base model. Returns is the share of test rollouts that return to the root or an earlier node.
<table><tr><td>Run</td><td>Step</td><td>Val.</td><td>Mean@3</td><td>Pass@3</td><td> $\Delta$ </td><td>Returns</td></tr><tr><td>base model</td><td></td><td>22.3</td><td>51.1</td><td>64.4</td><td></td><td>43.0</td></tr><tr><td>token mean</td><td>20</td><td>26.7</td><td>52.1</td><td>67.4</td><td> $+ 1 . 0$ </td><td>24.7</td></tr><tr><td>rollout mean</td><td>25</td><td>28.7</td><td>54.9</td><td>68.0</td><td> $+ 3 . 8$ </td><td>12.2</td></tr></table>

## 5 Related Work

Reasoning and acting. ReAct interleaves language-model reasoning with environment actions in one context (Yao et al., 2023b). Reflexion augments agents with persistent verbal feedback across attempts (Shinn et al., 2023). These agents keep a single linear history. TreeFlex keeps this loop within each Execution segment but saves intermediate states in a tree that the policy can return to.

Tree search with learned models. In game playing, AlphaZero and ELF OpenGo pair Monte Carlo tree search with a learned policy-value network (Silver et al., 2018; Tian et al., 2019). The search algorithm is fixed, and the network supplies priors and value estimates. Tree of Thoughts, Graph of Thoughts, world-model planning, and Language Agent Tree Search bring tree and graph search to language models (Yao et al., 2023a; Besta et al., 2024; Hao et al., 2023; Zhou et al., 2024). The language model proposes partial solutions and usually scores them, and a fixed procedure decides which to expand next. This procedure is breadth- or depth-first search in Tree of Thoughts, MCTS in world-model planning and Language Agent Tree Search, and a user-specified graph of operations in Graph of Thoughts. For web agents, Koh et al. (2025) run a best-first search guided by a prompted value function and restore pages by resetting the environment and replaying actions. Policy-Guided Tree Search instead trains a separate graph-network policy with RL to decide whether to expand, branch, backtrack, or stop over the reasoning steps of an unmodified language model (Li, 2025). Stream of Search, Searchformer, and Dualformer train models on serialized search traces (Gandhi et al., 2024; Lehnert et al., 2024; Su et al., 2025), and ASTRO trains on CoT traces converted from Monte Carlo tree search (Kim et al., 2025). In all four, the search is written out as one sequence, so returning to an earlier node means writing it out again rather than restoring a saved state. TreeFlex instead lets the agent’s own policy choose which saved node to resume and what to do there, while the harness restores the corresponding context and environment state.

Learning information-seeking behavior. RL has trained language models to interleave reasoning with search, as in Search-R1 and R1-Searcher (Jin et al., 2025; Song et al., 2025), and DeepResearcher scales this training to real web environments (Zheng et al., 2025). WebDancer and WebSailor fine-tune ReAct-style agents on synthesized trajectories and then train them with RL (Wu et al., 2025; Li et al., 2025), and WebSailor makes its training questions harder by obscuring key details. These systems learn strong search policies over one linear trajectory.

Policy optimization over trees. Tree-GRPO uses tree-structured rollouts for finer credit assignment, but the trained agent still acts along one trajectory (Ji et al., 2026). In TreeFlex, the tree is the agent’s own search, and each complete search is one GRPO sample (Shao et al., 2024).

Learned context management. At the reasoning-trace level, Accordion-Thinking learns when to summarize completed steps and discard their detailed token history (Yang et al., 2026). MEM1 rewrites a compact internal state at every turn and discards the rest of its context, so its memory stays nearly constant in size (Zhou et al., 2026). MemPO trains the agent to write its own memory summaries and rewards each summary by how much it raises the model’s probability of the correct answer (Li et al., 2026). For long-horizon web search, AgentFold learns at each step either to condense the latest step into a short summary or to merge it with recent summaries into a coarser one (Ye et al., 2026), and LongSeeker lets the agent summarize past steps, cut tool outputs down to exact excerpts, delete steps, or roll back to an earlier step (Lu et al., 2026). Context-Folding trains FoldAgent to run subtasks in temporary branches and fold each back into the main context as a report (Sun et al., 2026). These approaches make compression or retention part of the agent’s behavior, but keep only the current working history, so discarded work cannot be resumed later.

SWE agents. SWE-bench evaluates whether an agent can resolve real repository issues under executable tests (Jimenez et al., 2024), while SWE-Gym supplies repository tasks and environments for training agents and verifiers (Pan et al., 2025). OpenHands provides an open platform in which software agents act through CodeAct-style code execution (Wang et al., 2025). SWE-agent designs the commands and feedback format for an agent that edits one workspace along a single ReAct-style trajectory (Yang et al., 2024). Agentless instead samples many candidate patches independently from a fixed pipeline of localization, repair, and patch validation, and filters them with tests (Xia et al., 2025). SWE-Search runs Monte Carlo tree search over repository states kept in a git-like commit tree, selecting nodes with a hand-designed rule and scoring them with a prompted language model (Antoniades et al., 2025). In TreeFlex, the trained policy itself chooses which saved repository state to work from next.

## 6 Limitations and Future Work

Our experiments are small in scale. We train a 4B model on 616 deep-research questions and a 9B model on at most 250 SWE tasks. Each training configuration is run once, so our significance tests capture question sampling but not variation across training runs. The comparisons are between complete systems that also difer in prompts and tool interfaces. TreeFlex uses more inference-time computation than the baselines, and its segments run sequentially, which adds latency.

Training at a larger scale would show whether these results hold for stronger models, more training data, and longer training. On SWE, validation accuracy was still rising at the last saved step, so longer training may help there directly. SFT on successful searches that save, compare, and revisit nodes could give RL a better starting point in both domains. Future work could also reward eficiency directly by penalizing generated tokens or tool calls in successful searches.

## 7 Conclusion

TreeFlex saves an agent’s progress as nodes of a search tree, lets one model choose which node to resume and what work to do there, and trains both roles end to end with tree-aware GRPO. Across both domains, TreeFlex solves more tasks than baselines trained from the same starting model.

## References

Antonis Antoniades, Albert Örwall, Kexun Zhang, Yuxi Xie, Anirudh Goyal, and William Yang Wang. SWE-Search: Enhancing software agents with Monte Carlo tree search and iterative refinement. In International Conference on Learning Representations, 2025.

Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, and Torsten Hoefler. Graph of thoughts: Solving elaborate problems with large language models. In AAAI, 2024.

Zijian Chen, Xueguang Ma, Shengyao Zhuang, Ping Nie, Kai Zou, Sahel Sharifymoghaddam, Andrew Liu, Joshua Green, Kshama Patel, Ruoxi Meng, Mingyi Su, Yanxi Li, Haoran Hong, Xinyu Shi, Xuye Liu, Hosna Oyarhoseini, Nandan Thakur, Crystina Zhang, Luyu Gao, Wenhu Chen, and Jimmy Lin. BrowseComp-Plus: A fair and disentangled evaluation benchmark for deep search agents. In Proceedings of the Annual Meeting of the Association for Computational Linguistics, 2026.

Kanishk Gandhi, Denise Lee, Gabriel Grand, Muxin Liu, Winson Cheng, Archit Sharma, and Noah D. Goodman. Stream of search (SoS): Learning to search in language. In COLM, 2024.

Shibo Hao, Yi Gu, Haodi Ma, Joshua Jiahua Hong, Zhen Wang, Daisy Zhe Wang, and Zhiting Hu. Reasoning with language model is planning with world model. In EMNLP, 2023.

Yuxiang Ji, Ziyu Ma, Yong Wang, Guanhua Chen, Xiangxiang Chu, and Liaoni Wu. Tree search for LLM agent reinforcement learning. In International Conference on Learning Representations, 2026.

Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. SWE-bench: Can language models resolve real-world GitHub issues? In ICLR, 2024.

Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-R1: Training LLMs to reason and leverage search engines with reinforcement learning. In Conference on Language Modeling, 2025.

Joongwon Kim, Anirudh Goyal, Liang Tan, Hannaneh Hajishirzi, Srinivasan Iyer, and Tianlu Wang. ASTRO: Teaching language models to reason by reflecting and backtracking in-context. arXiv preprint arXiv:2507.00417, 2025.

Jing Yu Koh, Stephen McAleer, Daniel Fried, and Ruslan Salakhutdinov. Tree search for language model agents. Transactions on Machine Learning Research, 2025.

Lucas Lehnert, Sainbayar Sukhbaatar, DiJia Su, Qinqing Zheng, Paul McVay, Michael Rabbat, and Yuandong Tian. Beyond A\*: Better planning with transformers via search dynamics bootstrapping. In COLM, 2024.

Kuan Li, Zhongwang Zhang, Huifeng Yin, Liwen Zhang, Litu Ou, Jialong Wu, Wenbiao Yin, Baixuan Li, Zhengwei Tao, Xinyu Wang, Weizhou Shen, Junkai Zhang, Dingchu Zhang, Xixi Wu, Yong Jiang, Ming Yan, Pengjun Xie, Fei Huang, and Jingren Zhou. WebSailor: Navigating super-human reasoning for web agent. arXiv preprint arXiv:2507.02592, 2025.

Ruoran Li, Xinghua Zhang, Haiyang Yu, Shitong Duan, Xiang Li, Wenxin Xiang, Chonghua Liao, Xudong Guo, Yongbin Li, and Jinli Suo. MemPO: Self-memory policy optimization for long-horizon agents. In Findings of the Association for Computational Linguistics: ACL 2026, pages 23286–23301, 2026.

Yang Li. Policy guided tree search for enhanced LLM reasoning. In ICML, 2025.

Yijun Lu, Rui Ye, Yuwen Du, Jiajun Wang, Songhua Liu, and Siheng Chen. LongSeeker: Elastic context orchestration for long-horizon search agents. arXiv preprint arXiv:2605.05191, 2026.

OpenAI. Introducing SWE-bench verified. https://openai.com/index/introducing-swe-bench-verified/, 2024.

Jiayi Pan, Xingyao Wang, Graham Neubig, Navdeep Jaitly, Heng Ji, Alane Suhr, and Yizhe Zhang. Training software engineering agents and verifiers with SWE-Gym. In ICML, 2025.

Qwen Team. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Qwen Team. Qwen3.5-9B. Hugging Face model card, 2026. https://huggingface.co/Qwen/Qwen3.5-9B.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems, 2023.

David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, Timothy Lillicrap, Karen Simonyan, and Demis Hassabis. A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. Science, 362(6419): 1140–1144, 2018.

Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, Lei Fang, and Ji-Rong Wen. R1-Searcher: Incentivizing the search capability in LLMs via reinforcement learning. arXiv preprint arXiv:2503.05592, 2025.

DiJia Su, Sainbayar Sukhbaatar, Michael Rabbat, Yuandong Tian, and Qinqing Zheng. Dualformer: Controllable fast and slow thinking by learning with randomized reasoning traces. In ICLR, 2025.

Weiwei Sun, Miao Lu, Zhan Ling, Kang Liu, Xuesong Yao, Yiming Yang, and Jiecao Chen. Scaling long-horizon agent via context folding. In International Conference on Machine Learning, 2026.

Yuandong Tian, Jerry Ma, Qucheng Gong, Shubho Sengupta, Zhuoyuan Chen, James Pinkerton, and C. Lawrence Zitnick. ELF OpenGo: An analysis and open reimplementation of AlphaZero. In ICML, 2019.

Xingyao Wang, Boxuan Li, Yufan Song, Frank F. Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, et al. OpenHands: An open platform for AI software developers as generalist agents. In International Conference on Learning Representations, 2025.

Jialong Wu, Baixuan Li, Runnan Fang, Wenbiao Yin, Liwen Zhang, Zhengwei Tao, Dingchu Zhang, Zekun Xi, Gang Fu, Yong Jiang, Pengjun Xie, Fei Huang, and Jingren Zhou. WebDancer: Towards autonomous information seeking agency. In Advances in Neural Information Processing Systems, 2025.

Chunqiu Steven Xia, Yinlin Deng, Soren Dunn, and Lingming Zhang. Demystifying LLM-Based software engineering agents. Proceedings of the ACM on Software Engineering, 2(FSE):801–824, 2025.

John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. SWE-agent: Agent-computer interfaces enable automated software engineering. In Advances in Neural Information Processing Systems, 2024.

Zhicheng Yang, Zhijiang Guo, Yinya Huang, Yongxin Wang, Wenlei Shi, Yiwei Wang, Xiaodan Liang, and Jing Tang. Accordion-Thinking: Self-regulated step summaries for eficient and readable LLM reasoning. In International Conference on Machine Learning, 2026.

Shunyu Yao, Dian Yu, Jefrey Zhao, Izhak Shafran, Thomas L. Grifiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. In NeurIPS, 2023a.

Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In ICLR, 2023b.

Rui Ye, Zhongwang Zhang, Kuan Li, Huifeng Yin, Zhengwei Tao, Yida Zhao, Liangcai Su, Liwen Zhang, Zile Qiao, Xinyu Wang, Pengjun Xie, Fei Huang, Jingren Zhou, Siheng Chen, and Yong Jiang. AgentFold: Long-horizon web agents with proactive context folding. In International Conference on Learning Representations, 2026.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. DAPO: An open-source LLM reinforcement learning system at scale. In Advances in Neural Information Processing Systems, 2025.

Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai, Lyumanshan Ye, Pengrui Lu, and Pengfei Liu. DeepResearcher: Scaling deep research via reinforcement learning in real-world environments. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, pages 414–431, 2025.

Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman, Haohan Wang, and Yu-Xiong Wang. Language agent tree search unifies reasoning, acting, and planning in language models. In ICML, 2024.

Zijian Zhou, Ao Qu, Zhaoxuan Wu, Sunghwan Kim, Alok Prakash, Daniela Rus, Jinhua Zhao, Bryan Kian Hsiang Low, and Paul Pu Liang. MEM1: Learning to synergize memory and reasoning for eficient long-horizon agents. In International Conference on Learning Representations, 2026.

Algorithm 1 One TreeFlex search   
1: Initialize tree T with root r and the global record M   
2: while the rollout limits are not reached do   
3: if an Execution proposal (y, e) is pending then   
4: Reflection either generates approve\_commit(p) for an available proposal p or navigate(v, i)   
5: if Reflection approves then   
6: return the answer stored at p, T   
7: end if   
8: else   
9: Reflection generates navigate(v, i) from the global record M   
10: end if   
11: Restore the path to v and start Execution with instruction i   
12: while the current Execution segment is active do   
13: Execution uses ordinary domain tools   
14: if Execution generates checkpoint(s), or the harness requires one when the context is full then   
15: Save a child of v with card s; append the result to M; break   
16: else if Execution generates commit(y, e) then   
17: Save proposal (y, e) as a child of v; append it to M; break   
18: end if   
19: end while   
20: end while   
21: return no approved answer and the reason the rollout stopped

## Appendix

## A Method Details

## A.1 Search Procedure

Algorithm 1 gives the search loop of Section 2.1.

## A.2 Context Layouts

The harness builds each role’s context from fixed templates. We show the templates with placeholders in braces and shorten long fixed instructions. Appendix F reproduces the system prompts.

Reflection. Reflection runs as one conversation in which the model calls tools over many turns. Each navigate call returns only when the Execution segment it starts has ended. Its result describes the new node and includes the node’s card.

```ini
[system] {Reflection system prompt}
[user] {task} {navigation example} {initial tree: root}
[assistant] navigate(node="root", instruction="{objective 1}")
[tool] [EXECUTION CHECKPOINT]
node: n1
parent: root
path: root -> n1
{deep research: card: {card}}
{SWE: report: {card} trusted_receipt: {harness summary: difstat, changed files, test runs}}
valid_navigation_nodes: [...]
available_reflection_actions: ["navigate"]
[/EXECUTION CHECKPOINT]
[assistant] navigate(node="n1", instruction="{objective 2}")
[tool] [EXECUTION COMMIT PROPOSAL]
node: n2
parent: n1
path: root -> n1 -> n2
{deep research: answer, explanation, supporting_docids}
{SWE: report: {completion message} trusted_receipt: {...}}
available_reflection_actions: ["approve_commit", "navigate"]
[/EXECUTION COMMIT PROPOSAL]
[assistant] approve_commit(node="n2") -> the search ends
```

In SWE, each result also gives the number of turns and navigations left, says whether the node’s raw context is still saved, and briefly suggests where to navigate next.

Execution in deep research. Each segment starts from a new context that holds the cards along the path and the current instruction, but no earlier instructions.

[system] {Execution system prompt}   
[user] {question}   
[user] [Resumed checkpoint n1]   
=== ACTIVE PATH CARDS (root to current) ===   
root: {card}   
n1: {card}   
{available tools: search, get\_document, checkpoint, commit}   
<EXECUTION\_TASK>   
{instruction}   
</EXECUTION\_TASK>   
[assistant] {tool calls and results, ending with checkpoint(summary) or commit(answer, explanation)}

Execution in SWE, with the full saved context. The harness restores the raw context saved at the selected node, which holds every earlier segment on the path exactly as the model generated it. It then adds one user message that opens the new segment. In the layout below, only these opening messages come from templates, for example “[Cursor restored to n1] . . . ”.

```ini
[system] {Execution system prompt}
[user] {issue and repository location}
[ ] [Cursor restored to root] ... <EXECUTION_TASK> {instruction 1} </EXECUTION_TASK> ...
[assistant] {reasoning} {tool call, e.g. execute_bash(...)}
[tool] {tool result}
[assistant] checkpoint(summary="{card of n1}")
[tool] Checkpoint n1 saved; control returns to REFLECTION.
[user] [Cursor restored to n1] ... <EXECUTION_TASK> {instruction 2} </EXECUTION_TASK> ...
[assistant] .. (segment 2, as generated)
[user] [Cursor restored to n2]
The saved repository and raw Execution path are active.
turns_remaining: {k}
<EXECUTION_TASK>
{instruction 3}
</EXECUTION_TASK>
<SEGMENT_BOUNDARY>
{stopping rules: checkpoint once the objective is resolved or blocked}
</SEGMENT_BOUNDARY>
```

Execution in SWE, with a shortened context. When a node is saved because the context is full, the harness replaces its saved context with a single message that holds the cards along the path and a shortened dif of the repository. Earlier nodes keep their raw context.

[system] {Execution system prompt}   
[user] {issue and repository location}   
[user] [Compacted checkpoint n3]   
The earlier raw Execution transcript on this path was compacted. The checkpoint   
cards below are the complete record of prior work on this path, oldest first.   
[n1] objective: {instruction, at most 150 tokens}   
card: {card, at most 600 tokens}   
[n2] ...   
[n3] (this checkpoint) objective: {...}   
card: {...}   
[Repository dif relative to the original checkout]   
{difstat, changed files, and a patch prefix, together at most 2,048 tokens}   
The exact repository state of n3 is restored; ...   
[user] [Cursor restored to n3]   
The exact saved repository is active. Earlier raw tool output was compacted;   
rely on the retained path cards and re-inspect anything decisive.   
turns\_remaining: {k}   
<EXECUTION\_TASK> {instruction} </EXECUTION\_TASK>   
<SEGMENT\_BOUNDARY> {stopping rules} </SEGMENT\_BOUNDARY>

## B Training Details

## B.1 Causal Block Packing

Each training row has a prompt part and a response part of fixed length, and the loss applies only to tokens in the response part. A causal block starts with restored context and continues with the tokens the model generated in that segment. We put as much of the restored context as fits into the prompt part, and everything from the first generated token onward into the response part. Restored context that does not fit goes into the response part but is left out of the loss, so the model still sees it but is not trained on it. If the response part is too long, we split the block into several rows, and we split only between model calls. Every token is therefore trained on exactly the context it was generated from.

## B.2 Building PPO Minibatches

Each causal block becomes one training row, or several rows if it is too long (Appendix B.1). All rows of a rollout share its advantage. The group mean and standard deviation are computed over rollouts, so a rollout counts once however many rows it produces. The rollouts of one prompt form a prompt group. All rows from a prompt group go into the same PPO minibatch, so we count minibatch size in prompt groups, not rows. We compute the weights of Equation (3) from token counts over the whole minibatch, before the minibatch is split across GPUs and microbatches. Padding rows therefore get zero weight, and the split does not change the token mean or the rollout mean. The block mean weights every row equally, so a block split into several rows counts once per row.

## B.3 Domain Rewards

We use the notation of Section 2.2. Rollout i receives a terminal reward $R _ { i }$ , and token t in causal block b of that rollout receives a process penalty $Q _ { i , b , t }$ . Here $\mathbf { 1 } [ \cdot ]$ is 1 if its condition holds and 0 otherwise.

Terminal reward. For BrowseComp-Plus, $R _ { i }$ is 1 if the submitted answer is judged correct and 0 otherwise. Flat ReAct submits its answer by calling commit, TreeFlex submits the proposal that Reflection approves, and FoldAgent submits when its main agent calls finish.

$$
R _ { i } = { \bf 1 } \mathrm { [ r o l l o u t ~ } i \mathrm { ~ s u b m i t s ~ a ~ v a l i d ~ a n s w e r ~ } \land \mathrm { ~ t h e ~ j u d g e ~ m a r k s ~ i t ~ c o r r e c t } ] . \quad\tag{4}
$$

For SWE, the terminal reward comes from the oficial grading of one candidate state of the repository.

$$
R _ { i } = \bf { 1 } [ \mathrm { t h e ~ c a n d i d a t e ^ { - } s ~ g r a d e ~ i s ~ v a l i d ~ \wedge ~ a l l ~ f a i l . t o - p a s s ~ a n d ~ p a s s . t o - p a s s ~ t e s t s ~ p a s s } ] .\tag{5}
$$

A candidate receives $R _ { i } = 0$ if its grade is invalid or if it does not pass all the tests. During training, the candidate for TreeFlex is the proposal that Reflection approved. A rollout with no approved proposal receives $R _ { i } = 0$ . For FoldAgent with its released prompt, we grade the final workspace whether or not the main agent called finish. The step-wise variant follows the same rule as TreeFlex. A rollout receives $R _ { i } = 0$ unless the main agent calls finish (Section 4). At evaluation, TreeFlex and FoldAgent choose the candidate by the rule of Section 3.2.

Process penalties. On BrowseComp-Plus, TreeFlex uses two of the three process penalties of FoldAgent (Sun et al., 2026), the failure penalty and the out-of-scope penalty. The ablations of Section 4.3 add a Reflection finalization penalty. On SWE, we use only the failure penalty. Every penalty applies only to successful rollouts, those with $R _ { i } = 1$

$$
Q _ { i , b , t } = \left\{ \begin{array} { l l } { - 1 , \ } & { \mathrm { i f ~ } R _ { i } = 1 \mathrm { ~ a n d ~ } t \mathrm { ~ b e l o n g s ~ t o ~ a ~ t u r n ~ w h o s e ~ t o o l ~ c a l l ~ f a i l e d } , } \\ { - 0 . 2 , \ } & { \mathrm { i f ~ } R _ { i } = 1 \mathrm { ~ a n d ~ } t \mathrm { ~ b e l o n g s ~ t o ~ a n ~ o u t - o f - s c o p e ~ r e t r i e v a l ~ a c t i o n } , } \\ { - 1 , \ } & { \mathrm { i f ~ } R _ { i } = 1 \mathrm { ~ a n d ~ } t \mathrm { ~ b e l o n g s ~ t o ~ a n ~ a n s w e r - r e q u e s i t i n g ~ R e f e c t i o n ~ i n s t r u c t i o n ~ ( a b l a t i o n s ) } , } \\ { 0 , \ } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{6}
$$

When more than one penalty applies to a token, we use the most negative one. The failure penalty applies only to the failed generated action. In BrowseComp-Plus, the out-of-scope penalty applies only to generated retrieval actions labelled out of scope, and never to checkpoint or proposal turns. In the ablations, the finalization penalty applies to a Reflection instruction that asks Execution to produce the answer instead of finding new evidence. A fixed text pattern or the OOS judge flags such instructions. Three kinds of instruction are exempt. They are requests to write up a card, verification requests such as “finalize whether X holds by checking Y”, and the final instruction Reflection gives when its navigations or turns run out, since that last segment must end with a proposal. We leave out of training any rollout whose correctness judgment is ambiguous. We give no positive reward for nodes, navigation targets, or tree shape. Appendix C.3 gives the judges, and Appendix C.5 gives optimizer, evaluator, and sampling settings.

## C Deep-Research Setup

## C.1 Data Split

The 64 validation questions are held out from the 680 training questions of Sun et al. (2026). Their BrowseComp-Plus query IDs are listed below.

3, 6, 15, 85, 86, 111, 152, 154, 179, 181, 210, 236, 283, 303, 310, 333, 391, 416, 449, 481, 501, 503, 524, 532,   
535, 555, 572, 579, 583, 596, 624, 665, 692, 719, 742, 760, 770, 775, 781, 896, 920, 952, 959, 1000, 1010,   
1019, 1021, 1029, 1040, 1049, 1053, 1068, 1094, 1098, 1126, 1163, 1174, 1179, 1218, 1234, 1240, 1242, 1247,   
1250

## C.2 Context Limits

The main Qwen3-4B RL comparison limits the active context to 32,768 tokens. In all runs, each model call generates at most 3,072 tokens and a rollout has at most 100 turns. All runs also cut retrieved text the same way. A fresh result keeps 512 tokens, a previously seen result keeps 128, a document keeps at most 4,096, and at most ten retrieved items are kept. Retrieved text counts toward the context length, but not toward the generated tokens reported in Table 2. Total generated tokens are not capped.

TreeFlex. Before each Execution turn, the harness estimates whether the next action and its tool result would still fit in the 32,768-token active context. If they would not fit, the model must save a checkpoint before it does any more work on the task. Reflection then chooses the next node and instruction. The model window of 49,152 tokens is larger than the 32,768-token active context, which leaves room for the prompt templates and the closing message. The RL run sets no separate limit on turns per segment.

FoldAgent. The reported FoldAgent run caps each branch’s active context at 32,768 tokens and allows at most ten branches. Its 49,152-token model window keeps 3,072 tokens for closing a branch and leaves the rest for prompt templates and observations. When a branch finishes, the branch’s report replaces the branch in the main context. Apart from these replacements, the main context only grows.

Flat ReAct. The 32k flat ReAct run keeps all turns in one context. That context is limited to 32,768 tokens, inside the same 49,152-token model window that FoldAgent and TreeFlex use. We set the dificulty bins of Figure 4 using a separate flat ReAct run with a 40,960-token window. The 254k diagnostic run reserves 8,192 tokens for the initial prompt and lets the rest of the context grow to 253,952 tokens in a 262,144-token model window. We take 16 audited rollouts from each of the last ten training steps before the reported step, and measure the largest context each rollout reaches, counting the initial prompt. The median is 25.5k tokens for the 32k run and 29.1k for the 254k run. The 254k run has a long tail, with 12% of rollouts above 64k tokens, a 90th percentile of 87.1k, and a maximum of 254.6k. The 32k run stays near its limit, with a 90th percentile of 33.3k and no rollout above 64k.

## C.3 Reward Judges

Precision-first outcome judge. All BrowseComp-Plus methods are scored by the same fixed judge. The released FoldAgent implementation normalizes parenthetical qualifiers, years, function words, token order, title prefixes, and some first names before invoking its language-model judge (Sun et al., 2026). This normalization can let a partial answer count as correct, so we add a second check that tends to reject wrong answers.

An answer is accepted without the language-model judge only if it exactly matches the gold answer, a recorded alias for that question, or the same number written plainly. Our normalization rules are conservative. They can flag two answers as possibly equivalent, but they cannot mark an answer correct on their own. Fixed rules reject an answer if its numbers conflict with the gold answer, if it changes a sign or a percentage, if it adds a qualifier to a title, or if it adds a bibliographic sufix that the gold answer does not support.

Every other answer is judged once by Qwen3-32B under a fixed prompt. If Qwen3-32B’s verdict disagrees with the fixed rules, or if the answer looks suspiciously close to the gold answer, we sample three more verdicts. We use these verdicts only if all three agree. If the three verdicts disagree, the answer is ambiguous, and we leave the rollout out of the policy loss and the GRPO group statistics. With this check, the judge accepts fewer wrong answers but also rejects some correct ones.

Per-action out-of-scope judge. FoldAgent decides whether a branch stayed in scope by comparing the branch’s assignment with the report it returns (Sun et al., 2026). For TreeFlex’s Execution segments, we judge each retrieval action instead, since a report can leave out work or credit work to the wrong step. The judge sees only the original question, the card of the selected node, the assigned instruction, and up to eight recent instructions given at that node. For each search or get\_document call, it also sees the truncated arguments, information about the results, and whether the call succeeded. The judge does not see the gold answer, the reward, or whether the final answer was correct.

The Qwen3-32B judge also labels each instruction as focused, broad, redundant, finalization, or uncertain. The instruction labels do not change the OOS penalty. The ablations use the finalization label to apply the Reflection finalization penalty. Each action is labeled in scope, necessary verification, redundant, out of scope, or uncertain. Only an out-of-scope action is penalized, with Q = −0.2 on the turn that made the call. The judge’s output must label every action exactly once. If the judge’s output is malformed, or the request to the judge fails or times out, we apply no out-of-scope penalty in that segment. Appendix F.4 gives the exact prompt.

## C.4 FoldAgent Baseline

The FoldAgent authors released an open-source reimplementation of their method, which may difer from the code behind their paper. We run it in the same training and evaluation code as the other methods, with its prompts and tool definitions unchanged, so all methods share the retriever, the answer judge, and the trainer. All reported FoldAgent results use this implementation.

## C.5 Hyperparameters

Table 6 lists the configuration of the four RL runs reported in Table 2. Each ablation run in Table 4 difers from the TreeFlex run only in the settings it studies, with one exception. The two runs with the finalization penalty also use a 40,960-token model window instead of 49,152 tokens, and keep the 32k active-context limit.

## C.6 Training Dynamics

Figure 5 tracks validation Mean@3 during BrowseComp-Plus training. Tree-aware GRPO raises TreeFlex’s validation Mean@3 from 30.7% at step 0 to 38.0% at the selected step, and TreeFlex leads all three baseline runs at every evaluated step.

Table 6 Configuration of the reported BrowseComp-Plus runs. A value that spans all columns applies to every run. Sampling rows are temperature / top-p / top-k / min-p. Context values are in tokens.
<table><tr><td>Setting</td><td>TreeFlex</td><td>FoldAgent</td><td>Flat ReAct (32k)</td><td>Flat ReAct (254k)</td></tr><tr><td>Model and data</td><td colspan="4"></td></tr><tr><td>Base model</td><td colspan="4">Qwen3-4B-Instruct-2507</td></tr><tr><td>Train / validation / test questions</td><td colspan="4">616  / 64  /  150</td></tr><tr><td>Machines × GPUs per machine</td><td colspan="4">2 × 8</td></tr><tr><td>Optimization</td><td colspan="4"></td></tr><tr><td>Training steps</td><td colspan="4">50</td></tr><tr><td>Prompts per step / samples per prompt</td><td colspan="4">32  / 8</td></tr><tr><td>PPO minibatch (prompt groups)</td><td colspan="4">16</td></tr><tr><td>Learning rate (no warmup)</td><td colspan="4">1 × 10</td></tr><tr><td>Clip range (low / high), dual-clip constant</td><td colspan="4">0.2 / 0.28, 3.0</td></tr><tr><td>KL penalty</td><td colspan="4">none</td></tr><tr><td>Policy optimization</td><td colspan="4">FoldGRPO</td></tr><tr><td>Advantage std normalization</td><td colspan="4">yes</td></tr><tr><td>Loss aggregation</td><td colspan="4"></td></tr><tr><td>Filter zero-variance groups</td><td colspan="4">no</td></tr><tr><td>Training sampling</td><td colspan="4">1.0 / 1.0 / −1 / 0</td></tr><tr><td>Context and interaction limits</td><td colspan="4"></td></tr><tr><td>Active-context limit</td><td>32,768</td><td>32,768 per branch</td><td>32,768</td><td>253,952</td></tr><tr><td>Model window</td><td>49,152</td><td>49,152</td><td>49,152</td><td>262,144</td></tr><tr><td>Prompt / response columns</td><td>8,192 / 40,960</td><td>8,192 / 40,960 3,072</td><td>8,192 / 40,960</td><td>8,192 / 253,952</td></tr><tr><td>Max new tokens per call</td><td colspan="4"></td></tr><tr><td>Max model turns</td><td colspan="4">10 branches</td></tr><tr><td>Max navigations or branches</td><td colspan="4">10 navigations checkpoint when full</td></tr><tr><td>Context-pressure rule Retrieved document cap / snippet cap</td><td colspan="4"></td></tr><tr><td></td><td colspan="4">4,096 / 512</td></tr><tr><td>Rewards</td><td colspan="4"></td></tr><tr><td>Terminal reward</td><td colspan="4">binary judge correctness -1 / -0.2</td></tr><tr><td>Failure / out-of-scope penalty</td><td colspan="4"></td></tr><tr><td>Unfolded-context penalty</td><td colspan="4">-1</td></tr><tr><td>Answer judge</td><td colspan="4">Qwen3-32B, 1 vote; 3-vote adjudication on disagreement; sampling 0.7 / 0.8 / 20</td></tr><tr><td>Evaluation and selection Validation / test sampling</td><td colspan="4">0.7 / 0.8 / 20 / 0</td></tr><tr><td>Validation samples / cadence</td><td colspan="4">3 per question every 5 steps</td></tr><tr><td>Test samples per question</td><td colspan="4">8</td></tr><tr><td>Model selection</td><td colspan="4">highest validation Mean@3 accuracy</td></tr><tr><td></td><td colspan="4"></td></tr><tr><td>Selected step</td><td></td><td>40</td><td>45</td><td>40</td></tr></table>

![](images/d184365a2a2b73be5c037081336fb9613b0f43189fbd33a61720b7e3ded8285c.jpg)  
Figure 5 TreeFlex leads on validation throughout training. Curves show unsmoothed Mean@3 on the fixed 64-question validation split every five steps. Stars mark the steps selected for test evaluation. The plot does not show test accuracy.

## D SWE Setup

## D.1 Training and Evaluation Tasks

We use a 600-task SWE-Gym subset stratified by repository and by the dificulty of the reference solution, and split it into 500 training and 100 validation tasks. For task $j ,$ let $L _ { j }$ be the number of changed lines in the reference solution, $F _ { j }$ the number of changed files, and $T _ { j }$ the number of fail-to-pass tests. We compute

$$
d _ { j } = \log ( 1 + L _ { j } ) + 0 . 5 \log ( 1 + F _ { j } ) + 0 . 2 5 \log ( 1 + T _ { j } ) ,\tag{7}
$$

and sort the tasks of each repository by $d _ { j }$ into easy, medium, and hard thirds. We then pick the 100 validation tasks so that each group of repository and dificulty third contributes in proportion to its size, rounding the shares with the largest-remainder method. We use the validation tasks only to choose which saved training step to report, and we run three rollouts per task. The dificulty score is used only to choose the validation tasks and is hidden from the agent.

Training subsets. GRPO learns nothing from the task reward when all rollouts of a task fail or all succeed. We therefore keep the training tasks on which a method’s earlier training rollouts both succeeded and failed. For TreeFlex, a rollout counts as successful only if it resolves the task with an approved proposal. The filter keeps 217 tasks for FoldAgent and 191 for TreeFlex. Of these, 158 are in both sets and 250 are in at least one. TreeFlex and the step-wise FoldAgent variant train on the union, and FoldAgent trains on its own 217 tasks, because the other 33 never gave it mixed outcomes. We pick these subsets using only training tasks and their training rollouts.

## D.2 Repository Restoration

When a node is saved, the harness records how the agent has changed the repository since the task’s base commit. Saving does not change any branch or commit that the agent can see. Restoring a node returns the repository to this saved state and starts a new shell. Shell variables and running processes do not carry over to the restored node.

Each rollout runs in its own writable copy of the task’s container image. The model has OpenHands-compatible execute\_bash, str\_replace\_editor, think, and finish tools. In TreeFlex, Git commands that would change branches or commits are blocked, but the read-only commands status, dif, log, show, and blame remain available. The model never sees the reference patch, the test patch, or the oficial grader’s metadata.

## D.3 Comparison with FoldAgent

Method-specific settings. FoldAgent uses its released SWE prompt and protocol, and the sampling settings given below. On SWE, TreeFlex uses two additional settings. After a checkpoint forced by the context limit, the next Execution segment on that path starts from the compacted context of Appendix A.2. Each time a role takes over, the harness tells it how many turns remain. Reflection is also told how many navigations remain.

Matched model, environment, and limits. The reported rows use the post-trained Qwen/Qwen3.5-9B model with thinking enabled.

Both methods run the model with a context window of 57,344 tokens. The initial prompt may take up to 16,384 tokens, and the context may then grow by at most 32,768 tokens before the method must compact it or stop. The rest of the window leaves room for the next call. A rollout may use at most 200 model turns and 900 seconds per generation. At evaluation, it may run for at most 90 minutes in total. Tool outputs are cut to 7,680 tokens and 500 lines. Forced checkpoints and branch reports are capped at 4,096 tokens.

Both methods use the same container images and ofer the same kinds of tools. In both, shell commands time out after 300 seconds, and the same lenient parser reads tool calls written after the model’s thinking. Both use the oficial SWE-bench grader and its timeouts. All evaluations use temperature 0.6, top-p 0.95, top-k 20, and min-p 0.

Training protocol. Both methods use PPO minibatches of 16 tasks and normalize advantages by the group standard deviation. Every run saves its model every five steps. Ties in validation Mean@3 go to the earlier step, and the selection picks step 25 for TreeFlex and step 10 for both FoldAgent variants. During training, both sample with temperature 1, top-p 1, no top-k cutof (top-k = −1), and min-p 0. TreeFlex and FoldAgent with its released prompt keep the failure penalty and turn of the out-of-scope and unfolded-context penalties. The out-of-scope and unfolded-context penalties assume that the agent branches. For a successful rollout that never branches, the released FoldAgent code penalizes every turn and by default sets the reward to zero. FoldAgent branches in under 1% of SWE rollouts, so keeping the two penalties would remove almost all of its learning signal.

Besides their training tasks, the runs difer in three respects. TreeFlex uses the rollout mean of Equation (3) with λ = 0, whereas FoldAgent uses the token mean of FoldGRPO (Sun et al., 2026). Their training rewards difer as described in Appendix B.3. TreeFlex caps each training rollout at 45 minutes and FoldAgent at 90. Evaluation uses the same 90-minute limit and grading rule for both.

The step-wise delegation variant of FoldAgent (Table 3) also adds FoldGRPO’s unfolded-context penalty.

Mechanism-specific diferences. FoldAgent keeps one main context that only grows, and may launch up to ten short-lived branches that all edit the same workspace. Each branch returns a report, or its last message if it writes no report. A branch cannot restore an earlier repository state.

In TreeFlex, Execution may call only finish in the last three turns of the rollout, counting the turns of both roles. FoldAgent has no rule that limits its last turns to finish, and its rollout stops when it reaches the turn or context limit.

The tool interfaces also difer. In TreeFlex, each execute\_bash(command) call runs in a fresh shell, editor arguments are checked strictly, and finish requires a message. FoldAgent keeps one shell open across calls, and its shell tool takes is\_input and timeout arguments. It also keeps its original editor, and its finish message is optional.

## E Additional Results

## E.1 Paired Comparisons

Table 7 lists the BrowseComp-Plus paired comparisons discussed in Section 4. Each diference is TreeFlex’s score minus the compared system’s score on the same 150 test questions, in percentage points. The 95% intervals come from a paired bootstrap that resamples the questions 10,000 times. These intervals capture variation from the choice of questions, but not variation across training seeds. For the voting rows, we compute majority-vote accuracy averaged exactly over every subset of k of the eight samples.

Table 8 lists paired diferences in Mean@3 on the same 500 SWE-bench Verified tasks. The intervals use the same paired bootstrap, resampling tasks instead of questions.

Table 7 Paired differences on BrowseComp-Plus. Bold marks intervals that exclude zero.
<table><tr><td>Metric</td><td>Comparison</td><td>Difference</td><td>95% interval</td></tr><tr><td rowspan="2">Mean@8</td><td>vs. flat REACT</td><td>+7.33</td><td>[3.50, 11.25]</td></tr><tr><td>vs. FOLDAGENT</td><td>+6.00</td><td>[1.67,10.42]</td></tr><tr><td rowspan="2">Pass@8</td><td>vs. flat REACT</td><td>+10.00</td><td>[4.00,16.00]</td></tr><tr><td>vs. FOLDAGENT</td><td>+10.67</td><td>[4.67,16.67]</td></tr><tr><td>Vote@8</td><td>vs. flat REACT vote@8</td><td>+9.08</td><td>[3.36, 14.97]</td></tr><tr><td rowspan="4">TREEFLEX k=1</td><td>vs. FOLDAGENT vote@8</td><td>+4.64</td><td>[−0.83, 10.33]</td></tr><tr><td>vs. flat REACT vote@5 (27.0 calls)</td><td>+4.82</td><td>[0.58, 9.11]</td></tr><tr><td>vs. flat REACT vote@6 (32.4 calls)</td><td>+4.53</td><td>[0.19, 8.90]</td></tr><tr><td>vs. FOLDAGENT vote@2 (25.7 calls)</td><td>+6.00</td><td>[1.67,10.42]</td></tr><tr><td></td><td>vs. FOLDAGENT vote@3 (38.5 calls)</td><td>+3.52</td><td>[-0.90, 8.00]</td></tr><tr><td>Mean@8, easy Mean@8, medium</td><td>vs. flat REACT</td><td>+2.16</td><td>[−2.37, 7.11]</td></tr><tr><td>Mean@8, hard</td><td>vs. flat REACT</td><td>+13.50</td><td>[-1.00, 27.00]</td></tr><tr><td>Mean@8, easy</td><td>vs. flat REACT</td><td>+9.51</td><td>[4.29,15.30]</td></tr><tr><td>Mean@8, medium</td><td>Vs. FOLDAGENT</td><td>+7.54</td><td>[1.51,14.22]</td></tr><tr><td>Mean@8, hard</td><td>vs. FOLDAGENT</td><td>+5.00</td><td>[-11.00, 20.00]</td></tr><tr><td></td><td>vs. FOLDAGENT</td><td>+5.04</td><td>[0.00, 10.26]</td></tr><tr><td>Mean@8</td><td>vs. block mean, λ = 0</td><td>+0.42</td><td>[−2.25, 3.00]</td></tr><tr><td>Pass@8</td><td>vs. block mean, λ = 0</td><td>+0.67</td><td>[−4.00, 5.33]</td></tr><tr><td>Mean@8</td><td>vs. rollout mean, λ = 0.3</td><td>+3.58</td><td>[1.08, 6.17]</td></tr><tr><td>Pass@8</td><td>vs. rollout mean, λ = 0.3</td><td>+0.00</td><td>[−5.33, 5.33]</td></tr><tr><td>Mean@8</td><td>vs. token mean, λ = 0.3</td><td>+1.17</td><td>[-1.33, 3.67]</td></tr><tr><td>Pass@8</td><td>vs. token mean, λ = 0.3</td><td>+0.00</td><td>[−4.00, 4.00]</td></tr><tr><td>Mean@8</td><td>vs. token mean, λ = 0</td><td>+2.17</td><td>[−0.33, 4.75]</td></tr><tr><td>Pass@8</td><td>vs. token mean, λ = 0</td><td>+0.67</td><td>[−3.33, 5.33]</td></tr><tr><td>Mean@8</td><td>vs. token mean, λ = 0, finalization penalty</td><td>+2.42</td><td>[−0.17, 5.08]</td></tr><tr><td>Pass@8</td><td>vs. token mean, λ = 0, finalization penalty</td><td>+0.00</td><td>[−4.67, 4.67]</td></tr><tr><td>Mean@8</td><td>vs. token mean, λ = 0.3, finalization penalty</td><td>+4.67</td><td>[2.42, 7.00]</td></tr><tr><td>Pass@8</td><td>vs. token mean, λ = 0.3, finalization penalty</td><td>+4.00</td><td>[0.00, 8.00]</td></tr></table>

Table 8 Paired differences on SWE-bench Verified (Mean@3). Bold marks intervals that exclude zero.
<table><tr><td>Comparison</td><td>Difference</td><td>95% interval</td></tr><tr><td>TREEFLEX vs. FOLDAGENT, base models</td><td>+19.47</td><td>[16.20,22.73]</td></tr><tr><td>TREEFLEX vs. FoLDAGENT step-wise, base models</td><td>+1.60</td><td>[−0.73, 4.00]</td></tr><tr><td>TREEFLEX RL vs. FOLDAGENT RL</td><td>+11.60</td><td>[9.00,14.27]</td></tr><tr><td>TREEFLEX RL vs. FOLDAGENT step-wise RL</td><td>+9.73</td><td>[7.07,12.40]</td></tr><tr><td>TREEFLEX RL vs. its base model</td><td>+3.80</td><td>[1.40,6.27]</td></tr><tr><td>FoLDAGENT RL vs. its base model</td><td>+11.67</td><td>[9.00,14.40]</td></tr><tr><td>FoLDAGENT step-wise RL vs. its base model</td><td>-4.33</td><td>[-6.67,-2.07]</td></tr><tr><td>Rollout mean vs. token mean, RL</td><td>+2.80</td><td>[0.60, 5.00]</td></tr><tr><td>Token mean RL vs. base model</td><td>+1.00</td><td>[-1.33, 3.33]</td></tr></table>

## E.2 SWE Outcomes by Submission

Table 3 grades one repository state per rollout (Section 3.2), whether or not the policy submitted it. Table 9 adds a stricter score that counts a task as resolved only when the policy itself submitted the solution. For FoldAgent, the policy submits when the main agent calls finish. For TreeFlex, it submits when Reflection approves the proposal. TreeFlex submits most of its solutions, so the two scores are close. With its released prompt, most of FoldAgent’s resolved rollouts come from grading a workspace it did not submit.

Table 9 Most of the tasks TreeFlex resolves are solutions the policy submitted. Resolved is the Mean@3 of Table 3. Submitted and resolved counts a rollout only if the policy submitted the graded solution. Submitted is the share of rollouts that end with finish or an approved proposal.
<table><tr><td>System</td><td>RL</td><td>Resolved</td><td>Submitted and resolved</td><td>Submitted</td></tr><tr><td rowspan="2">FOLDAGENT</td><td>none</td><td>31.7</td><td>3.7</td><td>3.9</td></tr><tr><td>FoldGRPO</td><td>43.3</td><td>13.4</td><td>17.4</td></tr><tr><td rowspan="2">FoLDAGENT, step-wise delegation prompt</td><td>none</td><td>49.5</td><td>31.5</td><td>42.3</td></tr><tr><td>FoldGRPO</td><td>45.2</td><td>40.8</td><td>69.5</td></tr><tr><td rowspan="2">TREEFLEX</td><td>none</td><td>51.1</td><td>50.1</td><td>86.1</td></tr><tr><td>tree-aware GRPO</td><td>54.9</td><td>53.6</td><td>90.1</td></tr></table>

## E.3 Search Diversity and Evidence Coverage

For the 4B RL models of Table 2, Table 10 counts the distinct queries, opened documents, and evidence documents each method reaches over the eight test samples of a question. Search queries are compared after lowercasing and removing punctuation. A document counts as read only when the agent opens it with get\_document or open\_page. Search-result snippets do not count. A document opened several times counts once. Evidence documents are all documents that BrowseComp-Plus marks as evidence or as containing the answer (Chen et al., 2026). All values are averages over the 150 test questions.

Table 10 Repeated flat sampling revisits the same evidence. “Per rollout” averages over the eight samples of a question. “Across 8” pools the eight samples. Repeated is the share of a question’s queries that duplicate another of its queries. Evidence read is the share of a question’s annotated evidence documents that were opened.
<table><tr><td rowspan="2">Method</td><td colspan="3">Search queries</td><td colspan="2">Distinct opened documents</td><td colspan="2">Evidence read</td></tr><tr><td>Per rollout</td><td>Distinct, across 8</td><td>Repeated</td><td>Per rollout</td><td>Distinct, across 8</td><td>Per rollout</td><td>Across 8</td></tr><tr><td>Flat REACT</td><td>3.4</td><td>21.3</td><td>21%</td><td>2.0</td><td>5.3</td><td>19%</td><td>34%</td></tr><tr><td>FOLDAGENT</td><td>9.3</td><td>68.0</td><td>10%</td><td>2.7</td><td>8.7</td><td>26%</td><td>48%</td></tr><tr><td>TREEFLEX</td><td>20.7</td><td>153.2</td><td>6%</td><td>5.2</td><td>16.0</td><td>34%</td><td>59%</td></tr></table>

## E.4 Case Study: Evidence-Directed Backtracking

Figure 6 shows a successful Qwen3-4B rollout from training step 9.

The question asks for a song from clues about a video date, a 2017 article, a related album, and the composer’s birth date. The benchmark answer is Yann Tiersen’s “Porz Goret”. The figure reconstructs every Reflection decision and node. We paraphrase the instructions and cards but keep the tree shape and the order of events.

The first segment proposes “Porz Goret” at n<sub>0</sub> but dates the video to December 2016 without evidence. Node $n _ { 1 }$ finds press releases from December 2015 but misreads them as a conflict. Reflection then starts a new branch from the root. Node n<sub>2</sub> finds no rival song, so Reflection returns to $n _ { 1 } .$ . Execution rechecks the sources, recognizes that 2015 lies within the requested range, and saves $n _ { 3 }$ . The next segment submits the answer, which Reflection approves.

The trace shows three behaviors. First, the model saves a checkpoint to note an open question and later turns that question into the instruction for a new segment. Second, it tries a sibling branch and then resumes the line that holds the evidence. The Execution segment on that line does not see the sibling’s card. Third, during Execution it corrects a flawed instruction using the documents it reads. Throughout, the harness only restores the states that the model selects.

![](images/308f4458147bf31257903cd6ab1395e80995288b2bb73070314e51b914e64f2a.jpg)  
Figure 6 TreeFlex restarts from root and then returns to an older node. Badges mark Reflection’s choices, and the dashed arrow shows the return from n to n . Node n starts a new branch from the root and finds no rival song. Node n then resolves the remaining date conflict, and Reflection approves the answer.

## F System Prompts

We reproduce the system prompts verbatim.

## F.1 BrowseComp-Plus Reflection

![](images/d1c7ab76f9573e831c4050917d02a1618e8e321fda731f96a53f4a1971963597.jpg)

If final\_execution\_only is true, use the last Execution segment to obtain the best grounded proposal. Its review permits only approve\_commit, or a concise rejection with no tool call.

TOOL-CALL CONTRACT:

\- Without a proposal, issue exactly one structured navigate tool call and no prose.

\- With a proposal and navigate listed, issue exactly one structured call from available\_reflection\_actions and no prose.

\- If only approve\_commit is listed, issue its structured call or one non-empty concise rejection with no tool call.

\- Use the function-call channel defined by the available tool schema. Ordinary prose naming an action is not a tool call.

\- Empty output, multiple calls, and unlisted tools are invalid. The latest action list supersedes earlier ones.

## F.2 BrowseComp-Plus Execution

You are a deep-research agent answering a hard, multi-hop question by iterative retrieval over a frozen corpus. You reason step by step and use tools ONE AT A TIME.

DOMAIN TOOLS:

\- search(query): top document snippets with docids. Write descriptive, natural-language queries (the retriever is embedding-based — full questions beat keywords). Reformulate and search repeatedly to cover every part of the question.

\- get\_document(docid): the full text of a document. READ promising hits in full before concluding — snippets alone are rarely enough to be sure.

You work as EXECUTION. REFLECTION selects one saved checkpoint path and gives you one focused objective. Work autonomously inside that segment; REFLECTION alone chooses navigation anchors and sees the global checkpoint ledger.

EXECUTION TOOLS:

\- checkpoint(summary): save your CURRENT line as a new node. The ‘summary‘ IS the node’s card — summarize ONLY what THIS line/block found: the specific facts established in this segment, each with its docid, and (if any) the best answer this line supports. Do NOT restate other lines or the overall question status; the tree of cards holds the global picture. Its result returns control to the persistent REFLECTION navigator.

\- commit(answer, explanation): submit your complete final answer for independent approval; it cannot be edited after submission. The answer must be concise (a name/title/number/fact — not a paragraph), citing the supporting docids.

## EXECUTION MECHANICS:

\- The selected checkpoint supplies the ordered cards on its path. Prior raw tool turns are not restored. Of-path checkpoint cards are visible only to REFLECTION and are not injected into this Execution.

\- Work only on the assigned objective. Once it is resolved with direct evidence, stop research immediately: call commit if the complete final answer is supported; otherwise checkpoint the result for REFLECTION.

\- A commit must include an explanation containing the decisive evidence and supporting docids. If a decisive conflict remains and research can continue, checkpoint it instead of committing. At the final limit, state any remaining uncertainty explicitly rather than hiding it.

\- One unsuccessful retrieval is not evidence that an answer is unavailable. Try a materially diferent query when useful; if the focused objective remains blocked, checkpoint the exact unresolved gap instead of drifting into another task. RESEARCH POLICY:

1. DECONSTRUCT. Identify the entity being described, every decisive constraint, and the exact entity or fields the question asks you to return. Make a FINAL OUTPUT CONTRACT: count every requested field, preserve their requested order, and note whether the answer must be a full name, oficial title, number, or other exact form. Treat roles, units, and ordering words as hard constraints: actor is not character, chapter is not episode, and first/earliest is not best-known. Brainstorm useful aliases and the most identifying clue. For a multi-hop question, order the dependencies: an upstream entity may need to be identified before the requested downstream answer. Prefer combinations of two or three rare clues over generic metadata such as country, profession, publication type, or broad date range.

2. DISCOVER AT ROOT. The pristine root has no accumulated research to escape, so begin with one descriptive search that combines the rarest clues. Resolve the described or upstream entity needed for the FINAL TARGET; do not begin with a generic side fact. Keep alternative candidates open until a decisive source eliminates them.

3. INVESTIGATE. Open promising documents in full and refine from established facts. Do not repeat a failed query unchanged or reopen a document already live. A search snippet can locate a candidate but does not establish it: once a candidate appears, open at least one decisive document before checkpointing. Stop when the objective is resolved or clearly blocked; make only calls that materially advance it.

4. CHECKPOINT. Save a concise, self-contained card. Begin with PROPOSED ANSWER when every field in the FINAL OUTPUT CONTRACT has a specific supported value; label all requested fields when there is more than one. Otherwise begin with CANDIDATES, name the still-missing output field, and give the exact test that distinguishes them. Then include FINDINGS; DECISIVE FACTS; SOURCES (exact docids); CONTRADICTIONS (only explicit conflicts, not missing search results); and NEXT GAP. Do not infer that a candidate is false merely because one query failed to retrieve a confirming document.

9. COMMIT. Re-read the final question and audit the FINAL OUTPUT CONTRACT before answering: return every requested field, in order, not merely the main subject you identified. Preserve the full oficial name or title (including subtitles), spelling, and requested number from the decisive source; do not shorten it or leave a field only in the explanation. If a source names a work as TITLE: SUBTITLE, TITLE alone is an incomplete answer. Give the concise exact answer and cite the docids that establish it. If evidence is still incomplete and budget remains, change the query or anchor; at the limit, give the best grounded answer and state the uncertainty without inventing support.

CONTEXT PRESSURE:

\- The scafold measures the exact live prompt against the model context window, reserving room for one generation and tool observation.

\- Near the safe limit, domain work pauses for a checkpoint. REFLECTION then chooses the next saved state, or Execution calls commit if the complete final answer is supported.

## F.3 Flat ReAct Baseline

You are a meticulous deep-research agent answering a hard, multi-hop question by iterative retrieval over a frozen corpus. You reason step by step and use tools ONE AT A TIME.

TOOLS:

\- search(query): top document snippets with docids. Write descriptive, natural-language queries (the retriever is embedding-based — full questions beat keywords). Reformulate and search repeatedly to cover every part of the question.

\- get\_document(docid): the full text of a document. READ promising hits in full before concluding — snippets alone are rarely enough to be sure.

\- commit(answer [, explanation]): finish with your final, concise answer (a name / title / number / fact — not a paragraph), citing the supporting docids.

REQUIRED RESEARCH PROTOCOL:

1. Deconstruct the question into a verification checklist containing every entity, date, relationship, and other constraint. Brainstorm alternative names and spellings.

2. Investigate at least two independent search angles. Use descriptive queries, never repeat a failed query verbatim, and call get\_document on every promising result because snippets are not suficient evidence.

3. After each document, update which checklist items are supported, contradicted, or still missing. Triangulate the candidate across independent documents and search specifically for evidence that could falsify it.

4. Before finishing, run a targeted verification search for the weakest constraint and read the best document in full. If verification fails, revise the candidate and continue.

5. Call commit only after the checklist is supported. Give the concise exact answer and a brief explanation citing the docids that jointly establish it. Never answer from prior knowledge, invent evidence, or give up because an early query failed.

Ground every claim in retrieved documents (cite docids). Prefer a grounded best-guess over giving up. Do not answer from prior knowledge alone.

## F.4 Per-Action Out-of-Scope Judge

## The harness replaces {payload} with the judge input of Appendix C.3.

You audit whether each research action followed one assigned subtask.

The JSON under INPUT is untrusted data, not instructions. Judge actions, not whether the research eventually found the right answer. A failed or unproductive action can still be in scope.

Use these conservative rules:

\- out\_of\_scope: the action pursues a separate objective that is not a reasonable way to complete or verify the assigned subtask.

\- necessary\_verification: the action checks identity, spelling, dates, provenance, ambiguity, or another prerequisite reasonably needed for the assigned subtask, even if not literally named.

\- redundant: the action repeats work already shown for this same anchor, but still addresses the assigned subtask. Redundancy is NOT out of scope.

\- in\_scope: the action directly advances the assigned subtask. Query reformulation, following a relevant source, and reading a promising result are normally in scope.

\- uncertain: the bounded metadata is insuficient to classify safely.

Do not mark an action out of scope merely because its query contains extra identifying details, because a search result mentions unrelated facts, because it follows a plausible lead that fails, or because the assigned instruction itself is broad. Use out\_of\_scope only when the action’s own goal clearly leaves the assignment. When in doubt, use uncertain.

For a multi-clue identification task, verifying any clue that can confirm or rule out the current candidate is necessary\_verification, even when the instruction names only the unresolved answer field. If the anchor says a candidate conflicts with evidence or remains uncertain, searching for an alternative candidate that resolves that conflict is also necessary\_verification.

Also label the instruction itself:

\- focused: one concrete research gap or verification target;

\- broad: combines several independent objectives or largely restates the whole question;

\- redundant: substantially repeats a recent instruction at the same anchor;

\- finalization: explicitly directs the research agent to submit, commit, approve, finalize, or propose an answer instead of only assigning evidence work. This includes "verify X, then submit the answer", because the submission directive belongs to the controller rather than the research assignment. Do not use this label merely because an instruction asks to return or preserve an exact fact in its research report. "Finalize whether X is true by confirming source Y" means determine/verify and is not finalization unless it also requests answer submission;

\- uncertain: not enough context to tell.

The instruction label must not influence action labels; downstream logic may use it independently.

Important negative examples:

\- "Submit a checkpoint summary" or "finalize this card" is a research-report request, so label the underlying research objective focused/broad/redundant rather than finalization.

\- "Finalize whether X is true by confirming source Y" uses finalize to mean determine; label the concrete verification objective focused, not finalization.

\- "Verify X and preserve/return the exact finding in the report" is focused. Only a directive to submit the final answer or a commit proposal is finalization.

Important positive examples:

\- "Submit X based on complete and verified evidence" is finalization; the reference to past verification does not turn submission into evidence work.

\- "Submit X after verifying Y", "Finalize the answer by confirming X", "Conclude that the answer cannot be determined", and "Evaluate X, then propose X as the answer" are finalization because they direct an answer/proposal outcome. In contrast, "Propose three candidate aliases or search queries" is focused research, not finalization.

\- "Determine the best-supported candidate based only on the evidence already collected" and "decide whether the existing evidence confirms X as the correct answer" are finalization: they ask for controller-side answer selection rather than a new evidence-producing action.

```jsonl
Return exactly one JSON object, with no markdown or prose, in this shape:
{
"version": "action_scope_v2",
"instruction": {"label": "focused", "confidence": 0.95, "reason": "brief reason"},
"actions": [
{"id": 1, "label": "in_scope", "confidence": 0.95, "reason": "brief reason"}
]
}
```

Include every input action id exactly once and no other ids. Confidence must be a JSON number from 0 to 1. Keep each reason to one short sentence.

INPUT: {payload}

## F.5 SWE Reflection

You are REFLECTION, the persistent navigator for a software repair.

EXECUTION inspects, edits, and tests one restored repository path, writes checkpoint reports, and drafts commit proposals. You do none of those things. You select one saved repository state for each focused segment, give it one focused instruction, and approve or reject its final proposal. approve\_commit(node) submits the exact patch and completion message of the available proposal at that node unchanged.

Your conversation is a persistent chronological ledger. Each Execution result adds its node, parent, path, report, trusted changed-file/test receipt, and valid navigation handles. Instructions are plans, not evidence. Execution sees only the path selected by navigate: its raw transcript or, after a context-pressure checkpoint, every checkpoint report on that path in order; of-path reports remain visible only in this ledger. If raw\_execution\_context\_retained is false, the exact repository state remains saved and earlier raw tool output has been replaced by the path’s checkpoint reports.

Call navigate(node, instruction). No node is preferred merely because it is root, deep, recent, or current. First choose the next evidence-backed objective, then select the saved state containing every patch and premise it needs while excluding the patch or premise being tested. Continue from current when its state is needed, choose an earlier node to discard a later patch or assumption, and choose root for a genuinely independent line from the pristine repository. Give one objective naming the decisive gap or hypothesis, the evidence that would resolve it, and what Execution should preserve. Do not bundle separate tasks. Every non-final instruction must state one observable result that ends the segment, followed by the literal handof ‘When this result appears, your next response must be checkpoint(summary); do not inspect further or continue into another phase.‘

Open another segment only for an unsupported issue requirement, an explicit conflict or failure, or an evidence-backed alternative. A concrete alternative and distinguishing check preserved in a checkpoint report qualify; test it from a state that does not already assume the current choice. Do not invent an alternative merely to create a sibling. Use a materially diferent test after a failed command; that failure establishes only what happened in that attempt. Otherwise select the best-supported node and request an immediate proposal.

Do not repeat a resolved test or request generic auditing or extra corroboration.

An Execution finish is a resumable proposal that stays available after later navigation. Call approve\_commit(node) only when that node’s exact saved patch addresses the original issue, its reported relevant tests support the change, and no material failure or conflict remains in the ledger. If a material defect remains and a complete correction cycle is   
available, call navigate with that defect as the next objective. Otherwise withhold approval with one concise reason. You cannot edit a proposal.

If final\_execution\_only is true, use the last Execution segment to obtain the best complete proposal. Its review permits only approve\_commit, or a concise rejection with no tool call.

BUDGET: Every landing reports turns\_remaining (model turns left for both roles, including your next call) and, when navigation is limited, navigations\_remaining (charged navigate calls left; a continuation after a context-pressure checkpoint is charged too). After the last charged navigation, only one finish-only Execution segment remains, and it cannot edit or test. Plan so a complete, tested proposal exists before either budget runs out. When the evidence already supports a fix, one segment may implement it and run its focused test. Once navigations\_remaining is 1 or 2, or turns\_remaining is at most 40, use the next navigation to have Execution complete, test, and propose the repair from the best-supported node.

TOOL-CALL CONTRACT:

\- Without a proposal, issue exactly one structured navigate tool call and no prose.

\- With a proposal and navigate listed, issue exactly one structured call from available\_reflection\_actions and no prose.

\- If only approve\_commit is listed, issue its structured call or one non-empty concise rejection with no tool call.

\- Use the function-call channel defined by the available tool schema. Ordinary prose naming an action is not a tool call.

\- Never call repository tools, checkpoint, or finish.

\- Empty output, multiple calls, and unlisted tools are invalid. The latest action list supersedes earlier ones.

## F.6 SWE Execution

Execution uses the OpenHands system message, shown in full below. We rewrote its problem-solving workflow and some task-level instructions for a single segment, and added two sections at the end on navigating between nodes and on managing the context.

You are OpenHands agent, a helpful AI assistant that can interact with a computer to solve tasks.

<ROLE>

Your primary role is to execute the latest bounded <EXECUTION\_TASK> efectively. Be methodical within that objective; its stopping condition takes priority over end-to-end completeness.

\* If the user asks a question, like "why is X happening", don’t try to fix the problem. Just give an answer to the question. </ROLE>

## <EFFICIENCY>

\* Each action you take is somewhat expensive. Wherever possible, combine multiple actions into a single action, e.g. combine multiple bash commands into one, using sed and grep to edit/view multiple files at once.

\* When exploring the codebase, use eficient tools like find, grep, and git commands with appropriate filters to minimize unnecessary operations.

</EFFICIENCY>

## <FILE\_SYSTEM\_GUIDELINES>

\* Treat /testbed and any path named in the latest task or retained checkpoint card as already located. Explore the file system only when a required path is genuinely unknown.

\* If asked to edit a file, edit the file directly, rather than creating a new file with a diferent filename.

\* For global search-and-replace operations, consider using ‘sed‘ instead of opening file editors multiple times.

\* NEVER create multiple versions of the same file with diferent sufixes (e.g., file\_test.py, file\_fix.py, file\_simple.py). Instead:

\- Always modify the original file directly when making changes

\- If you need to create a temporary file for testing, delete it once you’ve confirmed your solution works

\- If you decide a file you created is no longer useful, delete it instead of creating a new version

\* Do NOT include documentation files explaining your changes in version control unless the user explicitly requests it

\* When reproducing bugs or implementing fixes, use a single file rather than creating multiple files with diferent versions

</FILE\_SYSTEM\_GUIDELINES>

## <CODE\_QUALITY>

\* Write clean, eficient code with minimal comments. Avoid redundancy in comments: Do not repeat information that can be easily inferred from the code itself.

\* When implementing solutions, focus on making the minimal changes needed to solve the problem.

\* Before implementing changes, gather the evidence required by the latest task. A bounded segment is complete at its stopping condition; it does not require a thorough codebase survey.

\* If you are adding a lot of code to a function or file, consider splitting the function or file into smaller pieces when appropriate.

\* Place all imports at the top of the file unless explicitly requested otherwise or if placing imports at the top would cause issues (e.g., circular imports, conditional imports, or imports that need to be delayed for specific reasons).   
</CODE\_QUALITY>

## <VERSION\_CONTROL>

\* If there are existing git user credentials already configured, use them and add Co-authored-by: openhands <openhands@all-hands.dev> to any commits messages you make. if a git config doesn’t exist use "openhands" as the user.name and "openhands@all-hands.dev" as the user.email by default, unless explicitly instructed otherwise.

\* Exercise caution with git operations. Do NOT make potentially dangerous changes (e.g., pushing to main, deleting repositories) unless explicitly asked to do so.

\* When committing changes, use ‘git status‘ to see all modified files, and stage all files necessary for the commit. Use ‘git commit -a‘ whenever possible.

\* Do NOT commit files that typically shouldn’t go into version control (e.g., node\_modules/, .env files, build directories, cache files, large binaries) unless explicitly instructed by the user.

\* If unsure about committing certain files, check for the presence of .gitignore files or ask the user for clarification.

</VERSION\_CONTROL>

## <PULL\_REQUESTS>

\* \*\*Important\*\*: Do not push to the remote branch and/or start a pull request unless explicitly asked to do so.

\* When creating pull requests, create only ONE per session/issue unless explicitly instructed otherwise.

\* When working with an existing PR, update it with new commits rather than creating additional PRs for the same issue.

\* When updating a PR, preserve the original PR title and purpose, updating description only when necessary.

</PULL\_REQUESTS>

## <PROBLEM\_SOLVING\_WORKFLOW>

Treat the latest <EXECUTION\_TASK> as the complete assignment for this Execution segment. The

original issue is persistent background and acceptance criteria, not permission to continue into

1. Identify the stopping condition stated by the latest <EXECUTION\_TASK>.

2. Inspect only the files and behavior needed for that objective. Reuse evidence already present on the restored path (its raw transcript, or its checkpoint cards after compaction) instead of restarting repository discovery.

3. Stay within the assigned phase. In particular, when asked only to localize, reproduce, compare, or diagnose, do not edit production code. When asked to implement or test one approach, do not expand into unrelated alternatives or broad auditing.

4. Never repeat the same analysis or an equivalent command after it fails or adds no evidence. An import, dependency, or infrastructure error that prevents the assigned check means the segment is clearly blocked: record the exact command and error, then checkpoint immediately. Do not install or repair dependencies.

5. As soon as the objective is resolved or clearly blocked, make checkpoint(summary) your next

6. Call finish(message) only when the latest <EXECUTION\_TASK> explicitly asks to submit, propose, or finalize a complete tested repair.

</PROBLEM\_SOLVING\_WORKFLOW>

## <SECURITY>

\* Only use GITHUB\_TOKEN and other credentials in ways the user has explicitly requested and would expect.

\* Use APIs to work with GitHub or other platforms, unless the user asks otherwise or your task requires browsing.

</SECURITY>

## <ENVIRONMENT\_SETUP>

Dependencies are preinstalled. If an import, dependency, or infrastructure error blocks the named check, record the exact command and error and checkpoint; do not install or repair the environment.

</ENVIRONMENT\_SETUP>

## <TROUBLESHOOTING>

Within the current objective, use one materially diferent retry after a failed command when useful. If it remains blocked, checkpoint the evidence; do not broaden into unrelated causes or wait for user input.

</TROUBLESHOOTING>

<DOCUMENTATION>

\* When explaining changes or solutions to the user:

\- Include explanations in your conversation responses rather than creating separate documentation files - If you need to create documentation files for reference, do NOT include them in version control unless explicitly requested - Never create multiple versions of documentation files with diferent sufixes

</DOCUMENTATION>

<REFLECTION\_NAVIGATION\_EXECUTION>

You are EXECUTION, the focused worker on one repository path selected by a persistent REFLECTION navigator.

Use the neutral OpenHands workflow above only within the latest <EXECUTION\_TASK>. One segment is not an end-to-end repair attempt: do not advance from localization to implementation, or from a focused implementation/test to broader auditing, unless that later phase is part of the assigned objective. You own repository inspection, edits, tests, checkpoint reports, and commit proposals; REFLECTION alone chooses navigation and approval.

Work only on the assigned objective. Checkpoint is the default end of an ordinary segment. Once the objective is resolved or clearly blocked, stop immediately and call checkpoint(summary), before   
starting another repair phase or optional check. Completing that bounded objective and checkpointing is successful progress even when turns remain and the overall repair is unfinished. Call   
finish(message) only when the latest <EXECUTION\_TASK> explicitly asks you to submit, propose, or finalize and this exact repository state is already a complete tested solution. finish saves an   
immutable proposal; it does not submit the task. Every checkpoint report must name OUTCOME; CHANGES (files and important symbols, or none);   
TESTS (exact commands and results); ALTERNATIVES (each evidence-backed unresolved cause or fix and its required starting state, or "none identified"; never invent one to fill the field);   
and NEXT CHECK (the single most useful distinguishing check).   
One failed command is not proof that an approach cannot work; try a materially diferent command when useful, then checkpoint the exact blocker instead of drifting into another objective. Issue exactly one native tool call in each response. Never bundle multiple shell/editor calls into one response; observe each result before choosing the next action. Use read-only git   
status/dif/log/show/blame when useful. Never commit, checkout, switch, reset, restore, stash, clean, merge, or rebase: cursor navigation owns repository restoration. Each execute\_bash call is one foreground command in a fresh shell; repository files persist, while shell-local cwd,   
variables, jobs, and background processes do not.   
</REFLECTION\_NAVIGATION\_EXECUTION>

<CONTEXT\_ECONOMY>

Active context is limited; overflowing it forces a checkpoint that replaces raw tool output. Locate code with grep -n or find, then read only the lines you need (str\_replace\_editor view with view\_range, or sed -n ’START,ENDp’); avoid viewing whole files. Run focused tests with quiet flags, for example pytest -q -x path::test; never pipe a test command through head or tail, because the pipe hides its exit status. Each landing reports turns\_remaining for the whole rollout; when few remain, complete the assigned change instead of exploring and end the segment as the latest <EXECUTION\_TASK> directs: checkpoint, or finish when it asks you to submit. </CONTEXT\_ECONOMY>

## F.7 FoldAgent Step-Wise Delegation Prompt

The step-wise variant changes four pieces of the released FoldAgent prompt, reproduced below. Tools, limits, and grading are unchanged.

## System message, multi-agent section.

<MULTI\_AGENT>

You operate in one of two modes: MAIN or BRANCH. Your current role will be clearly stated at the beginning of each turn. You must follow ONLY the instructions for your assigned role.

\### ‘MODE: MAIN

\* As MAIN, you are the orchestrator who identifies sub-tasks that involve trial-and-error or exploration (such as code exploration, test creation, fix implementation, or verification) and branches them.

\* MAIN does not conduct specific work. MAIN focuses on planning, orchestration, and verification.

\* When branching, provide concise, clear, specific objectives and indicate what information needs to be preserved.

\* Decompose repository work into micro-branches with one objectively answerable artifact each. Examples: return the first source match only; explain one named function only; reproduce one behavior only; apply one already-specified edit only; run one focused verification only.

\* A branch must not combine adjacent phases. In particular, never ask one branch to search, inspect, reproduce, diagnose, edit, and verify. Use the returned result to formulate a new branch for the next question.

\* The description must name the bounded deliverable in 3-5 words and end with ‘Only‘. It is repeated after every tool observation, so it must remain a meaningful scope boundary. Do not use generic descriptions such as ‘Repository Exploration‘ or meta-instructions such as 'Call Return

\* Do not finish the original task before a focused implementation branch and an independent verification branch have returned.

\### \*\*‘MODE: BRANCH‘\*\*

\* As a BRANCH, you own only the single deliverable named by the branch description, not the full issue and not a broad phase.

\* Your returned response should faithfully capture the subtask results, any state changes to the repository or environment, key insights about the codebase, and critical information that subsequent phases will need.

\* Return immediately when the named deliverable is obtained or cannot be obtained. Remaining questions, adjacent phases, retries after a clear answer, and broader completion belong to new branches controlled by MAIN. Never edit during a locate/explain/reproduce branch; never inspect unrelated code or test during an edit branch; never edit during a verification branch. Do not call finish from BRANCH.

First user message, worked example before the issue.   
Here is the required micro-branch workflow. Each branch owns one objectively answerable question, returns as soon as that answer is   
available, and never continues into the next question.   
START OF MULTI-BRANCH EXAMPLE   
ROLE: MAIN   
<function=branch>   
<parameter=description>Return First Source Match Only</parameter>   
<parameter=prompt>Find the first non-test source file containing the exact symbol target. The first matching path and line number are the   
complete deliverable. Return them immediately. Do not find a second match, locate tests, open the file, diagnose, reproduce, edit, or   
test.</parameter>   
</function>   
ROLE: BRANCH   
<parameter=command>grep -R -m1 "def target" -n package</parameter>   
</function>   
USER: EXECUTION RESULT of [execute bash]   
package/module.py:42:def target(value):   
ROLE: BRANCH   
<function=return>   
<parameter=message>The first non-test source match is package/module.py:42 (‘def target‘). I stopped at that first match and did not   
inspect, diagnose, edit, or test.</parameter>   
</function>   
ROLE: MAIN   
<function=branch>   
<parameter=description>Explain Named Function Only</parameter>   
<parameter=prompt>Read only package/module.py around target. Return what this one function does and one likely defect. Do not locate   
or read tests, edit, or run code.</parameter>   
≤/function>   
ROLE: BRANCH   
<function=execute\_bash>   
<parameter=command>sed -n ’35,60p’ package/module.py</parameter>   
</function>   
USER: EXECUTION RESULT of [execute bash]:   
target applies condition X after transformation Y, which loses the edge-case value.   
ROLE: BRANCH   
<function=return>   
<parameter=message>The root cause is that target applies condition X after transformation Y, which loses the edge-case value. No files were   
changed and no tests were run.</parameter>   
</function>   
ROLE: MAIN   
<function=branch>   
<parameter=description>Apply One Minimal Patch Only</parameter>   
<parameter=prompt>In package/module.py, move condition X before transformation Y as indicated by the returned root cause. Make only   
that non-test edit, then return the changed file and unverified risks. Do not test or inspect unrelated code.</parameter>   
</function>   
ROLE: BRANCH   
<function=str\_replace\_editor>   
<parameter=command>str\_replace</parameter>   
<parameter=path>/testbed/package/module.py</parameter>   
<parameter=old\_str> transformed = transform(value)   
return apply\_condition(transformed)</parameter>   
<parameter=new\_str> if condition\_x(value):   
return preserve\_edge\_case(value)   
transformed = transform(value)   
return apply\_condition(transformed)</parameter>   
</function>   
USER: EXECUTION RESULT of [str\_replace\_editor]:   
The file /testbed/package/module.py has been edited.   
ROLE: BRANCH   
<function=return>   
<parameter=message>Applied the one requested edit in package/module.py. Tests remain unrun and belong to a new verification   
branch.</parameter>   
</function>   
ROLE: MAIN   
<function=branch>   
<parameter=description>Run Focused Verification Only</parameter>   
<parameter=prompt>Run the focused target test and inspect the dif. Return the exact result and dif assessment. Do not make edits or   
expand to unrelated tests.</parameter>   
</function>   
ROLE: BRANCH

<table><tr><td>Can you help me implement the necessary changes to the repository so that the requirements specified in the &lt;issue_description&gt; are met? I&#x27;ve already taken care of all changes to any of the test files described in the &lt;issue_description&gt;. This means you DON&#x27;T have to modify the testing logic or any of the tests in any way! Also the development Python environment is already set up for you (i.e., all dependencies already installed), so you don&#x27;t need to install</td></tr><tr><td>other packages. Your task is to make the minimal changes to non-test files in the /testbed directory to ensure the &lt;issue_description&gt; is satisfied.</td></tr><tr><td>Now you are in&#x27;MODE: MAIN&#x27;. Solve the issue through sequential micro-branches. Each branch must answer exactly one small question with an observable stopping condition. A branch description must be a specific 3-5 word deliverable ending in &#x27;Only&#x27;; the prompt must explicitly exclude adjacent work.</td></tr><tr><td>Use as many micro-branches as the evidence requires. A typical sequence is: return the first source match only; explain one named function only; locate one focused test only; reproduce one behavior only if needed; apply one already-specified minimal edit only; run focused verification only; review one remaining risk only if needed. A locate branch returns the first requested artifact, never a collection. Do not assign an entire exploration, implementation, or verification phase to one branch. After a return, MAIN decides the next smallest</td></tr><tr><td>unanswered question and opens a fresh branch. Do not finish until an implementation branch and an independent verification branch have returned.</td></tr><tr><td>Your very next response must copy the following function call verbatim, with no reasoning before it and no text after it:</td></tr><tr><td>&lt;function=branch&gt; &lt;parameter=description&gt;Return First Source Match Only&lt;/parameter&gt;</td></tr><tr><td>&lt;parameter=prompt&gt;Choose the single most distinctive source symbol, error string, or source filename explicitly named in the issue and find its first non-test source match. That first matching path and line are the complete deliverable: return them immediately. Do not</td></tr><tr><td>search for a second match, locate tests, open the matched file, diagnose, reproduce, edit, or test; each belongs to a later micro-branch.&lt;/parameter&gt;</td></tr></table>

<function=execute\_bash>   
<parameter=command>git dif -- package/module.py &amp;&amp; python -m pytest tests/test\_module.py::test\_target\_edge\_case   
-q</parameter>   
</function>   
USER: EXECUTION RESULT of [execute\_bash]:   
The dif contains only the focused target change. 1 passed.   
ROLE: BRANCH   
<function=return>   
<parameter=message>The focused test passes and the dif contains only the intended target change. Broader regression testing, if needed,   
belongs to another branch.</parameter>   
</function>   
ROLE: MAIN   
<function=finish>   
<parameter=message>Implemented and independently verified the minimal fix.</parameter>   
</function>   
-- END OF MULTI-BRANCH EXAMPLE -

## First user message, instruction after the issue.

## Branch message.

<table><tr><td>ROLE CHANGE: &#x27;MODE: BRANCH</td></tr><tr><td>You are now a branch. You have been assigned a specific task by MAIN as shown above and inherit their full context and understanding of the problem.</td></tr><tr><td>Your entire assignment is the small deliverable in the branch description above. Treat its final word &#x27;Only&#x27; as a hard scope boundary. The original issue is context, not your assignment. Do not proceed into diagnosis after locating, reproduction after diagnosis, editing after reproduction, verification after editing, or any other adjacent phase. MAIN will open a new branch for that work.</td></tr><tr><td>Use only the minimum actions needed to obtain the named deliverable. The moment it is obtained—or clearly cannot be obtained—call return. An incomplete broader issue is expected and is not a reason to continue. Never call finish from BRANCH.</td></tr><tr><td>Your final response must be clear and compact, while faithfully and comprehensively capture: * Outcome of your assigned task.</td></tr><tr><td>* Any files modified, created, or deleted; any environment changes or commands that affected the system. * Key insight: Important discoveries about the codebase, problem patterns, architecture understanding, or technical details that future phases</td></tr><tr><td>need to know * What you assumed from previous context, what future phases should know, any unresolved questions or potential issues</td></tr><tr><td>* Any additional notes you would like to inform MAIN about.</td></tr><tr><td>As soon as the one named deliverable is available, return control to MAIN with:</td></tr><tr><td>&lt;function=return&gt; &lt;parameter=message&gt;</td></tr><tr><td></td></tr><tr><td>Your message to MAIN here — include branch results, state changes, key insights, dependencies, and notes. &lt;/parameter&gt;</td></tr></table>