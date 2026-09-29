# WebPageBench: Event-Level Verification and Controlled UI-Variant Generation for Web Agents

Anton Emelyanov<sup>1</sup> Maria Tikhonova<sup>1,2</sup> Zaven Martirosian<sup>1,3</sup>

Sergei Averkiev<sup>1</sup> Alena Fenogenova<sup>1,2</sup>

<sup>1</sup>DAIMLD <sup>2</sup>HSE University <sup>3</sup>NUST MISIS

login-const@mail.ru

## Abstract

We present WebPageBench, an open framework for evaluating web agents in which every task is verified from the interface’s own event log. Six instrumented mock sites with brand identifiers removed (a marketplace, a bookstore, a grocery service, rail ticketing, hotel search and a document cabinet) emit typed events with parameters as a user or an agent acts. A task declares the events it requires, and success is decided by matching them, with no judge model and no scraping of rendered pages. The same instrumentation supports controlled UI variation: one configuration switch re-renders a task through a diferent implementation of a single control while the prompt and the success conditions stay completely identical, so sensitivity to interface form can be measured under a fixed task specification. The WebPageBench release consists of three components: 152 tasks, divided into 65 canonical scenarios and 87 control variants across light/dark UI-modes; a common runner evaluated with six browser/DOM harness configurations and five screenshot-only GUIagent families; and a public leaderboard of 24 model–harness pairs. On the public 152-task leaderboard the gap between what agents declare finished and what the log confirms reaches 41 points (one configuration declares every task finished and satisfies the conditions on 59%).

## 1 Introduction

Benchmarks for web agents trade verifiability against scale. Ofline trajectory matching (Deng et al., 2023) is reproducible but scores the path instead of the outcome. Live-web benchmarks (He et al., 2024; Yoran et al., 2024; Trabucco et al., 2025) reach hundreds of sites and pay for it with a judge model, drifting content and anti-bot defenses. Self-hosted environments (Zhou et al., 2023; Boisvert et al., 2024; Garg et al., 2025; Xu et al., 2024) check application state directly, but only the state that remains when the episode ends, so the steps an agent took and the state it left behind are indistinguishable.

Existing benchmarks additionally fail to isolate agent sensitivity to the implementation of individual UI controls. Figure 1 shows what such a comparison requires: the same prompt and the same success condition rendered once through a popup calendar and once through a native date field. An aggregate score over ordinary benchmarks cannot tell a failure on the second interface from a misunderstood task.

WebPageBench addresses both. It is intended for researchers and engineers who develop web agents and need to tell model failures from harness and interface failures, and to reproduce those failures locally. The environment is a set of mock sites we control and instrument: every action a user or an agent performs writes a typed event with parameters to a backend, and a task’s success condition is a list of such events. Because the interface is configurable, one switch produces a variant of a task in which only the control changes; the prompt and the conditions are unchanged, so a diference in success rate reflects the interface realisation. Recording the agent’s completion signal separately from the verifier already gives a result: in the runs we report, agents declare success far more often than the log confirms it, and the size of the gap depends on the harness. Interfaces, prompts and data are Russian; the verification contract reads typed events and never the interface text, so it does not depend on the language.

The main contribution of this work is Web-PageBench, an evaluation system for web agents built around one methodology: the interface is instrumented, so success is decided from the typed events it emits, and the same instrumentation makes the interface itself a controlled variable. The system consists of three components, each of which follows from that principle.

• The event-level verification contract (§3.2, §3.3) defines what counts as success: 19 event types with typed parameters, per-task conditions with exact, wildcard, comparison and last-state matching, and a primary metric, the Event-Match Score (EMS), that needs nojudge model. The agent’s own completion signal is recorded separately, so over-claiming becomes measurable.

![](images/2446185a102386f4d7d6aa626734082f3635b16c91a0e9ee0b62c4d46727a412.jpg)  
Figure 1: One task, one condition, two interfaces. The prompt (translated) and the single success condition shown above the screenshots are identical in both task files; the released variant difers from its donor only in the profile it names (rail\_date\_native). On the left the departure date is picked from the default popup grid, where the month and the day are on screen and one click settles it. On the right the same date goes into a native <input type="date">, so the agent has to render “4 October 2026” from the prompt in the numeric format the field expects (the screenshot shows the field with an arbitrary date typed in). The figure illustrates the variant mechanism only; the leaderboard submissions do not separate donors from variants, so no per-variant result is reported.

• The generator of controlled UI variants (§3.4) makes the interface a variable of the experiment: nine UI configuration keys (eight controls and the colour theme) with 33 implementations produce 87 variant tasks from 25 donors, 82 of them changing exactly one control, while the prompt and the conditions stay fixed.

• The runner and leaderboard (§4) apply the contract uniformly to both action spaces: six evaluated browser/DOM harness configurations and five screenshot-only GUI-agent families run under one verifier, so a comparison can hold the model, the harness or the interface fixed; the release adds a reproducible local deployment, automatic repair of timedependent tasks, and a public leaderboard of 24 model–harness pairs on the 152-task suite.

## 2 Related Work

Table 6 groups web-agent benchmarks by what decides success, its first column. Reference matching never inspects the environment. Judge models scale to the open web but inherit the judge’s failure modes, and their reliability outside English has, to our knowledge, not been established. Hybrid designs check the state of the application but fall back on a model for information-seeking tasks or images. Deterministic designs decide success programmatically: a reward over product attributes (Yao et al., 2022), backend API checks (Boisvert et al., 2024), replay against a state machine (Wu et al., 2026), final-state comparison (Yuan et al., 2026). WebPageBench belongs to this last group and difers within it: conditions are matched against the typed events the interface emitted during the run, including intermediate steps and their parameters, instead of the state left at the end. Other work automates task construction (Pahuja et al., 2025; Xie et al., 2025; Chen et al., 2025; Fan et al., 2026; Huang et al., 2026); we author tasks by hand and automate the variants.

The two right-hand columns of Table 6 separate WebPageBench from the rest of its group. Among the systems surveyed, none renders one task through alternative implementations of a control while holding prompt and verifier fixed; the closest is WorkArena++, which samples one of ten fictitious company brands per episode, a repaint of the same interface with the controls unchanged. Where several agents can be run against a benchmark, they usually arrive through BrowserGym and AgentLab (Le Sellier De Chezelles et al., 2025), an ecosystem that wraps benchmarks behind a shared agent interface. That work is complementary: our environment could be exposed through it, and the frameworks and GUI models our runner drives are not among the agents it ships. Compared with a WebArena-style environment plus logging, four things difer: tasks are written against the event contract, the verdict is inspectable per condition, the task specification stays fixed across UI realisations, and one runner serves both action spaces.

## 3 Framework Methodology

Figure 3 shows one run end to end. A task file supplies the prompt and the success conditions, the domain configuration supplies the control implementations, and the two are opened as an isolated track on a mock site. The agent acts on the pages that track serves. The site writes typed events to the track log, and check() decides success by matching the conditions against that log alone.

## 3.1 Mock sites and anonymization

The canonical scenarios were written by hand, not generated. The contributors who built each mock site worked out the frequent purchases and the other ordinary actions on that site, and wrote the prompt and the success conditions for every scenario. Variants (§3.4) copy those scenarios and change a control; they are not new scenarios.

Tasks live under one neutral hub with six tabs, each a mock site covering an everyday online activity: marketplace, books and audiobooks, grocery delivery, rail travel, hotel search, and a file document cabinet. The sites are written as one front end over a back end that serves per-task configuration and stores events.

Anonymization goes beyond the logos. The mock sites are based on real, widely used platforms and are anonymized: route and state names are neutral (bench\_catalog\_main, bench\_hotel\_ search), so neither the URL nor the page structure gives a model any navigational or branding cue to a well-known platform whose flow it may have memorized. The book catalog is additionally rewritten by a deterministic script (seed 42): 226 items, 75 author names, 229 titles, and 86 series are replaced with synthetic Russian names, generated together with their grammatical case forms so that prompts naming an author remain well-formed. Book covers are dropped in that same pass. By contrast, the marketplace and grocery catalogs are not textually rewritten: their accompanying text is left intact, while the card photographs are replaced with openly licensed files stored locally. After image pruning the marketplace holds 42 products and the grocery service 277 products across 117 category files. The release contains 590 active card-image paths:

277 grocery, 41 marketplace and 272 hotel images, linked to 444 distinct source URLs. The attribution workbook maps every local path to its source URL and license<sup>1</sup>; 543 rows also include an Openverse image-information page. The workbook contains neither similarity scores nor automatic/manual decision flags, so we do not report a 0.5 threshold or an automatic-vs.-manual split. The verifier itself does not inspect image content: conditions match typed events rather than pixels.

## 3.2 Events and UI configuration keys

A track is one isolated run of one task: it has its own URL, application state and event log, so parallel runs never see each other. An event is a typed record the interface writes to the track when something happens: a name and a dictionary of parameters. basket\_add carries the item identifier and quantity, select\_tariff the tarif name, bench\_hotel\_select\_city the city identifier, display name and country. Events are emitted by the pages themselves and never reconstructed from the DOM, so they record what the application registered, independently of how the page looked. Low-level activity (click, keypress, scroll) is logged too, but conditions refer only to semantic events: the suite uses 19 such event types, and the 152 released tasks declare 521 conditions over them, 3.4 per task on average.

A UI configuration key names a control whose implementation is selected by configuration instead of being fixed in the markup; eight keys are controls and the ninth is the colour theme. Picking a date is one interaction; the hotel site can render it as a split popup calendar (the default), an always-visible inline calendar, a typed DD.MM.YYYY string, or a single-field range picker, and the rail site as a grid popup (its default), a native <input type="date"> or the same typed string, so the date key has six implementations. Choosing a city is an autocomplete or a native select; a guest counter is a per-room popup, an inline stepper, a compact select or a row of pill buttons. The nine keys have 33 implementations in total (Appendix B). A control emits the same event whichever implementation is mounted (Figure 2), so the condition that checks it stays valid across implementations.

Figure 3 shows how the two meet: a task JSON and the domain configuration are instantiated as

```jsonl
Condition declared by the task
{"event_name": "bench_hotel_select_guests",
"parameters": {"roomsCount": 1,
"guestsCount": 2}}
Event written by the interface
{"type": "bench_hotel_select_guests",
"roomsCount": 1, "guestsCount": 2,
"widget": "pill_buttons"}
```

Figure 2: A condition and the event that satisfies it. All four implementations of the guest counter (per-room popup, inline stepper, compact select, pill buttons) emit the same event with the same parameters, so the condition holds for all of them. The widget field is recorded for analysis and never enters the verdict.

![](images/5981b9d9a5c5a8d14f0185800e11367c9bb11e8395abebcab0665742e847b45c.jpg)  
Figure 3: One run. The task supplies the prompt and the conditions, the domain configuration supplies the control implementations, and the event log is the only evidence used to decide success.

a track, the agent acts against the served pages, and the client library checks the log against the conditions.

## 3.3 The verification contract

A condition is a predicate over the events emitted during a track: it names an event and the parameters that must match, and it holds if some event of that name carries the required parameters. Conditions are evaluated independently over the track log. Strings match exactly or through a \* wildcard, numeric fields may carry a comparison operator, and conditions can be grouped so that satisfying any member satisfies the group. The verifier does not require a particular ordering unless the ordering is encoded indirectly in event parameters or in the application’s own flow; what a task pins down is which steps happened and with which values. Quantities are the exception to “any event”: every basket, seat and guest-count event carries the resulting quantity, and a condition on that quantity is compared with the last such event for the item: adding two units and then a third fails a condition asking for two. A task can override this per condition with match: any when the intermediate action is what is being tested.

The primary metric, the binary task-level Event-Match Score (EMS), is 1 for a task when every condition is satisfied and 0 otherwise; we report its mean over tasks, and the per-condition verdicts are kept as diagnostic (§4): in the rail task of Appendix A, a wrong tarif satisfies six of seven conditions and scores 0. No model is consulted. Because the contract reads typed events and never the interface text, it does not depend on the language of the interface, whereas a judge model’s accuracy does.

The evaluation covers 11 Harness identifiers (Table 1): six browser integrations and five screenshotonly GUI adapters. DOM is an indexed element tree the agent acts on by element handle; text is extracted page text without that tree. Browser-use, Ouroboros-cut and OpenManus receive DOM plus a screenshot, except the DeepSeek pairs, which receive DOM text only; the two full Ouroboros runtimes receive text plus a screenshot; OpenHands receives DOM text only. The GUI adapters receive screenshots only and return pointer coordinates through our Playwright driver. Several integrations reuse browser-use as the execution layer and change only prompting, planning or memory. Both action spaces run over one suite with one verifier. The public leaderboard reports 24 evaluated model–harness pairs.

Two further signals are recorded. Completion is the harness’s own claim that it finished, for example, browser-use’s done action; a run that ends in a harness error counts as not completed. Where the two disagree, we distinguish an overclaim (completed, EMS 0) from an under-claim (EMS 1, not completed). We call the diference between Completion and EMS the completion– verification gap; §5.2 shows that it varies widely with the harness. Steps, time, tokens and cost are recorded as well. The published submission stores Completion and EMS as rates, together with steps, duration and cost. It does not store over-claim, under-claim or harness-error counts. Done&pass, counted from the run logs, recovers the first two: Completion minus Done&pass is the over-claim rate, and EMS minus Done&pass is the under-claim rate.

Time-dependent tasks are repaired automatically. Booking interfaces disable past dates. Before a track is created, all date conditions are shifted forward by one shared ofset so the earliest falls on or after today, and the dates in the Russian prompt (dotted, long-form and range spellings) are rewritten to

<table><tr><td>Harness</td><td>Obs.</td><td>Notes</td></tr><tr><td>browser-use</td><td>DOM+ screenshot‡</td><td>baseline executor</td></tr><tr><td>ouroboros-cut</td><td>DOM+ screenshot</td><td>Ouroboros prompts, browser-use</td></tr><tr><td>ouroboros-full-isolated screenshot</td><td>text+</td><td>Ouroboros, fresh memory</td></tr><tr><td>ouroboros-full-evolving</td><td>text+ screenshot</td><td>Ouroboros, shared memory†</td></tr><tr><td></td><td>DOM+</td><td></td></tr><tr><td>openmanus openhands</td><td>screenshot‡ DOM text</td><td>OpenManus prompts, browser-use OpenHands SDK, browser tools†</td></tr><tr><td>qwen3-v1</td><td>screenshot</td><td>Qwen3.8-27B</td></tr><tr><td>uitars</td><td>screenshot</td><td>UI-TARS-1.5-7B</td></tr><tr><td>opencua</td><td>screenshot</td><td>OpenCUA-32B / 72B</td></tr><tr><td>evocua</td><td>screenshot</td><td>EvoCUA-32B S1 / S2</td></tr><tr><td>fara</td><td>screenshot</td><td>Fara-1.5 9B / 27B</td></tr></table>

Table 1: Harnesses evaluated in this snapshot; all are selected by the AGENT\_HARNESS switch: browser-use (Browser Use, 2024), OpenManus (FoundationAgents, 2025), OpenHands (Wang et al., 2025a), Qwen3-VL (Bai et al., 2025), UI-TARS (Qin et al., 2025), OpenCUA (Wang et al., 2025b), EvoCUA (Xue et al., 2026) and Fara-1.5 (Awadallah et al., 2026); Ouroboros is an inhouse agent runtime. DOM is an indexed element tree the agent can act on by element handle; text is extracted page text without that tree. A screenshot, when listed, is sent in addition. <sup>‡</sup>The DeepSeek pairs receive DOM text only, without screenshots. <sup>†</sup>Runs on a single worker; for the evolving runtime because tasks share one memory.

match, so the suite stays runnable without editing task files.

## 3.4 Generating controlled UI variants

Operationally, a task is a variant when its file carries a variant profile and canonical otherwise; every canonical task in the release was authored by hand. A variant is a materialised copy of a canonical task (its donor) with a profile applied, a named assignment of UI configuration keys such as hotels\_date\_inline. The prompt and the conditions are copied, so the two share one task specification and difer only in the implementation of the control and what it brings with it: DOM structure, action count, layout, screenshot coordinates. The released suite has 87 variants from 25 donors through 20 profiles. Nineteen profiles set a single key, so 82 variants change exactly one control; one profile for the document cabinet sets three keys (layout, year selector, button style) that share a screen, giving five three-key variants. The dark theme is never combined with a control change. Table 2 gives the per-domain counts, and Table 4 in Appendix A shows one donor per domain with its variants. Every task also carries interaction classes from a 19-class taxonomy (navigation, basket, date, counter, payment, and so on), one of them designated as primary; 13 classes occur in the released tasks and 9 as primary (Appendix C).

<table><tr><td>Tab</td><td>Activity</td><td>Canon.</td><td>Var.</td><td>Total</td></tr><tr><td>M</td><td>marketplace</td><td>18</td><td>14</td><td>32</td></tr><tr><td></td><td>books, audiobooks</td><td>11</td><td>10</td><td>21</td></tr><tr><td></td><td>document cabinet</td><td>11</td><td>7</td><td>18</td></tr><tr><td></td><td>rail travel</td><td>9</td><td>17</td><td>26</td></tr><tr><td></td><td>grocery delivery</td><td>8</td><td>2</td><td>10</td></tr><tr><td></td><td>hotel search</td><td>8</td><td>37</td><td>45</td></tr><tr><td>Total</td><td></td><td>65</td><td>87</td><td>152</td></tr></table>

Table 2: Tasks per domain. Variant counts reflect the profiles available in a domain and the donors eligible for them. For example, the hotel domain combines date, city and guest-counter controls and yields 37 variants from 8 canonical tasks.

For a harness h and a key w, let $D _ { w }$ be the donor tasks that have a variant on w and $V _ { w }$ those variants. The sensitivity gap is

$$
\Delta _ { h , w } \ = \ \mathrm { E M S } _ { h } ( D _ { w } ) - \mathrm { E M S } _ { h } ( V _ { w } ) ,
$$

the change in success on the same task specifications when the implementation of w changes. A large $| \Delta _ { h , w } |$ shows that performance is sensitive to the realisation of that control; whether the agent misunderstood the task or could not operate the control is a hypothesis to examine on the paired traces; ∆ alone does not decide it. Restricting the baseline to $D _ { w }$ would keep the task mix identical. The published submissions do not split donors from variants, so this snapshot does not report $\Delta _ { h , w }$

## 4 System Demonstration

Running the benchmark. The stack is one Docker composition: backend with event store, front end, evaluation container. A run is one command; the harness and the model are chosen by two environment variables, and the runner creates a track per task, hands its URL to the agent, waits for the agent to stop, and calls check(). Evaluation is a pytest suite built on DeepEval (Confident AI, 2024), parallel over workers except for the two harnesses marked in Table 1. On the public leaderboard a 152-task run with gemini-3.8-flash and openmanus averages 7.8 steps and 221 s of harness time per task (median 131 s; 16.3 tasks/hour/worker at that mean; \$17.88 for the run). The submission does not record a separate cold-start time. Open-Hands runs on one worker. Local GUI checkpoints in this snapshot do not report USD cost.

Inspecting a run. A user picks a task, a model, a harness and a UI profile, starts a track and follows the browser episode. When the agent stops, the track page lists every event with its parameters next to the conditions and marks which matched, so the visitor sees the agent declare the task finished and, on the same page, the conditions no event satisfied. The visitor then re-runs the identical task with another implementation of one control, compares the two traces, and opens the leaderboard slice for that harness, class or key.

Leaderboard and extension. Finished runs are exported to a public leaderboard with views for success, speed, and cost, broken down per domain and per interaction class. In the submission file the domain lists are the canonical tasks, and the interaction-class scores are the nine primary classes in ui\_classes. A new task is a JSON file; a new control implementation is one more registered value for a key, and every task in the domain can be cloned against it. The public leaderboard can be found on HuggingFace<sup>2</sup>. The codebase can be found in our repository<sup>3</sup>.

## 5 Evaluation

## 5.1 Protocol

A run fixes a harness, a model and a task set, and reports the EMS, Completion, steps, duration and cost. Those are the fields in the published submission, plus per-domain sections for its canonical task list and primary-class ui\_classes. Done&pass is counted from the run logs. Over-claim, under-claim and harness-error counts are not in the submission. The runs ask whether event verification exposes failures hidden by the completion signal, how much performance varies across harnesses for one model, and how sensitive agents are to a control’s implementation. Each submission’s EMS covers all 152 tasks, variants included, but donors and variants are not scored separately. Static validation (test\_verify\_bench) checks the structure, routes and mounted profiles of all 152 task files. Decoding settings are the harness defaults; each pair is a single run. Model identifiers are those in Table 3.

## 5.2 Results

Table 3 reports the 24 public submissions. EMS, Completion, steps, duration and cost are submission fields; Done&pass, the share of tasks the agent reported done and the log confirmed, is counted from the run logs. Within a model, bold marks the highest EMS, the shortest duration and the lowest cost.

Completion and EMS diverge, and the sign of the gap depends on the harness. Completion minus EMS reaches 41 points for gpt-5.6-luna with openmanus (1.00 vs. 0.59); its Done&pass equals its EMS, so every verified success is hidden inside a completion signal that fires on every task. The same model with openhands is almost calibrated (0.638 vs. 0.625). On OpenHands the gap changes sign: Gemini’s completion is 0.461 against an EMS of 0.572, and its Done&pass of 0.454 means that 18 of its 87 verified successes ended without a completion signal. A protocol that equates the completion signal with success would inherit this harness-specific gap; the submission records the two rates separately.

Harness ranking is model-dependent, and GUI still trails DOM. openmanus is the best configuration for Gemini-3.8-flash (0.822) but not for GPT-5.6-luna, which peaks at ouroboros-fullisolated (0.796) and is worst on openmanus (0.592). Screenshot agents on the same 152-task EMS sit lower: Fara-1.5-9B at 0.638, OpenCUA-72B at 0.421, UI-TARS-1.5-7B at 0.336, EvoCUA below 0.10; Qwen3.8-27B scores 0.737–0.763 through DOM harnesses and 0.480 through screenshots.

GUI agents also under-report the tasks they solve. For the DOM harnesses Done&pass is within one point of EMS: a solved task is a reported task. For the screenshot agents it is not: Fara-1.5-9B solves 0.638 but reports and solves only 0.493, Qwen3-VL 0.480 against 0.375, UI-TARS 0.336 against 0.289. Between one in nine and one in four verified successes of these agents ends without a completion signal.

Cost and speed do not track EMS. The most expensive run, glm-5.2 with openhands (\$76.08), scores 0.500, while gpt-5.6-luna with browser-use scores 0.704 for \$3.23. Within one model the cheapest harness is not the best: for GPT-5.6-luna, openmanus costs least (\$3.04) and scores lowest. In each of the three model groups where it appears, OpenHands is the fastest harness per task (132–158 s) and at the same time the most expensive, because its per-task step count is high (131.7 steps for Gemini).

Primary-class scores are the taxonomy the submission publishes. On gemini-3.8-flash × openmanus, ui\_classes is BASKET 48/66, COUNTER 26/33, FILES 18/18, FAV 14/14, CARD 9/9, DATE 4/4, PAY 2/4, SELECT\_AC 2/2 and SELECT\_LIST 2/2 (Table 5). Classes with four or fewer tasks are diagnostics, not a ranking. The submission has no donor–variant split, so $\Delta _ { h , u }$ is not reported.

<table><tr><td>Model</td><td>Harness</td><td>EMS</td><td>Compl.</td><td>Done&amp;pass</td><td>Steps</td><td>Dur.</td><td>Cost $</td></tr><tr><td rowspan="3">gemini-3.8-flash</td><td>openmanus</td><td>0.822</td><td>1.000</td><td>0.822</td><td>7.8</td><td>221</td><td>17.88</td></tr><tr><td>browser-use</td><td>0.711</td><td>0.993</td><td>0.711</td><td>9.5</td><td>220</td><td>23.65</td></tr><tr><td>openhands</td><td>0.572</td><td>0.461</td><td>0.454</td><td>131.7</td><td>132</td><td>53.12</td></tr><tr><td rowspan="5">gpt-5.6-luna</td><td>ouro-full-iso</td><td>0.796</td><td>0.868</td><td>0.796</td><td>27.8</td><td>177</td><td>6.69</td></tr><tr><td>browser-use</td><td>0.704</td><td>0.993</td><td>0.704</td><td>10.8</td><td>356</td><td>3.23</td></tr><tr><td>ouro-full-env</td><td>0.697</td><td>0.717</td><td>0.678</td><td>13.1</td><td>297</td><td>3.91</td></tr><tr><td>openhands</td><td>0.625</td><td>0.638</td><td>0.605</td><td>18.1</td><td>158</td><td>23.19</td></tr><tr><td>openmanus</td><td>0.592</td><td>1.000</td><td>0.592</td><td>10.8</td><td>296</td><td>3.04</td></tr><tr><td rowspan="3">deepseek-v4.1-flash</td><td>browser-use</td><td>0.789</td><td>0.993</td><td>0.783</td><td>14.2</td><td>467</td><td>8.28</td></tr><tr><td>openmanus</td><td>0.711</td><td>0.974</td><td>0.704</td><td>14.7</td><td>472</td><td>9.03</td></tr><tr><td>openhands</td><td>0.454</td><td>0.342</td><td>0.336</td><td>23.5</td><td>148</td><td>39.66</td></tr><tr><td rowspan="4">qwen3.8-27b</td><td>openmanus</td><td>0.763</td><td>1.000</td><td>0.763</td><td>9.9</td><td>84</td><td></td></tr><tr><td>ouro-cut</td><td>0.750</td><td>1.000</td><td>0.750</td><td>10.4</td><td>83</td><td></td></tr><tr><td>browser-use</td><td>0.737</td><td>1.000</td><td>0.737</td><td>10.8</td><td>91</td><td></td></tr><tr><td>qwen3-vl</td><td>0.480</td><td>0.493</td><td>0.375</td><td>17.0</td><td>74</td><td></td></tr><tr><td>fara-1.5-9b fara-1.5-27b</td><td>fara</td><td>0.638</td><td>0.572</td><td>0.493</td><td>16.4</td><td>44</td><td></td></tr><tr><td></td><td>fara</td><td>0.625</td><td>0.625</td><td>0.553</td><td>16.5</td><td>72</td><td></td></tr><tr><td>glm-5.2</td><td>openhands</td><td>0.500</td><td>0.428</td><td>0.408</td><td>22.7</td><td>90</td><td>76.08</td></tr><tr><td>opencua-72b</td><td>opencua</td><td>0.421</td><td>0.724</td><td>0.421</td><td>13.2</td><td>437</td><td></td></tr><tr><td>minimax-m2.7</td><td>openhands</td><td>0.368</td><td>0.257</td><td>0.230</td><td>21.5</td><td>103</td><td>24.96</td></tr><tr><td>opencua-32b</td><td>opencua</td><td>0.336</td><td>0.559</td><td>0.336</td><td>15.3</td><td>271</td><td></td></tr><tr><td>uitars-1.5-7b</td><td>uitars</td><td>0.336</td><td>0.382</td><td>0.289</td><td>19.1</td><td>42</td><td></td></tr><tr><td>evocua-32b-s1</td><td>evocua</td><td>0.092</td><td>0.112</td><td>0.079</td><td>23.5</td><td>259</td><td></td></tr><tr><td>evocua-32b-s2</td><td>evocua</td><td>0.079</td><td>0.191</td><td>0.072</td><td>22.5</td><td>143</td><td></td></tr></table>

Table 3: Public leaderboard submissions (24 model–harness pairs), grouped by model. Models are ordered by their best EMS, rows within a model by EMS. Where a model has several harnesses, bold marks the highest EMS, the shortest duration and the lowest cost in that group; single-harness rows are not bold. ouro-full-iso is ouroboros-full-isolated, ouro-full-env is ouroboros-full-evolving, ouro-cut is ouroboros-cut. EMS is passed\_tasks/152. Compl. is agent\_completion\_rate. Done&pass is the share of tasks the agent reported done and whose conditions all passed, counted from the run logs; it is the same count the public leaderboard shows. Steps and Dur. are per-task means: agent steps and harness time in seconds. Cost is total\_cost\_usd for the whole 152-task run; — when the submission has no price.

## 6 Conclusion

WebPageBench verifies web-agent tasks from the interface’s own event log: no model decides success, and because the contract reads typed events and never the interface text, it does not depend on the language. Rendering one task through diferent implementations of one control makes sensitivity to interface form measurable. On the public 152- task leaderboard the completion–verification gap reaches 41 points, and the sign of the gap flips with the harness.

## Limitations

WebPageBench is a controlled set of mock sites, not a measurement of agents on production websites; anti-bot defenses, real payment rails and content drift are outside what it models. Events go to the backend, but application state (basket, favourites, login) lives in the browser’s storage, so a scenario that spans two browser sessions is not expressible. The suite is Russian-only and does not test codeswitching or transliteration beyond a single task family.

The published submissions do not split donor tasks from their variants, so this snapshot does not report $\Delta _ { h , w }$ . A variant changes DOM structure, action count and screenshot coordinates together, and a future split should not be read as proof that the agent misunderstood the task. The leaderboard covers 24 model–harness pairs. Each run is timebounded. We accept that an agent may take a long time to answer; if it does not finish within the shared time limit, the task scores 0, and we treat that outcome as a normal part of the comparison. Event matching is not proof of understanding. Conditions are checked independently and without order, so a task that must distinguish two orderings of the same events cannot be expressed, and an agent could in principle emit the required events without understanding the task. Conditions that pin taskspecific parameter values (item identifiers, dates, tarif names) make this unlikely, but we have not run an adversarial audit to confirm it.

## Ethics Statement

The environment contains no personal data: accounts, payment cards and passenger records are synthetic fixtures, and the payment flow is a mock that reaches no external system. Book catalog entries are replaced with synthetic authors and titles (§3.1). All 590 catalog photographs have a source URL and redistribution-compatible license in the release attribution workbook: CC BY, CC BY-SA, CC0 or Public Domain (§3.1). The software is released under the Apache License 2.0; third-party images retain their source licenses. The evaluation runs headless browsers against a local deployment and issues no trafic to third-party websites.

We used generative AI tools while preparing this demonstration and paper (Cursor / Claude / GPT for drafting LaTeX and code review). All task files, event contracts, evaluation numbers and the wording of claims were checked by the authors; remaining errors are ours.

## Acknowledgments

We thank the contributors of the project: Maria Glushkova, Anna Kostikova, Elina Basyrova, Aleksandra Smolenkova, Nadezhda Shervarly, Alexander Mazurin, Anastasia Filonova and Dmitri Baliev for implementing the mock sites, the evaluation runner and the public leaderboard, for authoring and validating the task suite, and for running the leaderboard submissions. Compute for the 24 leaderboard pairs was provided by cloud.ru.

## References

Ahmed Awadallah, Sahil Gupta, Yash Lara, Yadong Lu, Hussein Mozannar, Akshay Nambi, Zach Nussbaum, Yash Pandya, Aravind Rajeswaran, Corby Rosset, Alexey Taymanov, Luiz do Valle, Vibhav Vineet, Spencer Whitehead, and Andrew Zhao. 2026.

Fara1.5: Scalable learning environments for computer use agents. arXiv preprint arXiv:2606.20785.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, et al. 2025. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631.

Leo Boisvert, Megh Thakkar, Maxime Gasse, Massimo´ Caccia, Thibault Le Sellier De Chezelles, Quentin Cappart, Nicolas Chapados, Alexandre Lacoste, and Alexandre Drouin. 2024. Workarena++: Towards compositional planning and reasoning-based common knowledge work tasks. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track.

Browser Use. 2024. browser-use: Make websites accessible for ai agents. https://github.com/browseruse/browser-use.

Yurun Chen, Xavier Hu, Yuhan Liu, Ziqi Wang, Zeyi Liao, Lin Chen, Feng Wei, Yuxi Qian, Bo Zheng, Keting Yin, et al. 2025. Graph2eval: Automatic multimodal task generation for agents via knowledge graphs. arXiv preprint arXiv:2510.00507. To appear in CVPR 2026.

Confident AI. 2024. Deepeval: The open-source llm evaluation framework. https : / / github . com / confident-ai/deepeval.

Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Samuel Stevens, Boshi Wang, Huan Sun, and Yu Su. 2023. Mind2web: Towards a generalist agent for the web. Advances in Neural Information Processing Systems (NeurIPS).

Sicheng Fan, Qingyun Shi, Shengze Xu, Shengbo Cai, Tieyong Zeng, Li Ling, Yanyi Shang, and Dehan Kong. 2026. Webfactory: Automated compression of foundational language intelligence into grounded web agents. arXiv preprint arXiv:2603.05044. ICLR 2026.

FoundationAgents. 2025. Openmanus: An open-source framework for building general ai agents. https: //github.com/FoundationAgents/OpenManus.

Divyansh Garg, Diego Caples, Andis Draguns, Nikil Ravi, Pranav Putta, Naman Garg, Prannay Hebbar, Youngchul Joo, Jindong Gu, Charles London, Christian Schroeder de Witt, and Sumeet Motwani. 2025. Real: Benchmarking autonomous agents on deterministic simulations of real websites. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track.

Hongliang He, Wenlin Yao, Kaixin Ma, Wenhao Yu, Yong Dai, Hongming Zhang, Zhenzhong Lan, and Dong Yu. 2024. Webvoyager: Building an end-toend web agent with large multimodal models. arXiv preprint arXiv:2401.13919.

Tenghao Huang, Kung-Hsiang Huang, Prafulla Kumar Choubey, Yilun Zhou, Muhao Chen, Jonathan May, and Chien-Sheng Wu. 2026. GTA: Generating longhorizon tasks for web agents at scale. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 18805–18820, Vienna, Austria. Association for Computational Linguistics.

Jing Yu Koh, Robert Lo, Lawrence Jang, Vikram Duvvur, Ming Chong Lim, Po-Yu Huang, Graham Neubig, Shuyan Zhou, Ruslan Salakhutdinov, and Daniel Fried. 2024. Visualwebarena: Evaluating multimodal agents on realistic visually grounded web tasks. arXiv preprint arXiv:2401.13649.

Thibault Le Sellier De Chezelles, Maxime Gasse, Alexandre Drouin, Massimo Caccia, Leo Boisvert,´ Megh Thakkar, Tom Marty, Rim Assouel, Sahar Omidi Shayegan, Lawrence Keunho Jang, Xing Han Lu, Ori Yoran, Dehan Kong, Frank F. Xu, Siva Reddy,\` Quentin Cappart, Graham Neubig, Ruslan Salakhutdinov, Nicolas Chapados, and Alexandre Lacoste. 2025. The BrowserGym ecosystem for web agent research. Transactions on Machine Learning Research (TMLR).

Lajanugen Logeswaran, Jaekyeom Kim, Sungryull Sohn, Creighton Glasscock, and Honglak Lee. 2026. Scaling web agent training through automatic data generation and fine-grained evaluation. arXiv preprint arXiv:2602.12544. Introduces the BookingArena benchmark.

Vardaan Pahuja, Yadong Lu, Corby Rosset, Boyu Gou, Arindam Mitra, Spencer Whitehead, Yu Su, and Ahmed Hassan Awadallah. 2025. Explorer: Scaling exploration-driven web trajectory synthesis for multimodal web agents. In Findings ofthe Associationfor Computational Linguistics: ACL 2025.

Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, Wanjun Zhong, Kuanye Li, Jiale Yang, Yu Miao, Woyu Lin, Longxiang Liu, Xu Jiang, Qianli Ma, Jingyu Li, and 16 others. 2025. UI-TARS: Pioneering automated gui interaction with native agents. arXiv:2501.12326.

Brandon Trabucco, Gunnar Sigurdsson, Robinson Piramuthu, and Ruslan Salakhutdinov. 2025. Insta: Towards internet-scale training for agents. arXiv preprint arXiv:2502.06776.

Xingyao Wang, Boxuan Li, Yufan Song, Frank F. Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, et al. 2025a. Openhands: An open platform for AI software developers as generalist agents. In International Conference on Learning Representations (ICLR).

Xinyuan Wang, Bowen Wang, Dunjie Lu, Junlin Yang, Tianbao Xie, Junli Wang, Jiaqi Deng, Xiaole Guo, et al. 2025b. OpenCUA: Open foundations for computer-use agents. arXiv preprint arXiv:2508.09123.

Yifan Wu, Yiran Peng, Yiyu Chen, Jianhao Ruan, Zijie Zhuang, Cheng Yang, Jiayi Zhang, Man Chen, Yenchi Tseng, Zhaoyang Yu, Liang Chen, Yuyao Zhai, Bang Liu, Chenglin Wu, and Yuyu Luo. 2026. Autowebworld: Synthesizing infinite verifiable web environments via finite state machines. arXivpreprint arXiv:2602.14296.

Jingxu Xie, Dylan Xu, Xuandong Zhao, and Dawn Song. 2025. Agentsynth: Scalable task generation for generalist computer-use agents. arXiv preprint arXiv:2506.14205. Accepted to ICLR 2026.

Frank F. Xu, Yufan Song, Boxuan Li, Yuxuan Tang, Kritanjali Jain, Mengxue Bao, Zora Z. Wang, Xuhui Zhou, Zhitong Guo, Murong Cao, Mingyang Yang, Hao Yang Lu, Amaad Martin, Zhe Su, Leander Maben, Raj Mehta, Wayne Chi, Lawrence Jang, Yiqing Xie, and 2 others. 2024. Theagentcompany: Benchmarking llm agents on consequential real world tasks. arXiv preprint arXiv:2412.14161.

Taofeng Xue, Chong Peng, Mianqiu Huang, Linsen Guo, Tiancheng Han, Haozhe Wang, Jianing Wang, Xiaocheng Zhang, Xin Yang, Dengchang Zhao, Jinrui Ding, Xiandi Ma, Yuchen Xie, Peng Pei, Xunliang Cai, and Xipeng Qiu. 2026. EvoCUA: Evolving computer use agents via learning from scalable synthetic experience. arXiv preprint arXiv:2601.15876.

Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. 2022. Webshop: Towards scalable realworld web interaction with grounded language agents. Advances in Neural Information Processing Systems (NeurIPS).

Ori Yoran, Samuel Joseph Amouyal, Chaitanya Malaviya, Ben Bogin, Ofir Press, and Jonathan Berant. 2024. Assistantbench: Can web agents solve realistic and time-consuming tasks? In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP).

Peng Yuan, Yuyang Yin, Yuxuan Cai, and Zheng Wei. 2026. Webforge: Breaking the realismreproducibility-scalability trilemma in browser agent benchmark. arXiv preprint arXiv:2604.10988.

Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. 2023. Webarena: A realistic web environment for building autonomous agents. arXiv preprint arXiv:2307.13854.

## A Example Tasks

Tasks, interfaces and catalog data are Russian; the examples below are translated for readability, and the conditions are quoted verbatim. Table 4 shows one donor task per domain together with the variants generated from it.<sup>4</sup>

<sup>4</sup>Original prompts ship with the suite; the translation afects this paper only, not the evaluation.

Marketplace, one action. Prompt: “add any catalog item to the cart.” Condition: one basket\_ add event, no parameters required.

Hotels, four parameters. Prompt: “show options in Hotels: Zurich, Switzerland, check-in 15.09, check-out 20.09, 2 guests.” Conditions: bench\_hotel\_select\_city (cityId, cityName, countryName), bench\_hotel\_select\_start\_ date, bench\_hotel\_select\_end\_date, bench\_ hotel\_select\_guests (roomsCount=1, guestsCount=2), and state\_changed to the search results.

Rail, seven conditions. Prompt: “buy a Moscow–St. Petersburg ticket for 1 passenger on the economy tarif: add it to the cart and pay.” Conditions: select\_city (from), select\_city (to), select\_date, select\_train, select\_ tariff (tarif name), basket\_add, submit\_ payment (result=success).

## B Variant Catalog

Nine UI configuration keys (eight controls and the theme) carry 33 implementations: date (6: four on the hotel site, three on the rail site, with the typedstring input shared), text search (6), guest counter (4), collection layout (4), download button (4), year selector (3), city selector (2), station selector (2), theme (2). The 87 released variants instantiate 20 profiles over 25 donor tasks; 12 are dark-theme clones, two per domain, and five document-cabinet variants change layout, year selector and button style together because these three controls share one screen.

Counts in this paper are computed directly from the released task files: a task is a variant when its test\_data carries ui\_variants and ui\_variant\_profile, and canonical otherwise; per-domain counts group by bench\_first\_domain. Counted on the release tree: 65 canonical, 87 variants, 152 total; 19 event types and 521 conditions; 33 registered implementations across nine UI configuration keys (and 33 materialised profile JSON files).

## C Interaction Classes

Each task is annotated with the interaction classes it exercises, one of them designated as primary. The taxonomy defines 19 classes. Table 5 lists only the nine primary classes present in every public submission, with the solved/total score from the top pair. NAV, RADIO, SEARCH and SEAT occur in the task files but not in ui\_classes, so they are omitted. The other six (button, free text, checkbox, filter, login, low-level DOM activity) carry no released task.

## D Related Benchmarks

Table 6 collects the comparison discussed in §2. Rows are grouped by what decides success: a stored reference, a judge model, programmatic state plus a model, or a program alone. WebPageBench belongs to the last group and difers within it, because a condition is matched against typed events emitted during the run, including intermediate steps and their parameters, rather than against the state left at the end. The two right-hand columns are where that comparison is sharpest. No other system in the table re-renders one fixed task through an alternative implementation of a control; WorkArena++ only changes colours and logos. The Agents column is not a count of models: it records how agents are attached, and where several can be run they usually arrive through one shared ecosystem, BrowserGym and AgentLab.

<table><tr><td>Donor task</td><td>Prompt (translated) and default controls</td><td>Released variants of this donor</td></tr><tr><td>Hotels hotel_search_ scenario</td><td>“In Hotels, show options: Zurich, Switzerland, check-in 15.09.2026, check-out 20.09.2026, 2 guests.&quot; Defaults: split popup calendar, autocomplete city field, per-room guest popup.</td><td>date → inline calendar, single-field range, typed string; select_city → native select; counter_guests → inline stepper, compact select, pill buttons; dark theme. (8 variants; four sibling hotel tasks with other cities and guest mixes carry the same profiles.)</td></tr><tr><td>Rail rail_book_ to_cart</td><td>“In Trains, find a Moscow-St. Petersburg train on 04.10.2026, 1 passenger, Economy tariff, and add it to the cart.&quot; Defaults: grid popup calendar, typeahead station field.</td><td>date → native &lt;input  $\mathrm { t y p e } { = } " \mathrm { d a t e } ^ { \prime \prime } { > } ,$  typed string; select_station → native select; dark theme. (4 variants.)</td></tr><tr><td>Books digital_books_ named_product_ basket</td><td>“In Books, find Quiet Amber in the Last Carriage by Vera Rudneva (text edition) and add it to the cart.&#x27; Default: standard search field. Author and title are synthetic (§3.1).</td><td>text_search → filled, outlined, pill, underlined; dark theme. (5 variants; the marketplace search tasks carry the same four search profiles.)</td></tr><tr><td>Files files_ download_pdf</td><td>“In Files, open Lab reports, choose 2024 and download lab-reports-2024.pdf.&quot;Defaults: card layout, year select, outline buttons.</td><td>one three-key profile: collections → list, years → buttons, buttons → icon; dark theme. (2 variants.)</td></tr></table>

Table 4: One donor per domain with its released variants. The prompt and the conditions of every variant are identical to the donor’s; only the named control changes. Prompts are translated from Russian; identifiers are the task file names without the profile sufix.

<table><tr><td>Class</td><td>Control</td><td>Solved</td></tr><tr><td>BASKET</td><td>add / remove / quantity</td><td>48/66</td></tr><tr><td>COUNTER</td><td>stepper, guest count</td><td>26/33</td></tr><tr><td>FILES</td><td>collections, year, download</td><td>18/18</td></tr><tr><td>FAV</td><td>favourites toggle</td><td>14/14</td></tr><tr><td>CARD</td><td>card in a grid or carousel</td><td>9/9</td></tr><tr><td>DATE</td><td>date or range picker</td><td>4/4</td></tr><tr><td>PAY</td><td>payment form</td><td>2/4</td></tr><tr><td>SELECT_AC</td><td>autocomplete list</td><td>2/2</td></tr><tr><td>SELECT_LIST</td><td>click-to-select list</td><td>2/2</td></tr></table>

Table 5: Primary classes in the public submissions. Solved is passed/total from ui\_classes for gemini-3.8-flash × openmanus (best EMS). Every submission contains these nine classes and no others. Classes with four or fewer tasks are not a ranking. NAV, RADIO, SEARCH and SEAT are absent from ui\_classes and are omitted.

<table><tr><td>Success decided by</td><td>Benchmark</td><td>Environment</td><td>Dom.</td><td>Tasks</td><td>Verification</td><td>Controlled UI variation</td><td>Agents</td></tr><tr><td rowspan="4">Reference match</td><td>Mind2Web (Deng et al., 2023)</td><td>offline, branded</td><td>31 258</td><td>2,350 214</td><td>trajectory match answer match</td><td></td><td></td></tr><tr><td>AssistantBench (Yoran et al., 2024)</td><td>live, branded</td><td></td><td></td><td></td><td></td><td>BrowserGym</td></tr><tr><td>WebVoyager (He et al., 2024)</td><td>live, branded</td><td>15</td><td>643</td><td>GPT-4V judge</td><td></td><td></td></tr><tr><td>InSTA (Trabucco et al., 2025)</td><td>live</td><td>~150K</td><td>146,441</td><td>LLM judge</td><td></td><td>Gym env</td></tr><tr><td rowspan="5">Judge model</td><td>BookingArena (Logeswaran et al., 2026)</td><td>live, branded</td><td>20</td><td>120</td><td>VLM constraint judge</td><td></td><td></td></tr><tr><td>GTA (Huang et al., 2026)</td><td>live, crawled</td><td>50+</td><td>5,600</td><td>answer + path replay</td><td></td><td></td></tr><tr><td>WebArena (Zhou et al., 2023)</td><td>hosted, de-branded</td><td>4(+3)</td><td>812</td><td>state + fuzzy LLM</td><td></td><td>BrowserGym</td></tr><tr><td>VisualWebArena (Koh et al., 2024)</td><td>hosted, de-branded</td><td>3</td><td>910</td><td>+ VQA / SSIM judge</td><td></td><td>BrowserGym</td></tr><tr><td>REAL (Garg et al., 2025)</td><td>hosted, renamed</td><td>11</td><td>112</td><td>state diff + LLM rubric</td><td></td><td>2, pluggable</td></tr><tr><td rowspan="5"></td><td>TheAgentCompany (Xu et al., 2024)</td><td>hosted, branded</td><td>4</td><td>175</td><td>scripts + LLM (29% tasks)</td><td></td><td>1 agent</td></tr><tr><td>WebShop (Yao et al., 2022)</td><td></td><td>1</td><td>12,087</td><td>attribute reward</td><td></td><td></td></tr><tr><td>WorkArena++ (Boisvert et al., 2024)</td><td>simulated, de-branded live, branded</td><td>1</td><td>682</td><td>backend final state</td><td>10 brands (theme only)</td><td>Gym env AgentLab</td></tr><tr><td>AutoWebWorld (Wu et al., 2026)</td><td>generated</td><td>29</td><td>11,663 traj.</td><td>state-machine replay</td><td></td><td></td></tr><tr><td>WebForge (Yuan et al., 2026)</td><td>generated</td><td>7</td><td>934</td><td>final-state compare</td><td></td><td></td></tr><tr><td>No environment</td><td>WebPageBench (ours)</td><td>hosted mocks, de-branded</td><td>6</td><td>65 + 87 var.</td><td>event-level match</td><td></td><td>8 controls + theme, 33 impl. 6 browser/DOM + 5 GUI eval.</td></tr></table>

Table 6: Web-agent benchmarks grouped by what decides success (first column): a reference that never inspects the environment, a judge model, programmatic state plus a model, or a program alone. Environment: live, ofline, self-hosted, simulated or generated sites and their branding; Dom.: distinct domains or sites; Agents: how agents are connected (descriptive, not a count). Task counts as reported by each paper. Among the systems surveyed, none re-renders a fixed task through an alternative implementation of a control (WorkArena++ changes colours and logos only), and multi-agent support comes through one shared ecosystem.