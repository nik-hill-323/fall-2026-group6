
# Capstone Proposal
## WM-Concord: Do World-Model Diagnostics Agree? A Cross-Diagnostic Validity Study of State-Equivalence Metrics, Inductive-Bias Probes, and Representation Probing
### Proposed by: Venkatesh Nagarjuna, Siddardha Reddy, Nikhil Obuleni, Viharika Vemparala
#### Email: s.yaraguti@gwu.edu
#### Advisor: Amir Jafari
#### The George Washington University, Washington DC  
#### Data Science Program


## 1 Objective:  

            The claim that sequence models implicitly acquire "world models" -- coherent internal
            representations of the state of the domain they were trained on -- is one of the load-bearing
            claims of the current foundation-model research program. The field has responded by inventing
            diagnostics to test it, and it now has at least four mutually incompatible families of them:
            next-token validity checks (does the model only propose legal continuations?), state-equivalence
            metrics derived from the Myhill-Nerode theorem (Vafa et al., 2024), inductive-bias probes that
            measure how a model adapts to synthetic tasks generated from a postulated world model (Vafa et
            al., 2025), and representation probing of internal activations in the Othello-GPT tradition (Li
            et al., 2023; Nanda et al., 2023; Hazineh et al., 2023). Each family has been used to support
            conclusions about whether models "have" world models. Crucially, these four families have never
            been run head-to-head on the same models in the same domains, and there is no published evidence
            that they agree with each other.

            They plausibly do not. State-equivalence metrics measure behavioral consistency across sequences
            that reach the same state; inductive-bias probes measure what survives adaptation to a new task;
            representation probes measure whether state is linearly decodable from activations. Those are
            three distinct claims about a model, and a model can pass one while failing another. If the four
            diagnostic families rank the same models differently, then the world-model evaluation literature
            has a measurement-validity problem that nobody has documented, and any paper citing one
            diagnostic to support a claim about world models is on weaker ground than it appears. If they
            agree, the community's toolkit is validated and the diagnostics can be used interchangeably --
            which is itself a result worth having in print.

            The goal of this project is to settle that question empirically, and then to go one step further:
            to determine which diagnostic has the best PREDICTIVE validity -- which one actually forecasts
            the downstream failure mode that motivated this whole literature, namely that a model with an
            incoherent world model breaks when applied to related but subtly different tasks. The entire
            study runs on a single AWS g5.2xlarge instance (1x NVIDIA A10G, 24 GB VRAM, 8 vCPUs, 32 GiB RAM)
            because the models involved are small sequence models, not frontier systems.

            Key Objectives:
            1. Reproduce the published state-equivalence results of Vafa et al. (2024) on their publicly
               released navigation model checkpoints, establishing a verified implementation of the
               compression-error and distinction-error metrics before any new claim is made.
            2. Construct a controlled model zoo: small sequence models trained on domains with a known
               ground-truth deterministic finite automaton (DFA), varying along four axes deliberately chosen
               to spread the diagnostics apart -- training-data distribution (shortest-path vs. noisy vs.
               random-walk traces), model scale, training budget, and architecture family (transformer vs.
               state-space model vs. LSTM).
            3. Implement all four diagnostic families against a common interface and run every diagnostic on
               every model in the zoo: next-token validity, Myhill-Nerode compression/distinction metrics,
               inductive-bias probes (including symbolic-regression recovery of the postulated law), and
               representation probing with control tasks and activation-patching validation.
            4. Measure cross-diagnostic concordance -- do the four families induce the same ranking over
               models? -- and characterize where and why concordance breaks down, treating disagreement as a
               phenomenon to be explained rather than noise to be averaged away.
            5. Establish predictive validity: build a downstream fragility suite of related-but-shifted tasks
               per domain and determine which diagnostic best predicts a model's failure on them, giving the
               field a defensible criterion for which instrument to trust.
            6. Extend the analysis to a domain class never probed by this literature (chemical reaction
               sequences and program-interpreter state), and package the whole pipeline as an open-source,
               config-driven repository (WM-Concord) with a reusable diagnostic interface.
            

![Figure 1: Example figure](2026_Fall_10.png)
*Figure 1: Caption*

## 2 Dataset:  

            All resources below are publicly available for research use with no restricted access, no IRB,
            and no data-sharing agreement. The project organizes them into four tiers: reproduction
            artifacts, the domain generators used to build the model zoo, the downstream fragility suite, and
            a held-out generalization tier.

            TIER 1 -- REPRODUCTION ARTIFACTS (published data, checkpoints, and reference code):
            1. Vafa et al. (2024) released data and model checkpoints for the navigation domain -- the
               shortest-paths, noisy-shortest-paths, and random-walks splits -- plus the reference
               implementation of the compression and distinction metrics:
               https://github.com/keyonvafa/world-model-evaluation
               These are used in Weeks 1-2 purely to verify this project's re-implementation against
               published numbers before any new claim is made.
            2. Othello-GPT community game corpora and reference checkpoints, the standard testbed for
               representation probing of world models (Li et al., 2023; Nanda et al., 2023):
               https://github.com/likenneth/othello_world

            TIER 2 -- DOMAIN GENERATORS FOR THE MODEL ZOO (ground-truth world model known by construction):
            3. Navigation / street networks: OpenStreetMap street graphs via OSMnx, from which shortest-path
               and random-walk token sequences are generated with a known transition structure:
               https://github.com/gboeing/osmnx
            4. Lattice traversal: synthetic agent movement over a finite line segment and 2-D grid, the
               spatial-structure setting used in both Vafa papers -- generated in-house, no download required
            5. Board games with exactly specified state: Othello (Tier 1 corpus) and cumulative Connect-4,
               the worked example from Vafa et al. (2024) whose Myhill-Nerode interior is enormous and whose
               state-pooling failure mode is therefore easy to detect -- generated in-house
            6. Program-interpreter state (NEW DOMAIN): token sequences from a small deterministic stack
               machine / register machine whose full state is known at every step, giving a DFA of tunable
               size and a domain this literature has never probed -- generated in-house
            7. Chemical reaction sequences (NEW DOMAIN): the Open Reaction Database and the USPTO-50k
               reaction corpus, used to build reaction-step sequences over a rule-defined state space
               (available reagents / functional groups present), the "chemistry" case that Vafa et al. (2024)
               name as within scope but do not test:
               https://open-reaction-database.org/  and  https://github.com/connorcoley/rexgen_direct

            TIER 3 -- DOWNSTREAM FRAGILITY SUITE (the predictive-validity criterion, H3):
            8. Per-domain held-out task variants that are structurally related but subtly different from the
               training task -- detour-constrained routing on the navigation graphs, altered legal-move rules
               on the game domains, modified instruction semantics on the interpreter domain, and shifted
               reagent-availability constraints on the chemistry domain. Each variant is specified as an
               explicit DFA modification so that "correct behavior" is unambiguous, and each is constructed
               in Weeks 3-4 and frozen before any diagnostic is run, so no diagnostic can be tuned against it

            TIER 4 -- HELD-OUT GENERALIZATION (evaluation only, not used for development):
            9. One architecture family and one domain held out of Weeks 3-12 entirely, used in Week 14 only,
               to test whether the concordance and predictive-validity findings generalize beyond the models
               and domains used to develop the analysis

            DATASET / PIPELINE PREPARATION:
            - Every domain is exposed behind a common generator interface emitting (token sequence, ground-
              truth DFA state, legal next-token set) triples, so all four diagnostics consume identical
              ground truth and no diagnostic gets a privileged view of the domain
            - Every DFA is specified in a machine-readable schema and version-controlled, with state-space
              size, alphabet size, and mean Myhill-Nerode boundary length recorded per domain, so the
              difficulty of each domain is a measured quantity rather than an assumption
            - A run_matrix.csv enumerating every (domain x architecture x scale x training-distribution x
              seed) model and every (model x diagnostic) evaluation is fixed in Week 1, bounding the total
              experiment count before any compute is spent
            - All corpora are downloaded once in Week 1 and cached locally; every other domain is generated
              in-house from seeded code, so the pipeline is fully reproducible with no external dependency
            

## 3 Rationale:  

            Whether sequence models learn world models is not a side question in contemporary machine
            learning -- it is the premise underlying claims that scaling prediction produces understanding.
            The literature testing that premise has developed quickly and, importantly, has converged on a
            recurring finding: models look better under weak diagnostics than under strong ones.

            Vafa et al. (2024, NeurIPS) formalized world-model recovery for domains governed by a
            deterministic finite automaton and derived two metrics from the Myhill-Nerode theorem -- a
            compression error, when a model fails to treat two sequences reaching the same state as
            equivalent, and a distinction error, when it fails to separate genuinely different states. Their
            central result is that models performed well on existing world-model diagnostics while the new
            metrics revealed their world models to be considerably less coherent than they appeared, and that
            this incoherence produces fragility: the model breaks when used for related but subtly different
            tasks. Their worked cumulative-Connect-4 example makes the mechanism vivid -- two boards with
            identical legal-move sets can be separated only by a length-4 sequence, while the set of
            sequences that fail to separate them is astronomically large, so next-token evaluation cannot see
            the difference.

            Vafa et al. (2025, ICML) then introduced an entirely different instrument: the inductive-bias
            probe, which adapts a model to synthetic datasets generated from a postulated world model and
            asks whether the model's inductive bias aligns with that world model. The finding again is
            negative -- models can excel at their training task and still fail to develop inductive biases
            toward the underlying world model, and models trained on orbital trajectories failed to apply
            Newtonian mechanics on new physics tasks, with symbolic regression recovering a force law that
            makes no physical sense. Independently, the representation-probing tradition around Othello-GPT
            (Li et al., 2023; Nanda et al., 2023; Hazineh et al., 2023) argues from linear decodability of
            board state, and Kang et al. (2024) report that video generation models perform case-based
            mimicry rather than learning generalizable physical laws even as they scale.

            Four instruments, four sets of conclusions, and no head-to-head comparison. This is a
            measurement-validity gap, and it is the kind of gap a careful four-person student team can
            actually close, because closing it requires breadth of implementation rather than frontier
            compute.

            The gap matters for three reasons. First, if the diagnostics disagree, then published claims
            resting on any single one of them are weaker than they appear, and the field needs to know which
            instrument to prefer -- a question this project answers via the downstream-fragility criterion
            (H3) rather than by argument. Second, representation probing in particular has a known
            confounding problem: a sufficiently expressive probe can recover structure the model does not
            actually use, which is why control tasks (Hewitt & Liang, 2019) and intervention-based validation
            are necessary and are built into this design. Third, the diagnostics have never been applied
            outside a narrow set of domains -- game playing, logic puzzles, navigation, lattices, and orbital
            mechanics -- so whether their conclusions are properties of models or artifacts of those specific
            domains is untested.

            WHY THIS PROJECT IS TIMELY AND PUBLISHABLE:
            - World models are among the most active research threads of 2026, with dedicated workshops at
              NeurIPS, ICML, and ICLR, and with a rapidly growing evaluation literature that has not yet
              turned its attention on itself.
            - Deliberately, this project proposes NO new diagnostic. The video-world-model benchmark space is
              saturated -- WorldModelBench, PhyWorldBench, WorldSimBench, WorldScore, WBench, CoW-Bench,
              T2VWorldBench, VideoPhy, PhyGenBench, and others have appeared within roughly eighteen months,
              and adding another leaderboard would be immediately derivative. Measurement validity ACROSS
              existing diagnostics is the uncrowded question, and a concordance study is a contribution that
              survives the release of the next diagnostic rather than being superseded by it.
            - Every outcome is a reportable finding. Disagreement is a warning to the field; agreement
              validates the toolkit and licenses cheaper diagnostics as proxies for expensive ones. The
              analysis pipeline is designed from Week 1 to report either, so the project has no failure
              condition, only different conclusions.
            - Compute risk is close to zero, which is unusual and is a genuine selling point for a
              single-GPU, fixed-deadline student project. The models involved are small sequence models on
              the scale of Othello-GPT; published checkpoints exist for the reproduction arm; and the
              expensive parts of the work are implementation breadth and statistical care, not GPU hours.
            - The project produces a reusable artifact: a common interface behind which any future
              world-model diagnostic can be dropped in and compared against the existing four on identical
              ground truth.
            

## 4 Approach:  

            PHASE 1: REPRODUCTION AND INFRASTRUCTURE (Weeks 1-2)

            [Week 1: Environment, Artifacts, and the Run Matrix]
            - Install PyTorch, TransformerLens (activation hooks and probing), PySR (symbolic regression),
              scikit-learn, automata tooling, RDKit (chemistry domain), and OSMnx (street graphs)
            - Download Tier 1 artifacts: the Vafa et al. (2024) navigation data and checkpoints, and the
              Othello-GPT corpora and reference checkpoints
            - Project structure: domains/, zoo/, diagnostics/, fragility/, analysis/, notebooks/
            - Fix run_matrix.csv enumerating every planned model and every planned (model x diagnostic)
              evaluation, so the experiment count is bounded and auditable before any compute is spent

            [Week 2: Reproduction Gate]
            - Re-implement the compression-error and distinction-error metrics behind the project's common
              diagnostic interface and reproduce the published navigation results on the released checkpoints
            - Reproduce a published linear-probe result on the reference Othello-GPT checkpoint as a second
              independent implementation check
            - STAGE GATE: if published numbers cannot be reproduced within a documented tolerance, the
              implementation is wrong and must be fixed before the model zoo is built. No new claim is made
              on top of an unverified implementation.


            PHASE 2: DOMAIN GENERATORS AND THE MODEL ZOO (Weeks 3-5)

            [Week 3: Domain Generators and DFA Specifications]
            - Implement the common generator interface for all six domains (navigation, lattice, Othello,
              cumulative Connect-4, interpreter state, chemical reaction sequences), each emitting (sequence,
              ground-truth state, legal next-token set)
            - Record per-domain difficulty statistics: state-space size, alphabet size, and estimated mean
              Myhill-Nerode boundary length, so cross-domain comparisons later are anchored to measured
              domain complexity rather than intuition

            [Week 4: Fragility Suite Construction and Freeze]
            - Build the Tier 3 downstream fragility suite as explicit DFA modifications per domain, and FREEZE
              it before any diagnostic is implemented, so that no diagnostic can be tuned against the
              criterion it will later be judged by
            - Pre-register the concordance and predictive-validity analyses (metrics, tests, thresholds) in
              the repository, so the Week 11-14 results are confirmatory rather than exploratory

            [Week 5: Model Zoo Training]
            - Train the controlled model zoo, varying four axes: training-data distribution (shortest-path,
              noisy, random-walk), model scale (three sizes), architecture family (transformer, state-space
              model, LSTM), and seed
            - These are small sequence models trained on synthetic token streams; the zoo is sized in Week 1
              against measured single-A10G throughput and capped so that the full zoo trains within the week
            - Validate every model on held-out next-token accuracy before it is admitted to the diagnostic
              sweep, and record the training curve and final loss per model in a zoo_registry.csv


            PHASE 3: THE FOUR DIAGNOSTIC FAMILIES (Weeks 6-10)

            [Week 6: Diagnostic 1 -- Next-Token Validity Baseline]
            - Implement the standard weak diagnostic: fraction of proposed continuations that are legal for
              the true underlying state, per domain and per model
            - This is the baseline the literature has repeatedly shown to be too permissive, and it is
              included precisely so the project can quantify by how much

            [Week 7: Diagnostic 2 -- Myhill-Nerode State-Equivalence Metrics]
            - Run compression-error and distinction-error metrics across the full zoo
            - Because the Myhill-Nerode interior is combinatorially enormous in most domains, both metrics are
              computed as sampling estimators with reported variance and explicit sample-size justification,
              rather than exhaustively -- the estimator design is itself documented as a reusable contribution

            [Week 8: Diagnostic 3 -- Inductive-Bias Probes]
            - Adapt each model to synthetic datasets generated from the postulated world model of its domain
              and measure whether its inductive bias aligns with that world model
            - Include the symbolic-regression arm: recover the law the adapted model has effectively
              implemented and compare it against the true generating rule, with multiple symbolic-regression
              seeds and reported stability, since symbolic regression is stochastic

            [Week 9: Diagnostic 4 -- Representation Probing]
            - Train linear and nonlinear probes to decode ground-truth state from internal activations at
              every layer, across the zoo
            - Pair every probe with a control task (Hewitt & Liang, 2019) so that probe expressivity is
              separated from genuine encoding, and report probe selectivity rather than raw accuracy

            [Week 10: Intervention-Based Validation of Probing]
            - Validate the probing results causally: patch or steer the decoded state direction and test
              whether the model's subsequent predictions change as the decoded state would imply
            - Decodability without causal relevance is recorded explicitly as a distinct outcome, since it is
              the central confound in the representation-probing tradition


            PHASE 4: CONCORDANCE AND PREDICTIVE VALIDITY (Weeks 11-14)

            [Week 11: Cross-Diagnostic Concordance -- H1]
            - Assemble the master table: (domain x architecture x scale x training-distribution x seed) ->
              four diagnostic scores
            - Measure rank concordance between every pair of diagnostics per domain (Kendall's tau, Spearman)
              and overall (Kendall's W), with bootstrapped confidence intervals, directly resolving H1

            [Week 12: Explaining Disagreement -- H2]
            - Characterize where concordance breaks down as a function of domain complexity (state-space
              size, mean boundary length), architecture family, model scale, and training distribution
            - Test whether disagreement is systematic and attributable, or consistent with estimator noise --
              the difference between "the diagnostics measure different constructs" and "the diagnostics are
              noisy", directly resolving H2

            [Week 13: Predictive Validity -- H3]
            - Evaluate every model on the frozen Tier 3 fragility suite and regress downstream failure on
              each diagnostic score, with model and domain effects, to determine which diagnostic best
              predicts real breakdown on related-but-shifted tasks
            - This is the arm that converts "the diagnostics disagree" into "here is the one to trust", and
              it is the project's most consequential result, resolving H3

            [Week 14: New Domains and Held-Out Check -- H4]
            - Analyze the interpreter-state and chemical-reaction domains, which this literature has never
              probed, and test whether the Weeks 11-13 conclusions hold there
            - Evaluate the Tier 4 held-out architecture family and domain, kept out of all prior phases
            - Formally state the project's four central hypotheses being tested against the master table:
              H1: Do the four diagnostic families induce concordant rankings over models, or does concordance
                  break down -- and between which specific pairs of diagnostics?
              H2: Is any disagreement systematic and attributable to identifiable model and domain properties
                  (indicating the diagnostics measure separable constructs), or is it consistent with
                  estimator noise?
              H3: Which diagnostic best predicts downstream failure on related-but-shifted tasks -- the
                  fragility that motivates this literature -- and does the weak next-token baseline predict
                  it at all?
              H4: Do the concordance and predictive-validity findings generalize to domain classes never
                  probed by this literature, and to a held-out architecture family?


            PHASE 5: PAPER AND CODE RELEASE (Weeks 15-16)

            [Week 15: Research Paper Draft]
            Paper structure (8-10 pages, NeurIPS/ICML/ICLR world-model or interpretability workshop, or TMLR
            format; Neural Computing & Applications as the journal target):
            1. Abstract: four diagnostics, one model zoo, concordance and predictive-validity findings
            2. Introduction: the world-model claim and the proliferation of instruments to test it
            3. Related Work: state-equivalence metrics (Vafa et al., 2024), inductive-bias probes (Vafa et
               al., 2025), representation probing (Li et al., 2023; Nanda et al., 2023; Hazineh et al.,
               2023), probing confounds (Hewitt & Liang, 2019), and the video-world-model benchmark
               literature that this project deliberately does not extend
            4. WM-Concord: domains and DFA specifications, the model zoo, the four diagnostics behind a
               common interface, the fragility suite
            5. Results: pairwise and overall concordance
            6. Where the Diagnostics Disagree, and Why
            7. Predictive Validity: which diagnostic forecasts downstream fragility
            8. New Domains: interpreter state and chemical reaction sequences
            9. Practical Guidance: which diagnostic to use, when, and what each one licenses you to claim
            10. Conclusion & Future Work: extension to continuous-state and video world models, and to
                non-deterministic (probabilistic automaton) settings

            [Week 16: Code Release & Documentation]
            - Publish the WM-Concord repository: config-driven
              run_diagnostics.py --domain all --model all --diagnostic all
            - A documented plug-in interface so any future world-model diagnostic can be added and compared
              against the existing four on identical ground truth
            - Jupyter notebooks: one per diagnostic and one concordance-analysis notebook
            - README with quickstart, single-GPU reproduction guide, and the pre-registered analysis plan
            - requirements.txt with pinned dependencies; final presentation
            

## 5 Timeline:  

            Week 1:    Environment, Tier 1 artifacts downloaded, run_matrix.csv fixed
            Week 2:    Published Vafa (2024) and Othello-GPT probe results reproduced -- REPRODUCTION GATE
            Week 3:    Six domain generators implemented; per-domain DFA specs and difficulty stats recorded
            Week 4:    Downstream fragility suite built and FROZEN; analysis plan pre-registered
            Week 5:    Model zoo trained across distribution x scale x architecture x seed; zoo_registry.csv
            Week 6:    Diagnostic 1: next-token validity baseline across the zoo
            Week 7:    Diagnostic 2: Myhill-Nerode compression/distinction metrics with sampling estimators
            Week 8:    Diagnostic 3: inductive-bias probes plus symbolic-regression law recovery
            Week 9:    Diagnostic 4: layer-wise linear/nonlinear probing with control tasks
            Week 10:   Intervention-based causal validation of the probing results
            Week 11:   Cross-diagnostic concordance analysis with bootstrapped CIs -- H1
            Week 12:   Attribution of disagreement to domain/architecture/scale properties -- H2
            Week 13:   Fragility-suite evaluation; which diagnostic predicts downstream failure -- H3
            Week 14:   New-domain analysis (interpreter, chemistry) and Tier 4 held-out check -- H4
            Week 15:   Research paper draft
            Week 16:   WM-Concord code release, plug-in diagnostic interface, notebooks, final presentation

            TOTAL: 16 weeks (one semester)

            KEY MILESTONES:
            - Week 2:  Implementation verified against published results before any new claim is made
            - Week 4:  Fragility criterion frozen and analysis pre-registered, ruling out post-hoc tuning
            - Week 5:  Model zoo complete and validated
            - Week 10: All four diagnostics run across the full zoo
            - Week 11: Concordance resolved (H1)
            - Week 13: Predictive validity resolved (H3) -- the project's headline result
            - Week 14: All hypotheses (H1-H4) resolved; new domains and held-out check complete
            - Week 16: Paper submitted; WM-Concord released

            STAGE GATES (decision points that change the plan, not status meetings):
            - Week 2:  If published results cannot be reproduced, halt and fix the implementation before
                       building the zoo
            - Week 5:  If the full model zoo will not train within the week on one A10G, cut the scale axis
                       first (keep distribution, architecture, and seed), since scale is the least
                       informative axis for separating diagnostics
            - Week 7:  If the Myhill-Nerode sampling estimators have variance too large to rank models,
                       reduce the number of domains rather than the number of diagnostics -- the
                       cross-diagnostic comparison is the contribution and cannot be narrowed
            - Week 11: If all four diagnostics agree closely, report the null and shift the paper's center of
                       mass to predictive validity (H3) and the new domains (H4), which remain contributions
                       regardless
            - Week 13: Freeze the master table; no new models or diagnostics admitted after this point

            DELIVERABLES BY WEEK 16:
            - A cross-diagnostic validity study spanning four diagnostic families, six domains, three
              architecture families, and multiple scales, seeds, and training distributions
            - Quantitative answers to whether world-model diagnostics agree, where they diverge, which one
              predicts downstream fragility, and whether any of this generalizes to new domain classes
            - A frozen, pre-registered downstream fragility suite usable as a criterion by future work
            - Sampling estimators for Myhill-Nerode metrics with documented variance behavior
            - Research paper draft (8-10 pages)
            - Open-source WM-Concord repository with a plug-in interface for future diagnostics
            

## 6 Expected Number Students:  

            RECOMMENDED: 4 students

            The four diagnostic families are genuinely independent implementation efforts sharing only the
            domain interface and the master table. Each requires different expertise -- automata theory and
            estimator design, adaptation experiments and symbolic regression, probing and causal
            interventions, systems and statistics -- and each can proceed in parallel once Phase 2 delivers
            the domains and the zoo. This is the structural reason the project needs four people rather than
            benefiting from them.

            ROLE DISTRIBUTION FOR 4 STUDENTS:

            Student 1: State-Equivalence Metrics & Automata Infrastructure
            - Responsibilities: DFA specifications for all six domains, the Myhill-Nerode compression and
              distinction metrics, the sampling-estimator design and its variance analysis, per-domain
              difficulty statistics, Diagnostic 1 (next-token validity baseline)
            - Skills: automata theory, algorithms, sampling and estimation, Python

            Student 2: Inductive-Bias Probes & Symbolic Regression
            - Responsibilities: the adaptation-based probe pipeline, synthetic task generation from postulated
              world models, symbolic-regression law recovery and its stability analysis, comparison of
              recovered laws against ground-truth generating rules
            - Skills: transfer learning / fine-tuning, symbolic regression (PySR), experiment design, Python

            Student 3: Representation Probing & Causal Interventions
            - Responsibilities: layer-wise linear and nonlinear probes, control-task selectivity, activation
              patching and steering for causal validation, the decodability-versus-causal-relevance analysis
            - Skills: transformer internals, TransformerLens, mechanistic interpretability methods, Python

            Student 4: Model Zoo, Domains, and Concordance Analysis
            - Responsibilities: the common domain generator interface, training and validating the full model
              zoo across all four axes, the run orchestrator and run_matrix.csv tracker, the frozen fragility
              suite, the concordance and predictive-validity statistics, all publication figures, and the
              WM-Concord repository and plug-in interface
            - Skills: PyTorch training pipelines, experiment orchestration, statistical testing, software
              engineering

            SHARED RESPONSIBILITIES (all four students):
            - Week 2 reproduction gate and Week 4 pre-registration, both of which require agreement across
              all four diagnostics before the project proceeds
            - Interpretation of disagreement, practical-guidance synthesis, paper writing, code
              documentation, final presentation
            - Weekly integration meetings: all four diagnostics consume the identical domain interface and
              write into the identical master table, so interface changes require joint sign-off
            

## 7 Possible Issues:  

            TECHNICAL CHALLENGES AND SOLUTIONS:

            1. The Diagnostics May Simply Agree (Null Result):
            - ISSUE: H1 could resolve in the boring direction -- all four instruments rank models
              consistently, leaving no measurement-validity problem to report
            - SOLUTION: This is the Week 11 stage gate, not a failure. Agreement is a positive validation of
              the field's toolkit and licenses cheap diagnostics as proxies for expensive ones, which is
              directly useful. The paper narrative is designed from Week 1 to accommodate either outcome, and
              predictive validity (H3) and the new domains (H4) remain contributions regardless of how H1
              resolves

            2. Myhill-Nerode Metrics Are Combinatorially Expensive:
            - ISSUE: The set of sequences that fail to distinguish two states can be astronomically large --
              Vafa et al.'s cumulative Connect-4 example has an interior on the order of 10^27 sequences --
              so exhaustive computation is impossible in most domains
            - SOLUTION: Both metrics are implemented as sampling estimators from the outset, with reported
              variance, explicit sample-size justification, and a Week 7 gate: if estimator variance is too
              large to rank models, the number of DOMAINS is cut rather than the number of diagnostics, since
              the cross-diagnostic comparison is the contribution

            3. Representation Probing Is Confounded by Probe Capacity:
            - ISSUE: An expressive probe can decode structure the model does not actually use, inflating
              apparent world-model quality and corrupting the concordance analysis
            - SOLUTION: Every probe is paired with a control task and reported as selectivity rather than raw
              accuracy, and Week 10 adds causal validation via activation patching and steering.
              Decodability without causal relevance is recorded as its own outcome rather than being
              collapsed into a single score

            4. Symbolic Regression Is Stochastic and Can Be Unstable:
            - ISSUE: The recovered law can vary across runs, making the inductive-bias probe's headline
              output unreliable
            - SOLUTION: Multiple symbolic-regression seeds per model with reported recovery stability, and the
              probe's primary score is the adaptation-based measure, with symbolic recovery reported as a
              qualitative, stability-annotated companion result rather than a load-bearing number

            5. Comparing Diagnostics Requires a Criterion, and the Criterion Could Be Gamed:
            - ISSUE: If the downstream fragility suite were built after the diagnostics, it could be
              unconsciously shaped to favor one of them
            - SOLUTION: The suite is constructed in Week 4 and FROZEN before any diagnostic is implemented,
              and the concordance and predictive-validity analyses are pre-registered in the repository in
              the same week. This ordering is a design commitment, not a convention

            6. DFA Specification for the New Domains May Be Ambiguous:
            - ISSUE: Chemical reaction sequences and interpreter state do not come with a canonical automaton;
              a poorly chosen state abstraction would make results uninterpretable
            - SOLUTION: The state abstraction for each new domain is specified explicitly, version-controlled,
              and justified in Week 3, with a deliberately simple and fully deterministic interpreter as the
              safest of the two new domains; if the chemistry abstraction proves contestable, it is reported
              as an exploratory arm rather than as a load-bearing result

            7. Model Zoo Size Can Balloon:
            - ISSUE: Four varying axes plus seeds, times four diagnostics, times six domains, could exceed a
              single-GPU semester even though individual models are small
            - SOLUTION: run_matrix.csv fixes the total in Week 1 against measured A10G throughput; the Week 5
              gate cuts the scale axis first if needed; and the Week 13 freeze prevents late additions

            8. Scooping Risk From an Active Research Group:
            - ISSUE: The authors of both primary diagnostics are active and could publish a comparison
              themselves, and the broader world-model evaluation literature is moving monthly
            - SOLUTION: The project's contribution is deliberately positioned as measurement validity across
              EXISTING instruments plus predictive validity against a frozen external criterion, which is a
              different object from proposing a fifth diagnostic; the literature is re-checked at Weeks 1, 8,
              and 15, and the new-domain and causal-validation arms provide differentiation even if a partial
              comparison appears mid-semester

            9. Library / Version Drift:
            - ISSUE: TransformerLens, PySR, and RDKit are all actively maintained, and an API change mid-
              semester could invalidate earlier diagnostic runs
            - SOLUTION: Pin every dependency in requirements.txt from Week 1 and record the exact library
              versions alongside every diagnostic score in the master table

            RISK MITIGATION TIMELINE:
            - Weeks 1-2:  Reproduce published results for two independent diagnostics before building
                          anything new; halt on failure
            - Weeks 3-4:  Freeze the fragility criterion and pre-register the analyses before the instruments
                          exist
            - Weeks 5-10: Monitor estimator variance and probe selectivity as each diagnostic comes online;
                          checkpoint the master table after every diagnostic
            - Weeks 11-13: Trigger the Week 11 null gate if the diagnostics agree; freeze the master table at
                          the end of Week 13
            - Weeks 14-16: Cross-check new-domain and held-out results against the main findings; 3-day code
                          freeze for README, version manifest, and notebook review before release
            

## Contact
- Author: Amir Jafari
- Email: [ajafari@gwu.edu](mailto:ajafari@gwu.edu)
- GitHub: [amir-jafari/Capstone](https://github.com/amir-jafari/Capstone)
