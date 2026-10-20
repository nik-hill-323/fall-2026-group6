Note: Use Markdown Cheat Sheet download in the directory as needed.

- Useful links
  - https://github.com/im-luka/markdown-cheatsheet

---
## Date: Week 1 - September 1, 2026
- Topics of discussion
    - Team formed: Nikhil, Siddu, Venkatesh, Viharika
    - First proposals (LongAgent Bench, SpecComp Bench) submitted as petitions
    - Advisor feedback: proposals too easy for four people over one semester

- Action Items:

* [x] Submit petition proposals
* [x] Collect advisor feedback
* [x] Search for a harder project on transformer internals
---

## Date: Week 2 - September 8, 2026
- Topics of discussion
    - Considered SAE Faithful (proposal 7), already taken by another group
    - Settled on WM Concord: do four world model diagnostics agree?
    - Proposal written and submitted as proposal 10

- Action Items:

* [x] Write and submit WM Concord proposal
* [x] Pick the four diagnostics (D1 to D4) and hypotheses (H1 to H4)
* [x] Collect core papers (Li 2023, Vafa 2024, Vafa 2025, Yuan and Sogaard 2025)
---
## Date: Week 3 - September 15, 2026
- Topics of discussion
    - Repository created with the required folder template and main branch ruleset
    - AWS g5.2xlarge access and environment setup
    - Project board, first milestone, and first issues created

- Action Items:

* [x] Create repo, ruleset, milestone, and project board
* [x] Pin requirements.txt on the A10G instance (PR #3)
* [x] Literature notes for 4 core papers (PR #10, closes #1)
* [ ] Draft run_matrix.csv (#4)
* [ ] Download Othello and Vafa 2024 artifacts (#2)
---
## Date: Week 4 - September 22, 2026
- Topics of discussion
    - Instructor Review #1 (#5): no code yet, about 3 weeks behind, two members without commits
    - New requirements: vector figures only, draw.io or SVG diagrams, LaTeX report with 10+ cited works by October 6
    - Plan: Othello first, with diagnostic D4 (linear probe) end to end before the October 20 presentation

- Action Items:

* [x] Give instructor Write access to the project board
* [x] Re date Week 3 milestone, add Preliminary and Final milestones
* [x] One owner per issue
* [x] Merge requirements.txt and references.bib
* [x] Move proposal to reports/proposal.md, rewrite README
* [x] Environment built on the g5.2xlarge, Vafa 2024 and Othello artifacts downloaded (#2 closed)
* [x] D1 explainer and diagram (PR #21), D4 explainer, diagram and toy probe (PR #16, PR #18)
* [x] D3 toy probe and diagram opened for review (PR #23)
* [x] Run matrix fixed at 3 domains, 243 models, 972 runs (PR #19, in review)
* [x] Week 3 milestone closed
* [ ] Start LaTeX report skeleton with literature review section (moved to Week 5)
* [ ] Add pytest and a GitHub Action (moved to Week 5)
* [ ] Every team member lands at least one reviewed pull request this week (3 of 4)
* [ ] Othello model zoo and D4 probe running end to end on the g5.2xlarge (moved to Week 5)
---
## Date: Week 5 - September 29, 2026
- Topics of discussion
    - Instructor Review #2 (#22): process caught up, still about 3 weeks behind the plan
    - Top priority: reproduction gate, one published number reproduced before building on it
    - Scope fixed in the proposal: 3 domains (navigation, Othello, interpreter), 15 weeks

- Action Items:

* [x] Diagram naming standard, one bib file, proposal fit to 15 weeks and 3 domains (PR #25, closes #24)
* [x] Reproduction gate: Othello GPT linear probe within 1 point of Nanda et al. 2023 on 5 of 6 values (PR #31)
* [x] Throughput measured for all 3 architectures and 3 scales, Mamba via mambapy (PR #19)
* [x] D4 toy probe: Hewitt and Liang control task, layer 0 baseline, Othello rules test (PR #29)
* [x] LaTeX report skeleton in Sample_Report.tex (PR #36)
* [x] pytest and a GitHub Action (PR #33)
* [x] Shared Othello domain module (PR #38) and three Othello distributions with WTHOR championship games (PR #40)
* [ ] D2 explainer (#7) (moved to Week 6)
* [ ] Literature review with 10+ works, 4 of 10 written (moved to Week 6, due October 6)
---
## Date: Week 6 - October 6, 2026
- Topics of discussion
    - Instructor Review #3 (#41): about 2 weeks behind (was 3), trend up; reproduction gate accepted
    - Scope decision written down before any model is scored: full 243 model matrix with Mamba large, Othello first (81 models, D1 and D4 by October 20), navigation and the interpreter after October 20, interpreter held out for H4, no architecture held out (PR #47)
    - Fragility suite v1 for Othello: rules kept fixed, situations shifted; frozen with tag fragility-v1 before any scoring (PR #43)

- Action Items:

* [x] Literature review to 10+ works and 6 new references.bib entries (PR #51)
* [x] First zoo models: one transformer, LSTM and Mamba on synthetic Othello with legal move rate (PR #47; LSTM retrained with the stacked LSTM in PR #61)
* [x] Freeze and tag the fragility suite (PR #43, tag fragility-v1)
* [x] Results schema, common diagnostic interface, model registry, activation hooks for all 3 architectures (PRs #53, #55, #57)
* [x] Preregistration of the H1 to H4 analysis (PR #59, tag prereg-v1)
* [x] Othello zoo training split across the 4 instances: launcher and slice list (PR #69); transformer small and Mamba small retrained with checkpoint hashes (PR #67)
* [x] Occupancy only baseline for the black/white probe: added to the D4 plan in the preregistration (PR #59), computed with the D4 run
* [x] D1 scoring protocol (PR #62) and Othello data and splits in the report (PR #66)
* [ ] D2 explainer (#7) (moved to Week 7, now owned by Siddu)
---
## Date: Week 7 - October 13, 2026
- Topics of discussion
    - Instructor Review #4 (#70): on schedule against the revised plan (was about 2 weeks behind); Review #3 items done
    - Othello zoo: slice 1 finished on October 7 (19 of 19, 0 failed); 23 of 81 models trained, all admitted (legal move rate 0.56 to 0.97); slices 2, 3 and 4 running on the other three instances
    - Plan for October 20: all four diagnostics (D1, D2, D3, D4) on every admitted Othello model, a four way concordance table, and a Streamlit demo of the project (asked for by Prof. Jafari)
    - Competence control (Review #4): diagnostics may agree only because better trained models score higher on all of them, so the preregistration is amended before any score

- Action Items:

* [x] Preregistration amendment prereg-v1.1: H1 also controlled for legal move rate, H3 fragility relative to each model's own baseline, D2 protocol (PR #72, tag prereg-v1.1)
* [x] D2 explainer, diagram and module (PR #73, closes #7)
* [x] D1, D3 and D4 modules with tests and anchor scores on both Othello GPTs (PRs #89, #85, #79)
* [x] Amendment 2 (prereg-v1.2): D4 nonlinear probe and causal check, D3 symbolic regression arm, D2 sample size justification, all fixed before any zoo score (PRs #93, #94, #95, tag prereg-v1.2)
* [x] Review #4 minor fixes: zip(strict=True), LSTM row in zoo_first_models.md (PR #77)
* [x] Streamlit demo skeleton in demo/ (PR #83)
* [ ] Checkpoints shared through S3 (s3://dats-capstone/shared/group6/zoo/ set up and tested; uploads in progress) (moved to Week 8)
* [ ] Slices 1 to 3 committed (PRs #75, #81, #87, 61 of 81 models, all admitted); slice 4 trained and committed; zoo_registry.csv rebuilt with admission per architecture and scale (moved to Week 8)
* [ ] D1, D2, D3 and D4 scored on every admitted Othello model, master table built (moved to Week 8)
* [ ] Four way concordance table (Kendall tau raw and partial, Kendall W, bootstrap intervals) (moved to Week 8)
* [ ] Streamlit demo on the real results (moved to Week 8)
* [ ] Methods section in Sample_Report.tex (moved to Week 8)
* [ ] Slides for the preliminary presentation in presentation/ (moved to Week 8)
* [x] Proposal timeline updated to the October 20 plan (PR #99, closes #98)
----
## Date: Week 8 - October 20, 2026
- Topics of discussion
    - Instructor Review #5 (2026-10-19): on schedule against the revised plan; every Review #4 item fixed; all four members committed. Blocker: the bootstrap treated the 81 models as independent, but they are 27 configurations x 3 seeds
    - Othello zoo complete: 81 of 81 trained and admitted (competence 0.559 to 0.969); all 81 scored by D1 to D4 and fragility v1 and v2 (486 results, 0 missing) with the hash checked, resumable scoring launcher
    - Fragility v2 (Amendment 3, before scoring): detours played to the end of the game, after pass positions matched on move number and number of legal moves
    - Amendment 4 (prereg-v1.4) after Review #5: configuration bootstrap, H3 also without the test model, fragility distribution before H3. Written after the zoo was scored but before any H1 to H3 statistic; it changes no score
    - Preregistered results on Othello: H1, the diagnostics agree (Kendall W 0.87) but through competence (W 0.27 with competence removed; D1 not interpretable under A5); H2, disagreement consistent with noise; H3, only D4 adds to D1, no single winner
    - Exploratory (not preregistered): the D4 signal is a training data effect. Championship trained models are both the most fragile and the least probe readable; within each training distribution the link disappears
    - Preliminary Presentation on October 20: slides, results and demo

- Action Items:

* [x] Zoo registry and admission table with all 81 models (PR #108, closes #101)
* [x] Scoring launcher score_slice.sh: D1 to D4 and fragility per slice, hash checked, resumable (PR #103, closes #102)
* [x] Amendment 3 (prereg-v1.3, fragility-v2) and analysis details E4 (PR #107, closes #104)
* [x] Analysis code for H1 to H3 with tests on fake zoos; CI installs scipy and matplotlib (PR #109, closes #105)
* [x] All 81 Othello models scored, one slice per member (PRs #112, #115, #116, #120)
* [x] Amendment 4 (prereg-v1.4): configuration bootstrap (PR #119, closes #118)
* [x] Master table, preregistered H1 to H3 analysis and exploratory analyses with a plain language summary in results/analysis/othello/RESULTS.md (PR #122, closes #121)
* [x] Report: Methods, Othello results, discussion and limitations in Sample_Report.tex; 12 cited works in the literature review; vector figures made by reports/Latex_report/make_figures.py
* [x] Paper draft in research_paper/Latex/Paper_Temp.tex that builds
* [ ] Slides for the preliminary presentation in presentation/ (pptx and pdf)
* [ ] Streamlit demo on the real results
* [ ] Reply to Instructor Review #5 item by item
* [ ] Checkpoints shared through S3 (uploads in progress)
---

## Date: Week 9 - Month Day Year 
- Topics of discussion
    - Item1
    - Item2
    - Item3

- Action Items:

* [ ] Action Item 1
* [ ] Action Item 2
* [ ] Action Item 3
* [ ] Action Item 4
* [ ] Action Item 5
---
## Date: Week 10 - Month Day Year 
- Topics of discussion
    - Item1
    - Item2
    - Item3

- Action Items:

* [ ] Action Item 1
* [ ] Action Item 2
* [ ] Action Item 3
* [ ] Action Item 4
* [ ] Action Item 5
---
## Date: Week 11 - Month Day Year 
- Topics of discussion
    - Item1
    - Item2
    - Item3

- Action Items:

* [ ] Action Item 1
* [ ] Action Item 2
* [ ] Action Item 3
* [ ] Action Item 4
* [ ] Action Item 5
---
## Date: Week 12 - Month Day Year 
- Topics of discussion
    - Item1
    - Item2
    - Item3

- Action Items:

* [ ] Action Item 1
* [ ] Action Item 2
* [ ] Action Item 3
* [ ] Action Item 4
* [ ] Action Item 5
---
## Date: Week 13 - Month Day Year 
- Topics of discussion
    - Item1
    - Item2
    - Item3

- Action Items:

* [ ] Action Item 1
* [ ] Action Item 2
* [ ] Action Item 3
* [ ] Action Item 4
* [ ] Action Item 5
---
## Date: Week 14 - Month Day Year 
- Topics of discussion
    - Item1
    - Item2
    - Item3

- Action Items:

* [ ] Action Item 1
* [ ] Action Item 2
* [ ] Action Item 3
* [ ] Action Item 4
* [ ] Action Item 5
---
- **_Add Images and Diagrams Using Excalidraw_**
  - Just draw it and then copy as png paster in the editor
![img_2.png](img_2.png)


- **_Add Equation_**
  - $e^{\pi i} + 1 = 0$



- **_Add Pyhton Code_**

```
import numpy as np
a = np.array()
```

- **_Add Tables as needed._**

| Checkbox Experiments | checked header | crossed header |
| ---------------------|:--------------:|:--------------:|
| checkbox             |  &check; Row   |  &cross; row   |


- **_Add Tables as needed._**


|checked|unchecked|crossed|
|---|---|---|
|&check;|_|&cross;|
|&#x2611;|&#x2610;|&#x2612;|


- **_Add Tables as needed._**

| Selection |        |
| --------- | ------ |
| &#x2610;  |

| Selection |        |
| --------- | ------ |
| &#x2611; |

- **_Create Links as needed_**
  - [link text](full url minus the en-us locale)

- **_Add Geo Json_**

```geojson
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "id": 1,
      "properties": {
        "ID": 0
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
              [-90,35],
              [-90,30],
              [-85,30],
              [-85,35],
              [-90,35]
          ]
        ]
      }
    }
  ]
}
```

- **_Add flow chart_**


```mermaid
graph TD;
    A-->B;
    A-->C;
    B-->D;
    C-->D;
```
