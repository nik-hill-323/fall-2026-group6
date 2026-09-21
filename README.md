# WM Concord

**Do world model diagnostics agree with each other?**

GWU Data Science Capstone, Fall 2026, Group 6
Advisor: Dr. Amir Jafari

## Overview

Many papers claim a sequence model has learned a "world model", but each uses its own test. WM Concord runs four diagnostic families on the same set of models and asks two questions:

1. Do the four diagnostics rank models the same way?
2. Which diagnostic actually predicts when a model fails on a shifted task?

The four diagnostics:

* **D1 Next token validity:** fraction of proposed next moves that are legal.
* **D2 State equivalence:** Myhill Nerode compression and distinction errors (Vafa et al. 2024).
* **D3 Inductive bias probes:** how well a model adapts to tasks built from the true world model (Vafa et al. 2025).
* **D4 Representation probing:** whether the true state can be read from model activations, checked with causal patching (Li et al. 2023).

All models are small sequence models trained on domains with a known ground truth automaton (DFA), so the true state is always known.

The full proposal is in [reports/proposal.md](reports/proposal.md).

## Hypotheses

* **H1:** The four diagnostics produce concordant model rankings.
* **H2:** Where they disagree, the disagreement is systematic, not noise.
* **H3:** One diagnostic predicts downstream failure better than the others.
* **H4:** The findings hold on new domains and held out architectures.

## Installation

Tested on AWS g5.2xlarge (NVIDIA A10G, 24 GB), Ubuntu 26.04, CUDA 13, Python 3.13.

    git clone https://github.com/nik-hill-323/fall-2026-group6.git
    cd fall-2026-group6
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130

Note: PySR installs a Julia backend on first import. This takes a few minutes the first time.

## Running

Instructions will be added as each component lands. The first target is diagnostic D4 running end to end on an Othello model.

## Repository map

* `src/` importable project code (domains, model zoo, diagnostics, analysis)
* `src/docs/` project notes, including literature notes
* `cookbooks/` notebooks that walk through the code in narrative form
* `demo/` small runnable demo
* `reports/` proposal, LaTeX report, and weekly progress reports
* `research_paper/` paper draft (LaTeX)
* `presentation/` slides for the preliminary and final presentations
* `requirements.txt` pinned Python dependencies

## Project status

Current phase: setup and first diagnostic.

Key dates:

* **October 6, 2026:** literature review draft
* **October 20, 2026:** preliminary presentation
* **December 8, 2026:** final presentation and journal submission

Progress is tracked in the [Group 6 Project Management](https://github.com/users/nik-hill-323/projects/2) board and weekly reports in `reports/Progress_Report/`.

## Team

* Nikhil Obuleni ([@nik-hill-323](https://github.com/nik-hill-323))
* Siddardha Reddy ([@SIDCAP777](https://github.com/SIDCAP777))
* Venkatesh Nagarjuna ([@Asbetos](https://github.com/Asbetos))
* Viharika Vemparala ([@viharikav](https://github.com/viharikav))

## References

Full bibliography: [reports/Latex_report/references.bib](reports/Latex_report/references.bib)
