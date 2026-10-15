# EnergyMind Tunisia

**An AI trust layer that sits between raw smart-meter data and automated action on the electricity grid.**

TSYP14 Technical Challenge, **Phase 1** (ideation and data exploration).

---

## 1. The problem

Tunisia's grid is under growing strain from seasonal peaks, fast-growing rooftop solar, more smart meters and more connected devices. At the summer peak (13:00 to 17:00, driven by air conditioning), operators increasingly rely on automated systems fed by meter data. The challenge asks a question that goes beyond prediction accuracy: **can that data, and the AI acting on it, be trusted, and who is accountable when it is wrong?**

A single deviation can have three very different causes, and each needs a different response:

| What the screen shows | Possible cause | Right response |
|---|---|---|
| House 7 jumps by 1.2 kW and its voltage dips | **Real grid event** (voltage sag, outage, demand step) | Inspect the grid or the customer. The data are good. |
| Same reading | **Measurement problem** (biased, drifting, frozen, replayed or falsified meter) | Fix the data. The grid is fine. |
| Same reading | **Benign case** (a new appliance, a solar ramp) | Do nothing. It only looks like a fault. |

Confusing them is costly. "Repairing" a real outage as if it were a sensor glitch hides it, and alarming on every new appliance teaches operators to ignore alarms. When physics cannot tell the cases apart, the right answer is to say so and ask a human.

## 2. Our solution: a trust layer

EnergyMind checks every reading before any automated system is allowed to use it. It does four things, which are the four goals of the challenge:

```
raw meter data -> [ 1 Detect -> 2 Explain -> 3 Decide -> 4 Share ] -> automated action
                                                 ^                ^
                                          human operator    other houses
                                    (approve / reject / override)  (case cards)
```

1. **Detect.** Decide whether a reading window is normal, a measurement problem or a real event. Physics checks (for example, apparent power must be at least active power) are combined with machine-learning models (statistical baseline, Isolation Forest, One-Class SVM, LightGBM, TCN, autoencoder).
2. **Explain.** Give a calibrated confidence, the physics checks that passed or failed, SHAP features and a plain-language reason. Everything is written to a hash-chained decision log that can be audited.
3. **Decide.** A deny-by-default policy sets what the AI may do alone and what needs a person. The operator interface lets a human approve, reject or override, with a mandatory reason.
4. **Share.** Houses can exchange reviewed anomaly "case cards" and calibration lessons, never raw household data.

The unit of the design is **one node = one house**. Three small experiments extend it: a feeder (a house plus its neighbours and the transformer), evidence fusion, and case-sharing between two houses.

### What a verdict looks like
> *Real event, confidence 80 %. Check passed: p <= s. The other relations hold by construction and cannot fail. Alternative explanation: a coordinated reporting error that keeps the relations consistent.* Proposed action: request an inspection (needs a human).

The system never counts a check it cannot run as passed, and it answers **`unresolved`** when confidence is below 0.55 or two causes are too close.

### What the AI can do alone, and what it cannot
| Action | Who decides |
|---|---|
| Flag a reading | AI alone |
| Replacement estimate | AI alone, confidence 0.70 or higher |
| Quarantine a channel | AI alone, confidence 0.90 or higher and two independent sources |
| Request an inspection | A human |
| Recalibrate a meter, switch or shed a load, promote a model or shared rule | A human only |
| Anything else | Denied |

This addresses over-trust (the AI cannot act beyond its table, whatever it says about itself) and under-trust (low-risk actions are automated, so operators are not flooded).

### What is built now and what is planned
- **Phase 1 prototype (this repo):** fault injector, detectors, calibration, SHAP and physics explanations, policy, hash-chained log, Streamlit operator interface, feeder, fusion and two-node sharing experiments.
- **Phase 2 design (not built):** investigation workflow (LangGraph), evidence graph (Neo4j), similar-case search (Qdrant), LIME, policy engine (OPA), service interfaces. Details in the report.

## 3. The data

**Our dataset** is the clean baseline published on [Kaggle](https://www.kaggle.com/datasets/azertyqsdfghqsdfgh/pes-test) (card: [`docs/DATASET_CARD.md`](docs/DATASET_CARD.md)):
- 13 houses on one 160 kVA low-voltage feeder, one year (2018) at 15 minutes, Tunis.
- Real consumption from **WPuQ** (German houses) and **AMPds2** (Canadian appliances) on a **SimBench** grid solved with pandapower, adapted to Tunisian conditions.
- Two views of every house: `ground_truth` (true values) and `measured` (what a meter reports).
- 14 physical validation checks, all passed.
- It contains **no injected anomalies yet**. Injecting them and running detection on it is the Phase 2 step.

**Anomaly simulation.** Real labelled grid faults are rare, so a seeded injector (`energymind/inject.py`) adds 16 documented fault types: measurement problems (bias, drift, scale, noise, frozen, dropout, delay, replay, falsified), real events (voltage sag, outage, demand step, duty cycle, phase imbalance) and benign cases (new load, PV ramp). See [`docs/INJECTION.md`](docs/INJECTION.md).

**ECO is not our data.** To develop and test the detectors in Phase 1, we used public recordings (ECO, plus AMPds2 and WPuQ) with injected faults. ECO is not part of our dataset. Raw third-party data is not in this repo; see [`data/DATA_NOTES.md`](data/DATA_NOTES.md).

## 4. Results so far

Event-level scores of the final detector on injected faults (`results/improve_final_summary.md`):

| Test | Event macro-F1 | False alarms per day |
|---|---|---|
| ECO houses seen in training | 0.786 | 0.14 |
| ECO unseen houses | 0.589 and 0.609 | 0.97 and 0.53 |
| AMPds2 | 0.724 | 0.21 |
| WPuQ (28 days, indicative) | 0.766 | 0.25 |

Read these with their limits:
- They measure detection of our own injected faults on public recordings, not field performance, and not yet performance on our dataset.
- The detector is tuned to avoid false alarms. It misses most `demand_step`, `duty_cycle` and `falsified` events, and its median detection delay is about 10 minutes.
- A new appliance and a demand step are the same physics, so some cases must stay `unresolved`.
- We audited our own benchmark, found that the first injector leaked labels, fixed it, and the scores dropped. See [`results/leakage_audit.md`](results/leakage_audit.md).

Full tables and limits: [`docs/TECHNICAL_README.md`](docs/TECHNICAL_README.md).

## 5. Phase 1 deliverables

| Required by the challenge | Where |
|---|---|
| Project proposal: scope, objectives, data and anomaly plan, design per goal | [`reports/EnergyMind_Phase1_Report.pdf`](reports/EnergyMind_Phase1_Report.pdf) |
| Dataset | [Kaggle](https://www.kaggle.com/datasets/azertyqsdfghqsdfgh/pes-test) |
| Code developed in Phase 1 | this repository |

| Goal | Main files |
|---|---|
| 1. Detection | `energymind/inject.py`, `features*.py`, `models_*.py`, `evaluate.py` |
| 2. Confidence and explanation | `energymind/explain.py`, `decision_log.py` |
| 3. Escalation and governance | `energymind/policy.py`, `configs/policy.yaml`, `app/operator.py`, [`docs/GOVERNANCE.md`](docs/GOVERNANCE.md) |
| 4. Shared learning | `energymind/sharing.py`, [`docs/SHARED_LEARNING.md`](docs/SHARED_LEARNING.md) |

## 6. Run it

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt     # PyTorch is installed separately
.venv\Scripts\python -m pytest tests              # unit tests
.venv\Scripts\streamlit run app\operator.py       # operator interface
```
The operator interface shows the review queue, the evidence behind each verdict, the confidence, and approve / reject / override buttons. Without a trained model in `outputs/models/improve_best/` it opens with hand-built demo incidents. Commands to reproduce every result: `docs/TECHNICAL_README.md`. Rebuild the dataset: `python scripts/build_baseline_dataset.py --seed 42` (needs the raw WPuQ and AMPds2 files).

## 7. Repository map

```
reports/          Phase 1 report (PDF + LaTeX source)
energymind/       trust layer: injection, features, models, explanation, policy, log, sharing
baseline_dataset/ builds the published dataset
app/              operator interface (Streamlit)
configs/          policy file
scripts/          runners for experiments and the dataset build
tests/            unit tests
results/          measured outputs (tables, figures, logs)
docs/             design notes and dataset card
data/             data notes, baseline validation report and sample
```

## 8. Phase 2 plan (until 12 December 2026)

**Starting point.** The detectors tested so far do not perform well enough yet, especially on unseen houses and on some fault types. Phase 2 focuses on improving them and showing the gain with measured results.

**Roadmap, in priority order** (aligned with the challenge brief: simulation is expected, a working single-node prototype is required, multi-node sharing may stay a documented design)

1. **Improve the detection models (main focus).** Detection was tested on data with injected anomalies, and the results are not good enough yet. Phase 2 improves them:
   - Per-house adaptation, because models do not transfer between houses.
   - Better recall on the missed fault types, using richer physics checks (feeder energy balance, circuit model) and optional appliance-level and graph models, each kept only if it beats a simpler baseline.
   - Faster detection, with fewer false alarms.
2. **Prove the improvement with measured results.** Compare against the current baselines on the brief's four evaluation criteria: precision and recall, false-positive rate, explanation quality and response latency. The anomaly-injection method stays documented ([`docs/INJECTION.md`](docs/INJECTION.md)).
3. **Complete explainability.** Add LIME as a model-independent second opinion next to SHAP and physics rules, with a plain-language explanation and an auditable decision log for every flagged event.
4. **Complete the single-node prototype and operator interface.** Explicit thresholds for automatic actions and for human approval, an interface for rapid review, override or approval, and safeguards against both over-trust and under-trust. Add the investigation workflow and policy engine, and the service layer (FastAPI, PostgreSQL ledger, Neo4j evidence graph, Qdrant case index).
5. **Collective-defense design across nodes** (factories, neighbourhoods, microgrids). Document the sharing protocol: which anomaly patterns and calibration insights are shared, what stays local, and why (privacy). Cover k-anonymity and asymmetric signatures. Multi-node sharing stays a documented design, as the brief allows.
6. **Business scope** (bonus point). Budget, solution implementation, production, continuous training, and integration and deployment cost, with the deployment path: shadow mode first, then reversible automatic actions, humans only for anything that changes an asset. Costs will come from quotes, not guesses.

**What we will show in the final demo**
- A simulated genuine grid event (sag or outage), an isolated meter bias and a coordinated reporting inconsistency, each with its evidence, explanation and authorization outcome.
- A benign case, an ambiguous case answered `unresolved`, and an unsafe request that the policy denies.
- One validated lesson shared between two houses.
- Measured comparisons against baselines.

**Phase 2 deliverables:** full project report (workflow, escalation logic, collective-defense design), this repository with the full solution code, trained models and application files, a 5-minute demo video, and the final presentation (5 minutes plus a 3-minute demo).

## Sources and license
SimBench (ODbL), WPuQ (Schlemminger et al., Scientific Data 2022, CC BY 4.0), AMPds2 (Makonin et al., Scientific Data 2016, CC BY 4.0), ECO (Beckel et al., 2014). Code: MIT. Data: ODbL.
