# EnergyMind Tunisia - Phase 1 report (Overleaf project)

1. In Overleaf: New Project -> Upload Project -> select this zip.
2. Menu -> Compiler: pdfLaTeX. Main document: main.tex. Recompile (twice if the table of contents looks empty).

Structure (the Phase 1 brief: scope, objectives, data/anomaly-simulation plan, preliminary design for each goal):
- sections/01_introduction.tex       introduction (with the trust-layer figure)
- sections/02_scope_objectives.tex   scope and objectives
- sections/03_data_anomaly_plan.tex  data, published dataset (Kaggle), anomaly-simulation plan
- sections/04_architecture.tex       global architecture: simple view, detailed view, tools, data flow, events
- sections/05_goal1_detection.tex    Goal 1 (tested models and curves)
- sections/06_goal2_confidence.tex   Goal 2: confidence, SHAP/LIME, GraphRAG, grounded writing
- sections/07_goal3_governance.tex   Goal 3: investigation workflow, action authority, incident trace
- sections/08_goal4_sharing.tex      Goal 4: case promotion, model promotion, threats
- sections/09_plan_risks.tex, 10_business_conclusion.tex, 11_appendix.tex, 12_references.tex
- figures/reliability.png            calibration figure
