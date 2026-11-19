# A9P3 - Multimodal On-Device AI Pipelines for Real-Time Accessibility Enhancement and Intelligent Screen-Reader Assistance in Mobile Applications

Author: Mug (Independent Researcher, USA) - ICE2CPT 2026, Track 7 (accessible smart-city service apps for visually impaired users).

Proposed method: multimodal late-fusion network (MLF) for screen-reader barrier detection - visual, text and
UI-context encoders, modality dropout, learned fusion head, auxiliary unimodal losses.
Baselines: rule-based a11y checker, visual-only NN, text-only NN (+ LightGBM reference).

## Contents
- `paper/`   - A9P3_multimodal_a11y_fusion.docx / .pdf (IEEE two-column, 6 pages)
- `data/`    - A9P3_cityapp_a11y_elements.csv (15,000 UI elements) + A9P3_README.md (data dictionary)
- `code/`    - run_all.py (whole pipeline), fig1_flowchart.dot, requirements.txt, README.md
- `results/` - all tables as CSV, metrics.json, out-of-fold predictions, all figures as JPG

## Reproduce
    pip install -r code/requirements.txt      # plus Graphviz `dot`
    cd code && python3 run_all.py             # ~7 min on 2 CPU cores; writes ../results and ../figures
