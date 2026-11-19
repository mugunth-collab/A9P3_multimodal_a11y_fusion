# A9P3 code - multimodal late-fusion barrier detection

Run (about 5-15 min on 2 CPU cores, depending on load):

    pip install -r requirements.txt   # plus Graphviz `dot`
    python3 run_all.py

Reads `../data/A9P3_cityapp_a11y_elements.csv`; writes to `../results/` and `../figures/`.

## What it does
1. Groups the 21 usable columns into three modalities (visual, text, UI context); IDs, timestamp and the
   second label `ai_description_adequate` are excluded. Per fold: median imputation + missing flags,
   1-99 % winsorisation of outlier-prone columns, z-scoring, one-hot categories (fitted on training rows only).
2. 5-fold StratifiedGroupKFold (grouped by `screen_id`, seed 2026).
3. Proposed: multimodal late-fusion network (MLF): one MLP encoder per modality, modality dropout,
   learned fusion head on the concatenated embeddings, auxiliary unimodal heads, early stopping,
   F1-optimal threshold on an inner validation split.
4. Baselines: rule-based checker (missing label OR contrast < 4.5 OR target < 48 dp), visual-only NN,
   text-only NN; LightGBM on all features as a non-neural reference.
5. Ablations: context-only NN, no auxiliary loss, no modality dropout, additive late fusion, early-fusion MLP,
   each modality removed. Missing-modality robustness test, subgroup recall, paired t-tests, CPU latency.

## Outputs (`../results/`)
- `fold_metrics.csv` - F1, precision, recall, ROC-AUC, PR-AUC per method and fold
- `summary_metrics.csv` - mean and std over folds
- `paired_ttests.csv` - paired t-tests of MLF vs every other method (F1, AUC)
- `robustness_fold.csv`, `robustness_summary.csv` - AUC/F1 with one modality masked at test time
- `subgroup_recall.csv` - recall on barrier mechanisms (pooled out-of-fold)
- `dataset_summary.csv`, `modality_features.csv`, `hyperparameters.csv`
- `metrics.json` - all of the above plus on-device cost and runtime
- `fig*.jpg` - JPG copies of all figures (PNG 300 dpi versions in `../figures/`)

`fig1_flowchart.dot` is the Graphviz source of Fig. 1 (rendered by run_all.py).
