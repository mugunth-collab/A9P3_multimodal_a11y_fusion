#!/usr/bin/env python3
"""A9P3 - Multimodal late-fusion network for screen-reader barrier detection in smart-city apps.

Runs everything: preprocessing (imputation, winsorisation, scaling fitted per fold), the proposed
multimodal late-fusion network (visual / text / UI-context encoders + learned fusion head + auxiliary
unimodal heads + modality dropout), baselines (rule-based a11y checker, visual-only NN, text-only NN,
LightGBM reference), ablations, missing-modality robustness, subgroup analysis, paired t-tests,
on-device cost, and all figures. Usage:  python3 run_all.py   (reads ../data, writes ../results, ../figures)
"""
import os, json, time, shutil, subprocess
import numpy as np, pandas as pd
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch, torch.nn as nn
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, precision_score,
                             recall_score, precision_recall_curve)
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "A9P3_cityapp_a11y_elements.csv")
RES = os.path.join(HERE, "..", "results"); FIG = os.path.join(HERE, "..", "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
SEED = 2026; K = 5; T0 = time.time()
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.titlesize": 9})

# ------------------------------------------------------------------ data + modality grouping
df = pd.read_csv(DATA)
y_all = df["screen_reader_barrier"].to_numpy().astype(np.float32)
groups = df["screen_id"].to_numpy()
# excluded: IDs, timestamp, and the second label ai_description_adequate (a different, post-test outcome)
df["log_lux"] = np.log1p(df["ambient_lux"]); df["log_latency"] = np.log1p(df["inference_latency_ms"])
df["log_text"] = np.log1p(df["visible_text_chars"])
for c in ["ambient_lux", "ocr_confidence", "caption_confidence"]:
    df[c + "_miss"] = df[c].isna().astype(float)
MOD = {
    "V": dict(num=["touch_target_dp", "contrast_ratio", "caption_confidence", "log_lux",
                   "caption_confidence_miss", "ambient_lux_miss"], cat=["element_type"]),
    "T": dict(num=["has_accessibility_label", "label_quality_score", "log_text", "ocr_confidence",
                   "ocr_confidence_miss"], cat=[]),
    "C": dict(num=["view_hierarchy_depth", "sibling_count", "dynamic_content", "focus_order_valid",
                   "user_font_scale", "thermal_state", "battery_pct", "log_latency", "vlm_fused_confidence"],
              cat=["city_app", "platform", "device_class", "on_device_model"]),
}
WINSOR = ["log_lux", "log_latency", "contrast_ratio", "touch_target_dp"]

def make_matrix(tr_idx, te_idx):
    """Fit imputation/winsorisation/scaling on the training rows only; return per-modality arrays."""
    out_tr, out_te, names = {}, {}, {}
    for m, spec in MOD.items():
        Xtr = df.loc[tr_idx, spec["num"]].astype(float).copy(); Xte = df.loc[te_idx, spec["num"]].astype(float).copy()
        med = Xtr.median(); Xtr = Xtr.fillna(med); Xte = Xte.fillna(med)
        for c in spec["num"]:
            if c in WINSOR:
                lo, hi = Xtr[c].quantile([0.01, 0.99]); Xtr[c] = Xtr[c].clip(lo, hi); Xte[c] = Xte[c].clip(lo, hi)
        mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
        Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
        for c in spec["cat"]:
            for lev in sorted(df[c].unique()):
                Xtr[f"{c}={lev}"] = (df.loc[tr_idx, c] == lev).astype(float).to_numpy()
                Xte[f"{c}={lev}"] = (df.loc[te_idx, c] == lev).astype(float).to_numpy()
        out_tr[m] = Xtr.to_numpy(np.float32); out_te[m] = Xte.to_numpy(np.float32); names[m] = list(Xtr.columns)
    return out_tr, out_te, names

# ------------------------------------------------------------------ models
H_ENC, H_FUS = 16, 32
class Encoder(nn.Module):
    def __init__(self, d):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, H_ENC), nn.ReLU())
    def forward(self, x): return self.net(x)

class LateFusionNet(nn.Module):
    """Modality encoders -> embeddings h_m and unimodal logits z_m -> fusion head.
    fusion='mlp' (proposed, learned non-linear head on [h_V,h_T,h_C]) or 'sum' (additive decision-level fusion)."""
    def __init__(self, dims, mods, fusion="mlp", p_drop=0.15):
        super().__init__(); self.mods = mods; self.fusion = fusion; self.p_drop = p_drop
        self.enc = nn.ModuleDict({m: Encoder(dims[m]) for m in mods})
        self.aux = nn.ModuleDict({m: nn.Linear(H_ENC, 1) for m in mods})
        if fusion == "mlp":
            self.head = nn.Sequential(nn.Linear(H_ENC * len(mods), H_FUS), nn.ReLU(), nn.Linear(H_FUS, 1))
        else:
            self.bias = nn.Parameter(torch.zeros(1))
    def forward(self, xs, mask=None):
        hs, zs = [], []
        for m in self.mods:
            h = self.enc[m](xs[m])
            if mask is not None: h = h * mask[m]
            hs.append(h); zs.append(self.aux[m](h).squeeze(-1))
        if self.fusion == "mlp": z = self.head(torch.cat(hs, -1)).squeeze(-1)
        else: z = torch.stack(zs, 0).sum(0) + self.bias
        return z, zs

class EarlyFusionMLP(nn.Module):
    def __init__(self, d):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, xs, mask=None): return self.net(torch.cat([xs[m] for m in sorted(xs)], -1)).squeeze(-1), []

class UniNet(nn.Module):
    def __init__(self, d):
        super().__init__(); self.enc = Encoder(d); self.head = nn.Sequential(nn.Linear(H_ENC, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, xs, mask=None): return self.head(self.enc(next(iter(xs.values())))).squeeze(-1), []

HP = dict(lr=3e-3, wd=1e-4, batch=512, epochs=30, patience=5, lam_aux=0.3, p_drop=0.15, val_frac=0.1)

def train_nn(kind, mods, Xtr, ytr, seed, aux=True, mdrop=True, fusion="mlp", log_curve=False):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    n = len(ytr); perm = rng.permutation(n); nv = int(HP["val_frac"] * n); vi, ti = perm[:nv], perm[nv:]
    T = {m: torch.from_numpy(Xtr[m]) for m in mods}; Y = torch.from_numpy(ytr)
    dims = {m: Xtr[m].shape[1] for m in mods}
    if kind == "late": net = LateFusionNet(dims, mods, fusion=fusion)
    elif kind == "early": net = EarlyFusionMLP(sum(dims.values()))
    else: net = UniNet(dims[mods[0]])
    opt = torch.optim.Adam(net.parameters(), lr=HP["lr"], weight_decay=HP["wd"]); bce = nn.BCEWithLogitsLoss()
    best, best_state, bad, curve = 1e9, None, 0, []
    for ep in range(HP["epochs"]):
        net.train(); order = torch.from_numpy(rng.permutation(ti)); tl = 0.0
        for b in range(0, len(order), HP["batch"]):
            idx = order[b:b + HP["batch"]]; xs = {m: T[m][idx] for m in mods}; yb = Y[idx]
            mask = None
            if kind == "late" and mdrop and len(mods) > 1:   # modality dropout: drop at most one modality per sample
                drop = torch.rand(len(idx)) < HP["p_drop"]; which = torch.randint(0, len(mods), (len(idx),))
                mask = {m: (~(drop & (which == j))).float().unsqueeze(-1) for j, m in enumerate(mods)}
            z, zs = net(xs, mask); loss = bce(z, yb)
            if kind == "late" and aux and fusion == "mlp": loss = loss + HP["lam_aux"] * sum(bce(q, yb) for q in zs) / len(zs)
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item() * len(idx)
        net.eval()
        with torch.no_grad():
            zv, _ = net({m: T[m][vi] for m in mods}); vl = bce(zv, Y[vi]).item()
        curve.append((ep + 1, tl / len(ti), vl))
        if vl < best - 1e-4: best, bad, best_state = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= HP["patience"]: break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad(): pv = torch.sigmoid(net({m: T[m][vi] for m in mods})[0]).numpy()
    thr = best_threshold(Y[vi].numpy(), pv)
    return net, thr, curve

def best_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p); f = 2 * pr * rc / np.maximum(pr + rc, 1e-9)
    return float(th[np.argmax(f[:-1])])

def predict(net, mods, Xte, mask_mod=None):
    xs = {m: torch.from_numpy(Xte[m]) for m in mods}; mask = None
    if mask_mod is not None:
        mask = {m: torch.full((len(next(iter(xs.values()))), 1), 0.0 if m == mask_mod else 1.0) for m in mods}
    with torch.no_grad(): return torch.sigmoid(net(xs, mask)[0]).numpy()

def metrics(y, p, thr):
    yh = (p >= thr).astype(int)
    return dict(F1=f1_score(y, yh), Precision=precision_score(y, yh, zero_division=0), Recall=recall_score(y, yh),
                AUC=roc_auc_score(y, p) if p.std() > 0 else 0.5, PRAUC=average_precision_score(y, p))

# ------------------------------------------------------------------ experiment configs
CFG = {  # name: (kind, mods, aux, mdrop, fusion)
    "Visual-only NN": ("uni", ["V"], 0, 0, None), "Text-only NN": ("uni", ["T"], 0, 0, None),
    "Context-only NN": ("uni", ["C"], 0, 0, None),
    "Proposed MLF": ("late", ["V", "T", "C"], 1, 1, "mlp"),
    "w/o aux loss": ("late", ["V", "T", "C"], 0, 1, "mlp"),
    "w/o modality dropout": ("late", ["V", "T", "C"], 1, 0, "mlp"),
    "Additive late fusion": ("late", ["V", "T", "C"], 0, 1, "sum"),
    "Early-fusion MLP": ("early", ["V", "T", "C"], 0, 0, None),
    "MLF w/o context (V+T)": ("late", ["V", "T"], 1, 1, "mlp"),
    "MLF w/o text (V+C)": ("late", ["V", "C"], 1, 1, "mlp"),
    "MLF w/o visual (T+C)": ("late", ["T", "C"], 1, 1, "mlp"),
}
skf = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=SEED)
folds = list(skf.split(df, y_all, groups))
rows, oof, curves, robust = [], {}, {}, []
names_ref = None
for f, (tr, te) in enumerate(folds):
    tr_idx, te_idx = df.index[tr], df.index[te]
    Xtr, Xte, names_ref = make_matrix(tr_idx, te_idx); ytr, yte = y_all[tr], y_all[te]
    # rule-based checker (README): missing label OR contrast < 4.5 OR target < 48 dp
    d = df.loc[te_idx]
    rule = ((d["has_accessibility_label"] == 0) | (d["contrast_ratio"] < 4.5) | (d["touch_target_dp"] < 48)).astype(float).to_numpy()
    rows.append(dict(method="Rule-based checker", fold=f, **metrics(yte, rule, 0.5)))
    oof.setdefault("Rule-based checker", np.zeros(len(df)))[te] = rule
    # LightGBM reference on all features
    Xa_tr = np.hstack([Xtr[m] for m in "VTC"]); Xa_te = np.hstack([Xte[m] for m in "VTC"])
    rs = np.random.default_rng(SEED + f); pr_ = rs.permutation(len(tr)); nv = int(0.1 * len(tr))
    g = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, num_threads=1, random_state=SEED, verbose=-1)
    g.fit(Xa_tr[pr_[nv:]], ytr[pr_[nv:]]); thr_g = best_threshold(ytr[pr_[:nv]], g.predict_proba(Xa_tr[pr_[:nv]])[:, 1])
    pg = g.predict_proba(Xa_te)[:, 1]; rows.append(dict(method="LightGBM (ref.)", fold=f, **metrics(yte, pg, thr_g)))
    oof.setdefault("LightGBM (ref.)", np.zeros(len(df)))[te] = pg
    for name, (kind, mods, aux, md, fus) in CFG.items():
        t1 = time.time()
        net, thr, curve = train_nn(kind, mods, Xtr, ytr, SEED + f, aux=bool(aux), mdrop=bool(md), fusion=fus or "mlp")
        p = predict(net, mods, Xte); r = metrics(yte, p, thr); r.update(method=name, fold=f, thr=thr,
                                                                        epochs=len(curve), train_s=time.time() - t1)
        rows.append(r); oof.setdefault(name, np.zeros(len(df)))[te] = p
        if f == 0: curves[name] = curve
        if name in ("Proposed MLF", "w/o modality dropout"):
            for mm in [None, "V", "T", "C"]:
                pm = predict(net, mods, Xte, mask_mod=mm)
                robust.append(dict(model=name, masked=mm or "none", fold=f, AUC=roc_auc_score(yte, pm),
                                   F1=f1_score(yte, (pm >= thr).astype(int))))
        if name == "Proposed MLF" and f == 0:
            prop_net, prop_mods, prop_X = net, mods, Xte
    print(f"fold {f} done  t={time.time() - T0:.0f}s", flush=True)

OOF = pd.DataFrame(oof); OOF.insert(0, "fold", 0)
for f, (tr, te) in enumerate(folds): OOF.loc[te, "fold"] = f
OOF.insert(0, "y", y_all.astype(int)); OOF.insert(0, "element_id", df.element_id)
OOF.round(5).to_csv(os.path.join(RES, "oof_predictions.csv"), index=False)
pd.DataFrame([dict(method=m, epoch=e, train_loss=a, val_loss=b) for m, cv in curves.items() for e, a, b in cv]).round(5).to_csv(
    os.path.join(RES, "learning_curves_fold0.csv"), index=False)
R = pd.DataFrame(rows); R.to_csv(os.path.join(RES, "fold_metrics.csv"), index=False)
MET = ["F1", "Precision", "Recall", "AUC", "PRAUC"]
summ = R.groupby("method")[MET].agg(["mean", "std"]); summ.columns = [f"{a}_{b}" for a, b in summ.columns]
summ.round(4).to_csv(os.path.join(RES, "summary_metrics.csv"))
RB = pd.DataFrame(robust); RB.to_csv(os.path.join(RES, "robustness_fold.csv"), index=False)
rbs = RB.groupby(["model", "masked"])[["AUC", "F1"]].agg(["mean", "std"]); rbs.columns = [f"{a}_{b}" for a, b in rbs.columns]
rbs.round(4).to_csv(os.path.join(RES, "robustness_summary.csv"))

# paired t-tests on fold F1 / AUC (proposed vs each other method)
tt = []
P = R[R.method == "Proposed MLF"].sort_values("fold")
for m in R.method.unique():
    if m == "Proposed MLF": continue
    Q = R[R.method == m].sort_values("fold")
    for met in ["F1", "AUC"]:
        t, pv = stats.ttest_rel(P[met].to_numpy(), Q[met].to_numpy())
        tt.append(dict(method=m, metric=met, mean_diff=float((P[met].to_numpy() - Q[met].to_numpy()).mean()), t=float(t), p=float(pv)))
TT = pd.DataFrame(tt); TT.round(5).to_csv(os.path.join(RES, "paired_ttests.csv"), index=False)

# subgroup recall on pooled out-of-fold predictions (barrier mechanisms defined from observed features)
thr_mean = R.groupby("method")["thr"].mean().to_dict() if "thr" in R else {}
thr_mean["Rule-based checker"] = 0.5
fold_of = np.zeros(len(df), int)
for f, (tr, te) in enumerate(folds): fold_of[te] = f
thr_fold = {(r.method, r.fold): r.thr for r in R.itertuples() if not pd.isna(getattr(r, "thr", np.nan))}
def oof_label(m):
    if m == "Rule-based checker": return oof[m]
    if m == "LightGBM (ref.)": return None
    return np.array([oof[m][i] >= thr_fold[(m, fold_of[i])] for i in range(len(df))]).astype(int)
lab = {m: oof_label(m) for m in ["Rule-based checker", "Visual-only NN", "Text-only NN", "Additive late fusion", "Proposed MLF"]}
nolab = df.has_accessibility_label == 0; iconl = df.element_type.isin(["icon", "image_button", "image"])
SG = {
    "Unlabelled small icon": nolab & iconl & (df.touch_target_dp < 40),
    "Unlabelled, no text": nolab & ~iconl & (df.visible_text_chars < 2),
    "Vague label (q<0.35)": (df.has_accessibility_label == 1) & (df.label_quality_score < 0.35),
    "Live map/chart": (df.dynamic_content == 1) & df.element_type.isin(["map_marker", "chart"]),
    "Focus break, deep tree": (df.focus_order_valid == 0) & ((df.view_hierarchy_depth > 11) | (df.sibling_count > 8)),
    "Clipped text field": (df.element_type == "text_field") & (df.user_font_scale >= 1.6) & (df.touch_target_dp < 44),
    "Android parking/Wi-Fi": df.city_app.isin(["parking_payment", "public_wifi_portal"]) & (df.platform == "android"),
}
sg_rows = []
for s, msk in SG.items():
    pos = (msk & (y_all == 1)).to_numpy()
    r = dict(subgroup=s, n=int(msk.sum()), n_pos=int(pos.sum()), barrier_rate=float(y_all[msk.to_numpy()].mean()))
    for m, l in lab.items(): r[m] = float(l[pos].mean())
    sg_rows.append(r)
SGD = pd.DataFrame(sg_rows); SGD.round(4).to_csv(os.path.join(RES, "subgroup_recall.csv"), index=False)

# on-device cost of the proposed model (single-element CPU latency, 1 thread)
nparam = sum(p.numel() for p in prop_net.parameters())
x1 = {m: torch.from_numpy(prop_X[m][:1]) for m in prop_mods}
with torch.no_grad():
    for _ in range(50): prop_net(x1)
    ts = []
    for i in range(500):
        xi = {m: torch.from_numpy(prop_X[m][i:i + 1]) for m in prop_mods}; t1 = time.perf_counter(); prop_net(xi); ts.append(time.perf_counter() - t1)
lat_ms = np.array(ts) * 1e3
cost = dict(params=int(nparam), size_kb_fp32=nparam * 4 / 1024, latency_ms_median=float(np.median(lat_ms)),
            latency_ms_p95=float(np.percentile(lat_ms, 95)))

# dataset summary
ds = dict(rows=len(df), screens=int(df.screen_id.nunique()), barrier_rate=float(y_all.mean()),
          n_features={m: len(v) for m, v in names_ref.items()},
          missing_pct={c: float(df[c].isna().mean() * 100) for c in ["ambient_lux", "ocr_confidence", "caption_confidence"]},
          per_app_barrier=df.groupby("city_app").screen_reader_barrier.mean().round(4).to_dict(),
          per_type_barrier=df.groupby("element_type").screen_reader_barrier.mean().round(4).to_dict(),
          unlabelled_pct=float((df.has_accessibility_label == 0).mean() * 100))
pd.DataFrame([dict(item=k, value=json.dumps(v) if isinstance(v, dict) else v) for k, v in ds.items()]).to_csv(
    os.path.join(RES, "dataset_summary.csv"), index=False)
pd.DataFrame([dict(modality=m, n_inputs=len(v), inputs="; ".join(v)) for m, v in names_ref.items()]).to_csv(
    os.path.join(RES, "modality_features.csv"), index=False)
hp_tab = dict(HP, H_enc=H_ENC, H_fusion=H_FUS, folds=K, seed=SEED)
pd.DataFrame([dict(param=k, value=v) for k, v in hp_tab.items()]).to_csv(os.path.join(RES, "hyperparameters.csv"), index=False)
ep = R.groupby("method")["epochs"].mean().dropna().to_dict(); trs = R.groupby("method")["train_s"].mean().dropna().to_dict()

metrics_json = dict(dataset=ds, hyperparameters=hp_tab, summary={m: {k: float(v) for k, v in summ.loc[m].items()} for m in summ.index},
                    paired_ttests=tt, robustness={f"{a}|{b}": {k: float(v) for k, v in rbs.loc[(a, b)].items()} for a, b in rbs.index},
                    subgroup_recall=sg_rows, on_device_cost=cost, mean_epochs=ep, mean_train_s=trs,
                    runtime_s=time.time() - T0)
json.dump(metrics_json, open(os.path.join(RES, "metrics.json"), "w"), indent=1)

# ------------------------------------------------------------------ figures
def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(RES, name + ".jpg"), dpi=300, bbox_inches="tight", pil_kwargs={"quality": 92}); plt.close(fig)
ACC = "#c0392b"; GREY = "#8c8c8c"
# Fig 2: main comparison bars
order = ["Rule-based checker", "Visual-only NN", "Text-only NN", "Additive late fusion", "LightGBM (ref.)", "Proposed MLF"]
lbl = ["Rule", "Visual", "Text", "Add.\nLF", "LGBM", "MLF"]
fig, ax = plt.subplots(figsize=(3.4, 2.2)); w = 0.38; x = np.arange(len(order))
for j, (met, hatch) in enumerate([("F1", ""), ("AUC", "//")]):
    mu = [summ.loc[m, f"{met}_mean"] for m in order]; sd = [summ.loc[m, f"{met}_std"] for m in order]
    cols = [ACC if m == "Proposed MLF" else GREY for m in order]
    ax.bar(x + (j - 0.5) * w, mu, w, yerr=sd, capsize=2, color=cols, alpha=1 if j == 0 else 0.55, hatch=hatch,
           edgecolor="k", linewidth=0.4, label=met)
ax.set_xticks(x); ax.set_xticklabels(lbl); ax.set_ylabel("Score (5-fold mean)"); ax.set_ylim(0.4, 1.0)
ax.legend(loc="upper left", ncol=2, frameon=False); ax.grid(axis="y", lw=0.3)
save(fig, "fig2_comparison")
# Fig 3: PR curves (pooled OOF)
fig, ax = plt.subplots(figsize=(3.0, 2.3))
for m, nm, ls, c in [("Visual-only NN", "Visual", ":", "#555"), ("Text-only NN", "Text", "-.", "#555"),
                     ("Additive late fusion", "Add. LF", "--", "#1f77b4"), ("LightGBM (ref.)", "LGBM", "-", "#2ca02c"),
                     ("Proposed MLF", "MLF", "-", ACC)]:
    pr, rc, _ = precision_recall_curve(y_all, oof[m]); ap = average_precision_score(y_all, oof[m])
    ax.plot(rc, pr, ls=ls, color=c, lw=1.3 if m == "Proposed MLF" else 1.0, label=f"{nm} ({ap:.3f})")
r_rule = recall_score(y_all, oof["Rule-based checker"]); p_rule = precision_score(y_all, oof["Rule-based checker"])
ax.plot(r_rule, p_rule, "ks", ms=4, label="Rule")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_ylim(0.2, 1.01)
ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=3, columnspacing=0.8, handlelength=1.6)
ax.grid(lw=0.3); save(fig, "fig3_pr_curves")
# Fig 4: subgroup recall heatmap
mlist = ["Rule-based checker", "Visual-only NN", "Text-only NN", "Additive late fusion", "Proposed MLF"]
mshort = ["Rule", "Vis.", "Text", "Add.", "MLF"]
short_sg = {"Unlabelled small icon": "Unlab. small icon", "Unlabelled, no text": "Unlab., no text",
            "Vague label (q<0.35)": "Vague label", "Live map/chart": "Live map/chart",
            "Focus break, deep tree": "Focus break, deep", "Clipped text field": "Clipped field",
            "Android parking/Wi-Fi": "Andr. park./Wi-Fi"}
M = SGD[mlist].to_numpy()
fig, ax = plt.subplots(figsize=(2.3, 2.3)); ax.imshow(M, cmap="Greys", vmin=0, vmax=1.15, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color="white" if M[i, j] > 0.6 else "black")
ax.set_xticks(range(len(mshort))); ax.set_xticklabels(mshort, fontsize=7.5)
ax.set_yticks(range(len(SGD))); ax.set_yticklabels([f"{short_sg[s]} ({n})" for s, n in zip(SGD.subgroup, SGD.n_pos)], fontsize=7.5)
ax.tick_params(length=0)
save(fig, "fig4_subgroup_recall")
# Fig 5: missing-modality robustness (F1 with one modality masked at test time)
fig, ax = plt.subplots(figsize=(3.4, 2.0)); cond = ["none", "V", "T", "C"]; cl = ["All\npresent", "Visual\nmissing", "Text\nmissing", "Context\nmissing"]
x = np.arange(4)
for j, (m, c, nm) in enumerate([("w/o modality dropout", GREY, "MLF w/o modality dropout"), ("Proposed MLF", ACC, "MLF (proposed)")]):
    mu = [rbs.loc[(m, k), "F1_mean"] for k in cond]; sd = [rbs.loc[(m, k), "F1_std"] for k in cond]
    ax.bar(x + (j - 0.5) * 0.38, mu, 0.38, yerr=sd, capsize=2, color=c, edgecolor="k", lw=0.4, label=nm)
ax.set_xticks(x); ax.set_xticklabels(cl); ax.set_ylabel("F1 score"); ax.set_ylim(0.3, 0.95)
ax.legend(frameon=False, fontsize=6.5, loc="upper right", ncol=1); ax.grid(axis="y", lw=0.3)
save(fig, "fig5_robustness")
# Fig 6: learning curves (fold 0)
fig, ax = plt.subplots(figsize=(3.4, 1.9))
for m, c, ls in [("Proposed MLF", ACC, "-"), ("Additive late fusion", "#1f77b4", "--"), ("Visual-only NN", "#555", ":"), ("Text-only NN", "#555", "-.")]:
    cv = np.array(curves[m]); ax.plot(cv[:, 0], cv[:, 2], color=c, ls=ls, lw=1.1, label=m)
ax.set_xlabel("Epoch"); ax.set_ylabel("Validation BCE loss"); ax.legend(frameon=False, fontsize=6.5); ax.grid(lw=0.3)
save(fig, "fig6_learning_curves")

# Fig 1 flowchart (Graphviz)
dot = os.path.join(HERE, "fig1_flowchart.dot")
if os.path.exists(dot) and shutil.which("dot"):
    subprocess.run(["dot", "-Tpng", "-Gdpi=300", dot, "-o", os.path.join(FIG, "fig1_flowchart.png")], check=True)
    from PIL import Image
    Image.open(os.path.join(FIG, "fig1_flowchart.png")).convert("RGB").save(os.path.join(RES, "fig1_flowchart.jpg"), quality=92)
print(summ[["F1_mean", "F1_std", "AUC_mean", "PRAUC_mean", "Precision_mean", "Recall_mean"]].round(4).to_string())
print(f"total runtime {time.time() - T0:.0f}s")
