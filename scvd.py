"""
scvd.py -- shared library for the smart-contract multi-label detection experiments.

Companion to the paper "Dependency-Aware Multi-Label Deep Learning for Smart
Contract Vulnerability Detection" (Methodology, Section 3). Every stage notebook
imports this module; all results, logs, tables and plots are written under a
per-stage output directory so that every figure can be regenerated later from
the saved CSV/JSON artifacts.

Credentials (read in this order):
  1. environment variables KAGGLE_USERNAME / KAGGLE_KEY (or KAGGLE_API_KEY)
  2. Kaggle User Secrets     (names KAGGLE_USERNAME / KAGGLE_KEY / HF_TOKEN)
  3. ~/.kaggle/kaggle.json
HF_TOKEN is only used when DIVE_HF_REPO is set (optional HF mirror of DIVE);
otherwise the data is read from /kaggle/input or downloaded from Zenodo.
"""
import os, sys, json, time, logging, hashlib, warnings, zipfile, io
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
RNG_SEEDS = (0, 1, 2)

# ---------------------------------------------------------------- schema ----
LABELS = ["Reentrancy", "Access Control", "Arithmetic", "Unchecked Return Values",
          "DoS", "Bad Randomness", "Front Running", "Time manipulation"]
LABEL_SHORT = ["Reent", "AccCtrl", "Arith", "UnchkRet", "DoS", "BadRnd", "FrontR", "TimeMan"]

CODE_META_PRE = ["CompilerVersion", "OptimizationUsed", "Runs", "EVMVersion",
                 "LicenseType", "NoOfExternalLib"]
TX_META_POST = ["NoOfTransactions", "nonce", "value", "gas", "gasPrice",
                "txreceipt_status", "cumulativeGasUsed", "gasUsed", "methodId",
                "confirmations"]
TEMPORAL_POST = ["hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos",
                 "quarter", "part_of_day", "relative_tx_position", "block_position"]
CATS_PRE = {"CompilerVersion": "vocab", "EVMVersion": "vocab", "LicenseType": "vocab"}
CATS_POST = {"methodId": "hash:64", "block_position": "vocab", "part_of_day": "vocab"}
VIEWS = ("pre", "post", "all")
ZENODO = ("https://zenodo.org/records/18519253/files/DIVE_Preprocessed_Data.zip?download=1",
          "https://zenodo.org/records/18519253/files/DIVE_Labels.zip?download=1")
NEEDED = {"pre": "pre_processeddata.csv", "post": "post_processeddata.csv",
          "labels": "dive_labels.csv", "tools": "tool_results.csv"}

def _is_opcode(c):
    return (c.endswith("_Distribution") or c.endswith(" Operation count")
            or c in {"opcodeLength", "noOfOpcodes", "noOfUniqueOpcodes",
                     "mostFrequentOpcode", "uniqueOpcodesRatio", "Stop Operation count"})

# ----------------------------------------------------------- credentials ----
def _secret(name):
    v = os.environ.get(name)
    if v:
        return v
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return None

def kaggle_creds():
    u, k = _secret("KAGGLE_USERNAME"), _secret("KAGGLE_KEY") or _secret("KAGGLE_API_KEY")
    if not (u and k):
        p = os.path.expanduser("~/.kaggle/kaggle.json")
        if os.path.exists(p):
            d = json.load(open(p)); u, k = d.get("username"), d.get("key")
    return u, k

def on_kaggle():
    return os.path.isdir("/kaggle/working")

def quick():
    return os.environ.get("SCVD_QUICK", "0") == "1"

def setup_stage(stage):
    base = "/kaggle/working" if on_kaggle() else os.environ.get("SCVD_OUT", "./out")
    root = os.path.join(base, stage)
    for sub in ("results", "plots", "logs", "tables"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(os.path.join(root, "logs", f"{stage}.log"))],
                        force=True)
    log = logging.getLogger(stage)
    import sklearn, scipy
    info = {"stage": stage, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "python": sys.version.split()[0], "numpy": np.__version__,
            "sklearn": sklearn.__version__, "scipy": scipy.__version__, "quick": quick()}
    try:
        import torch
        info.update(torch=torch.__version__, cuda=torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        info["torch"] = None
    json.dump(info, open(os.path.join(root, "logs", "run_env.json"), "w"), indent=1)
    log.info("stage %s initialised at %s", stage, root)
    return root, log

# ------------------------------------------------------------ data loading --
def _find_input(names):
    """Locate required files under /kaggle/input (or SCVD_DATA), walking nested
    owner/slug layouts and looking *inside* .zip members, because datasets pushed
    with dir_mode="zip" store each subfolder (results/, plots/, ...) as a zip."""
    import tempfile, zipfile
    found, tmp = {}, None
    for root in ("/kaggle/input", os.environ.get("SCVD_DATA", ".")):
        if not os.path.isdir(root):
            continue
        for dirpath, _d, files in os.walk(root):
            for f in files:
                fl = f.lower()
                if fl.startswith("._") or fl in (".ds_store",):
                    continue
                for k, v in names.items():
                    if fl == v and k not in found:
                        found[k] = os.path.join(dirpath, f)
            if len(found) < len(names):
                for f in files:
                    if not f.lower().endswith(".zip"):
                        continue
                    try:
                        z = zipfile.ZipFile(os.path.join(dirpath, f))
                    except Exception:
                        continue
                    for n in z.namelist():
                        b = os.path.basename(n).lower()
                        if not b or b.startswith("._") or n.endswith("/"):
                            continue
                        for k, v in names.items():
                            if b == v and k not in found:
                                tmp = tmp or tempfile.mkdtemp(prefix="scvd_")
                                z.extract(n, tmp)
                                found[k] = os.path.join(tmp, n)
            if len(found) == len(names):
                return found
    return found if len(found) == len(names) else None

def load_dive(log=None):
    hit = _find_input(NEEDED) or _download_fallback(log)
    pre = pd.read_csv(hit["pre"]); post = pd.read_csv(hit["post"])
    lab = pd.read_csv(hit["labels"]); tools = pd.read_csv(hit["tools"])
    pre.columns = [c.strip() for c in pre.columns]
    post.columns = [c.strip() for c in post.columns]
    lab.columns = [c.strip() for c in lab.columns]
    if log:
        log.info("loaded DIVE: pre=%s post=%s labels=%s tools=%s",
                 pre.shape, post.shape, lab.shape, tools.shape)
    return pre, post, lab, tools

def _download_fallback(log=None):
    repo = os.environ.get("DIVE_HF_REPO")
    if repo:
        from huggingface_hub import snapshot_download
        d = snapshot_download(repo, token=_secret("HF_TOKEN"))
        hit = _find_input_in(d)
        if hit:
            return hit
    if log:
        log.info("DIVE not found locally -- downloading from Zenodo (public)")
    import urllib.request
    out = {}
    for url in ZENODO:
        with zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url, timeout=900).read())) as z:
            for n in z.namelist():
                b = os.path.basename(n).lower()
                if b.startswith("._") or b in (".ds_store",):
                    continue
                if b.endswith(".zip"):
                    with zipfile.ZipFile(z.open(n)) as z2:
                        for n2 in z2.namelist():
                            b2 = os.path.basename(n2).lower()
                            if b2.endswith(".csv") and not b2.startswith("._"):
                                out["pre" if b2.startswith("pre_") else "post"] = io.BytesIO(z2.read(n2))
                elif b.endswith(".csv"):
                    out["labels" if b.startswith("dive_") else "tools"] = io.BytesIO(z.read(n))
    return out

def _find_input_in(d):
    hit = {}
    for dirpath, _d, files in os.walk(d):
        for f in files:
            fl = f.lower()
            for k, v in NEEDED.items():
                if fl == v:
                    hit[k] = os.path.join(dirpath, f)
    return hit if len(hit) == 4 else None

def family_columns(pre, post):
    pre_cols = [c for c in pre.columns if c not in ("contractID", "timeStamp")]
    post_cols = [c for c in post.columns if c != "contractID"]
    return {
        "code_metadata":      [c for c in CODE_META_PRE if c in pre_cols],
        "interface_source":   [c for c in pre_cols if c not in CODE_META_PRE and not _is_opcode(c)],
        "creation_opcodes":   [c for c in pre_cols if _is_opcode(c)],
        "tx_metadata":        [c for c in TX_META_POST if c in post_cols],
        "temporal_position":  [c for c in TEMPORAL_POST if c in post_cols],
        "runtime_opcodes":    [c for c in post_cols if _is_opcode(c)],
    }

PRE_FAMS = ("code_metadata", "interface_source", "creation_opcodes")
POST_FAMS = ("tx_metadata", "temporal_position", "runtime_opcodes")

def view_families(fam, view):
    if view == "pre":
        names = PRE_FAMS
    elif view == "post":
        names = POST_FAMS
    elif view == "all":
        names = PRE_FAMS + POST_FAMS
    else:
        raise ValueError(f"unknown view {view!r}; expected 'pre', 'post' or 'all'")
    return {n: fam[n] for n in names}

def view_columns(fam, view):
    vf = view_families(fam, view)
    return [c for v in vf.values() for c in v]

# -------------------------------------------------------- preprocessing ----
class Prep:
    """Quantile-transform numerics; ordinal-vocab / hash encode categoricals.
    Fit on the training split only (leakage-safe, Section 3.2)."""
    def __init__(self, cols, cats):
        self.cols, self.cats = cols, cats
    def fit(self, df):
        from sklearn.preprocessing import QuantileTransformer, LabelEncoder
        self.enc = {}
        for c in self.cols:
            mode = self.cats.get(c)
            if mode is None and df[c].dtype == object:      # auto-detect strings
                mode = "vocab" if df[c].nunique() <= 200 else "hash:128"
            if mode == "vocab":
                le = LabelEncoder().fit(df[c].astype(str).tolist() + ["__UNK__"])
                self.enc[c] = ("vocab", le)
            elif mode:
                self.enc[c] = ("hash", int(mode.split(":")[1]))
            else:
                qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                                         subsample=10000).fit(df[[c]].to_numpy(float))
                self.enc[c] = ("num", qt)
        return self
    def transform(self, df):
        out = np.zeros((len(df), len(self.cols)), dtype=np.float32)
        for j, c in enumerate(self.cols):
            kind, obj = self.enc[c]
            if kind == "num":
                out[:, j] = obj.transform(df[[c]].to_numpy(float)).ravel()
            elif kind == "vocab":
                m = {cl: i for i, cl in enumerate(obj.classes_)}
                unk = m.get("__UNK__", 0)
                out[:, j] = [m.get(str(v), unk) for v in df[c]]
            else:
                out[:, j] = [int(hashlib.md5(str(v).encode()).hexdigest(), 16) % obj
                             for v in df[c]]
        return out

# ---------------------------------------------------------------- splits ----
def iterstrat(Y, ratios, seed=0):
    """Iterative stratification for multi-label data (Sechidis et al. 2011).

    Per-example assignment: repeatedly take the label with the fewest
    remaining positive examples and give each of its still-unassigned rows to
    the fold with the largest remaining desired count for that label (ties /
    exhausted desires broken by remaining fold capacity). Rows with no
    positive labels are distributed by remaining fold capacity.
    """
    rs = np.random.RandomState(seed)
    Yb = (np.asarray(Y) > 0).astype(int)
    n, L = Yb.shape
    k = len(ratios)
    r = np.asarray(ratios, float)
    c = Yb.sum(0).astype(float)[:, None] * r[None, :]   # desired per label x fold
    fold_rem = n * r                                     # remaining fold capacity
    assign = -np.ones(n, int)
    unassigned = set(range(n))
    rows_with = [np.flatnonzero(Yb[:, j]) for j in range(L)]
    rem_pos = Yb.sum(0).copy()                           # unassigned positives per label

    def _place(i, s):
        assign[i] = s
        fold_rem[s] -= 1.0
        c[:, s] -= Yb[i]
        np.subtract(rem_pos, Yb[i], out=rem_pos)
        unassigned.discard(i)

    while unassigned:
        cand = np.flatnonzero(rem_pos > 0)
        if len(cand) == 0:                               # only all-zero rows left
            for i in sorted(unassigned):
                _place(i, int(np.argmax(fold_rem)))
            break
        j = int(cand[np.argmin(rem_pos[cand])])          # rarest label first
        D = [i for i in rows_with[j] if i in unassigned]
        rs.shuffle(D)
        for i in D:
            if c[j].max() > 0:
                s = int(np.argmax(c[j]))
            else:
                s = int(np.argmax(fold_rem))
            if fold_rem[s] <= 0:
                s = int(np.argmax(fold_rem))
            _place(i, s)
    return [np.flatnonzero(assign == s) for s in range(k)]

def compiler_minor(v):
    try:
        return int(str(v).replace("v", "").split(".")[1])
    except Exception:
        return -1

def build_splits(pre, lab, seed=0):
    ids = lab["contractID"].to_numpy()
    Y = lab[LABELS].to_numpy(int)
    tr, va, te = iterstrat(Y, [0.70, 0.15, 0.15], seed=seed)
    s1 = (ids[tr], ids[va], ids[te])
    meta = pre.set_index("contractID")
    maj = meta["CompilerVersion"].map(compiler_minor)
    old = maj[maj.between(4, 6)].index.to_numpy(); new = maj[maj.between(7, 8)].index.to_numpy()
    rs = np.random.RandomState(seed); rs.shuffle(old)
    cut = int(0.85 * len(old))
    s2 = (old[:cut], old[cut:], new)
    year = pd.to_datetime(meta["timeStamp"], unit="s").dt.year
    o = year[year <= 2021].index.to_numpy(); t = year[year >= 2022].index.to_numpy()
    rs.shuffle(o)
    cut = int(0.85 * len(o))
    s3 = (o[:cut], o[cut:], t)
    return {"S1": s1, "S2": s2, "S3": s3}

# --------------------------------------------------------------- metrics ----
def f1(yt, yp):
    tp = int(((yt == 1) & (yp == 1)).sum()); fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

def tune_thresholds(y_true, probs):
    t = np.full(y_true.shape[1], 0.5)
    for j in range(y_true.shape[1]):
        best, bt = -1.0, 0.5
        for th in np.arange(0.05, 0.96, 0.05):
            v = f1(y_true[:, j], (probs[:, j] >= th).astype(int))
            if v > best:
                best, bt = v, th
        t[j] = bt
    return t

def all_metrics(y_true, pred, probs):
    from sklearn.metrics import roc_auc_score, average_precision_score, hamming_loss
    pf = [f1(y_true[:, j], pred[:, j]) for j in range(8)]
    pa, pap = [], []
    for j in range(8):
        try: pa.append(roc_auc_score(y_true[:, j], probs[:, j]))
        except ValueError: pa.append(np.nan)
        try: pap.append(average_precision_score(y_true[:, j], probs[:, j]))
        except ValueError: pap.append(np.nan)
    out = {"macro_f1": float(np.mean(pf)),
           "micro_f1": f1(y_true.ravel(), pred.ravel()),
           "hamming": float(hamming_loss(y_true, pred)),
           "subset_acc": float((pred == y_true).all(1).mean()),
           "macro_ap": float(np.nanmean(pap))}
    out.update({f"f1_{LABEL_SHORT[j]}": pf[j] for j in range(8)})
    out.update({f"auc_{LABEL_SHORT[j]}": pa[j] for j in range(8)})
    return out

def bootstrap_delta(pred_a, pred_b, y_true, n=10000, seed=0):
    rs = np.random.RandomState(seed); N = len(y_true); d = []
    fa = [f1(y_true[:, j], pred_a[:, j]) for j in range(8)]
    fb = [f1(y_true[:, j], pred_b[:, j]) for j in range(8)]
    for _ in range(n):
        idx = rs.randint(0, N, N)
        da = np.mean([f1(y_true[idx, j], pred_a[idx, j]) for j in range(8)])
        db = np.mean([f1(y_true[idx, j], pred_b[idx, j]) for j in range(8)])
        d.append(da - db)
    d = np.array(d)
    return float(np.mean(fa) - np.mean(fb)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

# --------------------------------------------------------- results / io ----
def append_results(root, stage, rows):
    p = os.path.join(root, "results", f"{stage}_results.csv")
    df = pd.DataFrame(rows)
    if os.path.exists(p):
        old = pd.read_csv(p)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(p, index=False)
    return p

def write_summary(root, stage):
    p = os.path.join(root, "results", f"{stage}_results.csv")
    df = pd.read_csv(p)
    keys = [c for c in ["model", "view", "split", "seed"] if c in df.columns]
    df = df.drop_duplicates(subset=keys, keep="first")   # guard against re-run dupes
    num = df.select_dtypes(include=[np.number]).columns
    summ = df.groupby([c for c in keys if c != "seed"])[list(num)].mean().reset_index()
    sd = df.groupby([c for c in keys if c != "seed"])[list(num)].std().reset_index()
    for c in num:
        summ[c + "_sd"] = sd[c]
    sp = os.path.join(root, "results", f"{stage}_summary.csv")
    summ.to_csv(sp, index=False)
    return summ, sp

def tex_table(summ, path, cols, caption, label, metric="macro_f1"):
    s = summ.sort_values(metric, ascending=False)
    lines = ["\\begin{table*}[t]", "\\centering", f"\\caption{{{caption}}}",
             f"\\label{{{label}}}", "\\small",
             "\\begin{tabular}{@{}lll" + "c" * len(cols) + "@{}}", "\\toprule",
             "Model & View & Split & " + " & ".join(cols) + " \\\\", "\\midrule"]
    for _, r in s.iterrows():
        lines.append(f"{r['model']} & {r['view']} & {r['split']} & "
                     + " & ".join(f"{r[c]:.3f}" for c in cols) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    open(path, "w").write("\n".join(lines))
    return path

# ------------------------------------------------------- plotting (style) ---
OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00",
         "#F0E442", "#999999", "#000000"]

def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5,
        "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5, "axes.linewidth": 0.7, "xtick.major.width": 0.7,
        "ytick.major.width": 0.7, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.linewidth": 0.4,
        "grid.color": "#000000", "grid.alpha": 0.15, "legend.frameon": False,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.prop_cycle": matplotlib.cycler(color=OKABE)})
    return plt

def savefig(fig, root, name, log=None):
    import matplotlib.pyplot as plt
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(root, "plots", f"{name}.{ext}"))
    plt.close(fig)
    if log:
        log.info("plot saved: %s (.pdf/.png)", name)

def plot_label_prevalence(lab, root, log=None):
    plt = style(); fig, ax = plt.subplots(figsize=(3.5, 2.4))
    cnt = lab[LABELS].sum().sort_values()
    ax.barh(range(len(cnt)), cnt.values, color=OKABE[0], height=0.72)
    ax.set_yticks(range(len(cnt)))
    ax.set_yticklabels([LABEL_SHORT[LABELS.index(c)] for c in cnt.index])
    ax.set_xlabel("positive contracts")
    for i, v in enumerate(cnt.values):
        ax.text(v + cnt.max() * 0.01, i, f"{v:,} ({100 * v / len(lab):.1f}%)", va="center", fontsize=7)
    ax.set_xlim(0, cnt.max() * 1.30)
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); savefig(fig, root, "s1_label_prevalence", log)

def plot_cooccurrence(lab, root, log=None):
    plt = style(); Y = lab[LABELS].to_numpy(int)
    M = np.zeros((8, 8))
    for i in range(8):
        di = Y[:, i] == 1
        if di.sum():
            M[i] = Y[di].mean(0)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    im = ax.imshow(M, cmap="magma", vmin=0, vmax=max(M.max(), 1e-6))
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.set_xticklabels(LABEL_SHORT, rotation=45, ha="right")
    ax.set_yticklabels(LABEL_SHORT)
    for i in range(8):
        for j in range(8):
            if M[i, j] > 0:
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.6,
                        color="white" if M[i, j] > 0.4 * M.max() else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("P(label j $\\mid$ label i)", fontsize=7.5)
    ax.set_title("label co-occurrence", fontsize=8); ax.grid(visible=False)
    fig.tight_layout(); savefig(fig, root, "s1_cooccurrence", log)
    return M

def plot_split_sizes(sizes, root, log=None):
    plt = style(); fig, ax = plt.subplots(figsize=(3.5, 2.3))
    keys = list(sizes); w = 0.26
    for i, part in enumerate(["train", "val", "test"]):
        ax.bar(np.arange(len(keys)) + i * w, [sizes[k][part] for k in keys], w,
               label=part, color=OKABE[i])
    ax.set_xticks(np.arange(len(keys)) + w); ax.set_xticklabels(keys)
    ax.set_ylabel("contracts"); ax.set_yscale("log")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout(); savefig(fig, root, "s1_split_sizes", log)

def plot_model_macro_f1(summ, root, name, log=None, views=("pre", "post", "all"),
                        splits=("S1",)):
    plt = style()
    s = summ[summ["split"].isin(splits)] if "split" in summ.columns else summ
    models = list(s["model"].unique())
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    w = 0.8 / len(views)
    for i, v in enumerate(views):
        sub = s[s["view"] == v].set_index("model")
        xs = np.arange(len(models)) + i * w
        m = [sub["macro_f1"].get(mo, np.nan) for mo in models]
        sd = [sub["macro_f1_sd"].get(mo, np.nan) if "macro_f1_sd" in sub else np.nan for mo in models]
        ax.bar(xs, m, w, yerr=sd, capsize=2, color=OKABE[i % len(OKABE)],
               label=f"$\\mathcal{{F}}_{{\\mathrm{{{v}}}}}$", error_kw={"lw": 0.6})
    ax.set_xticks(np.arange(len(models)) + w * (len(views) - 1) / 2)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("macro-F1"); ax.set_ylim(0, 1); ax.legend(ncol=len(views))
    fig.tight_layout(); savefig(fig, root, name, log)

def plot_perlabel_f1(summ, models, root, name, log=None):
    plt = style(); fig, ax = plt.subplots(figsize=(7.0, 2.5))
    w = 0.8 / len(models); x = np.arange(8)
    for i, mo in enumerate(models):
        r = summ[summ["model"] == mo]
        vals = [r[f"f1_{ls}"].mean() for ls in LABEL_SHORT]
        sds = [r[f"f1_{ls}"].std() if len(r) > 1 else 0.0 for ls in LABEL_SHORT]
        ax.bar(x + i * w, vals, w, yerr=sds, capsize=2, color=OKABE[i % len(OKABE)],
               label=mo, error_kw={"lw": 0.6})
    ax.set_xticks(x + w * (len(models) - 1) / 2); ax.set_xticklabels(LABEL_SHORT)
    ax.set_ylabel("F1"); ax.set_ylim(0, 1); ax.legend(ncol=len(models), fontsize=6.8)
    fig.tight_layout(); savefig(fig, root, name, log)

def plot_shift_degradation(summ, root, log=None):
    plt = style(); fig, ax = plt.subplots(figsize=(5.6, 2.6))
    s = summ[summ["view"] == "all"]
    models = list(s["model"].unique()); w = 0.26
    s1 = s[s["split"] == "S1"].set_index("model")["macro_f1"]
    if len(s1):                       # order by in-distribution performance so
        models = sorted(models, key=lambda mo: -s1.get(mo, -1.0))  # the shift
    CLASS = {"LGBM-BR", "RF-BR", "CC-ens", "RAkEL", "ML-kNN"}      # ordering
    plt.rcParams["hatch.linewidth"] = 0.7                          # is visible
    for i, sp in enumerate(["S1", "S2", "S3"]):
        sub = s[s["split"] == sp].set_index("model")
        m = [sub["macro_f1"].get(mo, np.nan) for mo in models]
        sd = [sub["macro_f1_sd"].get(mo, np.nan) if "macro_f1_sd" in sub else np.nan for mo in models]
        ax.bar(np.arange(len(models)) + i * w, m, w, yerr=sd, capsize=2, color=OKABE[i],
               label=sp, error_kw={"lw": 0.6},
               hatch=["//" if mo in CLASS else None for mo in models],
               edgecolor="white")
    ax.set_xticks(np.arange(len(models)) + w)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("macro-F1 ($\\mathcal{F}_{\\mathrm{all}}$)"); ax.set_ylim(0, 1)
    from matplotlib.patches import Patch
    h, l = ax.get_legend_handles_labels()
    h += [Patch(facecolor="0.88", edgecolor="white", hatch="//",
                label="classical/transf."), Patch(facecolor="0.88", label="deep")]
    ax.legend(handles=h, ncol=5, fontsize=6.5)
    fig.tight_layout(); savefig(fig, root, "s4_shift_degradation", log)

def plot_dependency_maps(M, A, attn, root, log=None):
    plt = style(); fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35))
    for ax, mat, ti in zip(axes, [M, A, attn],
                           ["empirical co-occurrence M", "LGC adjacency A",
                            "LAT label--label attention"]):
        own = mat is attn
        im = ax.imshow(mat, cmap="magma",
                       vmin=mat.min() if own else 0,
                       vmax=mat.max() if own else max(mat.max(), 1e-6))
        ax.set_xticks(range(8)); ax.set_yticks(range(8))
        ax.set_xticklabels(LABEL_SHORT, rotation=45, ha="right", fontsize=6.5)
        ax.set_yticklabels(LABEL_SHORT, fontsize=6.5)
        ax.set_title(ti, fontsize=7.5); ax.grid(visible=False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("attention (own scale)" if mat is attn else
                     ("P(j|i)" if mat is M else "A"), fontsize=6.5)
    fig.tight_layout(); savefig(fig, root, "s5_dependency_maps", log)

def plot_robustness(res, root, log=None):
    plt = style(); fig, ax = plt.subplots(figsize=(4.6, 2.4))
    models = list(res); w = 0.26
    for i, c in enumerate(["full", "agreement", "majority"]):
        ax.bar(np.arange(len(models)) + i * w, [res[m][c] for m in models], w,
               color=OKABE[i], label=c)
    ax.set_xticks(np.arange(len(models)) + w)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("macro-F1"); ax.set_ylim(0, 1); ax.legend(ncol=3)
    fig.tight_layout(); savefig(fig, root, "s5_label_robustness", log)

def plot_family_ablation(drop, root, log=None):
    plt = style(); fig, ax = plt.subplots(figsize=(5.0, 2.5))
    fams = list(drop); models = list(next(iter(drop.values()))); w = 0.8 / len(models)
    for i, mo in enumerate(models):
        ax.bar(np.arange(len(fams)) + i * w, [drop[f][mo] for f in fams], w,
               color=OKABE[i % len(OKABE)], label=mo)
    ax.set_xticks(np.arange(len(fams)) + w * (len(models) - 1) / 2)
    ax.set_xticklabels(fams, rotation=25, ha="right")
    ax.set_ylabel("$\\Delta$ macro-F1 when family removed"); ax.legend()
    vals = [drop[f][mo] for f in fams for mo in models]
    ax.axhline(0, color="0.3", lw=0.7)          # negative = family hurts when
    ax.set_ylim(min(-0.01, min(vals) - 0.005),  # present; keep it visible
                max(vals) * 1.08)
    fig.tight_layout(); savefig(fig, root, "s3_family_ablation", log)

# ------------------------------------------------------------- torch models -
def _t():
    import torch, torch.nn as nn
    return torch, nn

def _make_classes():
    import torch, torch.nn as nn
    import torch.nn.functional as F

    class FamilyEncoder(nn.Module):
        def __init__(self, fams, dm=128):
            super().__init__()
            self.names = list(fams)
            self.proj = nn.ModuleList([
                nn.Sequential(nn.Linear(len(fams[n]), dm), nn.GELU(), nn.LayerNorm(dm))
                for n in self.names])
            self.pool = nn.Sequential(nn.Linear(dm * len(self.names), dm), nn.GELU(),
                                      nn.Linear(dm, dm))
        def forward(self, xd):
            z = torch.stack([p(xd[n]) for p, n in zip(self.proj, self.names)], 1)
            return z, self.pool(z.reshape(z.shape[0], -1))

    class LGC(nn.Module):
        """Explicit dependencies: GCN over the label co-occurrence graph."""
        def __init__(self, fams, M, dm=128, tau=0.2, layers=2):
            super().__init__()
            self.enc = FamilyEncoder(fams, dm)
            A = np.where(M >= tau, 1.0, np.where(M > 0, 0.02, 0.0)) + np.eye(8)
            D = np.diag(1.0 / np.sqrt(A.sum(1)))
            self.register_buffer("Ahat", torch.tensor(D @ A @ D, dtype=torch.float32))
            self.E = nn.Parameter(torch.randn(8, dm) * 0.08)
            self.W = nn.ModuleList([nn.Linear(dm, dm) for _ in range(layers)])
            self.s = nn.Parameter(torch.tensor(10.0))
        def forward(self, xd):
            E = self.E
            for w in self.W:
                E = F.gelu(w(self.Ahat @ E))
            return torch.sigmoid(self.s * F.normalize(self.enc(xd)[1], dim=1)
                                 @ F.normalize(E, dim=1).t())

    class LAT(nn.Module):
        """Implicit dependencies: joint self-attention over feature + label tokens."""
        def __init__(self, fams, dm=128, layers=3, heads=4, mask_ll=False, trunk_only=False):
            super().__init__()
            self.enc = FamilyEncoder(fams, dm)
            self.trunk_only = trunk_only
            self.k = len(fams)
            if not trunk_only:
                self.lab = nn.Parameter(torch.randn(8, dm) * 0.08)
            layer = nn.TransformerEncoderLayer(dm, heads, 2 * dm, 0.1, batch_first=True,
                                               norm_first=True)
            self.tf = nn.TransformerEncoder(layer, layers)
            n = self.k if trunk_only else self.k + 8
            mask = torch.zeros(n, n, dtype=torch.bool)
            if mask_ll:                       # block label<->label, keep self
                mask[self.k:, self.k:] = ~torch.eye(8, dtype=torch.bool)
            self.register_buffer("mask", mask)
            self.ln = nn.LayerNorm(dm)
            self.head = nn.ModuleList([nn.Linear(dm, 1) for _ in range(8)])
        def forward(self, xd):
            z, _h = self.enc(xd)
            m = self.mask if self.mask.any() else None
            if self.trunk_only:
                hp = self.tf(z, mask=m).mean(1)
                return torch.sigmoid(torch.cat([h(self.ln(hp)) for h in self.head], 1))
            t = torch.cat([z, self.lab.unsqueeze(0).expand(z.shape[0], -1, -1)], 1)
            lt = self.ln(self.tf(t, mask=m)[:, self.k:])
            return torch.sigmoid(torch.cat([h(lt[:, j]) for j, h in enumerate(self.head)], 1))

    class WIDENNET(nn.Module):
        """Re-implementation of Osei et al. 2024 (wide & deep over opcode counts)."""
        def __init__(self, n_op, dm=128):
            super().__init__()
            self.wide = nn.Linear(n_op, 8)
            self.deep = nn.Sequential(nn.Linear(n_op, dm), nn.ReLU(), nn.Linear(dm, dm),
                                      nn.ReLU(), nn.Linear(dm, 8))
        def forward(self, x):
            return torch.sigmoid(self.wide(x) + self.deep(x))

    return FamilyEncoder, LGC, LAT, WIDENNET

FamilyEncoder, LGC, LAT, WIDENNET = _make_classes()

# ---------------------------------------------------------------- training --
def cooccurrence(y):
    M = np.zeros((8, 8))
    for i in range(8):
        di = y[:, i] == 1
        if di.sum():
            M[i] = y[di].mean(0)
    np.fill_diagonal(M, 1.0)
    return M

def train_torch(model, data, cfg, log, kind="fam"):
    """data: x_tr/x_va/x_te (dict of family arrays, or ndarray if kind=='op'),
    y_tr/y_va/y_te. Returns (test probs at best val epoch, history)."""
    torch, nn = _t()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    data = _stage_data(data, kind, dev)
    pw = torch.tensor((1 - data["y_tr"].mean(0)) / np.clip(data["y_tr"].mean(0), 1e-6, None),
                      dtype=torch.float32).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    hist, best = [], (-1.0, None, None)
    idx_all = np.arange(len(data["y_tr"]))
    for ep in range(cfg["epochs"]):
        model.train()
        rs = np.random.RandomState(ep); rs.shuffle(idx_all)
        loss_v = np.nan
        for i in range(0, len(idx_all), cfg["batch"]):
            idx = idx_all[i:i + cfg["batch"]]
            xb = _mk_batch(data, "tr", idx, kind, dev)
            yb = torch.tensor(data["y_tr"][idx], dtype=torch.float32).to(dev)
            out = model(xb)
            loss = (F_binary_cross_entropy(out, yb) * pw).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            loss_v = float(loss)
        sched.step()
        model.eval()
        vp = predict(model, data, "va", kind)
        vm = all_metrics(data["y_va"], (vp >= 0.5).astype(int), vp)["macro_f1"]
        hist.append({"epoch": ep, "loss": loss_v, "val_macro_f1": vm})
        if vm > best[0]:
            best = (vm, {k: v.cpu().clone() for k, v in model.state_dict().items()},
                    predict(model, data, "te", kind))
        if log:
            log.info("  ep %03d loss %.4f val_macro_f1 %.4f%s", ep, loss_v, vm,
                     " *" if vm == best[0] else "")
    if best[1] is not None:
        model.load_state_dict({k: v.to(dev) for k, v in best[1].items()})
    return best[2], hist

def F_binary_cross_entropy(out, y):
    import torch.nn.functional as F
    return F.binary_cross_entropy(out, y, reduction="none")

def _stage_data(data, kind, dev):
    """GPU fast path: move feature parts to device tensors once per run so
    training steps index on-device instead of re-converting numpy slices and
    copying them host->device every batch. CPU runs keep the numpy path."""
    torch, _ = _t()
    if dev.type != "cuda":
        return data
    out = dict(data)
    for part in ("tr", "va", "te"):
        x = data["x_" + part]
        if kind == "op":
            out["x_" + part] = torch.tensor(x, dtype=torch.float32, device=dev)
        else:
            out["x_" + part] = {n: torch.tensor(v, dtype=torch.float32, device=dev)
                                for n, v in x.items()}
    out["_staged"] = True
    return out

def _mk_batch(data, part, idx, kind, dev):
    torch, _ = _t()
    if data.get("_staged"):
        tidx = torch.as_tensor(np.asarray(idx), device=dev)
        x = data["x_" + part]
        return x[tidx] if kind == "op" else {n: v[tidx] for n, v in x.items()}
    if kind == "op":
        return torch.tensor(data["x_" + part][idx], dtype=torch.float32).to(dev)
    return {n: torch.tensor(v[idx], dtype=torch.float32).to(dev)
            for n, v in data["x_" + part].items()}

def predict(model, data, part, kind="fam"):
    torch, _ = _t()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    n = len(data["y_" + part])
    outs = []
    with torch.no_grad():
        for i in range(0, n, 4096):
            idx = np.arange(i, min(i + 4096, n))
            xb = _mk_batch(data, part, idx, kind, dev)
            outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs)

# ------------------------------------------------- classical baselines ------
# Fit-once / predict-many: every classical model is trained a single time per
# (view, seed) and then applied to validation and test parts. This halves the
# classical runtime and makes incremental checkpointing possible.
def fit_classical(name, Xtr, ytr, seed=0):
    """Return a callable proba(X) -> (n, 8) for a fitted classical baseline."""
    if name == "RF-BR":
        from sklearn.ensemble import RandomForestClassifier
        ms = [RandomForestClassifier(300, n_jobs=4, random_state=seed)
              .fit(Xtr, ytr[:, j]) for j in range(8)]
        return lambda X: np.column_stack([m.predict_proba(X)[:, 1] for m in ms])
    if name == "LGBM-BR":
        import lightgbm as lgb
        ms = [lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                                 verbose=-1, random_state=seed)
              .fit(Xtr, ytr[:, j]) for j in range(8)]
        return lambda X: np.column_stack([m.predict_proba(X)[:, 1] for m in ms])
    if name == "CC-ens":
        sc, chains = _cc_fit(Xtr, ytr, seed)
        return lambda X: _cc_proba(chains, sc.transform(X))
    if name == "RAkEL":
        sc, sub = _rakel_fit(Xtr, ytr, seed)
        return lambda X: _rakel_proba(sub, sc.transform(X))
    if name == "ML-kNN":
        st = _mlknn_fit(Xtr, ytr)
        return lambda X: _mlknn_proba(st, X)
    raise ValueError(name)

def run_classical(name, Xtr, ytr, Xte, seed=0):
    return fit_classical(name, Xtr, ytr, seed)(Xte)

def _cc_fit(Xtr, ytr, seed, chains=10):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr); Xtr = sc.transform(Xtr)
    out = []
    for c in range(chains):
        rs = np.random.RandomState(seed * 100 + c); order = rs.permutation(8)
        chain, etr = [], Xtr.copy()
        for pos, j in enumerate(order):
            lr = LogisticRegression(max_iter=100).fit(etr, ytr[:, j])
            chain.append((j, lr))
            if pos < 7:
                etr = np.hstack([etr, ytr[:, order[:pos + 1]]])
        out.append(chain)
    return [sc, out]

def _cc_proba(chains, X):
    P = np.zeros((len(X), 8))
    for chain in chains:
        acc = np.zeros((len(X), 8)); ete = X.copy()
        for pos, (j, lr) in enumerate(chain):
            acc[:, j] = lr.predict_proba(ete)[:, 1]
            if pos < len(chain) - 1:
                ete = np.hstack([ete, acc[:, [c[0] for c in chain[:pos + 1]]]])
        P += acc / len(chains)
    return P

def _rakel_fit(Xtr, ytr, seed, k=3, n_sets=12):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr); Xtr = sc.transform(Xtr)
    rs = np.random.RandomState(seed); sub = []
    for _ in range(n_sets):
        ss = sorted(rs.choice(8, k, replace=False).tolist())
        yss = ytr[:, ss].dot(1 << np.arange(k - 1, -1, -1))
        sub.append((ss, LogisticRegression(max_iter=100, solver="liblinear").fit(Xtr, yss)))
    return [sc, sub]

def _rakel_proba(sub, X):
    P = np.zeros((len(X), 8)); C = np.zeros(8)
    for ss, lr in sub:
        k = len(ss); pr = lr.predict(X)
        for cls in np.unique(pr):
            bits = [(cls >> (k - 1 - b)) & 1 for b in range(k)]
            m = pr == cls
            for b, j in zip(bits, ss):
                if b:
                    P[m, j] += 1
                C[j] += 1
    return P / np.clip(C, 1, None)

def _mlknn_fit(Xtr, ytr, k=10, s=1.0):
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr); Xtr = sc.transform(Xtr)
    nn_ = NearestNeighbors(n_neighbors=k).fit(Xtr)
    cnt_tr = ytr[nn_.kneighbors(Xtr, return_distance=False)].sum(1)
    like = {}
    for j in range(ytr.shape[1]):
        for v in (0, 1):
            msk = ytr[:, j] == v
            b = np.bincount(cnt_tr[msk, j], minlength=k + 1).astype(float) + s
            like[(j, v)] = b / b.sum()
    return {"nn": nn_, "ytr": ytr, "like": like, "sc": sc,
            "p1": (ytr.sum(0) + s) / (len(ytr) + 2 * s)}

def _mlknn_proba(st, X):
    X = st["sc"].transform(X)
    cnt = st["ytr"][st["nn"].kneighbors(X, return_distance=False)].sum(1)
    P = np.zeros((len(X), st["ytr"].shape[1]))
    for j in range(P.shape[1]):
        num = st["like"][(j, 1)][cnt[:, j]] * st["p1"][j]
        den = num + st["like"][(j, 0)][cnt[:, j]] * (1 - st["p1"][j])
        P[:, j] = num / np.clip(den, 1e-12, None)
    return P

# ------------------------------------------------------------ kaggle push ---
def push_dataset(root, stage, slug, title, log=None):
    u, k = kaggle_creds()
    if not (u and k):
        if log:
            log.warning("no Kaggle credentials -- skipping push of %s", slug)
        return None

    # Kaggle hard limits: slug 6-50 chars, title 6-50 chars
    if not (6 <= len(slug) <= 50):
        raise ValueError(f"slug must be 6-50 chars, got {len(slug)}: {slug!r}")
    if len(title) > 50:
        trimmed = title[:50].rstrip()
        if log: log.info("title truncated to 50 chars: %r -> %r", title, trimmed)
        title = trimmed
    if len(title) < 6:
        raise ValueError(f"title must be at least 6 chars, got {len(title)}: {title!r}")

    os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = u, k
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    meta = os.path.join(root, "dataset-metadata.json")
    json.dump({"id": f"{u}/{slug}", "title": title,
               "licenses": [{"name": "CC-BY-4.0"}]}, open(meta, "w"), indent=1)
    url = f"https://www.kaggle.com/datasets/{u}/{slug}"
    try:
        api.dataset_create_version(root, f"update: {stage} run", quiet=False, dir_mode="zip")
        if log: log.info("pushed new version -> %s", url)
    except Exception as e:
        if log: log.info("create_version failed (%s); creating new dataset instead", e)
        if hasattr(api, "dataset_create_new"):          # kaggle >= 2.x
            api.dataset_create_new(folder=root, quiet=False, dir_mode="zip", public=False)
        else:                                           # kaggle 1.x fallback
            api.datasets_create_new(meta, root, quiet=False, dir_mode="zip")
        if log: log.info("created dataset -> %s", url)
    return url

# ------------------------------------------------- orchestration helpers ----
def make_data(pre, post, lab, fam, view, ids_tr, ids_va, ids_te):
    """Leakage-safe feature matrices for one view: per-family dict (neural)
    plus a flat matrix (classical baselines). Pre- and post-deployment sources
    are transformed separately because opcode column names collide."""
    cats = {**CATS_PRE, **CATS_POST}
    vf = view_families(fam, view)
    pre_need = [c for n in PRE_FAMS if n in vf for c in vf[n]]
    post_need = [c for n in POST_FAMS if n in vf for c in vf[n]]
    src = {}
    if pre_need:
        src["pre"] = {"df": pre.set_index("contractID")[pre_need], "cols": pre_need}
    if post_need:
        src["post"] = {"df": post.set_index("contractID")[post_need], "cols": post_need}
    lab_i = lab.set_index("contractID")
    parts = {}
    for part, ids in (("tr", ids_tr), ("va", ids_va), ("te", ids_te)):
        s = set(ids)
        parts[part] = {k: v["df"][v["df"].index.isin(s)] for k, v in src.items()}
    preps = {k: Prep(v["cols"], {c: cc for c, cc in cats.items() if c in v["cols"]})
             .fit(parts["tr"][k].reset_index()) for k, v in src.items()}
    out = {"fams": vf}
    for part in ("tr", "va", "te"):
        A, offs = {}, {}
        blocks = []
        for k in src:
            Ak = preps[k].transform(parts[part][k].reset_index())
            offs[k] = {c: j for j, c in enumerate(src[k]["cols"])}
            A[k] = Ak; blocks.append(Ak)
        out["x_" + part] = {}
        for n in vf:
            k = "pre" if n in PRE_FAMS else "post"
            out["x_" + part][n] = A[k][:, [offs[k][c] for c in vf[n]]]
        out["X_" + part] = np.hstack(blocks)
        ids_p = parts[part][next(iter(src))].index.to_numpy()
        out["ids_" + part] = ids_p
        out["y_" + part] = lab_i.loc[ids_p, LABELS].to_numpy(int)
    return out

def lgc_A(M, tau):
    return np.where(M >= tau, 1.0, np.where(M > 0, 0.02, 0.0))

def extract_label_attention(model, data, kind="fam"):
    """Mean label<->label attention of a trained LAT, averaged over layers,
    heads and test instances (row-normalised)."""
    torch, nn = _t()
    dev = next(model.parameters()).device
    store = []
    hooks = []
    origs = []
    for layer in model.tf.layers:
        m = layer.self_attn
        origs.append((m, m.forward))
        def _fw(*a, _o=None, **kw):         # torch>=2 encoder layers request
            kw = dict(kw)                   # need_weights=False (fast path),
            kw["need_weights"] = True       # which would starve the hooks
            kw["average_attn_weights"] = True
            return _o(*a, **kw)
        m.forward = lambda *a, _o=m.forward, **kw: _fw(*a, _o=_o, **kw)
        def hk(mod, inp, outp):
            if isinstance(outp, tuple) and len(outp) == 2 and outp[1] is not None:
                store.append(outp[1].detach().cpu())
        hooks.append(m.register_forward_hook(hk))
    k = model.k
    with torch.no_grad():
        for i in range(0, len(data["y_te"]), 2048):
            idx = np.arange(i, min(i + 2048, len(data["y_te"])))
            xb = _mk_batch(data, "te", idx, kind, dev)
            model(xb)
    for h in hooks:
        h.remove()
    for m, orig in origs:
        m.forward = orig
    if not store:
        warnings.warn("extract_label_attention: hooks captured no weights; "
                      "returning identity fallback")
        return np.eye(8)
    W = torch.cat(store, 0)                     # (N, n, n), head-averaged
    A = W[:, k:, k:].mean(0).numpy()            # label-label block, instance-mean
    A = A / np.clip(A.sum(1, keepdims=True), 1e-9, None)
    return A

def save_splits(root, splits, pre, lab):
    meta = pre.set_index("contractID")
    rows = {"contractID": lab["contractID"].to_numpy()}
    for s in ("S1", "S2", "S3"):
        tr, va, te = splits[s]
        part = pd.Series("train", index=tr).combine_first(
            pd.Series("val", index=va)).combine_first(pd.Series("test", index=te))
        rows[s + "_part"] = lab["contractID"].map(part).fillna("excl").to_numpy()
    rows["compiler_minor"] = lab["contractID"].map(meta["CompilerVersion"].map(compiler_minor)).to_numpy()
    rows["year"] = lab["contractID"].map(
        pd.to_datetime(meta["timeStamp"], unit="s").dt.year).to_numpy()
    df = pd.DataFrame(rows)
    p = os.path.join(root, "splits.csv")
    df.to_csv(p, index=False)
    return p

def find_splits():
    candidates = [
        "/kaggle/input/scvd-stage1-eda/splits.csv",
        os.path.join(os.environ.get("SCVD_OUT", "./out"), "stage1_eda", "splits.csv"),
        "./splits.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # fallback: walk /kaggle/input (handles the
    # /kaggle/input/datasets/<owner>/scvd-stage1-eda/splits.csv layout, and any
    # other owner/slug nesting) for a splits.csv under a scvd-stage1-eda dir
    for base in ("/kaggle/input", os.environ.get("SCVD_DATA", ".")):
        if not os.path.isdir(base):
            continue
        for dirpath, _d, files in os.walk(base):
            if "splits.csv" in files and "scvd-stage1-eda" in dirpath:
                return os.path.join(dirpath, "splits.csv")
    return None

def load_splits():
    p = find_splits()
    if p is None:
        return None, None
    df = pd.read_csv(p)
    splits = {}
    for s in ("S1", "S2", "S3"):
        splits[s] = tuple(df.loc[df[s + "_part"] == part, "contractID"].to_numpy()
                          for part in ("train", "val", "test"))
    return splits, df
