# SCVD Kaggle experiment pipeline

Companion code for the paper *Do Label Dependencies Help? A Systematic Evaluation of
Dependency-Aware Deep Learning for Smart-Contract Vulnerability Detection*
(Journal of Cyber Security, submission JCS-92825). One notebook per experiment
stage; every stage writes **results (CSV), logs, LaTeX tables and journal-quality
plots (vector PDF + 300 dpi PNG)** into its output folder and then **pushes that
folder as a versioned Kaggle dataset** using your credentials.

> **Version note.** This is the final code used for every result in the revised
> manuscript (runs of 24–25 September 2026). The notebooks are the executed ones,
> with their saved outputs. See `CHANGELOG.md` for what changed since the first
> public version.

| Stage | Notebook | Pushed dataset (slug) | Contents (experiments) |
|---|---|---|---|
| 1 | `stage1_data_eda.ipynb` | `scvd-stage1-eda` | schema check, label stats, co-occurrence matrix, S1/S2/S3 splits (`splits.csv` reused by all later stages) |
| 2 | `stage2_baselines.ipynb` | `scvd-stage2-baselines` | E1/E7: RF-BR, LGBM-BR, CC-ens, RAkEL, ML-kNN on S1 × 3 views × 3 seeds |
| 3 | `stage3_deep.ipynb` | `scvd-stage3-deep` | E1/E2: LGC, LAT, LAT-mask, BR-trunk, WIDENNET on S1 × 3 views × 3 seeds, with a 16-configuration hyper-search per (model, view); leave-one-family-out ablation; test predictions saved |
| 4 | `stage4_generalization.ipynb` | `scvd-stage4-generalization` | E3/E4 (RQ3): compiler-era shift S2 and temporal shift S3 for **all ten models** × 3 views × 3 seeds; per-label test counts; paired-bootstrap tests on S2/S3; the S1 paired-bootstrap and Holm tests of the paper (last cell) |
| 5 | `stage5_interpretability.ipynb` | `scvd-stage5-interpretability` | E5/E6: LGC adjacency vs LAT attention vs empirical co-occurrence over 3 seeds, layer- and head-resolved attention; tool-agreement and majority-vote robustness; opcode-free circularity ablation |

`paper_figures/` regenerates Figures 3 and 5 of the paper from the archived result
tables (`python make_fig3_fig5.py`, run inside that folder).

## Experimental settings (as run)

* **Seeds.** `scvd.RNG_SEEDS = (0, 1, 2)`. `scvd.set_seed(seed)` seeds Python, NumPy
  and PyTorch (CPU and CUDA) before every deep-model run. Non-deterministic GPU kernels
  are not forced deterministic, so repeated runs can differ slightly.
* **Hyper-search.** Stage 3 only (S1): 16 configurations per (model, view) over
  learning rate {3e-4, 1e-3}, d_m {128, 256}, layers {2, 3} and τ {0.1, 0.3}
  (τ affects only LGC; layers only LAT, LAT-mask and BR-trunk; WIDENNET only lr and
  d_m), i.e. 5 × 3 × 16 = 240 fits, one (unseeded) training run per configuration, selected on
  validation macro-F1; the selected configuration is then retrained with seeds 0, 1, 2.
* **Fixed configuration elsewhere.** Stage 4 (S2/S3), the stage-3 family ablation and
  stage 5 use lr 1e-3, d_m 128, 3 layers, τ = 0.2 for every deep model.
* **Training.** AdamW (weight decay 1e-4), cosine schedule, batch 256, 120 epochs,
  gradient clipping 1.0, BCE with per-label `pos_weight` = (N − N_j)/N_j; checkpoint
  chosen by validation macro-F1 at threshold 0.5; per-label thresholds then tuned on
  validation over {0.05, …, 0.95}.
* **Classical baselines** (fixed, untuned, no class weights): RF 300 trees;
  LightGBM 200 rounds, lr 0.05, 31 leaves; 10 classifier chains (logistic regression);
  RAkEL 12 subsets of size 3 (liblinear logistic regression); ML-kNN k = 10, s = 1.
* **S2** removes `CompilerVersion` from the inputs. **S3** removes no feature; the raw
  timestamp is never an input, but age-correlated transaction features
  (`confirmations`, `nonce`, `NoOfTransactions`) are retained.

## One-time setup (you, ~10 minutes)

1. **Download DIVE** (public, no account needed): <https://zenodo.org/records/18519253>
   * `DIVE_Preprocessed_Data.zip` (16.3 MB) → unzip, then also unzip the nested
     `PRE_ProcessedData.zip` and `POST_ProcessedData.zip`
   * `DIVE_Labels.zip` (0.3 MB) → unzip → `DIVE_Labels.csv`, `Tool_Results.csv`
   * optional: `Feature list.xlsx` (feature registry)
2. **Upload to Kaggle** as one dataset named e.g. *DIVE Smart Contracts*, slug
   `dive-smart-contracts`, containing the four CSVs (flat, no subfolders).
3. **Upload `scvd.py`** as a second (tiny) dataset, slug `scvd-lib`.
4. **Credentials.** Create an API token at <https://www.kaggle.com/settings>
   (“Create New Token”) and store the two values as notebook *Secrets* (or environment
   variables) named exactly `KAGGLE_USERNAME` and `KAGGLE_KEY`; optionally `HF_TOKEN`
   and `DIVE_HF_REPO` if you mirror DIVE on Hugging Face (the pipeline only touches
   Hugging Face when `DIVE_HF_REPO` is set — otherwise DIVE is read from
   `/kaggle/input` or downloaded straight from Zenodo). `scvd.py` reads, in order:
   environment variables → Kaggle User Secrets → `~/.kaggle/kaggle.json`.
5. Attach to each notebook: `scvd-lib`, `dive-smart-contracts`, and the previous
   stage’s dataset (stage 2 ← 1, stage 3 ← 1+2, stage 4 ← 1+3, stage 5 ← 1+3).

## Running

Run stages **in order 1 → 5**. Stages 3–5: enable a GPU accelerator (T4/P100 is
enough; all models are < 2 M parameters). Set the environment variable
`SCVD_QUICK=1` for a fast smoke run (1 seed, 30–60 epochs, reduced hyper-search).
Expected wall time on a T4: stage 1 ≈ 2 min, stage 2 ≈ 40–60 min (CPU-bound), stage 3 ≈ 2–4 h,
stage 4 longer than stage 3 (all ten models under both shifts), stage 5 ≈ 1–2 h.

**Classical baseline speed.** Linear base learners (chains, label-powerset) and
ML-kNN use train-only $z$-scoring and early-converging solvers; measured on the real
DIVE S1 split this cut CC-ens from ~43 min to ~1 min and RAkEL from ~32 min to ~2 min
per run (4-core CPU), with unchanged CC-ens scores.

**Checkpointing / resume.** Stages 2–4 append one row per finished run to
`results/stageN_results.csv`, so a killed or restarted kernel never loses completed
runs. Stage 2 additionally *skips* any (model, view, seed) combination already present
in that CSV when the cell is re-executed, i.e. you can simply re-run the notebook after a
restart and it continues where it stopped. Stage 4 skips already-checkpointed (model, view, split, seed) runs in the same way.
Stage 3 does not skip (hyper-parameter search state is not resumable), but
`write_summary` de-duplicates rows, so accidental double runs cannot corrupt the
summary tables.

Each run appends to `logs/<stage>.log` and `logs/run_env.json` (library versions,
GPU name, seed, quick flag), so any number in a plot can be traced back to a run.

## Output layout per stage (exactly what gets pushed)

```
<kaggle/working>/stageN_*/
├── splits.csv                  (stage 1 only; shared protocol)
├── dataset-metadata.json       (written at push time)
├── results/  stageN_results.csv  (long format: model × view × split × seed × metric)
│           stageN_summary.csv    (mean ± sd over seeds)
│           *.json / *.npz        (split sizes, co-occurrence, ablations, test probs)
├── plots/    s*_*.pdf  +  s*_*.png   (vector PDFs are paper-ready, 300 dpi PNGs for quick look)
├── logs/     stageN.log, run_env.json
└── tables/   stageN_*.tex      (booktabs snippets ready to paste into the paper)
```

## Pushed-dataset layout and Kaggle limits

Pushes use `dir_mode="zip"`, so each subfolder arrives as an archive
(`results.zip`, `plots.zip`, `logs.zip`, `tables.zip`) next to the flat files
(`splits.csv`, `dataset-metadata.json`). `scvd._find_input` is zip-aware: later
stages transparently extract what they need (e.g. `stage3_summary.csv` from a
previous stage's `results.zip`) into a temp dir. Slugs must be 6-50 chars and
titles 6-50 chars (longer titles are auto-truncated, both enforced in
`push_dataset` before any API call); `splits.csv` is additionally discoverable
under any `/kaggle/input/.../scvd-stage1-eda/` nesting.

## Plot style

`scvd.style()` enforces a single journal style: serif text, 8 pt base size,
Okabe–Ito colour-blind-safe palette, top/right spines off, light y-grid, error bars
= ±1 sd over seeds, `pdf.fonttype=42` (TrueType text, editable in Illustrator/Inkscape).
Single-column figures are 3.5 in wide, double-column 7.0 in.
