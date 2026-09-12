# SCVD Kaggle experiment pipeline

Companion code for the paper *Dependency-Aware Multi-Label Deep Learning for Smart
Contract Vulnerability Detection* (Methodology, Section 3). One notebook per
experiment stage; every stage writes **results (CSV), logs, LaTeX tables and
journal-quality plots (vector PDF + 300 dpi PNG)** into its output folder and then
**pushes that folder as a versioned Kaggle dataset** using your credentials.

| Stage | Notebook | Pushed dataset (slug) | Contents (experiments) |
|---|---|---|---|
| 1 | `stage1_data_eda.ipynb` | `scvd-stage1-eda` | schema check, label stats, co-occurrence matrix, S1/S2/S3 splits (`splits.csv` reused by all later stages) |
| 2 | `stage2_baselines.ipynb` | `scvd-stage2-baselines` | E1/E7: RF-BR, LGBM-BR, CC-ens, RAkEL, ML-kNN on S1 × 3 views × 3 seeds |
| 3 | `stage3_deep.ipynb` | `scvd-stage3-deep` | E1/E2: LGC, LAT, LAT-mask, BR-trunk, WIDENNET on S1; hyper-search; family-token ablation; test predictions saved |
| 4 | `stage4_generalization.ipynb` | `scvd-stage4-generalization` | E3/E4 (RQ3): compiler-era shift S2 and temporal shift S3, full suite |
| 5 | `stage5_interpretability.ipynb` | `scvd-stage5-interpretability` | E5/E6: LGC adjacency vs LAT attention vs empirical co-occurrence; tool-agreement and majority-vote robustness |

Every figure is regenerable from the saved artifacts, e.g.
`scvd.plot_model_macro_f1(pd.read_csv('results/stage3_summary.csv'), root, 's3_deep_macro_f1')`.

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
stage 4 ≈ 2–4 h, stage 5 ≈ 1 h.

**Classical baseline speed.** Linear base learners (chains, label-powerset) and
ML-kNN use train-only $z$-scoring and early-converging solvers; measured on the real
DIVE S1 split this cut CC-ens from ~43 min to ~1 min and RAkEL from ~32 min to ~2 min
per run (4-core CPU), with unchanged CC-ens scores.

**Checkpointing / resume.** Stages 2–4 append one row per finished run to
`results/stageN_results.csv`, so a killed or restarted kernel never loses completed
runs. Stage 2 additionally *skips* any (model, view, seed) combination already present
in that CSV when the cell is re-executed, i.e. you can simply re-run the notebook after a
restart and it continues where it stopped. Stages 3–4 do not skip (hyper-parameter search
state is not resumable) but `write_summary` de-duplicates rows, so accidental double runs
cannot corrupt the summary tables.

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
