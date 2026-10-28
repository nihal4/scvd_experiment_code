# Changelog

## v2.0 — final code for the revised manuscript (runs of 24–25 September 2026)

All results in the revised manuscript (JCS-92825) come from this version.

### Fixed
- **Seeding of deep models.** v1.0 called only `np.random.seed(seed)`, which does not
  seed PyTorch, so weight initialisation and dropout were not controlled by the seed.
  `scvd.set_seed(seed)` now seeds Python, NumPy and PyTorch (CPU + CUDA) and is called
  before every deep-model run in stages 3–5. All deep-model results were re-run.

### Added
- Stage 4: LAT-mask, WIDENNET, RAkEL and ML-kNN under S2/S3, so all ten models are
  evaluated under both shifts; per-label positive counts of each shifted test split;
  saved test predictions and paired-bootstrap tests on S2/S3.
- Stage 4 (last cell): the S1 paired-bootstrap and Holm-corrected per-label tests
  reported in the paper.
- Stage 5: Spearman correlations over three seeds instead of one;
  layer- and head-resolved attention (`scvd.extract_label_attention_full`);
  opcode-free feature/label circularity ablation (`scvd.make_data_excluding`).
- `macro_auc` in `scvd.all_metrics`.
- `paper_figures/`: script and data for Figures 3 and 5 of the paper.

### Unchanged
- Model architectures, training loop, classical baselines, feature families,
  preprocessing and split construction are identical to v1.0.

## v1.0 — first public version (12 September 2026)
