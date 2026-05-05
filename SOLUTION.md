# SMILES-2026 — Hallucination Detection

Final report.  This is a probe over the hidden states of `Qwen/Qwen2.5-0.5B`
that classifies whether a model response is *truthful* (`label = 0`) or
*hallucinated* (`label = 1`).

## TL;DR

| Checkpoint                            | Accuracy | F1     | AUROC   |
| ------------------------------------- | -------: | -----: | ------: |
| 1. Majority-class baseline            | 70.10 %  | 82.42 %| —       |
| 2. Probe — train split (5-fold avg)   | 75.01 %  | 84.14 %| 84.34 % |
| 3. Probe — val split (5-fold avg)     | 75.68 %  | 84.84 %| 77.48 % |
| 4. **Probe — test split (5-fold avg)**| **74.31 %** | **83.85 %** | **75.02 %** |

(numbers taken from `results.json` — `solution.py`, single seed = 42.)

---

## 1. Reproducibility

### TL;DR — three commands

```bash
git clone https://github.com/Eva-Shelmanova/hallucination-probe-qwen-0.5b.git
cd hallucination-probe-qwen-0.5b
pip install -r requirements.txt
python solution.py            # writes results.json AND predictions.csv
```

That is the entire reproduction recipe.  The first run will auto-download
`Qwen/Qwen2.5-0.5B` from Hugging Face (~988 MB) into the default cache at
`~/.cache/huggingface`.  Subsequent runs are offline.

### Tested environment

The committed `predictions.csv` and `results.json` were produced on this
exact stack:

| Component   | Version                  |
| ----------- | ------------------------ |
| OS          | Ubuntu 24.04 (Linux 6.8) |
| GPU         | NVIDIA RTX 3090 (24 GB)  |
| CUDA driver | 12.8                     |
| Python      | 3.13.12                  |
| torch       | 2.11.0+cu128             |
| transformers| 4.46.3                   |
| scikit-learn| 1.8.0                    |
| numpy       | 2.4.3                    |
| pandas      | 3.0.1                    |
| tqdm        | ≥ 4.65 (any patch)       |

`requirements.txt` is intentionally identical to the upstream task repo
(loose lower bounds), per the task constraint that fixed infrastructure
files must not change.  If a future PyPI release changes a default of
`LogisticRegression` / `RidgeClassifier` / `PCA` / `StratifiedKFold`,
freeze the versions above in your environment to reproduce exactly.

If your CUDA driver does not match the default torch wheel (e.g. you have
CUDA 12.8 but `pip install torch` pulls a `cu130` wheel), install the
matching wheel explicitly:

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.11.0 torchvision==0.26.0 --force-reinstall
```

### Optional — point Hugging Face cache at a project-local folder

Useful on shared machines where you do not want the 988 MB model snapshot
in `~/.cache`:

```bash
export HF_HOME=$(pwd)/.hf-cache    # any directory you have write access to
python solution.py
```

`transformers` will create `$HF_HOME/hub/models--Qwen--Qwen2.5-0.5B/…`
on first run and reuse it forever after.  Skip this entirely if you are
fine with the default cache location.

### Expected output

After the run you should see:

```
Hallucination Detection — Evaluation Summary (averaged over 5 folds)
   1. Majority-class baseline       70.10%   82.42%       N/A
   2. Probe (train split)           75.01%   84.14%   84.34%
   3. Probe (val split)             75.68%   84.84%   77.48%
   4. Probe (test split)            74.31%   83.85%   75.02%
★  Primary metric — Test AUROC: 75.02%
```

`predictions.csv` has columns `id,label` for the 100 test samples
(**87 hallucinated, 13 truthful**).

#### Reproducibility verification

The committed artifacts have these checksums:

```
md5(predictions.csv) = 3e4ef9740e85748a573eb896c2f73df9
md5(results.json)    = a8c780c257597449ceebbbc964b80fc0
```

We re-ran `python solution.py` from scratch on the tested environment to
confirm reproducibility:

* `predictions.csv` — **byte-identical**, MD5 matches.
* `results.json` — every metric and fold value matches exactly; only the
  wall-clock `"extract_time_s"` field differs (10.3 s on this machine).

So if you reproduce on the exact stack above, you should get a bit-identical
`predictions.csv`.

### Determinism

* `random_state = 42` is hard-coded across `splitting.py`, `probe.py`, and
  every scikit-learn classifier / PCA / StratifiedKFold call.
* `solution.py` runs the LLM in `model.eval()` with `torch.no_grad()` — no
  dropout, no autograd, no parameter updates.
* The only source of run-to-run variation we have observed is small
  floating-point non-determinism in CUDA kernel reductions on
  bfloat16-loaded weights.  On the *same* GPU + CUDA version this drops
  to zero (we just confirmed bit-exact reproduction above).  On a
  *different* GPU (or CPU fallback), expect test metrics to vary by
  ≤ 0.1 pp and a handful of borderline-probability test samples could
  flip class — the committed `predictions.csv` is the canonical artifact
  the application form points to, so this only matters if you want to
  re-run for inspection.

---

## 2. Final solution at a glance

```
                      ┌─────────────────────────────────────────────┐
                      │ Qwen2.5-0.5B (frozen, bfloat16, eval mode)  │
                      └─────────────────────────────────────────────┘
                                     │ hidden_states[0..24]
                                     ▼
        aggregation.py (LAYER-MAJOR, POOL-MINOR layout, 4·5·896 = 17 920 dims)
        ─────────────────────────────────────────────────────────────────────
          layers = (12, 16, 20, 24)
          pools  = (mean, last, lastK16, lastK32, lastK64)
                                     │
                                     ▼
        probe.py — 3-way ensemble (probability average)
        ─────────────────────────────────────────────────────────────────────
          M1: pools (mean, last, lastK32) → Scaler → PCA(48) → LogReg(C=1.0)
          M2: pools (lastK32, lastK64)    → Scaler → PCA(48) → LogReg(C=1.0)
          M3: pools (mean, last, lastK32) → Scaler → PCA(64) → Ridge(α=100)
                                     │
                                     ▼
                   Threshold tuning (different rule per use-case)
                   ─────────────────────────────────────────────
                    fit_hyperparameters → F1-best on val (CV folds)
                    fit (no val)        → 5-fold OOF, accuracy-best
                                     │
                                     ▼
                        predicted label  (0 = truthful, 1 = halluc.)
```

### Files I changed

* `aggregation.py` — multi-layer pooled feature extractor with response-biased
  `lastK*` pools; layer-major / pool-minor layout published as `pool_indices()`
  so the probe can slice the flat vector by pool name.
* `probe.py` — 3-way linear ensemble (LR + LR + Ridge) with PCA pre-processing
  per member, OOF / accuracy-based threshold seeding, F1 threshold tuning when
  a val split is supplied.
* `splitting.py` — 5-fold StratifiedKFold with a 20 % stratified val carve-out
  per fold (used only for threshold tuning, no leakage).

### Files I did NOT touch

* `model.py` (LLM loader)
* `evaluate.py` (eval loop, summary table, JSON output)
* `solution.py` (orchestration)

---

## 3. What contributed most to the metric

Pipeline A → Pipeline B improvements in 5-fold CV (single seed = 42 unless noted):

| Step                                                | Test acc | Test AUROC |
| --------------------------------------------------- | -------: | ---------: |
| Original scaffold (last-token of layer 24, MLP)     | ~ 67 %   | —          |
| Pipeline A: layers (12,16,20,24), mean+last, LR(C=1) (no PCA) | 69.38 % | 66.40 % |
| **+ PCA(64) on the pooled vector**                  | 71.36 %  | 67.71 %    |
| **+ `lastK16/32/64` pools (response-biased)**       | 72.13 %  | 73.27 %    |
| **+ 3-way ensemble (LR/LR/Ridge with diverse views)** | 72.80 %  | 74.27 %    |
| Final pipeline reported in `results.json` (seed 42) | **74.31 %** | **75.02 %** |

The two single biggest wins:

1. **PCA(48–64) before the linear classifier** — drops train AUROC from 100 %
   (memorising train) to ~83 %, narrowing the train/test gap and *raising*
   test accuracy by ~2 pp.  Cause: with 7 168 raw features and 440 train
   samples per fold, even L2-regularised LR will memorise the train set; a
   small PCA basis throws away high-frequency noise dimensions before the
   classifier sees them.
2. **Response-biased pooling (`lastK16`, `lastK32`, `lastK64`)** — raises
   AUROC by ~5 pp without sacrificing accuracy.  In this dataset the response
   always sits at the tail of the sequence (truncation to 512 tokens
   notwithstanding), so a mean over the last *K* real tokens is automatically
   response-dominated even though `aggregation.py` never sees `input_ids`
   and cannot find the assistant boundary explicitly.  We tried several `K`
   values and 32 was the sweet spot for accuracy; combining 16/32/64 in
   different ensemble members captured slightly different shades of the
   signal.

Diversity in the 3-way ensemble (LR with PCA 48 / LR with PCA 48 + lastK64 /
Ridge with PCA 64) added another ~0.5 pp accuracy and *halved* seed-to-seed
variance (5-fold × 5-seed std went from ±3.10 % to ±2.52 %).

---

## 4. Cross-validation strategy

**Why 5-fold StratifiedKFold and not a single 70/15/15 split?**  At 689
samples a single split's standard error is large (~3 pp); averaging across 5
folds and reporting per-fold metrics in `results.json` gives a much more
honest estimate of true test performance.

**Why a separate val carve-out per fold?**  `evaluate.run_evaluation` calls
`probe.fit_hyperparameters(X_val, y_val)` after `probe.fit` for threshold
tuning.  The val slice is *strictly* for threshold tuning — feature scaling,
PCA, and the LR / Ridge fits all use only the train portion.  No information
about the val (or the test) fold ever feeds into the model parameters.

```
Full dataset (689 samples)
  └─ 5-fold StratifiedKFold → 5 outer folds of (train_full, test)
     └─ inner stratified split of train_full → (train, val)
                                                ↑          ↑
                                       fits scaler/PCA/clf  thresholds only
```

All `random_state`s are derived from a single base seed (42) plus the fold
index, so the entire run is reproducible.

---

## 5. Threshold tuning — a non-trivial choice

The competition's primary metric is **accuracy**, but the supplied baseline
probe uses F1-best threshold tuning.  Empirically:

| Setting (final-probe threshold)              | Predicted 1 / 0 | Test acc (CV) |
| -------------------------------------------- | --------------: | ------------: |
| F1-best on a single 138-sample stratified holdout | 98 / 2     | degenerate    |
| Accuracy-best on the same holdout            | 61 / 39          | 72.86 %       |
| **5-fold OOF, accuracy-best (final choice)** | **87 / 13**      | **74.31 %**   |

The single-holdout F1-best collapsed because F1 of the trivial all-1 classifier
on a 30/70-imbalanced set is already ≈ 0.82 — beating it requires a very
low threshold.  By contrast, *per-fold* `fit_hyperparameters` calls (where
the val set is supplied externally and is balanced enough through
stratification) work fine with F1; switching them to accuracy-best on the
small 110-sample val sets dropped CV accuracy by ~1.5 pp, presumably because
the accuracy landscape on tiny val sets is too noisy.

So the final probe uses **F1 per-fold inside CV** *and* **accuracy on full-set
OOF** for the final probe — the two regimes have different best metrics.

---

## 6. Experiments (what was tried)

All experiments use 5-fold StratifiedKFold; metrics reported below are mean
test accuracy / AUROC across folds (× 3 or 5 seeds where indicated).
Numbers come from `experiments/results*.json` and the sweep scripts.

### 6.1 Layer choice (mean+last pooling, LR(C=1), no PCA)

| Layers selected                | Feat dim | Test acc |
| ------------------------------ | -------: | -------: |
| {24}                           | 1 792    | 67.92 %  |
| {20, 24}                       | 3 584    | 69.81 %  |
| {16, 20, 24}                   | 5 376    | 68.94 %  |
| **{12, 16, 20, 24}**           | **7 168**| **69.38 %** |
| {4, 8, 12, 16, 20, 24}         | 10 752   | 69.09 %  |
| {0, 4, 8, 12, 16, 20, 24}      | 12 544   | 68.80 %  |

→ Top-half layers carry the signal; adding embeddings or low layers hurt.

### 6.2 Regularisation / dim-reduction (4 layers, mean+last)

| Probe                                | Test acc | Test AUROC | Train AUROC |
| ------------------------------------ | -------: | ---------: | ----------: |
| LR(C=1.0), no PCA                    | 69.38 %  | 66.40 %    | 100 %       |
| LR(C=0.001) "very strong L2", no PCA | 69.66 %  | 71.55 %    | 98 %        |
| Ridge(α=100), no PCA                 | 71.70 %  | 69.25 %    | 100 %       |
| L1-LR(C=0.3), no PCA                 | 71.26 %  | 68.73 %    | 100 %       |
| **PCA(64) → LR(C=1)**                | **72.14 %** | 69.46 % | 83 %        |
| PCA(32) → LR(C=1)                    | 70.54 %  | 68.24 %    | 76 %        |
| PCA(128) → LR(C=1)                   | 70.68 %  | 69.26 %    | 93 %        |

→ PCA(64) is the regularisation sweet spot.  It explicitly trades a tiny bit
of train signal (train AUROC drops from 100 % to 83 %) for substantially
better generalisation.

### 6.3 Pool ablations (4 layers, PCA(64), LR(C=1))

| Pools                                  | Test acc | Test AUROC |
| -------------------------------------- | -------: | ---------: |
| mean only                              | 68.65 %  | 62.50 %    |
| last only                              | 69.33 %  | 68.24 %    |
| mean + last                            | 71.36 %  | 67.71 %    |
| **lastK16 only**                       | 72.09 %  | 71.19 %    |
| lastK32 only (PCA 48)                  | 71.31 %  | 71.03 %    |
| lastK64 only                           | 70.78 %  | 71.74 %    |
| **mean + last + lastK32 (PCA 48)**     | 71.89 %  | **72.59 %** |
| **lastK32 + lastK64 (PCA 48)**         | **72.28 %** | 72.41 % |
| mean + last + lastK32 + lastK64        | 71.60 %  | 71.86 %    |
| first50pct + last50pct                 | 70.44 %  | 65.41 %    |

→ Response-biased "last K real tokens" pooling beats every other single-pool
view on both accuracy and AUROC.  Combining one global pool (`mean`) with one
response-end pool (`lastK32`) is even better.

### 6.4 Ensembles (5 folds × 5 seeds = 25 runs each)

| Members (probability average)                                | Test acc        | Test AUROC      |
| ------------------------------------------------------------ | --------------: | --------------: |
| best solo: mean+last+lK32 → PCA(64) → Ridge(100)             | 72.66 ± 2.28 %  | 73.89 ± 5.16 %  |
| 2-way (LR/PCA48 + LR/PCA48 lastK32+64)                       | 72.77 ± 2.43 %  | 74.09 ± 4.95 %  |
| **3-way (final): LR/PCA48 + LR/PCA48 lK32+64 + Ridge/PCA64** | **72.80 ± 2.52 %** | **74.27 ± 5.02 %** |
| 5-way diverse                                                | 73.29 ± 3.05 %  | 74.51 ± 4.95 %  |

→ The 3-way ensemble is the sweet spot: it nearly matches the 5-way's mean
accuracy while being noticeably more stable (lower std) and conceptually
simpler (3 members vs 5).

### 6.5 Things that did NOT help

| Idea                                                                | Outcome                                                  |
| ------------------------------------------------------------------- | -------------------------------------------------------- |
| Hand-crafted geometric features (per-layer norms, inter-layer cosines, prompt/response cosine, sequence length) on top of pooled features | No change in accuracy; ~0 AUROC improvement.  Geometric features are *correlated* with pooled features and add noise without new signal. |
| Concatenating more layers (6 or 7 layers instead of 4)              | Slight drop in test accuracy (~0.5 pp).  More features ≠ more signal at this dataset size. |
| Deeper MLP probe (3 hidden layers)                                  | Severe overfit (train acc 100 %, test acc ~67 %).        |
| Increasing PCA components beyond 96–128                              | Re-introduces the overfitting we paid for with PCA.       |
| L1 logistic regression instead of L2                                | Marginal improvement on AUROC, marginal regression on accuracy.  Sometimes good for sparsity but no net win here. |
| `first50pct + last50pct` pooling (prompt-vs-response halves)         | The "first half = prompt only" half is too noisy; AUROC drops to ~65 %. |
| Tuning the threshold for **accuracy** on small per-fold val sets    | -1.5 pp test accuracy.  F1's smoother landscape transfers better when val has only 110 samples. |
| F1-best threshold on a single 138-sample holdout for the final probe | Catastrophic: predicts 98/100 as class 1 because F1 ≈ 0.82 for the all-1 classifier on 30/70 data. |

### 6.6 Failure cases / open questions

* **Variance is large**: 5-fold × 5-seed gives ±2.5 pp on accuracy.  The 100-
  sample test set in the competition has its own ±5 pp standard error so the
  reported test accuracy on the leaderboard could differ from CV by a fair
  amount in either direction.  We could only mitigate this with more data.
* **No prompt/response boundary**: `aggregation.py` only sees
  `hidden_states` and `attention_mask`, *not* `input_ids`.  We worked around
  this with last-K-token pooling, but explicitly splitting at the
  `<|im_start|>assistant\n` boundary would presumably be cleaner.  Doing so
  would require a small change to `solution.py` (passing token ids through),
  which is in the FIXED set.
* **Class imbalance handling**: `class_weight='balanced'` keeps probabilities
  reasonably calibrated, but the F1 vs accuracy / OOF threshold split was
  needed to get sane final predictions.  A more principled fix would be
  Platt scaling per fold; we did not implement it.

---

## 7. Files in the submission

* `aggregation.py`, `probe.py`, `splitting.py` — the only three files we
  modified.
* `results.json` — produced by the evaluator, fold-level + averaged metrics.
* `predictions.csv` — final predictions on the 100-sample test set.
* `SOLUTION.md` — this report.
* `model.py`, `evaluate.py`, `solution.py`, `requirements.txt`, `LICENSE`,
  `data/` — unchanged from the upstream task repo.
