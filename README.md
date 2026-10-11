# The Gradient That Top-k Distillation Never Computes

Code and results for the paper *The Gradient That Top-k Distillation Never
Computes* ([arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)).

Most LLM distillation pipelines keep only the teacher's top-k probabilities,
renormalize them and train the student on what remains. We prove that the
renormalized loss depends only on log-odds among the kept tokens, so at each
position its gradient is exactly zero along the directions that change how much
probability the student puts outside them, and that the full-vocabulary gradient
equals the truncated one plus a tail term the truncated loss never computes. On
seven open-weight teachers, the kept set determines the teacher's distribution
only from k = d+1 (d the hidden size), far above the k used in practice.
Caching the teacher's final hidden state and rebuilding exact logits avoids the
problem; this repository includes a memory-bounded implementation of that route.

## What is here

```
code/hsc/           library: rebuild logits from a cached hidden state, chunked
                    full-vocabulary divergences (forward/reverse KL, JSD, skew KL)
                    that never build a (batch x vocabulary) tensor, hidden-state and
                    top-k caches, one distillation step, tail-mass metrics
code/experiments/   one script per experiment in the paper
code/analysis/      statistics (exact paired permutation tests, bootstrap intervals)
code/theory/        numerical checks of the theorems
jobs/               the SLURM scripts behind every reported run
results/            per-run JSON results behind every number in the paper
scripts/            model and dataset download scripts
verify_reported_numbers.py   recomputes the paper's numbers from results/
```

## Check the paper's numbers (CPU, seconds)

```bash
pip install numpy
python3 verify_reported_numbers.py            # exit code 0 iff every check passes
python3 verify_reported_numbers.py --verbose  # computed vs. reported values
python3 code/theory/verify_theory.py          # numerical check of the theorems (about a minute)
```

## Use the library

```python
import torch
from hsc.divergences import forward_kl          # run from code/, or add code/ to PYTHONPATH

# g: cached teacher hidden states (B, d_t); U: teacher unembedding (V, d_t)
# h: student hidden states (B, d_s);       W: student unembedding (V, d_s)
kl = forward_kl(h, W, g, U, chunk=16384)       # exact full-vocabulary KL per position,
kl.mean().backward()                           # memory O(B x chunk), not O(B x V)
```

`hsc.reconstruct.reconstruct_logits(U, g, b, softcap)` rebuilds the teacher's
logits, and `hsc.cache.HiddenStateCache` / `TopKCache` store the two kinds of
cache. Capture `g` after the teacher's final normalization (see
`code/scripts/check_hidden_state_convention.py`).

## Setup for rerunning experiments

```bash
conda env create -f environment.yml && conda activate hsc
cp .env.example .env                 # add HF_TOKEN: the Llama and Gemma checkpoints are gated
bash scripts/download_models.sh      # all teachers and students (~300 GB); pass ids to fetch a subset
bash scripts/download_data.sh        # the four dataset files the experiments read
```

Exact versions are in `requirements-pinned.txt` (Python 3.11, PyTorch 2.13.0 with
CUDA 13.0, Transformers 5.15.1, datasets 5.0.1). Teacher and student must share a
tokenizer.

The job scripts target a SLURM cluster with NVIDIA H200 GPUs (one per job; two
for the Llama-3.1-70B Threshold run). Fill in `<gpu-partition>`, `<account>` and
`<qos>` in their `#SBATCH` lines (or delete those lines), then submit from the
`jobs/` folder, e.g. `cd jobs && sbatch 13_e5_seeds6.sh`. Each job writes to its
folder under `results/`. The whole study took about 294 H200 GPU-hours.

## Where each result comes from

| Paper | Job script(s) | Results |
|---|---|---|
| Exactness | `02_e1_exactness.sh` | `e1_exactness` |
| Threshold (7 teachers, error-vs-k curves) | `01_e2_threshold.sh`, `22_e2_llama70b.sh`, then `code/experiments/merge_e2_threshold.py` | `e2_threshold` |
| Blind-subspace probe | `t1_blind_subspace.sh` | `t1_blind_subspace` |
| Mechanism (alpha_ce = 0) | `03_e5_mechanism.sh` (three seeds), `13_e5_seeds6.sh` (six seeds) | `e5_mechanism_3seed`, `e5_mechanism_6seed` |
| Mechanism (alpha_ce = 1) | `26b_e5_ace1_fixed_topk.sh`, `26c_e5_ace1_fixed_full.sh` | `e5_mechanism_ace1_topk`, `e5_mechanism_ace1_full` |
| Which tokens absorb the drift | `18_r10_absorption.sh`; decode with `code/analysis/decode_absorption.py` | `e5_token_absorption` |
| Drift versus k | `14_t2_drift_vs_k.sh`, `14b_t2_k4096.sh` | `t2_drift_vs_k` |
| Student-selected retained sets | `15_t3_union_select.sh` | `t3_union_select` |
| Residual plateau | `16_t4_residual_plateau_seeds012.sh`, `..._seeds345.sh` | `t4_residual_plateau` |
| Off-policy (cached, online, top-256) | `04_e3_offpolicy.sh` (three seeds), `10_e3_seeds6.sh` (six seeds) | `e3_offpolicy_3seed`, `e3_offpolicy_6seed` |
| Matched storage vs. top-1024 (5 teachers) | `11_e3_m1_arms.sh`, `19_e3_m1_gemma.sh`, `20_e3_m1_qwen4b.sh`, `24_e3_m1_llama3b.sh`, `23_e3_m1_smollm2.sh` | `e3_matched_*` |
| Online arm on the Qwen3 pairs | `27_qwen_online.sh` | `e3_online_qwen3-1.7b`, `e3_online_qwen3-4b` |
| Tail mass beyond top-k | `28_tail_mass.sh` | `tail_mass` |
| Unembedding geometry | `python3 code/experiments/diag_teacher_geometry.py <teacher> --device cpu --out results/geometry` | `geometry` |
| On-policy | `05_e4_onpolicy.sh` (three seeds), `12_e4_seeds6.sh` (six seeds) | `e4_onpolicy_3seed`, `e4_onpolicy_6seed` |
| Systems (storage, write cost) | `06_e6_systems.sh` | `e6_systems` |
| Precision | `07_e8_precision.sh` (synthetic), `07b_e8_precision_real.sh` | `e8_precision_synthetic`, `e8_precision_real` |
| Capacity | `08_e9_capacity.sh` | `e9_capacity` |
| Disagreement | `09_e7_disagreement.sh`, `20_e7_wide_sweep.sh`, `25_e7_wide_sweep_6seed.sh` | `e7_disagreement*` |

## Notes

- **Cross-entropy runs.** An early version computed the cross-entropy term
  without a gradient through its normalizer, so runs with a cross-entropy weight
  above zero were redone or dropped; those rows are removed from
  `e5_mechanism_3seed` and `e5_mechanism_6seed`, and the alpha_ce = 1 arm in the
  paper is the corrected rerun.
- **p-values.** `e3_offpolicy_3seed` stores a Monte Carlo p-value from an older
  version of the code; the paper reports exact values, as does
  `code/analysis/stats.py` (`compare()`).
- **Not bit-identical.** cuDNN attention backward is non-deterministic, so reruns
  differ slightly (below 1e-4 in tail mass).
- **Caches are not included.** Hidden-state and top-k caches are tens of GB per
  experiment; each job rebuilds its own.

## Citation

```bibtex
@article{topkdistillation2026,
  title   = {The Gradient That Top-k Distillation Never Computes},
  author  = {AUTHORS},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## License

See `LICENSE`. Model weights and datasets are not included and remain under their
own licenses: Qwen3 and SmolLM2 (Apache-2.0), Llama 3.1 and 3.2 (Llama Community
License), Gemma 3 (Gemma Terms of Use), FineWeb-Edu (ODC-BY), UltraChat-200k (MIT),
FLAN (Apache-2.0), Glaive Function Calling v2 (Apache-2.0).
