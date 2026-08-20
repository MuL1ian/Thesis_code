# EXP1 and RQ3 — measured results

Both are diagnostics, not training runs. Numbers below come from the artefacts
listed under each section; nothing here is transcribed by hand from a log.

| | EXP1 | RQ3 / E3 |
|---|---|---|
| Question | does the bridge coupling $\alpha$ change the update? | is the dominant GradLoRA subspace the right target? |
| Verdict | `redirects` — keep the $\alpha$ axis | RQ3(c) supported — but `mean` beats `selected` |
| Artefact | `outputs/Thesis/EXP1/alpha_axis/` | `outputs/Thesis/RQ3/sd3_5_flowbp_sparse_hpsv3plusplus_rq3_e3/` |
| Run on | 2026-08-19 00:21 | 2026-08-19 11:12 |

---

## EXP1 — the $\alpha$ axis

Spec: `EXP1_alpha_diagnostics.md`. Driver: `configs/Thesis/run_exp1_alpha_axis.sh`.
Artefacts: `summary.json`, `measurement_a.csv`.

Measurement A (tier 1) only. Measurement B ($\sigma_{\max}(\partial_x v_j)$) was
not run, so a null result would have been ambiguous — it is not, so B is not
needed to interpret A.

### Configuration

Pretrained SD3.5-medium, no checkpoint, weights frozen. Deployed bridge config:
`select_idx_seed=114`, `min_idx=1`, `max_idx=26`, `jk_sampling_mode=uniform`,
$K=4$, `late_bias=0.0`, `grad_rescale=0.0`, `train_batch_size=4`,
deployed $\alpha=0.3$. 12 batches scored, 0 excluded for an empty pre-segment.

### Self-check

The whole measurement rests on $\nabla_\theta L$ being exactly affine in
$\alpha$. It is:

| Check | Value | Verdict |
|---|---|---|
| $g_{0.5}$ vs $\tfrac12(g_0+g_1)$, relative error | 0.0161 | passes, tolerance 0.0615 |
| noise floor | 0.0205 | — |
| endpoint $\tilde x_0$ bitwise identical across $\alpha$ | true | `sg` blocks the forward path |
| reward cotangent relative difference | 0.0164 | within noise |
| realised active set identical across passes | true | no silent resampling |

`problems` is empty. The $\rho$ figures below are therefore meaningful.

### Measurement A — $\rho$ decomposition at $\alpha=1$

Median and IQR over the 12 batches, as the spec requires (per-batch reward
gradients are noisy and a mean would follow one outlier).

| Quantity | Median | Q1 | Q3 | IQR |
|---|---|---|---|---|
| $\rho_\parallel$ | −0.0364 | −0.1707 | 0.0029 | 0.1736 |
| $\rho_\perp$ | **0.1297** | 0.0726 | 0.1824 | 0.1098 |
| $\rho$ | 0.1746 | 0.0743 | 0.2305 | 0.1562 |
| $\rho_{\rm null}$ | 0.0179 | 0.0165 | 0.0197 | 0.0032 |
| $\cos(g_{0.3}, g_0)$ | 0.99916 | 0.99842 | 0.99976 | 0.0013 |
| $\lVert g_{0.3}\rVert/\lVert g_0\rVert$ | 0.9891 | 0.9496 | 1.0010 | 0.0514 |
| $\lVert d\rVert$ | 108.7 | 78.6 | 195.6 | 117.0 |
| $\lVert n\rVert$ | 17.0 | 8.8 | 31.9 | 23.1 |

### Reading

$\rho_\perp = 0.130$ against a decision threshold of $0.05$, and against a
numerical noise floor $\rho_{\rm null} = 0.018$ that is seven times smaller. The
orthogonal component is real, not measurement error. $\lvert\rho_\parallel\rvert
= 0.036$ sits inside the threshold, so the nested term is **not** acting as a
learning-rate knob: the step-size ratio at the deployed $\alpha=0.3$ is 0.989,
i.e. about 1% — the $\alpha$ sweep is not confounded with step size.

Row 3 of the decision table in the spec applies:

> $\rho_\perp$ appreciable → the nested term genuinely redirects the update.
> Keep the bridge $K$ sweep. RQ2's $\alpha$ question is live.

The four `flowbp_bridge_k{2,3,4,5}` runs are therefore justified, and they have
since completed.

### The caveat worth stating in the text

$\rho_\perp$ is measured at $\alpha=1$; at the deployed $\alpha=0.3$ the induced
rotation is tiny — $\cos(g_{0.3}, g_0) = 0.9992$, about $2.4°$. So "$\alpha$
redirects" is a statement about the *mechanism* being non-degenerate, not a
claim that the deployed setting moves the optimiser far. That is consistent with
the Chapter 5 sweep, where $\alpha=0$ scored 14.9537 against 14.9558 at
$\alpha=0.3$ — a gap of 0.0021 on a between-method spread of 1.89.

Dispersion is wide (IQR on $\rho_\perp$ is 0.110, comparable to the median) and
$\rho_\perp$ ranges from 0.040 to 0.309 across the 12 batches, tracking where
the sampler placed $j$. This is tier 1 on pretrained weights only; the spec's
tier 2 — the same measurement every 25 steps along the Bridge trajectory — has
not been run, so nothing here rules out the term growing or shrinking during
training.

---

## RQ3 / E3 — held-out subspace probe

Driver: `configs/Thesis/Stage2/run_sd35_rq3.sh`.
Config: `configs/Thesis/Stage2/sd3_5_flowbp_sparse_hpsv3plusplus_rq3_e3.yaml`.
Report: `python emprical/report_rq3.py --config <cfg> --label rq3_e3`.

A measurement, not a training run: `max_train_steps: 1`, `use_ema: false`, and
the single optimizer step afterwards only proves the run is trainable. No usable
checkpoint comes out of it.

### Configuration

GradLoRA rank 8, `use_second_moment=true`, `second_moment_beta=0.5`,
`spectral_exponent=0.0`, oversample 8, 4 power iterations — the deployed
operator, so the probe scores what the method would ship. Calibrated on
$K_{\rm cal}=4$ batches (batch size 4), scored on $K_{\rm ho}=8$ disjoint
held-out batches (64 samples), global batch 32, trainer `flowbp_sparse`.

96 layers scored, 1 excluded
(`transformer_blocks.23.attn.add_q_proj`), 24 per module type across all 24
blocks. `layer_sample: 96` caps rank-0 retention at ~24 GiB instead of the ~50
GiB that all 282 target layers would need.

**`ky_fan_violations` is empty.** Had it not been, the report would be void.

### Four arms, normalised by the pooled held-out oracle

| Arm | capture, per-batch | capture, pooled | usefulness, per-batch |
|---|---|---|---|
| `selected` | 0.918 | 0.903 | 0.955 |
| `mean` | **0.940** | **0.922** | **0.971** |
| `random` | 0.008 | 0.006 | 0.060 |
| `complement` | 0.004 | 0.002 | 0.013 |

Raw (un-normalised) cosine to the held-out gradient, pooled budget: `selected`
0.620, `mean` 0.622, `random` 0.043, `complement` 0.005.

### Paired per-layer contrasts, `selected` vs arm

| vs | metric | win frac | median $d$ | sign-test $p$ |
|---|---|---|---|---|
| `complement` | capture | 1.000 | +0.922 | $\approx 0$ |
| `random` | capture | 1.000 | +0.918 | $\approx 0$ |
| `mean` | capture | 0.146 | **−0.020** | $\approx 0$ |
| `complement` | usefulness | 1.000 | +0.940 | $2.5\times10^{-29}$ |
| `random` | usefulness | 1.000 | +0.888 | $\approx 0$ |
| `mean` | usefulness | 0.167 | **−0.010** | $\approx 0$ |

### By module type (capture, pooled / usefulness, per-batch — normalised medians)

| Stratum | `selected` | `mean` | `random` | `complement` |
|---|---|---|---|---|
| `attn` | 0.877 / 0.919 | 0.904 / 0.927 | 0.007 / 0.061 | 0.002 / 0.012 |
| `attn_context` | 0.938 / 0.953 | 0.970 / 0.974 | 0.005 / 0.064 | 0.001 / 0.011 |
| `ff` | 0.935 / 0.954 | 0.953 / 0.962 | 0.007 / 0.065 | 0.002 / 0.010 |
| `ff_context` | 0.924 / 0.957 | 0.959 / 0.979 | 0.003 / 0.042 | 0.000 / 0.008 |

The ordering is the same in every stratum, so nothing below is driven by one
module type.

### Reading

**RQ3(c) is supported.** `selected` beats `complement` on 100% of 96 layers
(sign test $p = 2.5\times10^{-29}$, median paired difference 0.940 on
usefulness). The complement arm is the only one that can refute the method, and
it does not: the dominant subspace is the right target, and the complement
carries essentially nothing (0.013 of oracle).

**The subspace is estimable from few batches.** `selected` reaches 0.955 of the
held-out oracle on usefulness from 4 calibration batches, against 0.060 for a
random subspace of the same rank — a factor of 16.

**`mean` beats `selected`, and this must be reported.** The margin is small
(median 0.020 on capture, 0.010 on usefulness) but it is consistent: `mean` wins
85% of layers on capture and 83% on usefulness, in every stratum, under both
scoring budgets, at $p \approx 0$. The honest statement is that the *dominant*
subspace is clearly the right region of parameter space, while the specific
top-$r$ selection is very slightly worse than simply averaging — so the claim to
defend is "the dominant subspace", not "our selection rule beats the obvious
baseline".

### Not run

E7 ($K_{\rm cal}$ sweep) and E9 ($\beta$ sweep) need a harness that calibrates
several times from a single model load. E3 is the only entry in the RQ3 driver.

The `selected` vs `mean` gap is the obvious thing E7 would settle: if it closes
as $K_{\rm cal}$ grows, it is a small-sample artefact of calibrating on 4
batches; if it does not, the selection rule needs revisiting.

---

## Reproducing

```bash
cd workspace/FlowBP

# EXP1 (minutes, pretrained weights)
bash configs/Thesis/run_exp1_alpha_axis.sh
REPORT_ONLY=1 bash configs/Thesis/run_exp1_alpha_axis.sh   # re-print only

# RQ3 E3 (one calibration + probe + one smoke step)
bash configs/Thesis/Stage2/run_sd35_rq3.sh
python emprical/report_rq3.py \
    --config configs/Thesis/Stage2/sd3_5_flowbp_sparse_hpsv3plusplus_rq3_e3.yaml \
    --label rq3_e3
```

Both scripts hand the GPUs back to `gpu_occupy.py` when they finish.
