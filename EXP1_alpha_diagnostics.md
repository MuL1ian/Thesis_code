# Small experiment 1 — is the $\alpha$ axis doing anything?

Implementation spec. Not part of the thesis build.

**What this decides.** Whether the bridge coupling $\alpha$ changes the update at
all, and if so whether it changes its *direction* or only its *magnitude*. A
negative result retires the $\alpha$ axis and removes the four bridge runs from
the $K$ sweep. It also supplies the measured value of the amplification factor
$G$, which Chapter 4 currently only bounds.

**Cost.** Zero training runs. Tier 1 is minutes on pretrained weights. Tier 2
rides along on the clean Bridge run you are doing anyway, at roughly 4% overhead.

---

## 1. Setup

From `Chapter3.tex`, Eq. `eq:bridge-velocity` and `eq:bridge-grad`. The active
set is $\mathcal{A}=\mathcal{A}_{\rm pre}\cup\mathcal{A}_{\rm post}$ with
$K=|\mathcal{A}|$, split index $j$, Euler steps $h_i$. Velocities are

$$
\tilde v_i=\begin{cases}
v_\theta(\alpha x_j+(1-\alpha)\,\mathrm{sg}(x_j),\ \sigma_j,\ c) & i=j\\
v_\theta(\mathrm{sg}(x_i),\ \sigma_i,\ c) & i\in\mathcal{A},\ i\neq j\\
\mathrm{sg}(v_i) & i\notin\mathcal{A}
\end{cases}
$$

and the endpoint Jacobian splits into a direct sum and one nested term:

$$
\frac{\partial\tilde x_0}{\partial\theta}
= \underbrace{-\sum_{i\in\mathcal{A}} h_i\,\partial_\theta v_i}_{\text{direct}}
+ \underbrace{\alpha\,h_j\,(\partial_x v_j)\sum_{i\in\mathcal{A}_{\rm pre}} h_i\,\partial_\theta v_i}_{\text{nested}} .
$$

Deployed config (from the run sheet):

```yaml
flowbp:
  select_idx_seed: 114
  min_idx: 1
  max_idx: 26
  jk_sampling_mode: uniform
  alpha: 0.3
flowbp_sparse:
  num_active_steps: 4
  late_bias: 0.0
  grad_rescale: 0.0
```

---

## 2. The simplification: $\nabla_\theta L$ is exactly affine in $\alpha$

Two facts make this hold **exactly**, not approximately:

1. $\alpha$ multiplies the nested term linearly and does not appear in the
   direct term.
2. The forward value of $\alpha x_j+(1-\alpha)\mathrm{sg}(x_j)$ equals $x_j$ for
   every $\alpha$, so the forward trajectory, the endpoint $\tilde x_0$, and
   therefore the reward cotangent $g=\partial L/\partial\tilde x_0$ are all
   independent of $\alpha$.

Hence, writing $g_\alpha := \nabla_\theta L(\alpha)$,

$$
\boxed{\;g_\alpha = g_0 + \alpha\, n,\qquad n := g_1 - g_0\;}
$$

Three consequences for the implementation:

- **No JVP is needed for the $\rho$ measurement.** Run the existing backward
  twice at two values of $\alpha$ and subtract. (If you ever do hand-roll it,
  note it is a **VJP** — you need $(\partial_x v_j)^{\mathsf T}g$, not
  $(\partial_x v_j)u$.)
- **Measure at $\alpha=1$, not at the deployed $0.3$.** Affineness gives you the
  whole axis from one measurement, and $\alpha=1$ is the worst case: if the
  nested term is negligible there, it is negligible everywhere.
- **You get a free correctness test.** See §5.

---

## 3. What must be identical across the two backward passes

This is the single biggest way to get a meaningless number. Everything below
must be byte-identical between the $\alpha=0$ and $\alpha=1$ passes:

| Quantity | Why |
|---|---|
| prompt batch | obvious |
| initial noise $x_N$ | different noise ⇒ different trajectory |
| **active set $\mathcal{A}$** | governed by `select_idx_seed`, `min_idx`, `max_idx`, `jk_sampling_mode`. If resampled, you measure index-selection variance, not $\alpha$. |
| split index $j$ | the nested term is defined at $j$ |
| $K$ (`num_active_steps`) | changes the direct sum |
| cached rollout | reuse the same cache; do not re-roll |
| model weights | same checkpoint |

Assert on the realised active index list, not on the seed. Seeds are consumed by
other draws and a matching seed does not guarantee a matching index set.

---

## 4. Measurement A — the $\rho$ decomposition

### 4.1 Quantities

With $d := g_0$ (direct-only gradient) and $n := g_1 - d$ (nested term at
$\alpha=1$), both flattened over all trainable parameters:

$$
\rho_\parallel = \frac{\langle n, d\rangle}{\lVert d\rVert^2},\qquad
n_\perp = n - \rho_\parallel d,\qquad
\rho_\perp = \frac{\lVert n_\perp\rVert}{\lVert d\rVert},\qquad
\rho = \frac{\lVert n\rVert}{\lVert d\rVert} = \sqrt{\rho_\parallel^2+\rho_\perp^2}.
$$

At any deployed $\alpha$ these give, in closed form and with no extra passes:

$$
\cos\big(g_\alpha, g_0\big) = \frac{1+\alpha\rho_\parallel}
{\sqrt{(1+\alpha\rho_\parallel)^2+(\alpha\rho_\perp)^2}},
\qquad
\frac{\lVert g_\alpha\rVert}{\lVert g_0\rVert}
=\sqrt{(1+\alpha\rho_\parallel)^2+(\alpha\rho_\perp)^2}.
$$

The second is the **effective learning-rate multiplier** that turning on
$\alpha$ applies. Report it: it is the quantity that decides whether the
$\alpha$ sweep is confounded with the learning rate.

### 4.2 Why not just report $\cos$

$\cos$ only constrains the orthogonal component. If $n$ happens to be parallel
to $d$, $\cos$ can be $1.000$ while $\lVert n\rVert$ is large — the nested term
would then be rescaling the update, which is absorbed by the learning rate.
That is a substantive finding and a single cosine hides it. Report
$\rho_\parallel$ and $\rho_\perp$ separately; derive $\cos$ from them.

### 4.3 Tier 1 — pretrained weights, run this first

No checkpoint required, no GPU queue.

1. Load the pretrained backbone. Freeze.
2. For each of $B=12$–$16$ batches from the training prompt source:
   1. Roll out and cache once with a fixed noise seed.
   2. Backward at $\alpha=0$ → `d`.
   3. Backward at $\alpha=1$ on the **same cache and same active set** → `g1`.
   4. `n = g1 - d`; record $\rho_\parallel,\rho_\perp,\rho$.
3. Report **median and IQR** across batches, not the mean. Per-batch reward
   gradients are noisy; a mean will be dragged by one outlier.

### 4.4 Tier 2 — online during the clean Bridge run

Tier 1 answers "at initialisation". Tier 2 answers "throughout training", which
is the claim actually needed to retire the axis, and it needs no saved
checkpoints because the clean Bridge run is already on the schedule.

Every 25 steps, on the current batch and the cache the training step already
built:

1. You already have $g_{0.3}$ — it is the training gradient.
2. Take **one extra backward at $\alpha=0$** on the same cache → `d`.
3. Recover the canonical nested term by affineness:
   `n = (g_0.3 - d) / 0.3`.
4. Log $\rho_\parallel,\rho_\perp$, $\cos(g_{0.3},d)$, and
   $\lVert g_{0.3}\rVert/\lVert d\rVert$.

Overhead: one extra backward over $K=4$ active velocities every 25 steps,
≈ 4% wall clock.

This measures the nested term **along the trajectory Bridge actually followed**,
which is the correct counterfactual for auditing the deployed $\alpha=0.3$, and
it is what explains the $0.0021$ gap between $\alpha=0$ and $\alpha=0.3$ in the
Chapter 5 sweep.

---

## 5. Assertions (write these before the measurement code)

| Assertion | Catches |
|---|---|
| $g_{0.5} = \tfrac12(g_0+g_1)$ to machine precision | **The main one.** Any violation means the bridge implementation is broken — usually `sg` not blocking the forward path, or the active set being resampled between calls. Run it once on one batch. |
| realised active index list identical across passes | silent resampling |
| $\tilde x_0$ identical across $\alpha$ | `sg` implemented as a value change rather than a gradient block |
| $g$ (reward cotangent) identical across $\alpha$ | same as above, one level up |
| $\rho=\sqrt{\rho_\parallel^2+\rho_\perp^2}$ | decomposition arithmetic |
| $j\in\mathcal{A}$ and $\mathcal{A}_{\rm pre}\neq\emptyset$ | an empty pre-segment makes the nested term exactly zero for trivial reasons, which would look like a real negative result |

The last one matters: with `min_idx: 1`, `max_idx: 26`, `K=4` and uniform
sampling, some draws may place no active index above $j$. Log the frequency;
those batches must be excluded from the $\rho$ statistics and counted
separately.

---

## 6. Measurement B — $\lVert\partial_x v_j\rVert_2$ along $\sigma$

Independent of Measurement A: needs the rollout cache and $v_\theta$ only, no
reward and no second backward.

For each candidate split index $j$ (sweep or subsample $[1,26]$), on the cached
$x_j,\sigma_j$:

```
u ← random unit vector in x-space
repeat 5–10 times:
    w = J u          # torch.func.jvp of v_θ(·, σ_j, c) at x_j
    z = Jᵀ w         # torch.autograd.grad, cotangent w
    u = z / ||z||
σ_max ≈ ||J u||      # power iteration on JᵀJ; σ_max = sqrt(λ_max)
```

Report median over batches of $\sigma_{\max}(j)$ against $\sigma_j$.

**Two things this delivers.** First, the measured magnitude of the amplification
factor $G$ in the Chapter 4 bound, which currently has no empirical value at
all. Second, and more useful here, it *explains* Measurement A: if $\rho_\perp
\approx 0$, Measurement B says whether that is structural — $\lVert\partial_x
v_j\rVert_2$ is uniformly small, so no placement of $j$ can make $\alpha$
matter — or an artefact of a badly placed $j$, which is fixable by changing
`min_idx`/`max_idx`. Without B, a null result in A is ambiguous between those
two.

Run B on pretrained weights alongside Tier 1: choosing where $j$ goes is a
configuration decision that should be settled before any Bridge run.

---

## 7. Decision rules

Fix these before looking at the numbers. Thresholds are on the $\alpha=1$
quantities.

| Outcome | Conclusion | Action |
|---|---|---|
| $\rho_\perp\lesssim0.05$ **and** $\lvert\rho_\parallel\rvert\lesssim0.05$, at every tier-2 checkpoint | The $\alpha$ axis is structurally inert. | Retire $\alpha$. **Drop the four bridge runs from the $K$ sweep.** Report as a negative result with the measured $\rho$, not as an omission. |
| $\rho_\perp\lesssim0.05$ but $\lvert\rho_\parallel\rvert$ not small | $\alpha$ is a learning-rate knob, not a direction change. | The existing $\alpha$ sweep is confounded with step size. Either add a norm-matched control arm or report $\alpha$ as an effective-step-size axis. Do **not** report it as a fidelity axis. |
| $\rho_\perp$ appreciable | The nested term genuinely redirects the update. | Keep the bridge $K$ sweep. RQ2's $\alpha$ question is live and worth the runs. |
| $\rho$ small at step 0 but grows during tier 2 | Initialisation is unrepresentative. | Tier 1 alone would have given the wrong answer; quote the trajectory, not the endpoint. |
| $\sigma_{\max}(j)$ uniformly small across $[1,26]$ | The state Jacobian carries little signal anywhere on the trajectory. | Explains a null in A structurally; strengthens the retirement case. |
| $\sigma_{\max}(j)$ large somewhere but the deployed $j$ sits elsewhere | $j$ is badly placed. | Re-run A with $j$ in the high-Jacobian region before concluding anything about $\alpha$. |

Context for calibrating "appreciable": the Chapter 5 $\alpha$ sweep measured
$14.9537$ at $\alpha=0$ against $14.9558$ at $\alpha=0.3$, a difference of
$0.0021$, while the six Stage I methods span $13.065$ to $14.956$ on the same
metric — a range of $1.89$. The $\alpha$ effect is already known to be about
three orders of magnitude below the between-method spread. This experiment
explains *why*, and does so without another training run.

---

## 8. What to log

One row per (tier, checkpoint/step, batch):

```
tier, step, batch_idx, active_idx_list, j, K,
||d||, ||n||, rho_par, rho_perp, rho,
cos_at_deployed_alpha, stepsize_ratio_at_deployed_alpha,
pre_segment_empty  (bool)
```

One row per (batch, j) for Measurement B:

```
batch_idx, j, sigma_j, sigma_max, n_power_iters, converged (bool)
```

Aggregate with **median and IQR** across batches everywhere. Feeds
`Chapter5.tex` §`sec:exp-stage1-sweeps` (the $\alpha$ rows of the sweep table)
and the amplification discussion around `prop:amplification` in Chapter 3.
