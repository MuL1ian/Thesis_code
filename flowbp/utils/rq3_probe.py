"""Held-out subspace scoring for RQ3.

Answers three questions about a calibrated LoRA-A subspace, none of which need
a training run:

  (a) identifiability -- can it be estimated from few calibration batches,
  (b) transferability -- does it still capture on batches never used to select,
  (c) correctness of the target -- is the *dominant* subspace the right place
      to spend the rank budget at all.

Only (c) can refute the method, and only the ``complement`` arm tests it.

The scoring rules here are not interchangeable with the obvious alternatives,
so the reasons are recorded next to the code that depends on them: see
:class:`RQ3ProbeScorer` for the oracle choice and :func:`aggregate_arms` for
the cross-layer weighting.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import torch

from flowbp.utils.logging_ import main_print


# Layer counts differ by an order of magnitude between module types, so an
# unweighted layer mean is really a count-weighted mean over these groups.
# Every aggregate is therefore also reported per group.
_MODULE_TYPE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"add_q_proj|add_k_proj|add_v_proj|to_add_out", "attn_context"),
    (r"\bto_q\b|\bto_k\b|\bto_v\b|to_out", "attn"),
    (r"\bff_context\b", "ff_context"),
    (r"\bff\b|proj_mlp|mlp", "ff"),
    (r"norm.*linear|adaLN|proj_out", "modulation"),
)


def module_type(name: str) -> str:
    for pattern, label in _MODULE_TYPE_PATTERNS:
        if re.search(pattern, name):
            return label
    return "other"


@dataclass
class LayerBatchScore:
    """Raw, un-normalised scores for one (layer, arm, held-out batch)."""

    held_out_energy: float
    captured_energy: float
    reference_dot: float
    reference_norm: float

    @property
    def capture(self) -> float:
        """``||G V||_F^2 / ||G||_F^2``, sign-blind, in ``[0, 1]``."""
        if self.held_out_energy <= 0.0:
            return float("nan")
        return self.captured_energy / self.held_out_energy

    @property
    def usefulness(self) -> float:
        """``<G, D> / ||D||_F`` with ``D = Gbar_cal V V^T``.

        Because the FlowBP loss is exactly the negated reward, a positive value
        means a step along the subspace-projected calibration gradient raises
        held-out reward to first order. Unlike capture, this is signed, which
        is the whole reason both are reported: high capture with negative
        usefulness is the failure mode that would demote capture to a
        diagnostic.
        """
        if self.reference_norm <= 0.0:
            return float("nan")
        return self.reference_dot / self.reference_norm

    @property
    def cosine(self) -> float:
        if self.reference_norm <= 0.0 or self.held_out_energy <= 0.0:
            return float("nan")
        return self.reference_dot / (
            self.reference_norm * math.sqrt(self.held_out_energy)
        )


@dataclass
class LayerRecord:
    """Everything scored for one layer, across arms and held-out batches."""

    name: str
    rows: int
    cols: int
    rank: int
    module_type: str
    per_batch: dict[str, list[LayerBatchScore]] = field(default_factory=dict)
    pooled: dict[str, LayerBatchScore] = field(default_factory=dict)
    oracle_per_batch: list[LayerBatchScore] = field(default_factory=list)
    oracle_pooled: LayerBatchScore | None = None
    # mean_j C_j(V*_j) / C_j(V*): how idiosyncratic a single batch's own
    # leading subspace is, relative to the held-out mean's.
    batch_oracle_ratio: float = float("nan")


def _top_right_subspace(matrix: torch.Tensor, rank: int) -> torch.Tensor:
    """Exact top-``rank`` right singular vectors of ``matrix`` as columns.

    Exact rather than randomised on purpose: Ky Fan makes this subspace the
    true rank-``r`` maximiser of pooled capture, and the report asserts against
    that bound, so a randomised approximation would turn its own error into a
    spurious violation and quietly deflate every arm's normalised score.

    Routed through the smaller Gram matrix. A gradient is ``d_out x d_in`` with
    ``d_out`` typically far smaller, so an eigendecomposition of ``G G^T``
    costs ``O(d_out^3)`` where a full SVD would cost ``O(d_out^2 d_in)``.
    """
    rows, cols = matrix.shape
    rank = min(int(rank), rows, cols)
    if rows <= cols:
        gram = matrix @ matrix.transpose(0, 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        order = eigenvalues.argsort(descending=True)[:rank]
        left = eigenvectors[:, order]
        # G^T u_i = sigma_i v_i, so a QR both normalises the columns and
        # supplies an arbitrary orthonormal completion for any zero singular
        # value, which is what a degenerate layer needs.
        return torch.linalg.qr(
            matrix.transpose(0, 1) @ left,
            mode="reduced",
        ).Q.contiguous()
    gram = matrix.transpose(0, 1) @ matrix
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = eigenvalues.argsort(descending=True)[:rank]
    return eigenvectors[:, order].contiguous()


def _capture(gradient: torch.Tensor, basis: torch.Tensor) -> float:
    energy = float(gradient.pow(2).sum().item())
    if energy <= 0.0:
        return float("nan")
    return float((gradient @ basis).pow(2).sum().item()) / energy


def _score(
    gradient: torch.Tensor,
    basis: torch.Tensor,
    reference: torch.Tensor,
) -> LayerBatchScore:
    projection = gradient @ basis
    return LayerBatchScore(
        held_out_energy=float(gradient.pow(2).sum().item()),
        captured_energy=float(projection.pow(2).sum().item()),
        reference_dot=float((projection * reference).sum().item()),
        reference_norm=float(reference.norm().item()),
    )


class RQ3ProbeScorer:
    """Score held-out gradients against every calibrated arm.

    Feed one call to :meth:`observe_batch` per held-out batch, then call
    :meth:`finish`. Gradients must already be reduced across data-parallel
    ranks, so that one observed batch is one global batch -- the same unit a
    calibration batch is measured in.

    Two scoring budgets are kept for every metric because they are not equal
    and the difference is itself informative:

    * per-batch, averaged over the held-out batches, which measures capture of
      gradients of the kind the optimiser actually sees, and is primary;
    * pooled, scored once on the held-out mean, which is optimistic because
      averaging denoises and a mean is more low-rank than its terms.

    Their ratio is a held-out analogue of the calibration ``snr``.
    """

    def __init__(
        self,
        subspaces: Mapping[str, Mapping[str, torch.Tensor]],
        references: Mapping[str, Mapping[str, torch.Tensor]],
        calibration_gradients: Mapping[str, torch.Tensor],
        *,
        rank: int,
        excluded_layers: Iterable[str] = (),
        device: str | torch.device = "cpu",
        compute_batch_oracle_ratio: bool = False,
        expected_batches: int | None = None,
        max_retained_gib: float = 64.0,
        layer_sample: int | None = None,
    ) -> None:
        self.subspaces = {
            arm: dict(layers) for arm, layers in subspaces.items()
        }
        self.references = {
            arm: dict(layers) for arm, layers in references.items()
        }
        # Needed to give the oracle a reference direction on the same footing
        # as every other arm: only the subspace may differ between arms.
        self.calibration_gradients = dict(calibration_gradients)
        self.rank = int(rank)
        # Layers that fell back to random initialisation carry no calibrated
        # subspace, so including them would drag `selected` toward `random`.
        self.excluded_layers = set(excluded_layers)
        self.device = torch.device(device)
        # One exact thin eigendecomposition per (layer, held-out batch). At a
        # few hundred layers that is thousands of O(d_out^3) solves, so this is
        # off by default despite being conceptually a by-product of data
        # already in hand.
        self.compute_batch_oracle_ratio = bool(compute_batch_oracle_ratio)
        self.expected_batches = expected_batches
        self.max_retained_bytes = int(max_retained_gib * 2**30)
        self._scored_layers = self._select_layers(layer_sample)
        self.batch_indices: list[int] = []
        self._raw: dict[str, dict[str, list[LayerBatchScore]]] = {
            arm: {} for arm in self.subspaces
        }
        self._pooled_gradients: dict[str, torch.Tensor] = {}
        self._batch_gradient_cache: dict[str, list[torch.Tensor]] = {}
        self._shapes: dict[str, tuple[int, int]] = {}
        self._retained_bytes = 0
        self.ky_fan_violations: list[tuple[str, str, float]] = []

    def scored_layers(self) -> list[str]:
        return self._scored_layers

    def _select_layers(self, layer_sample: int | None) -> list[str]:
        arms = list(self.subspaces)
        if not arms:
            return []
        common = set(self.subspaces[arms[0]])
        for arm in arms[1:]:
            common &= set(self.subspaces[arm])
        eligible = sorted(common - self.excluded_layers)
        selected = select_probe_layers(eligible, layer_sample)
        if len(selected) < len(eligible):
            main_print(
                f"--> RQ3 probe scores {len(selected)} of {len(eligible)} "
                "layers to bound retained memory"
            )
        return selected

    @torch.no_grad()
    def observe_batch(
        self,
        batch_index: int,
        gradients: Mapping[str, torch.Tensor],
    ) -> None:
        if batch_index in self.batch_indices:
            raise ValueError(
                f"RQ3 probe observed held-out batch {batch_index} twice"
            )
        self.batch_indices.append(int(batch_index))
        for name in self.scored_layers():
            gradient = gradients.get(name)
            if gradient is None:
                continue
            gradient = gradient.to(self.device, dtype=torch.float32)
            self._shapes[name] = (gradient.shape[0], gradient.shape[1])
            for arm in self.subspaces:
                basis = self.subspaces[arm][name].to(self.device, torch.float32)
                reference = self.references[arm][name].to(
                    self.device, torch.float32
                )
                self._raw[arm].setdefault(name, []).append(
                    _score(gradient, basis, reference)
                )
            if name in self._pooled_gradients:
                self._pooled_gradients[name].add_(gradient)
            else:
                self._pooled_gradients[name] = gradient.clone()
                self._retained_bytes += gradient.numel() * gradient.element_size()
            # Every held-out batch is retained: the oracle it must be scored
            # against is only known once the held-out mean is complete.
            self._batch_gradient_cache.setdefault(name, []).append(gradient)
            self._retained_bytes += gradient.numel() * gradient.element_size()

        if len(self.batch_indices) == 1:
            self._project_memory()

    def _project_memory(self) -> None:
        """Report the measured footprint after the first batch.

        This only reports. The budget is enforced by
        :func:`assert_probe_cache_fits` before the probe starts, because this
        scorer lives on rank 0 alone and raising here would abort one rank
        while the others block forever in the gradient reduction.
        """
        per_batch = self._retained_bytes / 2
        batches = self.expected_batches or len(self.batch_indices)
        projected = per_batch * (batches + 1)
        main_print(
            f"--> RQ3 probe retains {per_batch / 2**30:.1f} GiB per held-out "
            f"batch on {self.device}; {batches} batches project to "
            f"{projected / 2**30:.1f} GiB"
        )
        if projected > self.max_retained_bytes:
            main_print(
                "[warning] RQ3 measured footprint exceeds the budget that was "
                "checked up front; the layer shapes must have been "
                "mis-estimated"
            )

    @torch.no_grad()
    def finish(self) -> list[LayerRecord]:
        """Resolve the oracle and emit one record per scored layer.

        The oracle is the *pooled* one, ``V* = top-r(Gbar_ho)``, regardless of
        whether a score is per-batch or pooled. A per-batch oracle
        ``top-r(G_j)`` is fitted to a single noisy batch; since a high ``snr``
        is the normal regime -- it is the entire reason for ``K_cal > 1`` --
        that denominator grows with the noise and every honest arm's normalised
        score would fall for reasons unrelated to method quality, confounding
        exactly what ``snr`` exists to separate. The pooled oracle instead
        means "the best rank-r subspace obtainable if one were allowed to look
        at held-out data", so the ratio measures the cost of having to estimate
        from calibration.
        """
        if not self.batch_indices:
            raise RuntimeError("RQ3 probe scored no held-out batch")

        batch_count = len(self.batch_indices)
        records: list[LayerRecord] = []
        for name in self.scored_layers():
            if name not in self._pooled_gradients:
                continue
            rows, cols = self._shapes[name]
            pooled_gradient = self._pooled_gradients[name] / batch_count
            oracle_basis = _top_right_subspace(pooled_gradient, self.rank)

            record = LayerRecord(
                name=name,
                rows=rows,
                cols=cols,
                rank=self.rank,
                module_type=module_type(name),
            )
            for arm in self.subspaces:
                record.per_batch[arm] = self._raw[arm][name]
                basis = self.subspaces[arm][name].to(self.device, torch.float32)
                reference = self.references[arm][name].to(
                    self.device, torch.float32
                )
                record.pooled[arm] = _score(pooled_gradient, basis, reference)

            # The oracle needs a reference direction too, and it has to come
            # from the calibration gradient like every other arm's -- only the
            # subspace is allowed to differ, or the comparison is not paired.
            oracle_reference = self._oracle_reference(name, oracle_basis)
            batch_ratios: list[float] = []
            for gradient in self._batch_gradient_cache[name]:
                on_batch = _score(gradient, oracle_basis, oracle_reference)
                record.oracle_per_batch.append(on_batch)
                if self.compute_batch_oracle_ratio and on_batch.capture > 0.0:
                    own_basis = _top_right_subspace(gradient, self.rank)
                    batch_ratios.append(
                        _capture(gradient, own_basis) / on_batch.capture
                    )
            record.oracle_pooled = _score(
                pooled_gradient,
                oracle_basis,
                oracle_reference,
            )
            self._check_ky_fan_bound(record)
            if batch_ratios:
                record.batch_oracle_ratio = sum(batch_ratios) / len(batch_ratios)
            records.append(record)

        self._batch_gradient_cache.clear()
        self._retained_bytes = 0
        if self.ky_fan_violations:
            main_print(
                "[warning] RQ3 pooled capture exceeded the pooled oracle on "
                f"{len(self.ky_fan_violations)} (layer, arm) pairs; the oracle "
                "is the exact rank-r maximiser, so this indicates a scoring "
                "bug rather than a finding"
            )
        return records

    def _check_ky_fan_bound(
        self,
        record: LayerRecord,
        tolerance: float = 1e-5,
    ) -> None:
        """Pooled capture cannot beat the pooled oracle (Ky Fan).

        Asserted for this combination only. A per-batch score may legitimately
        exceed the pooled oracle, so that case is recorded rather than flagged.
        """
        assert record.oracle_pooled is not None
        oracle = record.oracle_pooled.capture
        if not math.isfinite(oracle) or oracle <= 0.0:
            return
        for arm, score in record.pooled.items():
            ratio = score.capture / oracle
            if math.isfinite(ratio) and ratio > 1.0 + tolerance:
                self.ky_fan_violations.append((record.name, arm, ratio))

    def _oracle_reference(
        self,
        name: str,
        oracle_basis: torch.Tensor,
    ) -> torch.Tensor:
        """``Gbar_cal V*``, the oracle's reference direction."""
        calibration_gradient = self.calibration_gradients.get(name)
        if calibration_gradient is None:
            raise RuntimeError(
                f"RQ3 probe has no calibration gradient for {name}; "
                "the calibrator must expose it for oracle normalisation"
            )
        return calibration_gradient.to(self.device, torch.float32) @ oracle_basis


def select_probe_layers(
    names: Sequence[str],
    layer_sample: int | None,
) -> list[str]:
    """Thin a layer list to roughly ``layer_sample``, stratified by module type.

    Evenly spaced within each module type, never random: the report's headline
    is an unweighted mean over layers, so thinning one module type harder than
    another would silently reweight it. Depth coverage survives because the
    names sort by block index. Shared with the memory projection so that the
    budget is computed over exactly the layers that will be retained -- an MLP
    projection is several times larger than an attention one, so a proportional
    estimate would be badly wrong.
    """
    ordered = sorted(names)
    if layer_sample is None or not ordered or layer_sample >= len(ordered):
        return ordered

    groups: dict[str, list[str]] = {}
    for name in ordered:
        groups.setdefault(module_type(name), []).append(name)
    share = max(1, int(layer_sample) // len(groups))
    sampled: list[str] = []
    for group_names in groups.values():
        if len(group_names) <= share:
            sampled.extend(group_names)
            continue
        step = len(group_names) / share
        sampled.extend(
            group_names[int(index * step)] for index in range(share)
        )
    return sorted(set(sampled))


def assert_probe_cache_fits(
    shapes: Mapping[str, tuple[int, int]],
    *,
    batches: int,
    max_retained_gib: float,
    layer_sample: int | None = None,
    bytes_per_element: int = 4,
) -> float:
    """Check the probe's memory budget on every rank, before it allocates.

    Every held-out batch is retained until the pooled oracle is known, so the
    footprint is linear in the batch count and reaches tens of GiB on a full
    transformer. Called from all ranks with identical inputs so that they fail
    together: the scorer itself lives on rank 0, and a rank-0-only abort would
    leave the others blocked forever in the gradient reduction. Failing here
    with a number and a remedy also beats an OOM kill, which on this cluster
    takes the whole pod down and leaves nothing to diagnose.

    Returns the projected size in GiB.
    """
    if not shapes:
        return 0.0
    retained = select_probe_layers(list(shapes), layer_sample)
    per_batch = sum(shapes[name][0] * shapes[name][1] for name in retained)
    # Held-out batches and their running mean follow the sampled layers, but
    # the calibration gradients the oracle needs as reference directions are
    # committed during calibration, before the sample is known, so they are
    # charged in full.
    calibration = sum(rows * cols for rows, cols in shapes.values())
    projected = (
        (per_batch * (batches + 1) + calibration)
        * bytes_per_element
        / 2**30
    )
    if projected > max_retained_gib:
        raise RuntimeError(
            f"RQ3 probe would retain about {projected:.1f} GiB of held-out "
            f"gradients, above the {max_retained_gib:.1f} GiB budget. Lower "
            "rq3.heldout_batches, set rq3.layer_sample to score a stratified "
            "subset of layers, or raise rq3.max_retained_gib if the node "
            "really has the memory."
        )
    return projected


def _mean(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


def _across_layers(scores: Sequence[LayerBatchScore], metric: str) -> float:
    """Combine one score per layer into a single whole-model value.

    All three metrics read the layers as one block-diagonal matrix, which is
    what makes the result a property of the model rather than of an arbitrary
    layer weighting. Every input must belong to the *same* held-out batch: the
    usefulness denominator ``||D||_F`` is a norm over layers only, so folding
    ``K`` batches into the same sum multiplies the numerator by ``K`` but the
    denominator by only ``sqrt(K)``, inflating the result by ``sqrt(K)`` -- an
    artefact that grows with the held-out budget and reads as a real effect.
    """
    if metric == "capture":
        numerator = sum(score.captured_energy for score in scores)
        denominator = sum(score.held_out_energy for score in scores)
        if denominator <= 0.0:
            return float("nan")
        return numerator / denominator

    reference_energy = sum(score.reference_norm**2 for score in scores)
    if reference_energy <= 0.0:
        return float("nan")
    numerator = sum(score.reference_dot for score in scores)
    if metric == "usefulness":
        return numerator / math.sqrt(reference_energy)

    held_out_energy = sum(score.held_out_energy for score in scores)
    if held_out_energy <= 0.0:
        return float("nan")
    return numerator / math.sqrt(reference_energy * held_out_energy)


def _whole_model_raw(
    records: Sequence[LayerRecord],
    arm: str,
    metric: str,
    budget: str,
) -> float:
    """Whole-model value, averaged over held-out batches when per-batch.

    The order matters: one value per batch, then the mean over batches. The
    reverse -- pooling every (layer, batch) score into one sum -- is not the
    same quantity for usefulness, and is not a per-batch quantity at all.
    """
    if not records:
        return float("nan")
    if budget == "pooled":
        return _across_layers([record.pooled[arm] for record in records], metric)

    batch_counts = {len(record.per_batch[arm]) for record in records}
    if len(batch_counts) != 1:
        raise ValueError(
            f"RQ3 arm {arm!r} has layers scored on different batch counts: "
            f"{sorted(batch_counts)}"
        )
    return _mean(
        [
            _across_layers(
                [record.per_batch[arm][index] for record in records], metric
            )
            for index in range(batch_counts.pop())
        ]
    )


def _layer_metric(
    record: LayerRecord,
    arm: str,
    metric: str,
    budget: str,
) -> float:
    if budget == "pooled":
        return getattr(record.pooled[arm], metric)
    return _mean([getattr(score, metric) for score in record.per_batch[arm]])


def _layer_oracle(record: LayerRecord, metric: str, budget: str) -> float:
    if budget == "pooled":
        assert record.oracle_pooled is not None
        return getattr(record.oracle_pooled, metric)
    return _mean([getattr(score, metric) for score in record.oracle_per_batch])


# Cosine is already dimensionless and already comparable across layers, and
# its oracle can sit near zero or cross it, so a ratio against that oracle has
# no stable meaning. It is reported raw only.
_NORMALISABLE_METRICS = frozenset({"capture", "usefulness"})


def _normalised_ratios(
    record: LayerRecord,
    arm: str,
    metric: str,
    budget: str,
) -> list[float]:
    """Oracle-normalised values, one per held-out batch.

    Each batch is divided by the oracle *on that same batch* before any
    averaging. Averaging first and dividing the means afterwards is a different
    quantity -- a ratio of means, not a mean of ratios -- and it discards the
    per-batch spread that tells us whether a subspace is reliably good or
    merely good on average.
    """
    if metric not in _NORMALISABLE_METRICS:
        return []
    if budget == "pooled":
        assert record.oracle_pooled is not None
        oracle = getattr(record.oracle_pooled, metric)
        value = getattr(record.pooled[arm], metric)
        if oracle > 0.0 and math.isfinite(oracle) and math.isfinite(value):
            return [value / oracle]
        return []

    ratios: list[float] = []
    for score, oracle_score in zip(
        record.per_batch[arm], record.oracle_per_batch
    ):
        oracle = getattr(oracle_score, metric)
        value = getattr(score, metric)
        # A non-positive oracle flips the meaning of "above 1", so those layers
        # are dropped from the normalised reading rather than sign-corrected.
        if oracle > 0.0 and math.isfinite(oracle) and math.isfinite(value):
            ratios.append(value / oracle)
    return ratios


def aggregate_arms(
    records: Sequence[LayerRecord],
    *,
    metric: str = "capture",
    budget: str = "per_batch",
) -> dict[str, Any]:
    """Aggregate one metric across layers, in the only two coherent ways.

    ``pooled_raw`` reads the layers as one block matrix, per held-out batch,
    then averages over batches. It answers a deployment question -- what
    fraction of the model's total held-out gradient energy a subspace captures
    -- and it is the only reading in which the ``random`` arm's ``C ~= r/d_in``
    sanity check holds. It is dominated by the few layers with the largest
    gradient norm.

    ``normalised_layer_mean`` is the unweighted mean over layers of the
    oracle-normalised value. It treats layers as replicates of one phenomenon,
    which is the reading the RQ3 claim needs, and is the primary number.

    An oracle-normalised quantity is never energy-weighted: weighting a
    scale-free ratio by a scale reintroduces the layer heterogeneity the
    normalisation just removed, and the product has no interpretation.
    """
    if metric not in {"capture", "usefulness", "cosine"}:
        raise ValueError(f"unknown RQ3 metric {metric!r}")
    if budget not in {"per_batch", "pooled"}:
        raise ValueError(f"unknown RQ3 scoring budget {budget!r}")
    if not records:
        return {}

    arms = sorted(records[0].per_batch)
    normalisable = metric in _NORMALISABLE_METRICS
    summary: dict[str, Any] = {}
    for arm in arms:
        layer_means: list[float] = []
        raw_layer_values: list[float] = []
        above_oracle = 0
        ratio_count = 0
        for record in records:
            raw_layer_values.append(_layer_metric(record, arm, metric, budget))
            ratios = _normalised_ratios(record, arm, metric, budget)
            if not ratios:
                continue
            layer_means.append(sum(ratios) / len(ratios))
            ratio_count += len(ratios)
            above_oracle += sum(1 for ratio in ratios if ratio > 1.0)

        summary[arm] = {
            "pooled_raw": _whole_model_raw(records, arm, metric, budget),
            "raw_layer_mean": _mean(raw_layer_values),
            "layers": len(records),
        }
        if not normalisable:
            continue
        summary[arm].update(
            {
                "normalised_layer_mean": _mean(layer_means),
                "normalised_layer_median": (
                    median(layer_means) if layer_means else float("nan")
                ),
                # Only meaningful for the per-batch budget, where beating the
                # pooled oracle is legitimate; a high fraction says the
                # held-out batches are noisy, not that the arm beat optimal.
                "above_oracle_fraction": (
                    above_oracle / ratio_count if ratio_count else float("nan")
                ),
                "layers_normalised": len(layer_means),
            }
        )
    return summary


def paired_contrast(
    records: Sequence[LayerRecord],
    left: str,
    right: str,
    *,
    metric: str = "capture",
    budget: str = "per_batch",
    normalised: bool = True,
) -> dict[str, Any]:
    """Compare two arms as paired per-layer differences.

    Every arm is scored on identical layers and identical held-out batches, so
    the arms are paired and the contrast should be read that way rather than as
    a difference of aggregates: the median paired difference and the fraction
    of layers each arm wins are invariant to the cross-layer weighting choice,
    and they supply the within-run replicate structure a single-seed table
    lacks. The refutation criterion is "complement >= selected on what fraction
    of layers", with a sign test -- not "which mean is larger".

    Differences are oracle-normalised by default. The win fractions and the
    sign test only need a per-layer ordering, which normalisation preserves,
    but the reported median difference is otherwise an average over layers of
    incommensurable units and is dominated by whichever layers happen to have
    the largest gradients.
    """
    # Cosine has no usable oracle, so it falls back to raw rather than
    # silently dropping every layer.
    normalised = normalised and metric in _NORMALISABLE_METRICS
    differences: list[float] = []
    left_wins = 0
    right_wins = 0
    for record in records:
        if normalised:
            left_ratios = _normalised_ratios(record, left, metric, budget)
            right_ratios = _normalised_ratios(record, right, metric, budget)
            if not left_ratios or not right_ratios:
                continue
            left_value = sum(left_ratios) / len(left_ratios)
            right_value = sum(right_ratios) / len(right_ratios)
        else:
            left_value = _layer_metric(record, left, metric, budget)
            right_value = _layer_metric(record, right, metric, budget)
        if not (math.isfinite(left_value) and math.isfinite(right_value)):
            continue
        difference = left_value - right_value
        differences.append(difference)
        if difference > 0.0:
            left_wins += 1
        elif difference < 0.0:
            right_wins += 1

    decided = left_wins + right_wins
    return {
        "left": left,
        "right": right,
        "metric": metric,
        "budget": budget,
        "normalised": normalised,
        "layers": len(differences),
        "median_paired_difference": (
            median(differences) if differences else float("nan")
        ),
        "mean_paired_difference": _mean(differences),
        "left_win_fraction": left_wins / decided if decided else float("nan"),
        "right_win_fraction": right_wins / decided if decided else float("nan"),
        "sign_test_p_value": _sign_test_p_value(left_wins, decided),
    }


def _sign_test_p_value(successes: int, trials: int) -> float:
    """Two-sided exact binomial test against ``p = 0.5``."""
    if trials <= 0:
        return float("nan")
    observed = max(successes, trials - successes)
    tail = sum(math.comb(trials, k) for k in range(observed, trials + 1))
    return min(1.0, 2.0 * tail / (2.0**trials))


def stratify_by_module_type(
    records: Sequence[LayerRecord],
    *,
    metric: str = "capture",
    budget: str = "per_batch",
) -> dict[str, dict[str, Any]]:
    """Aggregate separately per module type.

    An unweighted layer mean is a count-weighted mean over module types, so if
    MLP projections outnumber attention projections the overall number mostly
    reports MLP.
    """
    groups: dict[str, list[LayerRecord]] = {}
    for record in records:
        groups.setdefault(record.module_type, []).append(record)
    return {
        group: aggregate_arms(members, metric=metric, budget=budget)
        for group, members in sorted(groups.items())
    }


def build_report(
    records: Sequence[LayerRecord],
    *,
    metadata: Mapping[str, Any] | None = None,
    ky_fan_violations: Sequence[tuple[str, str, float]] = (),
    excluded_layers: Iterable[str] = (),
) -> dict[str, Any]:
    """Assemble the full RQ3 held-out probe report."""
    arms = sorted(records[0].per_batch) if records else []
    report: dict[str, Any] = {
        "metadata": dict(metadata or {}),
        "arms": arms,
        "layers_scored": len(records),
        # Named, not just counted: a layer absent from the scored set is a
        # layer the method declined to initialise from the gradient, and which
        # ones those are is part of the RQ3 answer rather than bookkeeping.
        "layers_excluded": sorted(excluded_layers),
        # Must stay empty: the pooled oracle is the exact rank-r maximiser, so
        # any entry here is a scoring bug and invalidates the normalised
        # columns of this report.
        "ky_fan_violations": [
            {"layer": layer, "arm": arm, "ratio": ratio}
            for layer, arm, ratio in ky_fan_violations
        ],
        "aggregates": {},
        "stratified": {},
        "contrasts": [],
        "per_layer": [],
    }
    for metric in ("capture", "usefulness", "cosine"):
        for budget in ("per_batch", "pooled"):
            key = f"{metric}/{budget}"
            report["aggregates"][key] = aggregate_arms(
                records, metric=metric, budget=budget
            )
            report["stratified"][key] = stratify_by_module_type(
                records, metric=metric, budget=budget
            )

    for metric in ("capture", "usefulness"):
        for right in ("complement", "mean", "random"):
            if right in arms and "selected" in arms:
                report["contrasts"].append(
                    paired_contrast(
                        records,
                        "selected",
                        right,
                        metric=metric,
                        budget="per_batch",
                    )
                )

    for record in records:
        entry: dict[str, Any] = {
            "name": record.name,
            "module_type": record.module_type,
            "rows": record.rows,
            "cols": record.cols,
            "rank": record.rank,
            "batch_oracle_ratio": record.batch_oracle_ratio,
            "arms": {},
        }
        for arm in arms:
            capture_ratios = _normalised_ratios(
                record, arm, "capture", "per_batch"
            )
            entry["arms"][arm] = {
                "capture_per_batch": _layer_metric(
                    record, arm, "capture", "per_batch"
                ),
                "capture_pooled": _layer_metric(
                    record, arm, "capture", "pooled"
                ),
                "capture_per_batch_normalised": _mean(capture_ratios),
                "capture_pooled_normalised": _mean(
                    _normalised_ratios(record, arm, "capture", "pooled")
                ),
                "usefulness_per_batch": _layer_metric(
                    record, arm, "usefulness", "per_batch"
                ),
                "usefulness_pooled": _layer_metric(
                    record, arm, "usefulness", "pooled"
                ),
                "usefulness_per_batch_normalised": _mean(
                    _normalised_ratios(record, arm, "usefulness", "per_batch")
                ),
                "cosine_per_batch": _layer_metric(
                    record, arm, "cosine", "per_batch"
                ),
                "capture_batches_above_oracle": sum(
                    1 for ratio in capture_ratios if ratio > 1.0
                ),
            }
        entry["oracle"] = {
            "capture_per_batch": _layer_oracle(record, "capture", "per_batch"),
            "capture_pooled": _layer_oracle(record, "capture", "pooled"),
            "usefulness_per_batch": _layer_oracle(
                record, "usefulness", "per_batch"
            ),
            "usefulness_pooled": _layer_oracle(record, "usefulness", "pooled"),
        }
        report["per_layer"].append(entry)
    return report
