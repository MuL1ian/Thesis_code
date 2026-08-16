"""One argument parser shared by every FlowBP entrypoint.

FLUX.1, FLUX.2 and SD3.5 used to each carry their own copy of the CLI surface.
The FLUX.1 parser was the superset and SD3.5 imported it directly, while the
FLUX.2 copy had drifted: it was missing the adapter (LoRA / GradLoRA / LoRA-GA /
OFT / LoRA-Pro) flags and carried 26 arguments at different defaults.

This module defines the union of all arguments once. Per-backbone defaults come
from :mod:`flowbp.config.defaults`, applied on top via ``set_defaults`` so the
resolved value for any existing config is unchanged.

Precedence, lowest to highest:

1. the ``default=`` on each argument (the FLUX.1 / SD3.5 value)
2. ``BACKEND_DEFAULTS[backend]``
3. the YAML file named by ``--config``
4. explicit command-line flags
"""

from __future__ import annotations

import argparse

from flowbp.config.defaults import (
    ARG_ALIASES,
    BACKEND_DEFAULTS,
    SUPPORTED_BACKENDS,
)
from flowbp.config.flowbp import (
    FLOWBP_TRAINER_CHOICES,
    load_config_defaults,
    normalize_trainer_name,
)


def build_parser(backend: str) -> argparse.ArgumentParser:
    """Build the full FlowBP parser with ``backend``'s defaults applied."""
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unknown backend={backend!r}; expected one of {SUPPORTED_BACKENDS}."
        )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a FlowBP YAML config. CLI flags override values from this file.",
    )

    parser.add_argument(
        "--trainer",
        type=str,
        default="leapalign",
        choices=FLOWBP_TRAINER_CHOICES,
        help="Training algorithm to run.",
    )

    # dataset & dataloader
    parser.add_argument("--data_json_path", type=str, default=None)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=10,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_latent_t",
        type=int,
        default=1,
        help="number of latent frames",
    )
    # Model components.
    parser.add_argument("--pretrained_model_name_or_path", type=str)

    # diffusion setting
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--cfg", type=float, default=0.0)

    # validation & logs
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--connector_dump_interval",
        type=int,
        default=0,
        help="Save connector latent pairs every N training steps; 0 disables.",
    )
    parser.add_argument(
        "--connector_dump_dir",
        type=str,
        default=None,
        help="Directory for connector latent dumps; defaults to output_dir/connector_dumps.",
    )
    parser.add_argument(
        "--connector_wandb_interval",
        type=int,
        default=0,
        help="Log decoded connector examples to wandb every N training steps; 0 disables.",
    )
    parser.add_argument(
        "--connector_wandb_num_samples",
        type=int,
        default=2,
        help="Number of samples to decode per connector wandb logging step.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--save_optimizer_state",
        action="store_true",
        default=False,
        help=(
            "Also write AdamW moments into full-finetuning checkpoints. "
            "This roughly doubles checkpoint size; RNG and data-loader position "
            "are not restored."
        ),
    )

    # optimizer & scheduler & Training
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=10,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--max_grad_norm", default=2.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--use_cpu_offload",
        action="store_true",
        help="Whether to use CPU offload for param & gradient & optimizer states.",
    )

    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
    parser.add_argument(
        "--train_sp_batch_size",
        type=int,
        default=1,
        help="Batch size for sequence parallel training",
    )

    parser.add_argument("--fsdp_sharding_startegy", default="full")

    # lr_scheduler
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of cycles in the learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay to apply."
    )
    parser.add_argument(
        "--adam_eps",
        type=float,
        default=1e-8,
        help=(
            "AdamW epsilon. The default matches the value every completed thesis "
            "run used, so the table stays comparable. Reward backprop clips "
            "gradients hard (raw norms reach the hundreds against "
            "max_grad_norm=1.0), leaving per-element grads near 1e-8 where this "
            "eps damps sqrt(v) unevenly across layers; pass 1e-15 to remove that "
            "damping. Optimizer state is fp32, so 1e-15 is safe."
        ),
    )
    parser.add_argument(
        "--use_lora",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Wrap the transformer with a PEFT LoRA adapter before FSDP.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="default",
        help="Comma-separated LoRA target module suffixes, or preset 'default'/'attention'.",
    )
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=float, default=64.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_use_rslora",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use rank-stabilized alpha/sqrt(rank) LoRA scaling.",
    )
    parser.add_argument(
        "--use_loraga",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use LoRA-GA reward-gradient initialization in the SD3.5 trainer.",
    )
    parser.add_argument("--loraga_target_modules", type=str, default="default")
    parser.add_argument("--loraga_rank", type=int, default=32)
    parser.add_argument("--loraga_alpha", type=float, default=64.0)
    parser.add_argument("--loraga_dropout", type=float, default=0.0)
    parser.add_argument(
        "--loraga_use_rslora",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--loraga_stable_gamma", type=float, default=16.0)
    parser.add_argument(
        "--loraga_use_second_moment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Draw LoRA-GA's A from GradLoRA's shrinkage mixture of the "
            "mean-gradient subspace and per-calibration-batch gradient energy. "
            "B, the stable scaling, and the base weight compensation are "
            "unchanged, so this isolates subspace selection."
        ),
    )
    parser.add_argument(
        "--loraga_second_moment_beta",
        type=float,
        default=0.5,
        help=(
            "Second-moment mixture weight in (0, 1]. Used only when "
            "--loraga_use_second_moment is enabled."
        ),
    )
    parser.add_argument("--loraga_calibration_batches", type=int, default=4)
    parser.add_argument(
        "--loraga_calibration_mode",
        type=str,
        default="sparse",
        choices=("sparse", "fullbp"),
    )
    parser.add_argument("--loraga_fullbp_batch_size", type=int, default=1)
    parser.add_argument(
        "--loraga_fullbp_full_cfg_grad",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--loraga_accumulation_device",
        type=str,
        default="cpu",
        choices=("cpu", "cuda"),
    )
    parser.add_argument(
        "--loraga_svd_device",
        type=str,
        default="cuda",
        choices=("cpu", "cuda"),
    )
    parser.add_argument("--loraga_svd_oversample", type=int, default=32)
    parser.add_argument("--loraga_svd_niter", type=int, default=4)
    parser.add_argument("--loraga_diagnostics_path", type=str, default=None)
    parser.add_argument(
        "--use_lora_pro",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optimize an enabled standard LoRA adapter with LoRA-Pro.",
    )
    parser.add_argument("--lora_pro_damping", type=float, default=1e-6)
    parser.add_argument("--lora_pro_rcond", type=float, default=None)
    parser.add_argument(
        "--lora_pro_shard_dense_states",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--lora_pro_save_optimizer_state",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use_gradlora",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use reward-gradient-aligned LoRA initialization. Version 1 is "
            "integrated in the SD3.5 trainer."
        ),
    )
    parser.add_argument(
        "--gradlora_target_modules",
        type=str,
        default="default",
        help="Comma-separated GradLoRA target module suffixes, or preset 'default'/'attention'.",
    )
    parser.add_argument("--gradlora_rank", type=int, default=32)
    parser.add_argument("--gradlora_alpha", type=float, default=64.0)
    parser.add_argument("--gradlora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--gradlora_calibration_batches",
        type=int,
        default=4,
        help="Reward-backprop batches used to estimate each virtual dense gradient.",
    )
    parser.add_argument(
        "--gradlora_calibration_mode",
        type=str,
        default="sparse",
        choices=("sparse", "fullbp"),
        help=(
            "Gradient source for GradLoRA calibration: 'sparse' follows the "
            "configured training sampler; 'fullbp' backpropagates through the "
            "complete flow trajectory."
        ),
    )
    parser.add_argument(
        "--gradlora_fullbp_batch_size",
        type=int,
        default=1,
        help="Per-rank microbatch size used only for FullBP calibration.",
    )
    parser.add_argument(
        "--gradlora_fullbp_full_cfg_grad",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Backpropagate through both CFG branches during FullBP calibration. "
            "By default FullBP follows cfg_detach_neg from the training config."
        ),
    )
    parser.add_argument(
        "--gradlora_use_second_moment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Initialize GradLoRA from a shrinkage mixture of the mean-gradient "
            "subspace and per-calibration-batch gradient energy."
        ),
    )
    parser.add_argument(
        "--gradlora_second_moment_beta",
        type=float,
        default=0.5,
        help=(
            "Second-moment mixture weight in (0, 1]. Used only when "
            "--gradlora_use_second_moment is enabled."
        ),
    )
    parser.add_argument(
        "--gradlora_spectral_exponent",
        type=float,
        default=0.0,
        help=(
            "Exponent p for GradLoRA A initialization: A is proportional to "
            "spectrum**p times the right spectral vectors. The result is "
            "globally row-RMS normalized, so p changes spectral shape but not "
            "the original PEFT A scale."
        ),
    )
    parser.add_argument(
        "--gradlora_a_scale_mode",
        type=str,
        default="legacy",
        choices=("legacy", "relative", "gain"),
        help=(
            "How the GradLoRA A row scale c is chosen. 'legacy' keeps PEFT's "
            "random Kaiming row RMS, 'relative' multiplies it by "
            "--gradlora_a_scale, and 'gain' sets c so the write gain "
            "s*c equals --gradlora_a_scale. B starts at zero in every mode, so "
            "the model function is unchanged regardless of c."
        ),
    )
    parser.add_argument(
        "--gradlora_a_scale",
        type=float,
        default=1.0,
        help=(
            "Multiplier ('relative' mode) or target write gain s*c ('gain' "
            "mode) for the GradLoRA A initializer. Ignored when "
            "--gradlora_a_scale_mode is 'legacy'."
        ),
    )
    parser.add_argument(
        "--gradlora_accumulation_device",
        type=str,
        default="cpu",
        choices=("cpu", "cuda"),
        help="Where virtual dense calibration gradients are accumulated.",
    )
    parser.add_argument(
        "--gradlora_svd_device",
        type=str,
        default="cuda",
        choices=("cpu", "cuda"),
        help="Device used for the one-time randomized SVD.",
    )
    parser.add_argument("--gradlora_svd_oversample", type=int, default=8)
    parser.add_argument("--gradlora_svd_niter", type=int, default=4)
    parser.add_argument(
        "--gradlora_init_variant",
        type=str,
        default="selected",
        choices=("selected", "mean", "random"),
        help=(
            "Which subspace initializes LoRA A: the calibrated one, the "
            "first-order mean-gradient control, or a same-rank random "
            "control. Controls keep the original PEFT A row RMS."
        ),
    )
    parser.add_argument(
        "--gradlora_freeze_a",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Exclude lora_A from the optimizer after calibration so only B "
            "trains, attributing reward differences to the chosen subspace."
        ),
    )
    parser.add_argument(
        "--gradlora_usefulness_batches",
        type=int,
        default=0,
        help=(
            "Extra held-out reward-backprop batches scored against the "
            "calibrated subspace after finalization. Writes "
            "<output_dir>/gradlora_usefulness.json. 0 disables the probe."
        ),
    )
    parser.add_argument(
        "--gradlora_diagnostics_path",
        type=str,
        default=None,
        help=(
            "Optional JSON path for calibration spectra. Defaults to "
            "<output_dir>/gradlora_calibration.json."
        ),
    )
    parser.add_argument(
        "--use_rq3_probe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Score the held-out probe under the RQ3 protocol: adds the "
            "complement arm, normalizes every metric by the pooled held-out "
            "oracle, and reports per-batch and pooled budgets side by side "
            "with paired per-layer contrasts. Requires a positive "
            "--gradlora_usefulness_batches."
        ),
    )
    parser.add_argument(
        "--rq3_report_path",
        type=str,
        default=None,
        help=(
            "Optional JSON path for the RQ3 held-out probe report. Defaults "
            "to <output_dir>/rq3_heldout_probe.json."
        ),
    )
    parser.add_argument(
        "--rq3_score_device",
        type=str,
        default="cpu",
        help=(
            "Device holding held-out gradients during RQ3 scoring. Every "
            "held-out batch is retained until the pooled oracle is known, so "
            "this is the memory-heavy choice; cpu is the safe default."
        ),
    )
    parser.add_argument(
        "--rq3_layer_sample",
        type=int,
        default=None,
        help=(
            "Score only this many layers, sampled evenly within each module "
            "type. Cuts retained memory proportionally; the paired per-layer "
            "statistics stay valid on the subset. Default scores every layer."
        ),
    )
    parser.add_argument(
        "--rq3_max_retained_gib",
        type=float,
        default=64.0,
        help=(
            "Abort the RQ3 probe if the projected held-out gradient cache "
            "exceeds this many GiB, instead of risking an OOM kill."
        ),
    )
    parser.add_argument(
        "--rq3_batch_oracle_ratio",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also report how much better each held-out batch's own leading "
            "subspace is than the pooled oracle. Diagnostic only, and costs "
            "one exact eigendecomposition per layer per held-out batch, so it "
            "is off by default."
        ),
    )
    parser.add_argument(
        "--use_oft",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Wrap the transformer with a PEFT OFT adapter before FSDP.",
    )
    parser.add_argument(
        "--oft_target_modules",
        type=str,
        default="default",
        help="Comma-separated OFT target module suffixes, or preset 'default'/'attention'.",
    )
    parser.add_argument("--oft_block_size", type=int, default=32)
    parser.add_argument("--oft_module_dropout", type=float, default=0.0)
    parser.add_argument(
        "--oft_coft", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--oft_eps", type=float, default=6e-5)
    parser.add_argument(
        "--oft_block_share", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--master_weight_type",
        type=str,
        default="fp32",
        help="Weight type to use - fp32 or bf16.",
    )

    # GRPO training
    parser.add_argument(
        "--h",
        type=int,
        default=None,
        help="video height",
    )
    parser.add_argument(
        "--w",
        type=int,
        default=None,
        help="video width",
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=None,
        help="sampling steps",
    )
    parser.add_argument(
        "--rollout_steps",
        type=int,
        default=25,
        help="Number of rollout denoising steps for ReFL training.",
    )
    parser.add_argument(
        "--sampler_seed",
        type=int,
        default=None,
        help="seed of sampler",
    )
    parser.add_argument(
        "--use_hpsv2",
        action="store_true",
        default=False,
        help="Deprecated compatibility flag; prefer --reward_model_type hpsv2.",
    )
    parser.add_argument(
        "--reward_model_type",
        type=str,
        choices=("hpsv2", "mps", "hpsv3plusplus"),
        default="hpsv2",
        help="Differentiable reward used for generator back-propagation.",
    )
    parser.add_argument(
        "--reward_ckpt_path",
        type=str,
        default="models/reward_ckpts",
        help="Reward checkpoint file or root directory.",
    )
    parser.add_argument(
        "--mps_processor_name",
        type=str,
        default=None,
        help="Optional local path or Hugging Face id for the MPS CLIP processor.",
    )
    parser.add_argument("--mps_condition", type=str, default=None)
    parser.add_argument("--mps_mask_threshold", type=float, default=0.3)
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Enable Exponential Moving Average of model weights.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="flux_rlhf",
        help="project name for wandb",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="flux_grpo",
        help="run name for wandb",
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        nargs="*",
        default=None,
        help="Optional wandb tags, e.g. --wandb_tags mps lora.",
    )
    parser.add_argument(
        "--cfg_guidance",
        type=float,
        default=3.5,
        help="guidance scale for cfg",
    )

    # refl parameters:
    parser.add_argument(
        "--select_idx_seed",
        type=int,
        default=None,
        help="select index seed for refl",
    )
    parser.add_argument(
        "--min_idx",
        type=int,
        default=1,
        help="min index for refl",
    )
    parser.add_argument(
        "--max_idx",
        type=int,
        default=10,
        help="max index for refl",
    )
    parser.add_argument(
        "--train_step_tail_ratio",
        type=float,
        default=1.0,
        help=(
            "For FlowBP variants, expose only this final fraction of rollout "
            "steps to gradient-carrying active-step sampling. 1.0 keeps the "
            "full rollout."
        ),
    )
    parser.add_argument(
        "--loss_grad_scale",
        type=float,
        default=1.0,
        help="reward loss coefficient",
    )
    parser.add_argument(
        "--reward_loss_type",
        type=str,
        default="hinge",
        choices=["hinge", "relu_hinge", "neg_reward"],
        help=(
            "Reward optimization loss. 'hinge' keeps relu(-reward + "
            "loss_relu_clip); 'neg_reward' directly maximizes via -reward."
        ),
    )
    parser.add_argument(
        "--reward_input_mode",
        type=str,
        default="posterior_mean",
        choices=["posterior_mean", "sampled_x0"],
        help=(
            "Which image the reward model scores. 'posterior_mean' is the "
            "per-step x0 estimate used by ReFL and DRTune. 'sampled_x0' "
            "finishes the rollout under no_grad and swaps the forward value "
            "for the sampled endpoint via a straight-through residual, "
            "leaving the active set and gradient path untouched."
        ),
    )
    parser.add_argument(
        "--hpsv3plusplus_iter_start",
        type=float,
        default=0.3,
        help="Initial normalized HPSv3++ RL-iteration condition.",
    )
    parser.add_argument(
        "--hpsv3plusplus_iter_end",
        type=float,
        default=1.0,
        help="Final normalized HPSv3++ RL-iteration condition.",
    )
    parser.add_argument(
        "--loss_relu_clip",
        type=float,
        default=0.7,
        help="relu clip for reward loss",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Bridge coupling coefficient for LeapAlign and FlowBP bridged variants.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.1,
        help="Residual floor for trajectory similarity weighting.",
    )

    # j-k index sampling (shared by FlowBP-Bridge/Lagrange trainers)
    parser.add_argument(
        "--jk_sampling_mode",
        type=str,
        default="uniform",
        choices=("uniform",),
        help=(
            "How to draw the (j, k) step indices for FlowBP bridge/lagrange "
            "rollouts. Only the randperm-based 'uniform' sampler remains; the "
            "dirichlet / midpoint / midpoint_j samplers were removed."
        ),
    )

    parser.add_argument(
        "--refl_last_n_steps",
        type=int,
        default=11,
        help="For ReFL, randomly update one step among the final N rollout steps.",
    )
    parser.add_argument(
        "--flowbp_lagrange_connector_order",
        type=int,
        default=4,
        help="Number of trajectory-prior support velocities for FlowBP-Lagrange.",
    )
    parser.add_argument(
        "--flowbp_lagrange_detach_history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detach cached history velocities in FlowBP-Lagrange.",
    )
    parser.add_argument(
        "--flowbp_lagrange_grad_support_mode",
        type=str,
        default="start",
        choices=("none", "start", "midpoint", "all"),
        help="Which FlowBP-Lagrange support velocities participate in backward.",
    )
    parser.add_argument(
        "--flowbp_lagrange_grad_support_scale",
        type=float,
        default=1.0,
        help="Backward gradient scale for non-start active support velocities.",
    )
    parser.add_argument(
        "--flowbp_lagrange_max_active_supports",
        type=int,
        default=0,
        help="Maximum active supports per interval; 0 means no extra limit.",
    )
    parser.add_argument(
        "--debug_flowbp_lagrange_connector",
        action="store_true",
        default=False,
        help="Enable FlowBP-Lagrange connector sanity checks.",
    )
    parser.add_argument(
        "--flowbp_lagrange_weight_scheme",
        type=str,
        default="lagrange",
        choices=("lagrange", "uniform"),
        help="Weight scheme for FlowBP-Lagrange supports.",
    )
    parser.add_argument(
        "--flowbp_lagrange_grad_rescale",
        type=float,
        default=0.0,
        help=(
            "Gradient rescaling strength for active support velocities. "
            "0=off, 1=full rescale (Σ|w_all|/Σ|w_active|), "
            "values in (0,1) interpolate."
        ),
    )
    parser.add_argument(
        "--flowbp_lagrange_anchor_lambda",
        type=float,
        default=1.0,
        help=(
            "Blend factor for anchored FlowBP-Lagrange connector. "
            "1.0 = FlowBP-Lagrange, 0.0 = Euler/LeapAlign jump, "
            "values in (0,1) use Euler-anchored Lagrange correction."
        ),
    )

    # FlowBP-Sparse trainer
    parser.add_argument(
        "--flowbp_sparse_num_active_steps",
        type=int,
        default=3,
        help=(
            "FlowBP-Sparse trainer: number of active sampling steps to "
            "re-forward with grad per rollout (default 3)."
        ),
    )
    parser.add_argument(
        "--flowbp_sparse_late_bias",
        type=float,
        default=2.0,
        help=(
            "FlowBP-Sparse trainer: power-law bias for active step "
            "sampling. w_i ∝ (i+1)^bias. 0 = uniform, larger = more weight "
            "on late (close-to-clean) indices."
        ),
    )
    parser.add_argument(
        "--flowbp_sparse_grad_rescale",
        type=float,
        default=0.0,
        help=(
            "FlowBP-Sparse trainer: gradient rescale strength (>= 0). "
            "Formula: factor = 1 + gr * (Σ|Δσ_all|/Σ|Δσ_active| - 1). "
            "0 = raw small per-step gradient, 1 = full Euler/ReFL-magnitude "
            "gradient, >1 = push beyond full Euler (each active velocity gets "
            "more than its fair share). Use values >1 when even gr=1 yields "
            "small grad_norm under strong late-bias sampling."
        ),
    )
    # DRTune trainer
    parser.add_argument(
        "--drtune_num_train_steps",
        type=int,
        default=3,
        help=(
            "DRTune: number of equally spaced denoising steps to train per "
            "rollout. Paper default for 25-step SDXL is K=3."
        ),
    )
    parser.add_argument(
        "--drtune_early_stop_steps",
        type=int,
        default=None,
        help=(
            "DRTune: maximum reverse-index early-stop step m. None uses "
            "--drtune_early_stop_ratio * rollout_steps; 0 disables early stop."
        ),
    )
    parser.add_argument(
        "--drtune_early_stop_ratio",
        type=float,
        default=0.4,
        help=(
            "DRTune: default early-stop ratio used when "
            "--drtune_early_stop_steps is None. Paper suggests m ~= 0.4T."
        ),
    )

    # DRaFT-LV trainer
    parser.add_argument(
        "--draft_lv_num_noised_samples",
        type=int,
        default=2,
        help=(
            "DRaFT-LV: number of extra noised last-step examples. "
            "The paper default is n=2."
        ),
    )

    # evaluation
    parser.add_argument(
        "--evaluation_interval",
        type=int,
        default=500,
        help="evaluation interval",
    )
    parser.add_argument(
        "--evaluation_prompts",
        type=str,
        nargs="+",
        default=[
            "A man lying down on green grass, gazing at the stars during an evening at a countryside villa",
            "A grey tabby cat with yellow eyes rests on a weathered wooden log under bright sunlight.",
            "A photo of four giraffes",
            "A photo of a red rabbit on the left of a white stop sign",
        ],
        help=(
            "A list of prompts to use for generating sample images during training. "
            "To provide multiple prompts, separate them with spaces. "
            "If a prompt contains spaces, you MUST enclose it in quotes. "
            'Example: --evaluation_prompts "a photo of a cat" "an astronaut on a horse"'
        ),
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="whether use debug mode",
    )
    parser.add_argument(
        "--eval_prompts_file",
        type=str,
        default="./assets/eval_prompts.txt",
        help="Path to eval prompts file for online HPSv2.1 evaluation.",
    )
    parser.add_argument(
        "--eval_num_imgs_per_prompt",
        type=int,
        default=4,
        help="Number of images to generate per prompt during evaluation.",
    )
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=42,
        help="Random seed for evaluation image generation.",
    )
    parser.add_argument(
        "--eval_reward_fn",
        type=str,
        nargs="+",
        default=["hpsv2", "hpsv3"],
        help=(
            "Reward metrics to compute during evaluation, e.g. "
            "clipscore aesthetic hpsv2 hpsv3 pickscore imagereward."
        ),
    )
    parser.add_argument(
        "--eval_reward_ckpt_path",
        type=str,
        default="",
        help="Directory containing evaluation reward model checkpoints.",
    )
    parser.add_argument(
        "--eval_hpsv3_config_path",
        type=str,
        default="",
        help=(
            "Optional HPSv3 inferencer YAML config path. When empty, the "
            "online eval runner auto-discovers a config under "
            "<eval_reward_ckpt_path>/HPSv3/ and falls back to "
            "assets/eval/hpsv3/HPSv3_7B.yaml (which points at the local "
            "Qwen2-VL-7B-Instruct mirror)."
        ),
    )
    parser.add_argument(
        "--eval_hpsv3_checkpoint_path",
        type=str,
        default="",
        help=(
            "Optional HPSv3 .safetensors / .pt checkpoint path override. "
            "Defaults to <eval_reward_ckpt_path>/HPSv3/HPSv3.safetensors "
            "when present."
        ),
    )
    # ---- FLUX.2-only arguments -------------------------------------------
    # Ignored by the FLUX.1 / SD3.5 trainers, which never read them.

    parser.add_argument(
        "--caption_key",
        type=str,
        default="caption",
        help="JSON key that holds the prompt string in each dataset entry.",
    )

    parser.add_argument(
        "--cfg_detach_neg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether to detach the negative-prompt branch in "
            "``_cfg_aware_forward``. Default True (memory-efficient, gradient "
            "flows only through the conditional branch). Pass "
            "--no-cfg_detach_neg to compute pred_neg with grad enabled; the "
            "effective gradient becomes the mathematically standard CFG "
            "gradient ``(1-cfg)*d(pred_neg)/dθ + cfg*d(pred_cond)/dθ``. This "
            "costs roughly 1.8x activation memory due to the second backward "
            "graph."
        ),
    )

    parser.add_argument(
        "--cfg_grad_norm_compensate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether to divide the backward loss by ``cfg_guidance`` to undo "
            "the gradient amplification introduced by the detached negative "
            "branch in ``_cfg_aware_forward``. Default True (recommended). "
            "Pass --no-cfg_grad_norm_compensate to use the raw CFG-scaled gradient; "
            "effective gradient magnitude then scales linearly with cfg, "
            "equivalent to setting LR = base_lr * cfg_guidance."
        ),
    )

    parser.add_argument(
        "--clip_d0",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Switch: when on, drop samples whose one-step pred_x0 error d_0 "
            "exceeds --clip_d0_threshold from the reward loss (their "
            "gradient contribution is zeroed). Forward still runs for those "
            "samples; only the reward signal is masked. ANDs with the "
            "--clip_dj_threshold mask if both are active. Default: off."
        ),
    )

    parser.add_argument(
        "--clip_d0_threshold",
        type=float,
        default=0.2,
        help=(
            "Per-sample d_0 threshold used when --clip_d0 is enabled. "
            "Default: 0.2."
        ),
    )

    parser.add_argument(
        "--clip_dj_threshold",
        type=float,
        default=0.0,
        help=(
            "If > 0, drop samples whose connector d_j exceeds the threshold "
            "from the reward loss (their gradient contribution is zeroed). "
            "Forward still runs for those samples; only the reward signal is "
            "masked. Set 0 to disable (default)."
        ),
    )

    parser.add_argument(
        "--eval_guidance_scale",
        type=float,
        default=4.0,
        help="CFG scale for eval sampling (klein-base-4B default is 4.0).",
    )

    parser.add_argument(
        "--eval_num_steps",
        type=int,
        default=50,
        help="Number of denoising steps for eval sampling.",
    )

    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Max Qwen3 tokenizer context (FLUX.2 default is 512).",
    )

    parser.add_argument(
        "--text_encoder_out_layers",
        type=int,
        nargs="+",
        default=[9, 18, 27],
        help="Qwen3 hidden-state layer indices fused into prompt_embeds (3 layers -> 3*hidden_dim=7680).",
    )

    parser.add_argument(
        "--use_precomputed_embeds",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If set, load Qwen3 prompt embeddings precomputed by "
            "flowbp/data_preprocess/preprocess_flux2_embedding.py instead of "
            "running the text encoder on the fly. data_json_path should point "
            "to the resulting captions2embed.json."
        ),
    )

    # ---- legacy spellings -------------------------------------------------
    for alias, canonical in ARG_ALIASES.items():
        parser.add_argument(
            f"--{alias}",
            dest=canonical,
            default=argparse.SUPPRESS,
            help=f"Deprecated alias for --{canonical}.",
        )

    parser.set_defaults(**BACKEND_DEFAULTS[backend])
    return parser


def _enabled_adapter_flags(args: argparse.Namespace) -> list[str]:
    return [
        name
        for name in (
            "use_lora",
            "use_gradlora",
            "use_loraga",
            "use_lora_pro",
            "use_oft",
        )
        if bool(getattr(args, name, False))
    ]


def _validate_backend_args(backend: str, args: argparse.Namespace) -> None:
    enabled_adapters = _enabled_adapter_flags(args)
    if backend in {"flux1", "flux2"} and enabled_adapters:
        raise ValueError(
            f"{backend} does not implement adapter training, but the following "
            f"flags are enabled: {enabled_adapters}. Use the SD3.5 entrypoint "
            "or disable these flags; otherwise the run would silently perform "
            "full-parameter fine-tuning."
        )

    if backend == "sd35" and enabled_adapters:
        resume = getattr(args, "resume_from_checkpoint", None)
        if resume:
            raise NotImplementedError(
                "SD3.5 adapter checkpoints cannot currently be resumed. "
                f"Disable resume_from_checkpoint={resume!r} or run full "
                "fine-tuning. Refusing to silently restart adapter training."
            )
        if bool(getattr(args, "save_optimizer_state", False)):
            raise ValueError(
                "save_optimizer_state applies only to full-finetuning "
                "checkpoints; SD3.5 adapter checkpoints do not currently have "
                "a matching resume path."
            )


def parse_args(backend: str, argv: list[str] | None = None) -> argparse.Namespace:
    """Parse args for ``backend``, layering the YAML config under the CLI flags."""
    parser = build_parser(backend)
    probe_args, _ = parser.parse_known_args(argv)
    if probe_args.config:
        parser.set_defaults(**load_config_defaults(probe_args.config))
    args = parser.parse_args(argv)
    args.trainer = normalize_trainer_name(args.trainer)
    _validate_backend_args(backend, args)
    return args
