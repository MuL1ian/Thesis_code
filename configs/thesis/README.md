# Thesis configs

These are the training configs behind the experiments in Chapter 6 of the
thesis. Every run fine-tunes SD3.5-M on HPSv3++ for 300 updates with a global
batch of 32 (8 GPUs × batch 4). All training logs are in the W&B project
[diffusionRL-Curl/thesis](https://forge.coreweave.com/wandb/diffusionRL-Curl/thesis).

Launch a config with

```bash
CONFIG=configs/thesis/<config>.yaml bash scripts/finetune/train_sd35.sh
```

Flags after the script override the YAML values. Each sweep below reuses a
base config and changes one flag. When you rerun a sweep, also pass
`--run_name` and `--output_dir` so that runs do not overwrite each other.

## Paths

The configs use these repo-relative paths. Put the files there, or override
the keys.

| Key | Path |
| --- | --- |
| `model.pretrained_model_name_or_path` | `checkpoints/SD3.5_medium` |
| `data.data_json_path` | `data/sd3_5_rl_embeddings/videos2caption.json` (written by `scripts/preprocess/preprocess_sd35_embeddings.sh`) |
| `eval.prompts_file` | `drawbench/test.txt` (200 DrawBench prompts, five rounds) |
| `eval.reward_ckpt_path` | `checkpoints/reward_ckpts` |
| `eval.hpsv3_config_path` | `checkpoints/reward_ckpts/HPSv3-PlusPlus/config.json` |
| `eval.hpsv3_checkpoint_path` | `checkpoints/reward_ckpts/HPSv3-PlusPlus/hpsv3++.pth` |

## Approximation over the steps (Section 6.2)

| Config | Thesis | W&B run |
| --- | --- | --- |
| `sd3_5_flowbp_sparse_hpsv3plusplus_k3.yaml` | FlowBP, K = 3 (Table 6.3) | [`qh2uyta1`](https://forge.coreweave.com/wandb/diffusionRL-Curl/thesis/runs/qh2uyta1) |
| `sd3_5_refl_hpsv3plusplus.yaml` | ReFL (Table 6.3) | [`xpumveby`](https://forge.coreweave.com/wandb/diffusionRL-Curl/thesis/runs/xpumveby) |
| `sd3_5_drtune_hpsv3plusplus.yaml` | DRTune (Table 6.3) | [`ecsnv9h4`](https://forge.coreweave.com/wandb/diffusionRL-Curl/thesis/runs/ecsnv9h4) |
| `sd3_5_leapalign_hpsv3plusplus.yaml` | LeapAlign (Table 6.3) | [`g8mbg001`](https://forge.coreweave.com/wandb/diffusionRL-Curl/thesis/runs/g8mbg001) |
| `sd3_5_refl_hpsv3plusplus_endpoint.yaml` | ReFL, complete image (Table 6.4) | [`fwws0e6e`](https://forge.coreweave.com/wandb/diffusionRL-Curl/thesis/runs/fwws0e6e) |
| `sd3_5_drtune_hpsv3plusplus_endpoint.yaml` | DRTune, complete image, early window (Table 6.4) | [`qymn4be0`](https://forge.coreweave.com/wandb/diffusionRL-Curl/thesis/runs/qymn4be0) |
| `sd3_5_drtune_hpsv3plusplus_endpoint_nostop.yaml` | DRTune, complete image, whole trajectory (Table 6.4) | [`j4f3unr5`](https://forge.coreweave.com/wandb/diffusionRL-Curl/thesis/runs/j4f3unr5) |
| `sd3_5_flowbp_bridge_hpsv3plusplus_k4.yaml` | FlowBP-Bridge, K = 4, α = 0.3 (Section 6.2.3) | [`ut5joccq`](https://forge.coreweave.com/wandb/diffusionRL-Curl/thesis/runs/ut5joccq) |

Sweeps:

| Base config | Flag | Thesis | W&B runs |
| --- | --- | --- | --- |
| `..._flowbp_sparse_hpsv3plusplus_k3.yaml` | `--flowbp_sparse_num_active_steps 2` / `4` / `5` | Number of active steps (Section 6.2.4) | `2evrstks` / `8fmz3evg` / `85iho7wm` |
| `..._flowbp_sparse_hpsv3plusplus_k3.yaml` | `--flowbp_sparse_late_bias 2` | Late-biased draw (Table 6.4) | — |
| `..._flowbp_bridge_hpsv3plusplus_k4.yaml` | `--alpha 0` / `0.1` / `0.5` / `1.0` | Bridge weight (Section 6.2.3) | `svnxuuvn` / `7amzikj1` / `drzshkgd` / `k196ioqk` |

## Approximation over the parameters (Section 6.3)

All adapter runs use rank eight, `alpha: 16` (LoRA scale s = alpha / rank = 2),
learning rate 1e-4, and FlowBP K = 3 gradients.

| Config | Thesis | W&B run |
| --- | --- | --- |
| `sd3_5_flowbp_sparse_hpsv3plusplus_k3_gradlora_mean_all_r8_lr1e4.yaml` | GradLoRA, all layers (Tables 6.5, 6.6) | [`7sj9nmhd`](https://forge.coreweave.com/wandb/diffusionRL-Curl/thesis/runs/7sj9nmhd) |
| `sd3_5_flowbp_sparse_hpsv3plusplus_k3_lora_all_r8_lr1e4.yaml` | LoRA, all layers (Tables 6.5, 6.6) | [`oextpmom`](https://forge.coreweave.com/wandb/diffusionRL-Curl/thesis/runs/oextpmom) |
| `sd3_5_flowbp_sparse_hpsv3plusplus_k3_lora_attn_r8_lr1e4.yaml` | LoRA, attention only (Table 6.5) | [`w8b06ko6`](https://forge.coreweave.com/wandb/diffusionRL-Curl/thesis/runs/w8b06ko6) |

The basis controls of Table 6.6 start from the GradLoRA config:

| Initial basis | Input factor A | Flags | W&B run |
| --- | --- | --- | --- |
| Random orthogonal | Trainable | `--gradlora_init_variant random` | `1p4oaok4` |
| GradLoRA | Frozen | `--gradlora_freeze_a` | `nehgt611` |
| Random orthogonal | Frozen | `--gradlora_init_variant random --gradlora_freeze_a` | `o405pah4` |

## Gradient diagnostics (Section 6.4)

The held-out probe of Section 6.4.1 runs inside the trainer after GradLoRA
calibration. Turn it on with `--gradlora_usefulness_batches <n> --use_rq3_probe`;
it writes `<output_dir>/rq3_heldout_probe.json`. The protocol is in
Appendix C.3.
