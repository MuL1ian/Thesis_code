Path-stripped copies of `configs/Thesis/Thesis2` for publication.

Site-specific absolute paths have been replaced by placeholders. Hyperparameters, trainer settings, and relative `_base_` inheritance are unchanged.

| key | placeholder |
|---|---|
| `model.pretrained_model_name_or_path` | `checkpoints/SD3.5_medium` |
| `eval.prompts_file` | `prompts/drawbench/test.txt` |
| `eval.reward_ckpt_path` | `checkpoints/reward_ckpts` |
| `eval.hpsv3_config_path` | `checkpoints/reward_ckpts/HPSv3-PlusPlus/config.json` |
| `eval.hpsv3_checkpoint_path` | `checkpoints/reward_ckpts/HPSv3-PlusPlus/hpsv3++.pth` |

`data.data_json_path` and `train.output_dir` were already repo-relative and are left as-is.
