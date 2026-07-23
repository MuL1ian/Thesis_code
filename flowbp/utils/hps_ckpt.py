"""Resolve HPSv2 checkpoints: local first, else download from Hugging Face."""

from __future__ import annotations

import os
from pathlib import Path

CLIP_REPO_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
CLIP_FILENAME = "open_clip_pytorch_model.bin"
HPS_REPO_ID = "xswu/HPSv2"
HPS_FILENAME = "HPS_v2.1_compressed.pt"


def _default_hps_dir() -> Path:
    env = os.environ.get("FLOWBP_HPS_CKPT_DIR", "").strip()
    if env:
        return Path(os.path.expanduser(env))
    return Path("./hps_ckpt")


def _is_main_process() -> bool:
    return int(os.environ.get("LOCAL_RANK", "0") or "0") <= 0


def _barrier() -> None:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        pass


def _download_file(repo_id: str, filename: str, local_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(local_dir),
    )
    return Path(path)


def ensure_hpsv2_ckpts(
    local_dir: str | Path | None = None,
    *,
    download: bool = True,
) -> tuple[str, str]:
    """
    Return (open_clip_bin_path, hps_v2_1_pt_path).

    Resolution order:
    1. ``FLOWBP_HPS_CLIP_PATH`` / ``FLOWBP_HPS_PT_PATH`` if both exist
    2. files under ``local_dir`` (default ``./hps_ckpt`` or ``FLOWBP_HPS_CKPT_DIR``)
    3. Hugging Face download when ``download=True`` (LOCAL_RANK<=0 only)
    """
    clip_env = os.environ.get("FLOWBP_HPS_CLIP_PATH", "").strip()
    hps_env = os.environ.get("FLOWBP_HPS_PT_PATH", "").strip()
    if clip_env and hps_env:
        clip_p, hps_p = Path(os.path.expanduser(clip_env)), Path(os.path.expanduser(hps_env))
        if clip_p.is_file() and hps_p.is_file():
            return str(clip_p), str(hps_p)

    root = Path(local_dir) if local_dir is not None else _default_hps_dir()
    root = root.expanduser()
    clip_path = root / CLIP_FILENAME
    hps_path = root / HPS_FILENAME

    need_clip = not clip_path.is_file()
    need_hps = not hps_path.is_file()
    if not need_clip and not need_hps:
        return str(clip_path), str(hps_path)

    if not download:
        missing = []
        if need_clip:
            missing.append(str(clip_path))
        if need_hps:
            missing.append(str(hps_path))
        raise FileNotFoundError(
            "HPSv2 checkpoints missing and download disabled: " + ", ".join(missing)
        )

    if _is_main_process():
        root.mkdir(parents=True, exist_ok=True)
        if need_clip:
            print(
                f"[hps_ckpt] downloading {CLIP_FILENAME} from {CLIP_REPO_ID} -> {root}",
                flush=True,
            )
            _download_file(CLIP_REPO_ID, CLIP_FILENAME, root)
        if need_hps:
            print(
                f"[hps_ckpt] downloading {HPS_FILENAME} from {HPS_REPO_ID} -> {root}",
                flush=True,
            )
            _download_file(HPS_REPO_ID, HPS_FILENAME, root)

    _barrier()

    if not clip_path.is_file() or not hps_path.is_file():
        raise FileNotFoundError(
            f"HPSv2 checkpoints still missing after download under {root}: "
            f"clip={clip_path.is_file()} hps={hps_path.is_file()}"
        )
    return str(clip_path), str(hps_path)
