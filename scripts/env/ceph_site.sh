# Resolve Ceph root by SITE.
#   SITE=gz     (default) -> /apdcephfs_fsgm3/share_305110755
#   SITE=zw|local         -> /apdcephfs_zwfy8/share_305110755
#
# Usage:
#   SITE=zw source scripts/env/ceph_site.sh
# Exports: SITE, CEPH_ROOT, BOYE_ROOT, FLOWBP_HPS_CLIP_PATH, FLOWBP_HPS_PT_PATH

SITE="${SITE:-gz}"
SITE_LC="$(printf '%s' "$SITE" | tr '[:upper:]' '[:lower:]')"

case "$SITE_LC" in
  gz|guangzhou|fsgm3)
    CEPH_ROOT="/apdcephfs_fsgm3/share_305110755"
    SITE="gz"
    ;;
  zw|local|zwfy8)
    CEPH_ROOT="/apdcephfs_zwfy8/share_305110755"
    SITE="zw"
    ;;
  *)
    echo "[ceph_site] unknown SITE='$SITE' (expected: gz | zw | local)" >&2
    exit 1
    ;;
esac

BOYE_ROOT="${BOYE_ROOT:-${CEPH_ROOT}/hunyuan/boye}"

# HPS / OpenCLIP: prefer site-local Ceph copies.
_hps_clip_candidates=(
  "${BOYE_ROOT}/datasets/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"
  "${BOYE_ROOT}/workspace/FlowBP/hps_ckpt/open_clip_pytorch_model.bin"
  "${BOYE_ROOT}/hps_ckpt/open_clip_pytorch_model.bin"
)
_hps_pt_candidates=(
  "${BOYE_ROOT}/datasets/HPS_v2.1_compressed.pt"
  "${BOYE_ROOT}/workspace/FlowBP/hps_ckpt/HPS_v2.1_compressed.pt"
  "${BOYE_ROOT}/hps_ckpt/HPS_v2.1_compressed.pt"
)

_pick_existing() {
  local p
  for p in "$@"; do
    if [[ -f "$p" ]]; then
      printf '%s' "$p"
      return 0
    fi
  done
  # Fall back to first candidate (caller / HF helper may still recover).
  printf '%s' "$1"
  return 1
}

if [[ -z "${FLOWBP_HPS_CLIP_PATH:-}" ]]; then
  FLOWBP_HPS_CLIP_PATH="$(_pick_existing "${_hps_clip_candidates[@]}")"
fi
if [[ -z "${FLOWBP_HPS_PT_PATH:-}" ]]; then
  FLOWBP_HPS_PT_PATH="$(_pick_existing "${_hps_pt_candidates[@]}")"
fi

export SITE CEPH_ROOT BOYE_ROOT FLOWBP_HPS_CLIP_PATH FLOWBP_HPS_PT_PATH
echo "[ceph_site] SITE=${SITE} CEPH_ROOT=${CEPH_ROOT}" >&2
echo "[ceph_site] HPS_CLIP=${FLOWBP_HPS_CLIP_PATH}" >&2
echo "[ceph_site] HPS_PT=${FLOWBP_HPS_PT_PATH}" >&2

unset _hps_clip_candidates _hps_pt_candidates
unset -f _pick_existing 2>/dev/null || true
