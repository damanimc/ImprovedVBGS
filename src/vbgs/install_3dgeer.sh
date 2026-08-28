#!/bin/bash
# Install Bosch 3DGEER geer-rasterizer as the CUDA rasterizer used at eval time.
# Replaces graphdeco diff-gaussian-rasterization in the active Python env.
# See README "Eval with 3DGEER".
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
THIRD_PARTY_DIR="${VBGS_THIRD_PARTY_DIR:-${REPO_ROOT}/third_party}"
GEER_DIR="${VBGS_3DGEER_DIR:-${THIRD_PARTY_DIR}/3dgeer}"
GEER_RASTER="${GEER_DIR}/submodules/geer-rasterizer"

mkdir -p "${THIRD_PARTY_DIR}"
if [ ! -d "${GEER_DIR}/.git" ]; then
  git clone --recursive https://github.com/boschresearch/3dgeer.git "${GEER_DIR}"
fi

if [ ! -d "${GEER_RASTER}" ]; then
  echo "Missing ${GEER_RASTER}. Re-clone with --recursive." >&2
  exit 1
fi

pip install ninja
pip install --no-build-isolation "${GEER_RASTER}"

python - <<'PY'
from diff_gaussian_rasterization import GaussianRasterizationSettings as S
fields = set(S.__annotations__)
need = {"tan_theta", "render_mode", "focal_x"}
missing = need - fields
if missing:
    raise SystemExit(f"geer-rasterizer install looks wrong; missing {missing}")
print("3DGEER geer-rasterizer OK:", sorted(need))
PY

echo "Installed. Use: python scripts/eval_3dgeer.py <model.json> --data-path ... --save-images"
echo "To restore graphdeco rasterizer:"
echo "  cd ${THIRD_PARTY_DIR}/gaussian-splatting/submodules/diff-gaussian-rasterization && python setup.py install"
