#!/usr/bin/env bash
# Super-resolution experiment:
#   rendered novel views of each method (IN_DIR, e.g. qualitative/<dataset>-<method>.png) are
#   downscaled by 1/n (n = 2, 3, 4, 8) and restored with PFT (x2, x3, x4; x4 for n = 8),
#   then compared with the original views (PSNR / SSIM).
#
#   bash scripts/sr_experiment.sh <IN_DIR> [OUT_ROOT]
set -e
cd "$(dirname "$0")/.."
IN_DIR=${1:?usage: bash scripts/sr_experiment.sh <IN_DIR> [OUT_ROOT]}
OUT_ROOT=${2:-results/sr_experiment}
IN_DIR=$(realpath "$IN_DIR")
mkdir -p "$OUT_ROOT"
NAME=$(basename "$IN_DIR")

for n in 2 3 4 8; do
    python sr/downscale.py "$IN_DIR" $n
    rm -rf "$OUT_ROOT/${NAME}_x$n"
    mv "$(dirname "$IN_DIR")/${NAME}_x$n" "$OUT_ROOT/"
    s=$n; [ $n -gt 4 ] && s=4
    python sr/super_resolve.py -i "$OUT_ROOT/${NAME}_x$n" -o "$OUT_ROOT/result_x$n" --scale $s --task classical
    echo "== 1/$n -> PFT x$s"
    python sr/eval_sr.py --gt_dir "$IN_DIR" --sr_dir "$OUT_ROOT/result_x$n"
done
