#!/usr/bin/env bash
#
# Regenerate the committed examples/block_samples/ reference set.
#
# One subdirectory per block basis (the Virasoro, superconformal, torus, and
# global examples). The exact commands live only here, and each written table
# also records its own command in `args`; examples/block_samples/README.md
# describes the samples and their shared settings. This script is the single
# source of truth, so the samples stay reproducible after block-pipeline
# changes (e.g. the pole-selection default).
#
# Usage:
#   tools/regen_block_samples.sh            # regenerate every basis
#   tools/regen_block_samples.sh scalar_G ordered_B   # only the named bases
#
# Shared settings (also listed in examples/block_samples/README.md): sphere
# Virasoro runs use c = 10, --lambda 4, --order 20, --npoles 6,
# --precision 60, --format both, and chiral norm N = 16 (library default).
# Global runs use the same interpolation settings, omit central charge, and use
# q-order only to define their finite SL(2) pole universe. An integer --npoles
# selects the order-reduced interpolation path.
set -euo pipefail

ROOT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT_DIR"

python_bin="$ROOT_DIR/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  if command -v python3.12 >/dev/null 2>&1; then
    python_bin="$(command -v python3.12)"
  elif command -v python3 >/dev/null 2>&1; then
    python_bin="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    python_bin="$(command -v python)"
  else
    echo "could not find a Python interpreter (looked for .venv/bin/python, python3.12, python3, python)" >&2
    exit 1
  fi
fi

vb() { "$python_bin" -m virasoro_blocks "$@"; }

# Common flags shared by every run.
COMMON=(--lambda 4 --order 20 --npoles 6 --precision 60 \
        --format both)
COMMON_GLOBAL=(--lambda 4 --order 20 --npoles 6 --precision 60 --format both)

# The large-spin tower is emitted as JSON only (no variable='L' Mathematica
# schema), so it uses the same settings minus the .m output. The separate
# settings block keeps the tower's output contract explicit.
COMMON_LS=(--lambda 4 --order 20 --npoles 6 --precision 60 --format json)
COMMON_GLOBAL_LS=(--lambda 4 --order 20 --npoles 6 --precision 60 --format json)

gen_scalar_G() {
  echo "== scalar_G =="
  vb table --family virasoro --basis scalar_G \
     --central-charge 10 --spin-ranges 0 1 2 \
     --delta-ext 0.15 0.15 0.15 0.15 \
     "${COMMON[@]}" -o examples/block_samples/scalar_G/
}

gen_ordered_B() {
  echo "== ordered_B =="
  vb table --family virasoro --basis ordered_B \
     --central-charge 10 --spin-ranges 0 1 2 \
     --delta-ext 0.15 1.25 0.35 1.45 --spin-ext 0 1 0 1 \
     --coordinates zzb \
     "${COMMON[@]}" -o examples/block_samples/ordered_B/
}

gen_parity_projected_B() {
  echo "== parity_projected_B =="
  vb table --family virasoro --basis parity_projected_B \
     --central-charge 10 --spin-ranges 0 1 2 \
     --delta-ext 0.15 1.25 0.35 1.45 --spin-ext 0 1 0 1 \
     --parity-ext 1 1 1 1 --parity-project + --parity-exchanged 1 \
     --coordinates zzb \
     "${COMMON[@]}" -o examples/block_samples/parity_projected_B/
}

gen_chiral_F() {
  echo "== chiral_F =="
  vb table --family virasoro --basis chiral_F \
     --central-charge 10 \
     --h-ext 0.05 0.10 0.15 0.20 --coordinates z \
     "${COMMON[@]}" -o examples/block_samples/chiral_F/
}

gen_ns_scalar_G() {
  echo "== ns_scalar_G =="
  # Compact JSON-only smoke sample for the selectable N=1 NS family.
  vb table --family ns --basis ns_scalar_G \
     --central-charge 1.3 --spin-ranges 0 \
     --lambda 2 --delta-ext 0.25 0.25 0.25 0.25 --coordinates zzb \
     --order 4 --npoles 2 --precision 40 --format json \
     -o examples/block_samples/ns_scalar_G/
}

gen_offdiagonal_scalar_G() {
  echo "== offdiagonal_scalar_G =="
  # Off-diagonal example: the block expanded around an asymmetric point
  # z = 0.4 != zbar = 0.55 (vs the default crossing-symmetric z = zbar = 1/2).
  # The m-axis derivatives are taken around z0 = 0.4, the n-axis around
  # zbar0 = 0.55; the JSON records z_center = 0.4 and zbar_center = 0.55.
  vb table --family virasoro --basis scalar_G \
     --central-charge 10 --spin-ranges 0 1 2 \
     --delta-ext 0.15 0.15 0.15 0.15 \
     --coordinates zzb \
     --z-center 0.4 --zbar-center 0.55 \
     "${COMMON[@]}" -o examples/block_samples/offdiagonal_scalar_G/
}

gen_global_scalar_G() {
  echo "== global_scalar_G =="
  vb table --family global --basis global_scalar_G \
     --spin-ranges 0 1 2 \
     --delta-ext 0.15 0.15 0.15 0.15 \
     --coordinates zzb \
     "${COMMON_GLOBAL[@]}" -o examples/block_samples/global_scalar_G/
}

gen_global_ordered_B() {
  echo "== global_ordered_B =="
  vb table --family global --basis global_ordered_B \
     --spin-ranges 0 1 2 \
     --delta-ext 0.15 1.25 0.35 1.45 --spin-ext 0 1 0 1 \
     --coordinates zzb \
     "${COMMON_GLOBAL[@]}" -o examples/block_samples/global_ordered_B/
}

gen_global_parity_projected_B() {
  echo "== global_parity_projected_B =="
  vb table --family global --basis global_parity_projected_B \
     --spin-ranges 0 1 2 \
     --delta-ext 0.15 1.25 0.35 1.45 --spin-ext 0 1 0 1 \
     --parity-ext 1 1 1 1 --parity-project + --parity-exchanged 1 \
     --coordinates zzb \
     "${COMMON_GLOBAL[@]}" -o examples/block_samples/global_parity_projected_B/
}

gen_global_chiral_F() {
  echo "== global_chiral_F =="
  vb table --family global --basis global_chiral_F \
     --h-ext 0.05 0.10 0.15 0.20 --coordinates z \
     "${COMMON_GLOBAL[@]}" -o examples/block_samples/global_chiral_F/
}

# Large-spin asymptotic positivity towers (variable="L", one file per twist
# slice). Scalar externals only; tower is emitted as JSON. The leading three
# double-twist slices tau_n = 2*Delta_ext + 2n cover the tail L >= lmin.
gen_large_spin_scalar_G() {
  echo "== large_spin_scalar_G =="
  vb table --family virasoro --basis scalar_G \
     --central-charge 10 --delta-ext 0.15 0.15 0.15 0.15 \
     --large-spin-lmin 8 --large-spin-double-twists 3 --coordinates zzb \
     "${COMMON_LS[@]}" -o examples/block_samples/large_spin_scalar_G/
}

gen_global_large_spin_scalar_G() {
  echo "== global_large_spin_scalar_G =="
  vb table --family global --basis scalar_G \
     --delta-ext 0.15 0.15 0.15 0.15 \
     --large-spin-lmin 8 --large-spin-double-twists 3 --coordinates zzb yyb \
     "${COMMON_GLOBAL_LS[@]}" -o examples/block_samples/global_large_spin_scalar_G/
}

gen_torus_G() {
  echo "== torus_G =="
  # Exact torus reduced-character smoke sample. Torus output is JSON-only and
  # has no sphere q-order or interpolation-pole settings.
  vb table --surface torus --family torus-character \
     --central-charge 10 --spin-ranges 0 \
     --lambda 2 --precision 40 --format json \
     -o examples/block_samples/torus_G/
}

gen_ramond_scalar_G() {
  echo "== ramond_scalar_G =="
  # Compact JSON-only smoke sample for the N=1 Ramond family (default branch ++).
  vb table --family ramond --basis ramond_scalar_G \
     --central-charge 1.3 --spin-ranges 0 \
     --lambda 2 --delta-ext 0.25 0.25 0.25 0.25 --coordinates zzb \
     --order 4 --npoles 2 --precision 40 --format json \
     -o examples/block_samples/ramond_scalar_G/
}

gen_torus_onept_G() {
  echo "== torus_onept_G =="
  # Torus one-point tables for a scalar external of chiral weight 1/2. --npoles
  # is omitted, so the pole count is chosen (recorded as npoles_chosen).
  vb table --surface torus --family torus-onepoint \
     --central-charge 10 --torus-onept-ext 0.5 --spin-ranges 0 2 \
     --lambda 2 --order 10 --precision 40 --format json \
     -o examples/block_samples/torus_onept_G/
}

gen_point_table() {
  echo "== point_table =="
  # Point tables: every family at a single exchanged weight. The torus
  # character point tables are irreducible characters, chosen from the
  # quantum numbers: the vacuum, the current pair (1, 0) + (0, 1), and a Kac
  # weight at a singular c (the Ising spin field at c = 1/2).
  local out=examples/block_samples/point_table
  vb point-table --family virasoro --basis scalar_G \
     --central-charge 10 --delta-ext 0.15 0.15 0.15 0.15 \
     --exchange-delta 2.5 --exchange-spin 2 \
     --lambda 4 --order 20 --precision 60 --format both \
     --output-template scalar_G_point.json -o "$out"
  vb point-table --surface torus --family torus-character \
     --central-charge 10 --exchange-delta 0 --exchange-spin 0 \
     --lambda 2 --precision 40 --output-template torus_vacuum.json -o "$out"
  vb point-table --surface torus --family torus-character \
     --central-charge 10 --exchange-delta 1 --exchange-spin 1 \
     --lambda 2 --precision 40 --output-template torus_current.json -o "$out"
  vb point-table --surface torus --family torus-character \
     --central-charge 1/2 --exchange-delta 1/8 --exchange-spin 0 \
     --lambda 2 --precision 40 --output-template torus_ising_sigma.json -o "$out"
  vb point-table --surface torus --family torus-onepoint \
     --central-charge 10 --torus-onept-ext 0.5 --exchange-delta 2.5 --exchange-spin 0 \
     --lambda 2 --order 10 --precision 40 --output-template torus_onept_point.json -o "$out"
}

ALL=(scalar_G ordered_B parity_projected_B chiral_F offdiagonal_scalar_G \
     ns_scalar_G ramond_scalar_G global_scalar_G global_ordered_B \
     global_parity_projected_B global_chiral_F large_spin_scalar_G \
     global_large_spin_scalar_G torus_G torus_onept_G point_table)

targets=("$@")
if [[ ${#targets[@]} -eq 0 ]]; then
  targets=("${ALL[@]}")
fi

for name in "${targets[@]}"; do
  valid=0
  for candidate in "${ALL[@]}"; do
    if [[ "$name" == "$candidate" ]]; then
      valid=1
      break
    fi
  done
  if [[ "$valid" -eq 0 ]]; then
    echo "unknown basis '$name'; valid: ${ALL[*]}" >&2
    exit 2
  fi
done

for name in "${targets[@]}"; do
  case "$name" in
    scalar_G)                   gen_scalar_G ;;
    ordered_B)                  gen_ordered_B ;;
    parity_projected_B)         gen_parity_projected_B ;;
    chiral_F)                   gen_chiral_F ;;
    ns_scalar_G)                gen_ns_scalar_G ;;
    offdiagonal_scalar_G)       gen_offdiagonal_scalar_G ;;
    global_scalar_G)            gen_global_scalar_G ;;
    global_ordered_B)           gen_global_ordered_B ;;
    global_parity_projected_B)  gen_global_parity_projected_B ;;
    global_chiral_F)            gen_global_chiral_F ;;
    large_spin_scalar_G)        gen_large_spin_scalar_G ;;
    global_large_spin_scalar_G) gen_global_large_spin_scalar_G ;;
    torus_G)                    gen_torus_G ;;
    ramond_scalar_G)            gen_ramond_scalar_G ;;
    torus_onept_G)              gen_torus_onept_G ;;
    point_table)                gen_point_table ;;
    *)
      echo "unknown basis '$name'; valid: ${ALL[*]}" >&2
      exit 2
      ;;
  esac
done

"$python_bin" - "${targets[@]}" <<'PY'
import json
import sys
from pathlib import Path

files = []
for target in dict.fromkeys(sys.argv[1:]):
    selected = sorted((Path("examples/block_samples") / target).rglob("*.json"))
    if not selected:
        raise SystemExit(f"{target}: artifact validation found no JSON block samples")
    files.extend(selected)
for path in files:
    document = json.loads(path.read_text())
    invocation = str(document.get("args", ""))
    if not invocation.startswith(("virasoro_blocks table ", "virasoro_blocks point-table ")):
        raise SystemExit(f"{path}: artifact was not generated by `table` or `point-table`")
    for legacy in ("--method", "__main__.py", "--large-spin-mode"):
        if legacy in invocation:
            raise SystemExit(f"{path}: legacy CLI spelling in args: {legacy}")
print(f"validated {len(files)} canonical JSON artifacts")
PY

echo "=== block samples regenerated: ${targets[*]} ==="
