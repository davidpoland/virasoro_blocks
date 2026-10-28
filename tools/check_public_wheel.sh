#!/usr/bin/env bash
#
# Build and exercise the installable virasoro_blocks wheel without importing
# the repository's source tree. Check both public packages and ensure importing
# the block engine does not import bootstrap or model code.
set -euo pipefail

ROOT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

if [[ -n "${VB_PYTHON:-}" ]]; then
  python_bin="$VB_PYTHON"
elif command -v python3.12 >/dev/null 2>&1; then
  python_bin="$(command -v python3.12)"
else
  python_bin="$(command -v python3)"
fi

if ! python_path="$(command -v "$python_bin")"; then
  echo "Python interpreter is unavailable: $python_bin" >&2
  exit 1
fi
python_bin="$python_path"
if [[ "$python_bin" == */* && "$python_bin" != /* ]]; then
  python_bin="$PWD/$python_bin"
fi
cd "$ROOT_DIR"

build_root="$(mktemp -d)"
venv_root="$(mktemp -d)"
trap 'rm -r "$build_root" "$venv_root"' EXIT

"$python_bin" -m pip wheel . --no-deps --wheel-dir "$build_root"
wheel_path="$(find "$build_root" -maxdepth 1 -type f -name '*.whl' -print -quit)"
if [[ -z "$wheel_path" ]]; then
  echo "wheel build produced no .whl file" >&2
  exit 1
fi

# Import isolation alone does not prove release isolation: an unused package can
# still be present in the archive. Inspect what will be shipped: only the two
# release packages and the wheel's own metadata.
"$python_bin" - "$wheel_path" <<'PYCONTENTS'
import sys
from pathlib import PurePosixPath
from zipfile import ZipFile

released = {"virasoro_blocks", "virasoro_bootstrap"}


def top_level(name):
    parts = PurePosixPath(name).parts
    if parts[0].endswith(".dist-info"):
        return None
    if parts[0].endswith(".data"):
        return parts[2] if len(parts) > 2 and parts[1] in ("purelib", "platlib") else None
    return parts[0]


with ZipFile(sys.argv[1]) as wheel:
    extra = sorted(name for name in wheel.namelist()
                   if top_level(name) is not None and top_level(name) not in released)
if extra:
    raise SystemExit("packages other than virasoro_blocks and virasoro_bootstrap in the wheel: "
                     + ", ".join(extra[:5]) + "\nBuild and check the wheel from the exported release tree.")
print("wheel archive contains only virasoro_blocks and virasoro_bootstrap")
PYCONTENTS

# The numerical dependencies come from the calling interpreter, offline: the system site packages, and
# the caller's own site-packages (a virtualenv's are not "system" ones) appended after the new venv's,
# so the wheel installed here still shadows any other copy of the project (checked below).
"$python_bin" -m venv --system-site-packages "$venv_root"
"$venv_root/bin/python" -m pip install --force-reinstall --no-deps --no-index "$wheel_path"
"$python_bin" -c 'import sysconfig; p = sysconfig.get_paths(); print(p["purelib"]); print(p["platlib"])' \
  > "$("$venv_root/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/zz_caller_site_packages.pth"

cd /tmp
# An inherited source-tree PYTHONPATH must not mask an incomplete wheel.
unset PYTHONPATH
"$venv_root/bin/virasoro_blocks" info --family ramond --format json \
  > "$venv_root/ramond-info.json"
"$venv_root/bin/python" -c '
import json
import sys
from pathlib import Path

document = json.loads(Path(sys.argv[1]).read_text())
assert document["family"] == "ramond"
assert document["operations"]
import virasoro_blocks
assert Path(virasoro_blocks.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert "virasoro_bootstrap" not in sys.modules
import virasoro_bootstrap
assert Path(virasoro_bootstrap.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
from virasoro_bootstrap.defaults import load_default_params
assert load_default_params()
' "$venv_root/ramond-info.json"

echo "public wheel and console entry point: ok"
