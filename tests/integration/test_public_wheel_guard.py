"""The release checker inspects wheel contents before installing the artifact."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest


@pytest.mark.skipif(shutil.which("bash") is None, reason="release helper requires bash")
@pytest.mark.parametrize("member,relative", [("otherpkg/__init__.py", False),
                                             ("project.data/purelib/otherpkg.py", "path")])
def test_wheel_with_another_package_is_rejected_before_installation(tmp_path, member, relative):
    # Substitute only wheel production; execute the checker's real Python
    # archive inspection. Any attempted venv/install is an unexpected step.
    python = tmp_path / "wheel_producer"
    # the fake interpreter is a /bin/sh wrapper around a Python body: a #! line cannot name an interpreter
    # whose path contains spaces (a virtualenv under a directory with spaces)
    body = python.with_name(python.name + '_body.py')
    body.write_text(f'''import os
from pathlib import Path
import sys
from zipfile import ZipFile
args = sys.argv[1:]
if args[:3] == ["-m", "pip", "wheel"]:
    directory = Path(args[args.index("--wheel-dir") + 1])
    with ZipFile(directory / "private.whl", "w") as wheel:
        wheel.writestr({member!r}, "")
elif args[:1] == ["-"]:
    os.execv(sys.executable, [sys.executable, *args])
else:
    raise SystemExit("unexpected installation step")
''')
    python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(body))} "$@"\n')
    python.chmod(0o755)
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(["bash", str(root / "tools/check_public_wheel.sh")],
                            env={**os.environ,
                                 "VB_PYTHON": (python.name if relative == "path" else
                                               "./wheel_producer" if relative else str(python)),
                                 "PATH": ("." + os.pathsep + os.environ["PATH"]
                                          if relative == "path" else os.environ["PATH"])},
                            cwd=tmp_path if relative else None,
                            text=True, capture_output=True)
    assert result.returncode != 0
    assert "packages other than virasoro_blocks and virasoro_bootstrap in the wheel" in result.stderr
    assert member in result.stderr
    assert "unexpected installation step" not in result.stderr
