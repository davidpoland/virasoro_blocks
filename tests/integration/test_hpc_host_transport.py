"""--host laptop-to-cluster transport: git guard, ssh/scp argv, remote parsing.

Git-guard tests use a real throwaway git repo (fast, local, no network). The
ssh/scp calls themselves are never real: a fake subprocess.run intercepts
argv[0] in {"ssh", "scp"} and returns canned output, passing every other
command (git, etc.) through to the real subprocess.run.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from virasoro_bootstrap.hpc import collect, submit

# the remote transport works on git checkouts (commit pinning, clean-and-pushed checks)
pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

_REAL_RUN = subprocess.run


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def make_repo(tmp_path: Path, *, pushed: bool = True) -> Path:
    repo = tmp_path / "repo"
    (repo / "sweeps").mkdir(parents=True)
    (repo / "clusters").mkdir()
    (repo / "sweeps" / "s.toml").write_text(
        'name = "s"\ntask = "scalar_gap_bisection"\n[sweep]\nc = ["2"]\n'
    )
    (repo / "clusters" / "c.toml").write_text(
        'kind = "slurm"\n'
        'remote_repo = "$HOME/repo"\n'
        'runs_root = "$HOME/vb_runs"\n'
        'python = "$HOME/.conda/envs/vb/bin/python"\n'
    )
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "T")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    if pushed:
        bare = tmp_path / "origin.git"
        _git(repo, "init", "-q", "--bare", str(bare))
        _git(repo, "remote", "add", "origin", str(bare))
        _git(repo, "push", "-q", "-u", "origin", "HEAD")
    return repo


@pytest.fixture
def library_checkout(monkeypatch):
    """Pin what --host treats as the library checkout.

    submit._library_checkout() finds the checkout the package itself runs
    from, which under test is the real repository. Every --host test states
    which of the two modes it is exercising instead of inheriting that.
    """
    def _set(repo):
        monkeypatch.setattr(submit, "_library_checkout", lambda: repo)
    return _set


@pytest.fixture
def fake_ssh_scp(monkeypatch):
    """Intercept ssh/scp; pass every other subprocess.run call through for real."""
    calls: list[list[str]] = []
    responses: list[subprocess.CompletedProcess] = []

    def fake(cmd, *args, **kwargs):
        if cmd[0] in ("ssh", "scp"):
            calls.append(cmd)
            if responses:
                return responses.pop(0)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return _REAL_RUN(cmd, *args, **kwargs)

    monkeypatch.setattr(submit.subprocess, "run", fake)
    monkeypatch.setattr(collect.subprocess, "run", fake)
    return calls, responses


def _queue(fake_ssh_scp, *procs: subprocess.CompletedProcess) -> None:
    _, responses = fake_ssh_scp
    responses.extend(procs)


def cp(stdout: str = "", stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)


# --- git guard -----------------------------------------------------------------

def test_require_clean_and_pushed_ok(tmp_path):
    repo = make_repo(tmp_path)
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    assert submit._require_clean_and_pushed(repo) == head


def test_require_clean_and_pushed_dirty(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "sweeps" / "s.toml").write_text("dirty = true\n")
    with pytest.raises(SystemExit, match="uncommitted"):
        submit._require_clean_and_pushed(repo)


def test_require_clean_and_pushed_ignores_untracked_files(tmp_path):
    # Untracked scratch files do not affect the remote checkout of this commit.
    repo = make_repo(tmp_path)
    (repo / "scratch.py").write_text("# not tracked, not committed\n")
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    assert submit._require_clean_and_pushed(repo) == head


def test_require_clean_and_pushed_no_upstream(tmp_path):
    repo = make_repo(tmp_path, pushed=False)
    with pytest.raises(SystemExit, match="pushed upstream"):
        submit._require_clean_and_pushed(repo)


def test_require_clean_and_pushed_diverged(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "extra.txt").write_text("x")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "unpushed")
    with pytest.raises(SystemExit, match="git push"):
        submit._require_clean_and_pushed(repo)


def test_inputs_outside_any_repo_are_accepted(tmp_path, fake_ssh_scp, library_checkout, monkeypatch):
    """Accept sweep and cluster files outside a checkout and record their contents."""
    library_checkout(None)
    loose = tmp_path / "not-a-repo"
    loose.mkdir()
    (loose / "s.toml").write_text(
        'name = "s"\ntask = "scalar_gap_bisection"\n[sweep]\nc = ["2"]\n'
    )
    (loose / "c.toml").write_text('kind = "slurm"\nruns_root = "/scratch/runs"\n')
    _queue(
        fake_ssh_scp,
        cp(stdout="/scratch/runs"),                                     # expand (printf has no newline)
        cp(),                                                            # mkdir
        cp(), cp(),                                                      # scp x2
        cp(stdout="run x: 1 task(s) under /scratch/runs/x\n"),           # submit
        cp(),                                                            # manifest
    )
    submit.submit_remote(
        loose / "s.toml", loose / "c.toml",
        host="h", runs_root=tmp_path / "local", run_id="x",
    )
    calls, _ = fake_ssh_scp
    scps = [c for c in calls if c[0] == "scp"]
    assert [Path(c[2]).name for c in scps[:2]] == ["s.toml", "c.toml"]
    assert scps[0][3] == "h:/scratch/runs/x/inputs/sweep/s.toml"
    (loose / 'sweep').mkdir()
    (loose / 'profile').mkdir()
    same_sweep = loose / 'sweep' / 'input.toml'
    same_profile = loose / 'profile' / 'input.toml'
    same_sweep.write_text((loose / 's.toml').read_text())
    same_profile.write_text((loose / 'c.toml').read_text())
    calls.clear()
    _queue(fake_ssh_scp, cp(stdout='/scratch/runs'), cp(), cp(), cp(),
           cp(stdout='run y: 1 task(s) under /scratch/runs/y\n'), cp())
    submit.submit_remote(same_sweep, same_profile, host='h',
                         runs_root=tmp_path / 'local', run_id='y')
    uploads = [c for c in calls if c[0] == 'scp'][:2]
    assert uploads[0][3] != uploads[1][3], 'sweep and profile overwrote one another'
    # The sweep filename supplies the default job name when name is omitted.
    assert Path(uploads[0][3]).name == same_sweep.name
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('VB_TEST_MIRROR_ROOT', str(tmp_path / 'expanded local'))
    calls.clear()
    _queue(fake_ssh_scp, cp(stdout='/scratch/runs'), cp(), cp(), cp(),
           cp(stdout='run z: 1 task(s) under /scratch/runs/z\n'), cp())
    mirror = submit.submit_remote(same_sweep, same_profile, host='h',
                                  runs_root=Path('$VB_TEST_MIRROR_ROOT'), run_id='z')
    assert mirror == (tmp_path / 'expanded local/z').resolve()
    assert calls[-1][-1] == str(mirror / 'manifest.json')
    calls.clear()
    for bad_root in ('', '  ', '~__vb_missing_home_user_20261006__/runs'):
        with pytest.raises(ValueError, match='runs_root'):
            submit.submit_remote(same_sweep, same_profile, host='h', runs_root=bad_root)
        assert not calls, 'invalid local mirror root reached SSH'


def test_remote_staging_refuses_existing_run_before_copying_inputs(tmp_path, monkeypatch, library_checkout):
    library_checkout(None)
    root = tmp_path / 'remote runs'
    previous = root / 'same' / 'inputs'
    previous.mkdir(parents=True)
    original = previous / 'sweep.toml'
    original.write_text('original submitted input')
    sweep = tmp_path / 'sweep.toml'
    sweep.write_text('task = "scalar_gap_bisection"\n[params]\nc = "2"\n')
    profile = tmp_path / 'profile.toml'
    profile.write_text(f'runs_root = {json.dumps(str(root))}\n')
    uploads = []
    def transport(cmd, *args, **kwargs):
        if cmd[0] == 'ssh':
            return _REAL_RUN(['bash', '-c', cmd[2]], capture_output=True, text=True)
        if cmd[0] == 'scp':
            uploads.append(cmd)
            return cp(returncode=1, stderr='unexpected input copy')
        return _REAL_RUN(cmd, *args, **kwargs)
    monkeypatch.setattr(submit.subprocess, 'run', transport)
    with pytest.raises(SystemExit):
        submit.submit_remote(sweep, profile, host='example', run_id='same', dry_run=True)
    assert uploads == []
    assert original.read_text() == 'original submitted input'
    for run_id in ('', '.', '..', '../outside', '/absolute/run', 'nested/run'):
        with pytest.raises(ValueError, match='run_id'):
            submit.submit_remote(sweep, profile, host='example', run_id=run_id, dry_run=True)
        with pytest.raises(ValueError, match='run_id'):
            submit.create_run(sweep, profile, runs_root=root, run_id=run_id, dry_run=True)
    assert sorted(root.iterdir()) == [previous.parent]
    repo = tmp_path / 'checkout'
    repo.mkdir()
    profile.write_text(f'remote_repo = {json.dumps(str(repo))}\nruns_root = "relative runs"\n')
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):  # The transport intentionally refuses input copies.
        submit.submit_remote(sweep, profile, host='example', run_id='fresh', dry_run=True)
    assert (repo / 'relative runs' / 'fresh' / 'inputs').is_dir()
    assert not (tmp_path / 'relative runs').exists()


# --- _ssh_run argv shape ---------------------------------------------------------

def test_ssh_run_survives_ssh_argv_rejoining(monkeypatch, tmp_path):
    """SSH argument rejoining preserves the command and working directory.

    Simulate the remote shell parsing SSH's space-joined argument string.
    A directory containing spaces checks that the payload stays quoted.
    """
    target = tmp_path / 'remote "repo"\\ path'
    target.mkdir()

    calls: list[list[str]] = []
    monkeypatch.setattr(
        submit.subprocess, "run",
        lambda cmd, *a, **kw: (calls.append(cmd), cp())[1],
    )
    submit._ssh_run("mycluster", str(target), "pwd")

    assert len(calls) == 1
    ssh_argv = calls[0]
    assert ssh_argv[:2] == ["ssh", "mycluster"]
    assert len(ssh_argv) == 3, "must be one argv element or ssh will rejoin+corrupt it"

    # What the remote's sshd-invoked shell actually receives and parses.
    result = _REAL_RUN(["bash", "-c", ssh_argv[2]], capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout.strip() == str(target)
    submit._ssh_run('mycluster', str(tmp_path / 'missing'), "printf 'wrong directory'; pwd")
    result = _REAL_RUN(['bash', '-c', calls[-1][2]], capture_output=True, text=True)
    assert result.returncode != 0 and result.stdout == ''


def test_remote_inspection_reaches_profile_interpreter(tmp_path, monkeypatch):
    target = tmp_path / 'checkout "quoted"\\ path'
    target.mkdir()
    interpreter = tmp_path / 'environment with spaces' / 'py"thon'
    interpreter.parent.mkdir()
    interpreter.write_text('#!/bin/sh\npwd\nprintf "<%s>\\n" "$@"\n')
    interpreter.chmod(0o755)
    monkeypatch.setenv('VB_REMOTE_TEST_PYTHON', str(interpreter))
    calls = []
    monkeypatch.setattr(submit.subprocess, 'run',
                        lambda cmd, *args, **kw: (calls.append(cmd), cp())[1])
    profile = tmp_path / 'profile.toml'
    cases = ((str(target), str(interpreter)),
             ('~/' + os.path.relpath(os.path.realpath(target), os.path.realpath(Path.home())),
              '~/' + os.path.relpath(os.path.realpath(interpreter), os.path.realpath(Path.home()))),
             (None, '$VB_REMOTE_TEST_PYTHON'))
    for repo, python in cases:
        text = f'python = {json.dumps(python)}\n'
        if repo is not None:
            text += f'remote_repo = {json.dumps(repo)}\n'
        profile.write_text(text)
        collect._ssh_module_call('example', profile, 'status "run with spaces"')
        executed = _REAL_RUN(['bash', '-c', calls[-1][2]], capture_output=True, text=True)
        assert executed.returncode == 0, executed.stderr
        lines = executed.stdout.splitlines()
        if repo is not None:
            assert lines[0] == str(target)
        assert lines[1:] == ['<-m>', '<virasoro_bootstrap.hpc.collect>',
                             '<status>', '<run with spaces>']
    # printf emits no terminator: preserve literal quotes and path whitespace.
    def execute(cmd, *args, **kw):
        return _REAL_RUN(['bash', '-c', cmd[2]], capture_output=True, text=True)
    with monkeypatch.context() as patch:
        patch.setattr(submit.subprocess, 'run', execute)
        literal = str(target) + ' '
        assert submit._remote_expand('h', literal) == literal
        monkeypatch.setenv('VB_REMOTE_TEST_EMPTY', '')
        with pytest.raises(SystemExit, match='could not expand'):
            submit._remote_expand('h', '$VB_REMOTE_TEST_EMPTY')


# --- submit_remote / retry_remote -----------------------------------------------

def test_submit_remote_happy_path(tmp_path, fake_ssh_scp, library_checkout):
    repo = make_repo(tmp_path)
    library_checkout(repo)
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    _queue(
        fake_ssh_scp,
        cp(stdout="/home/x/vb_runs"),                                    # expand runs_root
        cp(),                                                            # mkdir inputs
        cp(), cp(),                                                      # scp sweep, cluster
        cp(stdout="run testrun: 1 task(s) under /home/x/vb_runs/testrun\n"
                  "submitted SLURM job 999\n"),                          # submit
        cp(),                                                            # manifest back
    )
    local_runs = tmp_path / "local_runs"
    result = submit.submit_remote(
        repo / "sweeps" / "s.toml", repo / "clusters" / "c.toml",
        host="mycluster", runs_root=local_runs, run_id="testrun",
    )
    calls, _ = fake_ssh_scp
    ssh_calls = [c for c in calls if c[0] == "ssh"]
    scp_calls = [c for c in calls if c[0] == "scp"]

    # Inputs are staged under the run directory, not named repo-relatively.
    assert scp_calls[0][2:] == [
        str(repo / "sweeps" / "s.toml"), "mycluster:/home/x/vb_runs/testrun/inputs/sweep/s.toml"
    ]
    assert scp_calls[1][2:] == [
        str(repo / "clusters" / "c.toml"),
        "mycluster:/home/x/vb_runs/testrun/inputs/cluster/c.toml",
    ]

    submit_call = ssh_calls[-1]
    # Exactly 3 argv elements to ssh (ssh, host, ONE command string) -- more
    # elements would make ssh rejoin them with spaces and the remote reparse,
    # splitting bash -lc's argument at the first space (see _ssh_run).
    assert len(submit_call) == 3
    assert submit_call[:2] == ["ssh", "mycluster"]
    full = submit_call[2]
    assert full.startswith("bash -lc ")
    assert 'cd -- "$HOME/repo"' in full
    # The CODE is still pinned to the pushed commit; only the inputs moved.
    assert "git fetch origin" in full
    assert f"git checkout {head}" in full
    assert "/home/x/vb_runs/testrun/inputs/sweep/s.toml" in full
    assert "--cluster /home/x/vb_runs/testrun/inputs/cluster/c.toml" in full
    assert "--run-id testrun" in full
    assert '"$HOME/.conda/envs/vb/bin/python"' in full

    assert scp_calls[-1] == [
        "scp", "-q", "mycluster:/home/x/vb_runs/testrun/manifest.json",
        str((local_runs / "testrun").resolve() / "manifest.json"),
    ]
    assert result == (local_runs / "testrun").resolve()


def test_submit_remote_without_remote_repo_pins_no_commit(
    tmp_path, fake_ssh_scp, library_checkout
):
    """A pip-installed library on the cluster is a supported arrangement.

    There is no checkout to pin, so --host must not try to fetch or check out
    anything -- and must not demand a clean, pushed local tree either.
    """
    repo = make_repo(tmp_path)
    library_checkout(repo)
    interpreter = tmp_path / 'environment with spaces' / 'python'
    interpreter.parent.mkdir()
    interpreter.write_text("#!/bin/sh\nprintf 'remote interpreter reached\\n'\nexit 1\n")
    interpreter.chmod(0o755)
    home_relative = '~/' + os.path.relpath(os.path.realpath(interpreter), os.path.realpath(Path.home()))
    (repo / "clusters" / "c.toml").write_text(
        'kind = "slurm"\nruns_root = "/scratch/runs"\n' + f'python = "{home_relative}"\n'
    )
    # Deliberately left dirty and unpushed: with no remote_repo there is
    # nothing for the remote to fetch, so the guard must not fire.
    _queue(
        fake_ssh_scp,
        cp(stdout="/scratch/runs"), cp(), cp(), cp(),
        cp(stdout="run x: 1 task(s) under /scratch/runs/x\n"),
        cp(),
    )
    submit.submit_remote(
        repo / "sweeps" / "s.toml", repo / "clusters" / "c.toml",
        host="mycluster", runs_root=tmp_path / "local", run_id="x",
    )
    calls, _ = fake_ssh_scp
    full = [c for c in calls if c[0] == "ssh"][-1][2]
    assert "git checkout" not in full
    assert "git fetch" not in full
    assert "-m virasoro_bootstrap.hpc.submit" in full
    executed = _REAL_RUN(['bash', '-c', full], capture_output=True, text=True)
    assert executed.returncode == 1 and executed.stdout.strip() == 'remote interpreter reached', executed.stderr


def test_submit_remote_rejects_local_kind(tmp_path, fake_ssh_scp):
    repo = make_repo(tmp_path)
    (repo / "clusters" / "c.toml").write_text('kind = "local"\nremote_repo = "$HOME/repo"\n')
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "local kind")
    with pytest.raises(SystemExit, match="in-process"):
        submit.submit_remote(
            repo / "sweeps" / "s.toml", repo / "clusters" / "c.toml", host="mycluster"
        )


def test_submit_remote_requires_runs_root(tmp_path, fake_ssh_scp, library_checkout):
    """runs_root is what --host stages the inputs under, so it is required."""
    repo = make_repo(tmp_path)
    library_checkout(None)
    (repo / "clusters" / "c.toml").write_text('kind = "slurm"\n')
    with pytest.raises(SystemExit, match="runs_root"):
        submit.submit_remote(
            repo / "sweeps" / "s.toml", repo / "clusters" / "c.toml", host="mycluster"
        )


def test_retry_remote(tmp_path, fake_ssh_scp):
    repo = make_repo(tmp_path)
    interpreter = tmp_path / 'environment with spaces' / 'python'
    interpreter.parent.mkdir()
    interpreter.write_text("#!/bin/sh\nprintf 'retry interpreter reached\\n'\nexit 1\n")
    interpreter.chmod(0o755)
    home_relative = '~/' + os.path.relpath(os.path.realpath(interpreter), os.path.realpath(Path.home()))
    profile = repo / 'clusters' / 'c.toml'
    profile.write_text(f'remote_repo = "{repo}"\npython = "{home_relative}"\n')
    _queue(fake_ssh_scp, cp(stdout="retrying 1 task(s): [2]\n"))
    submit.retry_remote("vb_runs/testrun", repo / "clusters" / "c.toml", host="mycluster")
    calls, _ = fake_ssh_scp
    assert len(calls) == 1
    assert len(calls[0]) == 3
    full = calls[0][2]
    assert "git fetch" not in full  # retry never re-checks-out code
    assert "--retry vb_runs/testrun" in full
    executed = _REAL_RUN(['bash', '-c', full], capture_output=True, text=True)
    assert executed.returncode == 1 and executed.stdout.strip() == 'retry interpreter reached', executed.stderr
    profile.write_text(f'python = "{home_relative}"\n')
    _queue(fake_ssh_scp, cp())
    submit.retry_remote('/absolute/run', profile, host='mycluster', dry_run=True)
    full = calls[-1][2]
    assert 'cd ' not in full and '--retry /absolute/run --dry-run' in full
    executed = _REAL_RUN(['bash', '-c', full], capture_output=True, text=True)
    assert executed.returncode == 1 and executed.stdout.strip() == 'retry interpreter reached', executed.stderr


def test_main_host_wiring(tmp_path, fake_ssh_scp, monkeypatch, library_checkout):
    repo = make_repo(tmp_path)
    library_checkout(repo)
    _queue(
        fake_ssh_scp,
        cp(stdout="/home/x/vb_runs"), cp(), cp(), cp(),
        cp(stdout="run x: 1 task(s) under /home/x/vb_runs/x\n"),
        cp(),
    )
    monkeypatch.chdir(repo)
    rc = submit.main([
        "research/sweeps/s.toml", "--cluster", "research/cluster/c.toml", "--host", "mycluster", "--run-id", "x",
    ])
    assert rc == 0
    calls, _ = fake_ssh_scp
    assert [c[0] for c in calls] == ["ssh", "ssh", "scp", "scp", "ssh", "scp"]


def test_main_host_uses_vb_cluster_env_var(
    tmp_path, fake_ssh_scp, monkeypatch, library_checkout
):
    repo = make_repo(tmp_path)
    library_checkout(repo)
    _queue(
        fake_ssh_scp,
        cp(stdout="/home/x/vb_runs"), cp(), cp(), cp(),
        cp(stdout="run x: 1 task(s) under /home/x/vb_runs/x\n"),
        cp(),
    )
    monkeypatch.chdir(repo)
    monkeypatch.setenv("VB_CLUSTER", "research/cluster/c.toml")
    rc = submit.main(["research/sweeps/s.toml", "--host", "mycluster", "--run-id", "x"])  # no --cluster
    assert rc == 0
    calls, _ = fake_ssh_scp
    assert [c[0] for c in calls] == ["ssh", "ssh", "scp", "scp", "ssh", "scp"]


def test_main_host_requires_cluster(tmp_path, monkeypatch):
    monkeypatch.delenv("VB_CLUSTER", raising=False)  # ambient env must not mask this
    repo = make_repo(tmp_path)
    monkeypatch.chdir(repo)
    with pytest.raises(SystemExit):
        submit.main(["research/sweeps/s.toml", "--host", "mycluster"])


# --- collect.py --host ----------------------------------------------------------

def test_status_host_dispatch(tmp_path, fake_ssh_scp, capsys):
    repo = make_repo(tmp_path)
    _queue(fake_ssh_scp, cp(stdout="run x  task=t  n_tasks=1  slurm_job=1\n  done=1\n"))
    rc = collect.status_main([
        "vb_runs/x", "--host", "mycluster", "--cluster", str(repo / "clusters" / "c.toml"),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "n_tasks=1" in out
    calls, _ = fake_ssh_scp
    assert len(calls) == 1
    assert len(calls[0]) == 3
    full = calls[0][2]
    # --runs-root is always relayed too (defaults to the consolidated artifact
    # root here since the test profile sets none) -- the remote side needs it to
    # resolve a bare job id itself; harmless when run_dir is already a path.
    assert "collect status vb_runs/x --requeue-warn 5 --runs-root" in full
    assert "$HOME/vb_runs" in full
    assert 'cd -- "$HOME/repo"' in full


def test_status_host_dispatch_relays_bare_job_id_and_runs_root(tmp_path, fake_ssh_scp, capsys):
    repo = make_repo(tmp_path)
    (repo / "clusters" / "c.toml").write_text(
        'kind = "slurm"\nremote_repo = "$HOME/repo"\nruns_root = "$HOME/vb_runs"\n'
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "add runs_root")
    _queue(fake_ssh_scp, cp(stdout="run x  task=t  n_tasks=1  slurm_job=12345678\n  done=1\n"))

    rc = collect.status_main([
        "12345678", "--host", "mycluster", "--cluster", str(repo / "clusters" / "c.toml"),
    ])
    assert rc == 0
    assert "n_tasks=1" in capsys.readouterr().out
    calls, _ = fake_ssh_scp
    full = calls[0][2]
    assert "collect status 12345678 --requeue-warn 5" in full
    assert "--runs-root" in full  # exact quoting of $HOME/vb_runs covered elsewhere


@pytest.mark.parametrize("remote_path", [
    "/home/x/repo/vb_runs/x/results.json",
    "/home/x/run (2/2 task(s) complete)/results.json",
])
def test_collect_host_dispatch_success(tmp_path, fake_ssh_scp, capsys, monkeypatch, remote_path):
    repo = make_repo(tmp_path)
    _queue(
        fake_ssh_scp,
        cp(stdout=f"wrote {remote_path} (2/2 task(s) complete)\n"),
        cp(),  # scp
    )
    workdir = tmp_path / "cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    rc = collect.collect_main([
        "vb_runs/x", "--host", "mycluster", "--cluster", str(repo / "clusters" / "c.toml"),
    ])
    assert rc == 0
    calls, _ = fake_ssh_scp
    assert [c[0] for c in calls] == ["ssh", "scp"]
    scp_call = calls[1]
    assert scp_call == [
        "scp", "-q", f"mycluster:{remote_path}",
        "artifacts/results/x-results.json",
    ]
    assert "copied to artifacts/results/x-results.json" in capsys.readouterr().out


def test_collect_host_unparseable_output_reports_failure(tmp_path, fake_ssh_scp, capsys):
    repo = make_repo(tmp_path)
    _queue(fake_ssh_scp, cp(stdout="something unexpected\n"))
    rc = collect.collect_main([
        "vb_runs/x", "--host", "mycluster", "--cluster", str(repo / "clusters" / "c.toml"),
    ])
    assert rc == 1
    assert "could not parse" in capsys.readouterr().err


# --- module dispatch (python -m virasoro_bootstrap.hpc.collect status|collect) --

def _make_local_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    (run_dir / "specs").mkdir(parents=True)
    (run_dir / "tasks" / "0000").mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": "x", "task": "t", "n_tasks": 1, "slurm_job_id": None,
    }))
    (run_dir / "specs" / "0000.json").write_text(json.dumps({
        "task": "t", "params": {"c": "2"}, "output_dir": "tasks/0000",
    }))
    (run_dir / "tasks" / "0000" / "result.json").write_text(json.dumps({
        "spec_hash": "abc", "result": {"delta_gap": "1.0"},
    }))
    return run_dir


def test_module_dispatch_status_subcommand(tmp_path):
    # both the explicit ``status`` subcommand and bare args (which default to status)
    run_dir = _make_local_run_dir(tmp_path)
    for args in (["status", str(run_dir)], [str(run_dir)]):
        out = subprocess.run(
            [sys.executable, "-m", "virasoro_bootstrap.hpc.collect", *args],
            capture_output=True, text=True, check=True,
        )
        assert "done=1" in out.stdout, args


def test_module_dispatch_collect_subcommand(tmp_path):
    run_dir = _make_local_run_dir(tmp_path)
    subprocess.run(
        [sys.executable, "-m", "virasoro_bootstrap.hpc.collect", "collect", str(run_dir)],
        capture_output=True, text=True, check=True,
    )
    aggregate = json.loads((run_dir / "results.json").read_text())
    assert aggregate == [{"delta_gap": "1.0"}]


def test_module_dispatch_list_subcommand(tmp_path):
    run_dir = _make_local_run_dir(tmp_path)  # a run under tmp_path, not tmp_path itself
    out = subprocess.run(
        [sys.executable, "-m", "virasoro_bootstrap.hpc.collect", "list",
         "--runs-root", str(tmp_path)],
        capture_output=True, text=True, check=True,
    )
    assert "run" in out.stdout and "task=t" in out.stdout


# --- vb-list --host --------------------------------------------------------

def test_list_host_dispatch_uses_cluster_runs_root(tmp_path, fake_ssh_scp):
    # A plain path (no $ or ~) here: shlex.quote leaves it unescaped, so a
    # substring check is meaningful. The $HOME/~-expansion mechanics
    # (Python-side os.path.expandvars, not bash-side, since shlex.quote's
    # single-quoting would suppress bash expansion) are covered separately by
    # test_list_expands_home_and_vars, and were exercised live against a SLURM cluster.
    repo = make_repo(tmp_path)
    (repo / "clusters" / "c.toml").write_text(
        'kind = "slurm"\nremote_repo = "$HOME/repo"\nruns_root = "/abs/vb_runs"\n'
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "add runs_root")
    _queue(fake_ssh_scp, cp(stdout="runs under /abs/vb_runs (newest first):\n  x  task=t\n"))

    rc = collect.list_main([
        "--host", "mycluster", "--cluster", str(repo / "clusters" / "c.toml"),
    ])
    assert rc == 0
    calls, _ = fake_ssh_scp
    assert len(calls) == 1
    full = calls[0][2]
    assert 'cd -- "$HOME/repo"' in full
    assert "list --runs-root /abs/vb_runs --limit 15" in full


def test_remote_inspection_defaults_use_the_remote_environment(tmp_path, monkeypatch):
    remote_root = tmp_path / 'remote runs'
    remote_run = _make_local_run_dir(remote_root)
    local_root = tmp_path / 'laptop runs'
    monkeypatch.setenv('VB_RUNS_ROOT', str(local_root))
    profile = tmp_path / 'profile.toml'
    profile.write_text(f'python = {json.dumps(sys.executable)}\n')
    # the fake remote is a login shell: with the real HOME it would source the user's ~/.bashrc, which on the cluster
    # exports VB_CLUSTER (a profile this test does not use) -- give it an empty home (and the real user site-packages)
    import site

    remote_home = tmp_path / 'remote home'
    remote_home.mkdir()
    remote_env = {**os.environ, 'VB_RUNS_ROOT': str(remote_root), 'HOME': str(remote_home),
                  'PYTHONUSERBASE': site.getuserbase(),
                  'PYTHONPATH': str(Path(collect.__file__).resolve().parents[2])}
    calls = []
    def transport(cmd, *args, **kwargs):
        calls.append(cmd)
        if cmd[0] == 'ssh':
            return _REAL_RUN(['bash', '-c', cmd[2]], capture_output=True, text=True,
                             cwd=tmp_path, env=remote_env)
        return cp()  # No real scp: collection only checks the source path below.
    monkeypatch.setattr(collect.subprocess, 'run', transport)
    for command in ('status', 'list', 'collect'):
        args = [] if command == 'list' else ['run']
        if command == 'collect':
            args += ['-o', str(tmp_path / 'collected.json')]
        args += ['--host', 'example', '--cluster', str(profile)]
        assert getattr(collect, command + '_main')(args) == 0
        ssh = [cmd for cmd in calls if cmd[0] == 'ssh'][-1]
        assert '--runs-root' not in ssh[2] and str(local_root) not in ssh[2]
    assert calls[-1][2] == f'example:{remote_run}/results.json'


@pytest.mark.parametrize("raw,sent", [
    ("~/vb_runs", "$HOME/vb_runs"),
    ("~", "$HOME"),
    ("$HOME/vb_runs", "$HOME/vb_runs"),
    ("/scratch/vb runs", "/scratch/vb runs"),
])
def test_remote_expand_rewrites_leading_tilde(monkeypatch, raw, sent):
    """Tilde expansion does not happen inside the double quotes we need.

    The quotes are what make $VARS expand and what keep a path containing
    spaces in one piece, but they also stop `~` expanding -- so a profile
    runs_root of "~/vb_runs" produced a literal ~ remotely and scp failed
    against a directory that had never been created. Caught only by a real
    submission; the fake-ssh tests happily accepted the unexpanded path.
    """
    seen: list[str] = []

    def fake(cmd, *a, **kw):
        seen.append(cmd[2])
        return subprocess.CompletedProcess(cmd, 0, stdout=" /expanded ", stderr="")

    monkeypatch.setattr(submit.subprocess, "run", fake)
    assert submit._remote_expand("h", raw) == " /expanded "
    assert f'vb_runs_path="{sent}"' in seen[0]


@pytest.mark.parametrize("command, root", [("status", "/override root/runs"), ("list", "$HOME/custom runs")])
def test_remote_inspection_honors_explicit_runs_root(tmp_path, fake_ssh_scp, monkeypatch, command, root):
    import shlex

    repo = make_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    _queue(fake_ssh_scp, cp(stdout="wrote /remote/results.json (1/1 task(s) complete)\n"), cp())
    args = ([] if command == "list" else ["run-id"])
    args += ["--host", "mycluster", "--cluster", str(repo / "clusters" / "c.toml"), "--runs-root", root]
    assert getattr(collect, f"{command}_main")(args) == 0
    calls, _ = fake_ssh_scp
    outer = shlex.split(calls[0][2])
    inner = shlex.split(outer[-1])
    assert inner[inner.index("--runs-root") + 1] == root
