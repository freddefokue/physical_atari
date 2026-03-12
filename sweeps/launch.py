"""Launch multi-GPU sweep workers against generated trial specs."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from sweeps.common import (
    HOSTS,
    REPO_ROOT,
    STAGES,
    ensure_sweep_dirs,
    load_trial_specs,
    parse_gpu_list,
    read_json,
    trial_spec_paths,
    utc_now_iso,
    write_json,
)

RUN_DIR_PATTERN = re.compile(r"Run complete:\s*(.+)")


@dataclass(frozen=True)
class LaunchContext:
    host: str
    family: str
    stage: str
    gpus: Sequence[int]
    dry_run: bool
    rerun_failed: bool
    python_exe: str
    paths: Dict[str, Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch one sweep worker per GPU.")
    parser.add_argument("--host", type=str, required=True, choices=sorted(HOSTS), help="Target host.")
    parser.add_argument(
        "--family",
        type=str,
        required=True,
        choices=["ppo", "delay_target", "bbf", "rainbow_dqn", "sac"],
        help="Agent family to launch.",
    )
    parser.add_argument("--stage", type=str, default="stage0", choices=sorted(STAGES), help="Sweep stage.")
    parser.add_argument("--gpus", type=str, required=True, help="Explicit comma-separated GPU ids.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without executing them.")
    parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help="Requeue trials whose existing result record has status=failed.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to use for benchmark and scoring subprocesses.",
    )
    return parser.parse_args()


def validate_host_gpu_rules(host: str, gpus: Sequence[int]) -> None:
    if host == "obsession" and len(gpus) != 2:
        raise ValueError("--host obsession requires exactly 2 GPUs in --gpus")
    if not gpus:
        raise ValueError("--gpus must not be empty")


def build_run_command(trial_spec: Dict[str, Any], config_path: Path, logdir: Path, gpu: int, python_exe: str) -> List[str]:
    family = str(trial_spec["family"])
    cmd = [python_exe, "-u", "-m", "benchmark.run_multigame", "--config", str(config_path), "--logdir", str(logdir)]
    if family == "delay_target":
        cmd.extend(["--delay-target-gpu", str(gpu)])
    elif family == "rainbow_dqn":
        cmd.extend(["--rainbow-dqn-gpu", str(gpu)])
    elif family == "sac":
        cmd.extend(["--sac-gpu", str(gpu)])
    elif family == "ppo":
        cmd.extend(["--ppo-device", "cuda"])
    elif family == "bbf":
        pass
    else:
        raise ValueError(f"Unsupported family: {family}")
    return cmd


def build_run_env(family: str, gpu: int) -> Dict[str, str]:
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    if family in {"ppo", "bbf"}:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def append_worker_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"[{utc_now_iso()}] {message}\n")


def load_result_status(result_path: Path) -> Optional[str]:
    if not result_path.exists():
        return None
    try:
        return str(read_json(result_path).get("status"))
    except Exception:
        return None


def launcher_lock_path(paths: Dict[str, Path]) -> Path:
    return paths["queue"] / "launcher.lock"


def _read_lock_payload(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return read_json(path)
    except Exception:
        return None


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except ProcessLookupError:
        return False
    return True


def acquire_launcher_lock(context: LaunchContext) -> Path:
    path = launcher_lock_path(context.paths)
    payload = {
        "pid": int(os.getpid()),
        "host": context.host,
        "family": context.family,
        "stage": context.stage,
        "acquired_at": utc_now_iso(),
    }

    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            existing = _read_lock_payload(path)
            if existing is None:
                raise RuntimeError(
                    f"Sweep lock already exists at {path}. Refusing to start a second launcher."
                ) from exc
            existing_pid = int(existing.get("pid", -1))
            if _pid_is_running(existing_pid):
                raise RuntimeError(
                    f"Sweep lock already held by pid={existing_pid} at {path}. "
                    "Refusing to start a second launcher."
                ) from exc
            path.unlink()
            continue
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
                fh.write("\n")
            return path
        except Exception:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            raise


def release_launcher_lock(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def sync_queue_state(context: LaunchContext, specs: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"pending": 0, "completed": 0, "failed": 0}
    for spec in specs:
        paths = trial_spec_paths(context.paths, spec["trial_id"])
        for stale_path in (paths["queue_pending_path"], paths["queue_claimed_path"], paths["queue_completed_path"], paths["queue_failed_path"]):
            if stale_path.exists():
                stale_path.unlink()

        result_status = load_result_status(paths["result_path"])
        if result_status == "completed":
            write_json(paths["queue_completed_path"], spec)
            counts["completed"] += 1
            continue
        if result_status == "failed" and not context.rerun_failed:
            write_json(paths["queue_failed_path"], spec)
            counts["failed"] += 1
            continue
        write_json(paths["queue_pending_path"], spec)
        counts["pending"] += 1
    return counts


def claim_next_trial(paths: Dict[str, Path]) -> Optional[Path]:
    for pending_path in sorted(paths["queue_pending"].glob("*.json")):
        claimed_path = paths["queue_claimed"] / pending_path.name
        try:
            os.replace(pending_path, claimed_path)
            return claimed_path
        except FileNotFoundError:
            continue
    return None


def detect_run_dir(trial_run_root: Path, stdout_path: Path) -> Optional[Path]:
    if stdout_path.exists():
        content = stdout_path.read_text(encoding="utf-8", errors="replace")
        matches = RUN_DIR_PATTERN.findall(content)
        if matches:
            return Path(matches[-1].strip())
    if not trial_run_root.exists():
        return None
    directories = [path for path in trial_run_root.iterdir() if path.is_dir()]
    if not directories:
        return None
    directories.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return directories[0]


def detect_new_run_dir(before_dirs: Sequence[Path], trial_run_root: Path, stdout_path: Path) -> Optional[Path]:
    before_set = {path.resolve() for path in before_dirs}
    if trial_run_root.exists():
        created_dirs = [path for path in trial_run_root.iterdir() if path.is_dir() and path.resolve() not in before_set]
        if created_dirs:
            created_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
            return created_dirs[0]
    return detect_run_dir(trial_run_root, stdout_path)


def execute_command(
    *,
    cmd: Sequence[str],
    env: Dict[str, str],
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    append: bool = False,
) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with stdout_path.open(mode, encoding="utf-8") as stdout_fh, stderr_path.open(mode, encoding="utf-8") as stderr_fh:
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd),
            env=env,
            stdout=stdout_fh,
            stderr=stderr_fh,
            check=False,
        )
    return int(proc.returncode)


def collect_score_fields(run_dir: Optional[Path]) -> Dict[str, Any]:
    if run_dir is None:
        return {}
    score_path = run_dir / "score.json"
    if not score_path.exists():
        return {}
    score = read_json(score_path)
    fields = {}
    for key in (
        "final_score",
        "mean_score",
        "bottom_k_score",
        "forgetting_index_mean",
        "plasticity_mean",
        "fps",
        "frames",
    ):
        if key in score:
            fields[key] = score[key]
    return fields


def write_result_record(path: Path, payload: Dict[str, Any]) -> None:
    write_json(path, payload)


def run_trial(context: LaunchContext, trial_spec: Dict[str, Any], gpu: int, worker_log_path: Path) -> Dict[str, Any]:
    paths = trial_spec_paths(context.paths, trial_spec["trial_id"])
    trial_run_root = paths["trial_run_root"]
    trial_run_root.mkdir(parents=True, exist_ok=True)
    before_dirs = [path for path in trial_run_root.iterdir() if path.is_dir()]

    benchmark_config = dict(trial_spec["benchmark_config"])
    benchmark_config["logdir"] = str(trial_run_root)
    write_json(paths["benchmark_config_path"], benchmark_config)

    cmd = build_run_command(
        trial_spec=trial_spec,
        config_path=paths["benchmark_config_path"],
        logdir=trial_run_root,
        gpu=gpu,
        python_exe=context.python_exe,
    )
    env = build_run_env(str(trial_spec["family"]), gpu)

    started_at = utc_now_iso()
    append_worker_log(worker_log_path, f"starting {trial_spec['trial_id']} on gpu={gpu}")
    benchmark_exit_code = execute_command(
        cmd=cmd,
        env=env,
        cwd=REPO_ROOT,
        stdout_path=paths["stdout_log_path"],
        stderr_path=paths["stderr_log_path"],
    )
    run_dir = detect_new_run_dir(before_dirs, trial_run_root, paths["stdout_log_path"])
    score_exit_code: Optional[int] = None
    scoring_cmd: Optional[List[str]] = None
    status = "failed"
    failure_reason: Optional[str] = None

    if benchmark_exit_code != 0:
        failure_reason = "benchmark_failed"
    elif run_dir is None or not run_dir.exists():
        failure_reason = "run_dir_not_found"
    else:
        scoring_cmd = [context.python_exe, "-m", "benchmark.score_run", "--run-dir", str(run_dir)]
        score_exit_code = execute_command(
            cmd=scoring_cmd,
            env=env,
            cwd=REPO_ROOT,
            stdout_path=paths["stdout_log_path"],
            stderr_path=paths["stderr_log_path"],
            append=True,
        )
        if score_exit_code != 0:
            failure_reason = "scoring_failed"
        elif not (run_dir / "score.json").exists():
            failure_reason = "score_json_missing"
        else:
            status = "completed"

    ended_at = utc_now_iso()
    score_fields = collect_score_fields(run_dir)
    result = {
        "trial_id": trial_spec["trial_id"],
        "family": trial_spec["family"],
        "stage": trial_spec["stage"],
        "training_seed": trial_spec["training_seed"],
        "sampled_hyperparameters": trial_spec["sampled_hyperparameters"],
        "benchmark_config_path": str(paths["benchmark_config_path"]),
        "benchmark_command": cmd,
        "score_command": scoring_cmd,
        "host": context.host,
        "gpu": int(gpu),
        "run_dir": None if run_dir is None else str(run_dir),
        "status": status,
        "start_time": started_at,
        "end_time": ended_at,
        "exit_code": int(score_exit_code if score_exit_code is not None else benchmark_exit_code),
        "benchmark_exit_code": int(benchmark_exit_code),
        "score_exit_code": score_exit_code,
        "stdout_log_path": str(paths["stdout_log_path"]),
        "stderr_log_path": str(paths["stderr_log_path"]),
        "failure_reason": failure_reason,
        "score_summary": score_fields,
    }
    result.update(score_fields)
    write_result_record(paths["result_path"], result)
    append_worker_log(worker_log_path, f"finished {trial_spec['trial_id']} status={status}")
    return result


def worker_loop(context: LaunchContext, gpu: int) -> None:
    worker_log_path = context.paths["logs"] / f"worker_gpu{gpu}.log"
    append_worker_log(worker_log_path, f"worker online for gpu={gpu}")
    while True:
        claimed_path = claim_next_trial(context.paths)
        if claimed_path is None:
            append_worker_log(worker_log_path, "queue empty")
            return
        trial_spec = read_json(claimed_path)
        queue_paths = trial_spec_paths(context.paths, trial_spec["trial_id"])
        try:
            result = run_trial(context, trial_spec, gpu, worker_log_path)
            destination = (
                queue_paths["queue_completed_path"] if result["status"] == "completed" else queue_paths["queue_failed_path"]
            )
            os.replace(claimed_path, destination)
        except Exception as exc:  # pragma: no cover - defensive launcher path
            append_worker_log(worker_log_path, f"exception for {trial_spec['trial_id']}: {exc}")
            failure_payload = {
                "trial_id": trial_spec["trial_id"],
                "family": trial_spec["family"],
                "stage": trial_spec["stage"],
                "training_seed": trial_spec["training_seed"],
                "sampled_hyperparameters": trial_spec["sampled_hyperparameters"],
                "benchmark_config_path": str(queue_paths["benchmark_config_path"]),
                "benchmark_command": None,
                "score_command": None,
                "host": context.host,
                "gpu": int(gpu),
                "run_dir": None,
                "status": "failed",
                "start_time": utc_now_iso(),
                "end_time": utc_now_iso(),
                "exit_code": -1,
                "benchmark_exit_code": -1,
                "score_exit_code": None,
                "stdout_log_path": str(queue_paths["stdout_log_path"]),
                "stderr_log_path": str(queue_paths["stderr_log_path"]),
                "failure_reason": f"launcher_exception: {exc}",
                "score_summary": {},
            }
            write_result_record(queue_paths["result_path"], failure_payload)
            os.replace(claimed_path, queue_paths["queue_failed_path"])


def dry_run_preview(context: LaunchContext, specs: Sequence[Dict[str, Any]]) -> None:
    pending_specs: List[Dict[str, Any]] = []
    for spec in specs:
        paths = trial_spec_paths(context.paths, spec["trial_id"])
        result_status = load_result_status(paths["result_path"])
        if result_status == "completed":
            continue
        if result_status == "failed" and not context.rerun_failed:
            continue
        pending_specs.append(spec)

    for index, spec in enumerate(pending_specs):
        gpu = context.gpus[index % len(context.gpus)]
        paths = trial_spec_paths(context.paths, spec["trial_id"])
        cmd = build_run_command(
            trial_spec=spec,
            config_path=paths["benchmark_config_path"],
            logdir=paths["trial_run_root"],
            gpu=gpu,
            python_exe=context.python_exe,
        )
        env = build_run_env(str(spec["family"]), gpu)
        print(json.dumps({"trial_id": spec["trial_id"], "gpu": gpu, "cmd": cmd, "env_overrides": env.get("CUDA_VISIBLE_DEVICES")}, sort_keys=True))


def main() -> None:
    args = parse_args()
    gpus = parse_gpu_list(args.gpus)
    validate_host_gpu_rules(args.host, gpus)
    paths = ensure_sweep_dirs(args.host, args.family, args.stage)
    context = LaunchContext(
        host=args.host,
        family=args.family,
        stage=args.stage,
        gpus=tuple(gpus),
        dry_run=bool(args.dry_run),
        rerun_failed=bool(args.rerun_failed),
        python_exe=str(args.python),
        paths=paths,
    )
    lock_path = acquire_launcher_lock(context)
    try:
        specs = load_trial_specs(paths)
        counts = sync_queue_state(context, specs)
        print(
            json.dumps(
                {
                    "host": args.host,
                    "family": args.family,
                    "stage": args.stage,
                    "gpus": gpus,
                    "pending": counts["pending"],
                    "completed": counts["completed"],
                    "failed": counts["failed"],
                    "dry_run": bool(args.dry_run),
                    "rerun_failed": bool(args.rerun_failed),
                },
                sort_keys=True,
            )
        )

        if args.dry_run:
            dry_run_preview(context, specs)
            return

        threads = [threading.Thread(target=worker_loop, args=(context, gpu), daemon=False) for gpu in gpus]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        print("Launch complete")
    finally:
        release_launcher_lock(lock_path)


if __name__ == "__main__":
    main()
