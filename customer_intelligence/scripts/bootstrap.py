"""One-time setup, run by setup.bat with the system Python (standard library only).

Creates/validates .venv, installs requirements WITH THAT venv's interpreter, verifies imports, downloads the
dataset and trains the demo tenants. Every command, its full output and exit code go to the console AND setup.log.
"""
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "setup.log"
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass


def out(msg: str = "") -> None:
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


class StepFailed(Exception):
    pass


def _fmt(cmd) -> str:
    return " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)


def run(cmd, desc: str, attempts: int = 1) -> None:
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    rc = -1
    for n in range(1, attempts + 1):
        out(f"\n>>> {desc}" + (f"  (attempt {n}/{attempts})" if attempts > 1 else ""))
        out("$ " + _fmt(cmd))
        try:
            p = subprocess.Popen([str(c) for c in cmd], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            for line in p.stdout:
                out(line.rstrip())
            rc = p.wait()
        except OSError as e:
            out(f"Could not start the command: {e!r}")
            rc = -1
        out(f"[exit code {rc}]")
        if rc == 0:
            return
        if n < attempts:
            out("Retrying in 5 seconds ...")
            time.sleep(5)
    raise StepFailed(f"{desc} FAILED with exit code {rc}\nFailing command: {_fmt(cmd)}")


def venv_python() -> Path:
    return ROOT / ".venv" / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")


def venv_ok(vpy: Path) -> bool:
    if not vpy.exists():
        return False
    try:
        r = subprocess.run([str(vpy), "-c", "import sys, pip; sys.exit(0 if sys.version_info[:2] >= (3, 11) else 3)"],
                           capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        out(f"Existing .venv interpreter could not run: {e!r}")
        return False
    if r.returncode != 0:
        out(f"Existing .venv is unusable (exit {r.returncode}): {(r.stderr or r.stdout).strip()}")
    return r.returncode == 0


def main() -> int:
    LOG.write_text("", encoding="utf-8")
    out(f"Setup started {time.strftime('%Y-%m-%d %H:%M:%S')}")
    out(f"Project folder: {ROOT}")
    out(f"Platform: {platform.platform()}")
    out(f"Bootstrap Python executable: {sys.executable}")
    out(f"Python version: {platform.python_version()}")
    if sys.version_info < (3, 11):
        raise StepFailed(f"Python 3.11 or newer is required (found {platform.python_version()}). Install Python 3.12 from python.org.")

    vpy = venv_python()
    if venv_ok(vpy):
        out("Virtual environment: VERIFIED")
    else:
        if (ROOT / ".venv").exists():
            out("Removing broken .venv ...")
            shutil.rmtree(ROOT / ".venv", ignore_errors=True)
        run([sys.executable, "-m", "venv", ".venv"], "Create virtual environment")
        if not venv_ok(vpy):
            raise StepFailed(f"Virtual environment was created but its interpreter is unusable: {vpy}")
        out("Virtual environment: CREATED")
    out(f"Project interpreter: {vpy}")

    run([vpy, "-m", "pip", "--version"], "Check pip in the virtual environment")
    run([vpy, "-m", "pip", "install", "--default-timeout=180", "--retries", "10", "-r", "requirements.txt"],
        "Install dependencies into the virtual environment", attempts=3)
    run([vpy, "-m", "scripts.verify_env"], "Verify dependencies with the same interpreter")
    run([vpy, "-m", "scripts.download_data"], "Obtain dataset (scripts/download_data.py)")
    out("download_data.py: EXECUTION SUCCESSFUL")
    run([vpy, "-m", "scripts.seed_demo"], "Create demo tenants and train models")
    out("\nSetup completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except StepFailed as e:
        out(f"\nSETUP FAILED: {e}")
        sys.exit(1)
    except Exception:  # noqa: BLE001
        out("\nSETUP FAILED with an unexpected error:\n" + traceback.format_exc())
        sys.exit(2)
