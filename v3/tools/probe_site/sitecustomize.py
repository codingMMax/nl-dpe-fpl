"""sitecustomize.py — isolation probe for v3 runs (PLAN R3: v3 copies from v2 and never imports it).

Active only when the environment sets V3_PROBE_LOG and PYTHONPATH contains this folder, so every Python process of a
probed run (the runners' generator and gate subprocesses included) loads it. It records, without changing behavior:
  - every file opened under the repository but outside v3/;
  - every subprocess argument that names such a path (plain, or after `=` as in +VDIR=...);
  - at exit, every imported module whose file lies there.
One JSON line per process is appended to $V3_PROBE_LOG. tools/f1_check.py reads the log; the run is isolated when no
line has a violation. (The system sitecustomize, Ubuntu's apport crash hook, is not loaded in probed processes.)
"""
import atexit
import json
import os
import sys

_LOG = os.environ.get("V3_PROBE_LOG")
if _LOG:
    _V3 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _REPO = os.path.dirname(_V3)
    _seen = {"open": 0, "exec": 0}
    _bad = []

    def _outside(p, base=None) -> bool:
        """p (relative paths against base, else the current directory) lies in the repo but outside v3/."""
        try:
            if isinstance(p, int):
                return False
            p = os.fsdecode(p)
            full = os.path.realpath(os.path.join(os.fsdecode(base) if base is not None else os.getcwd(), p))
        except Exception:
            return False
        return (full + os.sep).startswith(_REPO + os.sep) and not (full + os.sep).startswith(_V3 + os.sep)

    def _hook(event, args):
        try:
            if event == "open":
                _seen["open"] += 1
                if args and _outside(args[0]):
                    _bad.append(f"open {os.fsdecode(args[0])}")
            elif event == "subprocess.Popen":         # args: executable, args, cwd, env
                _seen["exec"] += 1
                argv = args[1] if len(args) > 1 else None
                cwd = args[2] if len(args) > 2 else None
                argv = [argv] if isinstance(argv, (str, bytes, os.PathLike)) else list(argv or [])
                for a in argv:
                    a = os.fsdecode(a) if isinstance(a, (bytes, os.PathLike)) else str(a)
                    part = a.split("=", 1)[1] if "=" in a else a      # +VDIR=path, --out=path: the value
                    if "/" in part and _outside(part, cwd):           # relative to the child's directory
                        _bad.append(f"exec arg {a}" + (f" (cwd {os.fsdecode(cwd)})" if cwd is not None else ""))
        except Exception:
            pass

    def _report():
        try:
            mods = sorted({m.__file__ for m in list(sys.modules.values())
                           if isinstance(getattr(m, "__file__", None), str) and _outside(m.__file__)})
            rec = {"pid": os.getpid(), "argv": sys.argv, "cwd": os.getcwd(), "opens": _seen["open"],
                   "execs": _seen["exec"], "violations": _bad + [f"import {f}" for f in mods]}
            with open(_LOG, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        except Exception:
            pass

    sys.addaudithook(_hook)
    atexit.register(_report)
