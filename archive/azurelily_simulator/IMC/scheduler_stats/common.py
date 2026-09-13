from __future__ import annotations

from pathlib import Path

DEBUG = True
DSP_WDITH = 4
_LOG_FILE: Path | None = None
_LOG_STDOUT = False


def cycles_to_ns(cycle, MHz):
    return cycle * (1e9 / (MHz * 1e6))


def configure_logging(
    debug: bool,
    log_file: str | None = None,
    also_stdout: bool = False,
    append: bool = False,
):
    global DEBUG, _LOG_FILE, _LOG_STDOUT
    DEBUG = debug
    _LOG_STDOUT = also_stdout
    if log_file:
        path = Path(log_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        if append:
            path.touch(exist_ok=True)
        else:
            path.write_text("", encoding="utf-8")
        _LOG_FILE = path
    else:
        _LOG_FILE = None


def log(msg):
    if not DEBUG:
        return
    line = str(msg)
    if _LOG_FILE is not None:
        with _LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(f"{line}\n")
    if _LOG_FILE is None or _LOG_STDOUT:
        print(msg)
