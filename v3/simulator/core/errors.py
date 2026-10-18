"""errors.py — the three errors the simulator raises on purpose."""


class EdgeError(RuntimeError):
    """A graph is wired against the port disciplines (docs/architecture.md §3.2), or a module breaks them at run time."""


class Deadlock(RuntimeError):
    """No process can step before every process has finished; the message lists what each one waits for."""


class ContractError(RuntimeError):
    """A producer broke a port contract of a module's spec (a converter overrun, rows out of order, a full register)."""
