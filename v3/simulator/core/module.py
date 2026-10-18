"""module.py — the base of every behavior model (docs/simulator.md §3.3).

A module is a set of processes; a process is an in-order sequence of steps (its units). A step declares what it needs
(`Need`): windows of an input port by count per channel, the credit of a given index on an output port, progress of
another process of the same module. Then `step()` executes it: it reads the inputs' values and times, computes the
outputs by the hardware's rules, emits windows and credits, records its events and adds its counts.

A module's parameters are a dataclass with the RTL parameter names (`p`); its derived values are a dataclass with the
RTL localparam names (`d`), computed by `derive(p)`; `checkDomain(p)` rejects what the RTL's elaboration guards reject.
"""
from __future__ import annotations

from dataclasses import fields

from simulator.core.records import Counts, Resources


class Need:
    """What step `unit` of a process reads before it can run."""

    def __init__(self, windows: dict | None = None, credits: dict | None = None, after: dict | None = None):
        self.windows = dict(windows or {})    # input port -> {channel: number of windows}
        self.credits = dict(credits or {})    # output port -> credit index (0-based), or None (no credit needed yet)
        self.after = dict(after or {})        # process -> finished steps needed

    @staticmethod
    def eachChannel(L: int, n: int = 1) -> dict:
        """n windows from each of L channels."""
        return {channel: n for channel in range(L)}


def checkRules(name: str, rules: list) -> None:
    """The domain check of a module: `rules` = [(holds, text)], every text whose rule fails goes into the error."""
    failed = [text for holds, text in rules if not holds]
    if failed:
        raise ValueError(f"{name}: parameters outside the domain: " + "; ".join(failed))


class Module:
    NAME = "module"             # the RTL module's name
    PARAMS_CLASS = None         # the dataclass of the RTL parameters (field names = the RTL's parameter names)
    BASIS = "model-only"        # "rtl-pinned" once GATE 2 has pinned the model's named latencies
    STEP_METHODS: dict = {}     # process name -> the method that runs one of its steps: method(unit, io) -> Step

    @classmethod
    def paramNames(cls) -> list:
        """The RTL parameter names, in the RTL's order."""
        return [f.name for f in fields(cls.PARAMS_CLASS)] if cls.PARAMS_CLASS else []

    def __init__(self, instance: str, params=None, *, withValues: bool = True, role: str | None = None):
        self.instance = self.baseName = instance      # baseName inside its graph; instance = the flattened name
        self.p = params
        if params is not None:
            self.checkDomain(params)
        self.d = self.derive(params)
        self.withValues = withValues
        self.role = role or self.NAME                 # what the instance does in its workload (for the breakdowns)
        self.counts = Counts()
        self.resources = Resources()

    # -- declaration ------------------------------------------------------------------------------------------------
    @staticmethod
    def derive(p):
        """The RTL's localparams (a dataclass), from the parameters."""
        return None

    @staticmethod
    def checkDomain(p) -> None:
        """Rejects the parameter sets the RTL's elaboration guards reject."""

    def ports(self) -> dict:
        """port name -> PortSpec."""
        raise NotImplementedError

    def processes(self) -> dict:
        """process name -> number of units (steps)."""
        raise NotImplementedError

    # -- behavior ---------------------------------------------------------------------------------------------------
    def need(self, process: str, unit: int) -> Need:
        raise NotImplementedError

    def step(self, process: str, unit: int, io):
        return getattr(self, self.STEP_METHODS[process])(unit, io)
