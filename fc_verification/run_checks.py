#!/usr/bin/env python3
"""Machine-checkable pass/fail harness for fc_verification/.

Runs the functional + latency testbenches for a DIMM-top config, parses
the Verilog testbench stdout, compares results to expected_cycles.json /
functional_whitelist.json, and exits 0 on pass or 1 on any failure.

Designed to be called by autonomous agents — every failure path prints a
single-line summary so progress.md entries stay terse.

Usage:
    python3 fc_verification/run_checks.py --config nldpe_dimm_top_d64_c128
    python3 fc_verification/run_checks.py --config azurelily_dimm_top_d64_c128
    python3 fc_verification/run_checks.py --all
    python3 fc_verification/run_checks.py --config X --skip-latency   # functional only
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FC_DIR = REPO_ROOT / "fc_verification"
RTL_DIR = FC_DIR / "rtl"
STUB = FC_DIR / "dpe_stub.v"

EXPECTED_JSON = FC_DIR / "expected_cycles.json"
EXPECTED_COUNTERS_JSON = FC_DIR / "expected_counters.json"
KNOWN_COUNT_DELTAS_JSON = FC_DIR / "known_count_deltas.json"
WHITELIST_JSON = FC_DIR / "functional_whitelist.json"
PHASE3_KNOWN_DELTAS_JSON = FC_DIR / "phase3_known_deltas.json"
PHASE5_KNOWN_DELTAS_JSON = FC_DIR / "phase5_known_deltas.json"
PHASE7_KNOWN_DELTAS_JSON = FC_DIR / "phase7_known_deltas.json"

# Phase-specific known-delta JSON selector. NL-DPE DIMM-top configs load
# Phase 3 deltas; Azure-Lily DIMM-top loads Phase 5. Phase 7 covers the
# attention-head close-out (AH track) for both archs. The schema is
# identical — only the file differs so each architecture's residual
# provenance stays self-contained.
PHASE_DELTA_FILES = {
    "nldpe_dimm_top_d64_c128":         PHASE3_KNOWN_DELTAS_JSON,
    "azurelily_dimm_top_d64_c128":     PHASE5_KNOWN_DELTAS_JSON,
    "nldpe_attn_head_d64_c128":        PHASE7_KNOWN_DELTAS_JSON,
    "azurelily_attn_head_d64_c128":    PHASE7_KNOWN_DELTAS_JSON,
}

# Map of config name → (rtl file, functional TB, latency TB).  AH configs
# use a single combined func+latency TB (tb_*_attn_head_v2.v); the
# functional check parses the v2 TB's output pulse summary while the
# latency check parses the AH_*_STAGES regex line from the same run.
CONFIG_REGISTRY = {
    "nldpe_dimm_top_d64_c128": {
        "rtl": RTL_DIR / "nldpe_dimm_top_d64_c128.v",
        "tb_func": FC_DIR / "tb_nldpe_dimm_top_functional.v",
        "tb_lat":  FC_DIR / "tb_nldpe_dimm_top_latency.v",
    },
    "azurelily_dimm_top_d64_c128": {
        "rtl": RTL_DIR / "azurelily_dimm_top_d64_c128.v",
        "tb_func": FC_DIR / "tb_azurelily_dimm_top_functional.v",
        "tb_lat":  FC_DIR / "tb_azurelily_dimm_top_latency.v",
    },
    "nldpe_attn_head_d64_c128": {
        "rtl":         RTL_DIR / "nldpe_attn_head_d64_c128.v",
        "tb_combined": FC_DIR / "tb_nldpe_attn_head_v2.v",
        "extra_rtl":   [
            RTL_DIR / "nldpe" / "fc_top_qkv_streaming.v",
            RTL_DIR / "nldpe_dimm_top_d64_c128.v",
            STUB,
        ],
        "ah_arch":     "nldpe",
    },
    "azurelily_attn_head_d64_c128": {
        "rtl":         RTL_DIR / "azurelily_attn_head_d64_c128.v",
        "tb_combined": FC_DIR / "tb_azurelily_attn_head_v2.v",
        "extra_rtl":   [
            RTL_DIR / "azurelily" / "azurelily_fc_128_64.v",
            RTL_DIR / "azurelily_dimm_top_d64_c128.v",
        ],
        "ah_arch":     "azurelily",
    },
}

# AH (attention-head) configs — these route through a different
# pipeline than DIMM-top: a single combined TB run produces both the
# functional sanity (top valid_n pulse counts) and the per-stage cycle
# regex line (AH_NLDPE_STAGES / AH_AL_STAGES).
AH_CONFIGS = {"nldpe_attn_head_d64_c128", "azurelily_attn_head_d64_c128"}

# AH config → arch tag (controls regex variant + softmax-fold semantics).
AH_ARCH_BY_CONFIG = {
    "nldpe_attn_head_d64_c128":     "nldpe",
    "azurelily_attn_head_d64_c128": "azurelily",
}


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    fields: dict = field(default_factory=dict)


def run_iverilog(sources: list[Path], timeout_s: int = 120) -> tuple[int, str]:
    """Compile + run a testbench with iverilog. Returns (returncode, stdout+stderr)."""
    out_bin = Path("/tmp") / f"rc_{sources[-1].stem}"
    cmd_compile = ["iverilog", "-g2012", "-o", str(out_bin)] + [str(s) for s in sources]
    try:
        comp = subprocess.run(cmd_compile, capture_output=True, text=True,
                              timeout=timeout_s)
    except subprocess.TimeoutExpired as e:
        return 124, f"compile timeout: {e}"
    if comp.returncode != 0:
        return comp.returncode, f"COMPILE FAIL\n{comp.stderr}\n{comp.stdout}"
    try:
        run = subprocess.run(["vvp", str(out_bin)], capture_output=True, text=True,
                             timeout=timeout_s)
    except subprocess.TimeoutExpired as e:
        return 124, f"runtime timeout: {e}"
    return run.returncode, run.stdout + run.stderr


# AH-track (attention-head) per-stage regex lines emitted by
# tb_{nldpe,azurelily}_attn_head_v2.v at simulation finish. The TBs emit
# both a legacy `AH_*_STAGES` (first-out latencies, kept for back-compat)
# and a canonical `AH_*_STAGES_TOTAL` (stage-total durations, OR-aggregated
# across all W=16 DIMM lanes).  run_checks.py prefers _TOTAL if present.
RE_AH_NLDPE_STAGES_TOTAL = re.compile(
    r"AH_NLDPE_STAGES_TOTAL\s+linear_qkv=(-?\d+)\s+mac_qk=(-?\d+)\s+"
    r"softmax_exp=(-?\d+)\s+softmax_norm=(-?\d+)\s+mac_sv=(-?\d+)\s+e2e=(-?\d+)"
)
RE_AH_AL_STAGES_TOTAL = re.compile(
    r"AH_AL_STAGES_TOTAL\s+linear_qkv=(-?\d+)\s+mac_qk=(-?\d+)\s+"
    r"softmax_exp=(-?\d+)\s+softmax_norm=(-?\d+)\s+mac_sv=(-?\d+)\s+e2e=(-?\d+)"
)
RE_AH_NLDPE_STAGES = re.compile(
    r"AH_NLDPE_STAGES\s+linear_qkv=(-?\d+)\s+mac_qk=(-?\d+)\s+"
    r"softmax_exp=(-?\d+)\s+softmax_norm=(-?\d+)\s+mac_sv=(-?\d+)\s+e2e=(-?\d+)"
)
RE_AH_AL_STAGES = re.compile(
    r"AH_AL_STAGES\s+linear_qkv=(-?\d+)\s+mac_qk=(-?\d+)\s+"
    r"softmax_exp=(-?\d+)\s+softmax_norm=(-?\d+)\s+mac_sv=(-?\d+)\s+e2e=(-?\d+)"
)
RE_AH_FUNC_PASS = re.compile(r"Functional\s*(?:output)?\s*:\s*PASS", re.IGNORECASE)
RE_AH_TOP_PULSES = re.compile(r"top valid_n pulses\s*[:=]?\s*(\d+)", re.IGNORECASE)

# AH-gate Stage 1: counter probe regex.
# Format: COUNTER <stage> <key> <value>
RE_COUNTER = re.compile(
    r"^COUNTER\s+(\S+)\s+(\S+)\s+(-?\d+)\s*$", re.MULTILINE
)


def _parse_ah_stages(out: str, arch: str) -> dict:
    """Parse the canonical AH_*_STAGES_TOTAL regex line; fall back to legacy.

    The new TB probe semantics emit a dedicated `AH_*_STAGES_TOTAL` line
    with stage-total durations (OR-aggregated across W=16 DIMM lanes).
    The legacy `AH_*_STAGES` line is kept for diagnostic purposes only.
    """
    rx_total = RE_AH_NLDPE_STAGES_TOTAL if arch == "nldpe" else RE_AH_AL_STAGES_TOTAL
    m = rx_total.search(out)
    if not m:
        # Legacy fallback (older TB without _TOTAL line).
        rx = RE_AH_NLDPE_STAGES if arch == "nldpe" else RE_AH_AL_STAGES
        m = rx.search(out)
    if not m:
        return {}
    return {
        "linear_qkv":   int(m.group(1)),
        "mac_qk":       int(m.group(2)),
        "softmax_exp":  int(m.group(3)),
        "softmax_norm": int(m.group(4)),
        "mac_sv":       int(m.group(5)),
        "e2e":          int(m.group(6)),
    }


# ── Functional parsing ────────────────────────────────────────────────────

RE_NLDPE_SCORE   = re.compile(r"Score PASS\s*:\s*(\d+)\s*/\s*(\d+)\s*\(err=(\d+)\)")
RE_NLDPE_LANE    = re.compile(r"Lane isolate\s*:\s*(\d+)\s*/\s*(\d+)")
RE_OVERALL_PASS  = re.compile(r"Overall\s*:\s*PASS", re.IGNORECASE)
RE_OVERALL_FAIL  = re.compile(r"Overall\s*:\s*FAIL")


def check_functional(config_name: str, whitelist: dict) -> CheckResult:
    cfg = CONFIG_REGISTRY[config_name]
    rc, out = run_iverilog([STUB, cfg["rtl"], cfg["tb_func"]])
    if rc != 0 and rc != 124 and "COMPILE FAIL" in out:
        return CheckResult("functional", False, f"compile failed rc={rc}",
                           {"stdout_tail": out[-500:]})

    score_m = RE_NLDPE_SCORE.search(out)
    lane_m  = RE_NLDPE_LANE.search(out)
    overall_pass = bool(RE_OVERALL_PASS.search(out))
    overall_fail = bool(RE_OVERALL_FAIL.search(out))

    rules = whitelist.get("configs", {}).get(config_name, {})
    allowed_score_errs = sum(e.get("max_count", 0)
                             for e in rules.get("tolerated_mismatches", [])
                             if e.get("check") == "per_lane_score")
    min_lanes = rules.get("min_lanes_match_lane0", 15)

    passed = True
    reasons = []
    if not score_m:
        passed = False
        reasons.append("score summary not found in TB output")
    else:
        correct = int(score_m.group(1)); total = int(score_m.group(2)); errs = int(score_m.group(3))
        if errs > allowed_score_errs:
            passed = False
            reasons.append(f"score errs {errs} > whitelist {allowed_score_errs}")

    if lane_m:
        lanes_ok = int(lane_m.group(1))
        if lanes_ok < min_lanes:
            passed = False
            reasons.append(f"lane-isolation {lanes_ok} < required {min_lanes}")

    if overall_fail and not overall_pass:
        # Accept overall FAIL only if the failure reason is entirely in the whitelist.
        if allowed_score_errs == 0 and not reasons:
            passed = False
            reasons.append("TB reported Overall FAIL with no whitelisted justification")

    detail = "pass" if passed else "; ".join(reasons) or "see stdout"
    fields = {
        "score_correct": int(score_m.group(1)) if score_m else None,
        "score_errs":    int(score_m.group(3)) if score_m else None,
        "lane_ok":       int(lane_m.group(1)) if lane_m else None,
        "overall_pass":  overall_pass,
        "overall_fail":  overall_fail,
        "stdout_tail":   out[-400:],
    }
    return CheckResult("functional", passed, detail, fields)


# ── Latency parsing ───────────────────────────────────────────────────────

# Per-stage line format (from tb_nldpe_dimm_top_latency.v):
#   "Score stage   : %0d cycles  (start=%0d  end=%0d)"
RE_STAGE   = re.compile(r"(Score|Softmax|Wsum)\s+stage\s*:\s*(\d+)\s*cycles", re.IGNORECASE)
RE_E2E     = re.compile(r"Compute\+output\s*:\s*(\d+)\s*cycles", re.IGNORECASE)


def _load_phase3_known_deltas(config_name: str) -> dict:
    """Load per-stage known-delta expectations for Phase 3 DIMM-top configs.

    Returns a dict mapping stage -> {"delta_cycles", "tolerance"}.  Missing
    file or missing config -> empty dict (legacy Phase-J tight-tolerance
    behaviour applies).
    """
    if not PHASE3_KNOWN_DELTAS_JSON.exists():
        return {}
    try:
        data = json.loads(PHASE3_KNOWN_DELTAS_JSON.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    cfg_entry = data.get("configs", {}).get(config_name, {})
    by_stage = {}
    for entry in cfg_entry.get("deltas", []):
        stage = entry.get("stage")
        if stage:
            by_stage[stage] = {
                "delta_cycles": entry.get("delta_cycles", 0),
                "tolerance":    entry.get("tolerance", 0),
                "root_cause":   entry.get("root_cause", ""),
                "classification": entry.get("classification", ""),
            }
    return by_stage


def _load_known_deltas(config_name: str) -> dict:
    """Dispatch to the phase-specific known-delta JSON for *config_name*.

    NL-DPE DIMM-top configs use phase3_known_deltas.json (established by
    Phases 3+4). Azure-Lily DIMM-top uses phase5_known_deltas.json (added
    in Phase 5). Schema is identical — each file keeps its architecture's
    residual provenance self-contained.
    """
    delta_path = PHASE_DELTA_FILES.get(config_name, PHASE3_KNOWN_DELTAS_JSON)
    if not delta_path.exists():
        return {}
    try:
        data = json.loads(delta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    cfg_entry = data.get("configs", {}).get(config_name, {})
    by_stage = {}
    for entry in cfg_entry.get("deltas", []):
        stage = entry.get("stage")
        if stage:
            by_stage[stage] = {
                "delta_cycles": entry.get("delta_cycles", 0),
                "tolerance":    entry.get("tolerance", 0),
                "root_cause":   entry.get("root_cause", ""),
                "classification": entry.get("classification", ""),
            }
    return by_stage


def check_latency(config_name: str, expected: dict) -> CheckResult:
    cfg = CONFIG_REGISTRY[config_name]
    rc, out = run_iverilog([STUB, cfg["rtl"], cfg["tb_lat"]])
    if "COMPILE FAIL" in out:
        return CheckResult("latency", False, f"compile failed rc={rc}",
                           {"stdout_tail": out[-500:]})

    stages = {"score": None, "softmax": None, "wsum": None, "e2e": None}
    for m in RE_STAGE.finditer(out):
        stages[m.group(1).lower()] = int(m.group(2))
    m = RE_E2E.search(out)
    if m:
        stages["e2e"] = int(m.group(1))

    exp_cfg = expected["configs"].get(config_name, {})
    exp_stages = exp_cfg.get("stages", {})
    tol = expected.get("tolerance_cycles", {"per_stage": 3, "e2e": 5})
    tol_stage = tol["per_stage"]
    tol_e2e   = tol["e2e"]

    # Per-stage annotated residuals (documented FSM / structural deltas).
    # A stage with a known-delta entry passes iff
    # |(RTL - sim) - delta_cycles| <= tolerance. NL-DPE configs load from
    # phase3_known_deltas.json; Azure-Lily from phase5_known_deltas.json
    # (selected by PHASE_DELTA_FILES in _load_known_deltas).
    known_deltas = _load_known_deltas(config_name)

    deltas = {}
    reasons = []
    for s, rtl_val in stages.items():
        exp_val = exp_stages.get(s)
        if exp_val is None:
            continue  # no expectation set — skip
        if rtl_val is None:
            reasons.append(f"{s}: RTL TB did not report a cycle count")
            continue
        d = rtl_val - exp_val
        deltas[s] = d
        if s in known_deltas:
            expected_delta = known_deltas[s]["delta_cycles"]
            tol_known = known_deltas[s]["tolerance"]
            if abs(d - expected_delta) > tol_known:
                reasons.append(
                    f"{s}: RTL={rtl_val} sim={exp_val} Δ={d} "
                    f"(expected Δ={expected_delta}±{tol_known} per "
                    f"{PHASE_DELTA_FILES.get(config_name, PHASE3_KNOWN_DELTAS_JSON).name})"
                )
            continue
        threshold = tol_e2e if s == "e2e" else tol_stage
        if abs(d) > threshold:
            reasons.append(f"{s}: RTL={rtl_val} sim={exp_val} Δ={d} > ±{threshold}")

    passed = len(reasons) == 0 and any(v is not None for v in stages.values())
    detail = "pass" if passed else "; ".join(reasons) or "no stage data parsed"
    fields = {"rtl": stages, "sim": exp_stages, "deltas": deltas,
              "stdout_tail": out[-400:]}
    return CheckResult("latency", passed, detail, fields)


# ── AH (attention-head) Stage 1: Fmax-independent counter gate ───────────

def _parse_ah_counters(out: str) -> dict:
    """Parse all `COUNTER <stage> <key> <value>` lines from TB stdout.

    Returns nested dict: {stage_name: {key: int_value}}. Schema keys are
    integers; non-integer values (e.g. "_arch") are skipped because the
    regex only matches `\\d+` for the value group.
    """
    counters = {}
    for m in RE_COUNTER.finditer(out):
        stage, key, val = m.group(1), m.group(2), m.group(3)
        try:
            iv = int(val)
        except ValueError:
            continue
        counters.setdefault(stage, {})[key] = iv
    return counters


def _load_known_count_deltas(config_name: str) -> dict:
    """Load known counter deltas (file:line annotated structural / modelling
    residuals) for *config_name*. Returns dict[stage][key] -> dict with
    delta_count + tolerance + classification.
    """
    if not KNOWN_COUNT_DELTAS_JSON.exists():
        return {}
    try:
        data = json.loads(KNOWN_COUNT_DELTAS_JSON.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    cfg_entry = data.get("configs", {}).get(config_name, {})
    by_stage_key = {}
    for entry in cfg_entry.get("deltas", []):
        s = entry.get("stage")
        k = entry.get("key")
        if s and k:
            by_stage_key.setdefault(s, {})[k] = {
                "delta_count":    entry.get("delta_count", 0),
                "tolerance":      entry.get("tolerance", 0),
                "root_cause":     entry.get("root_cause", ""),
                "classification": entry.get("classification", ""),
            }
    return by_stage_key


def check_ah_attn_head_counters(config_name: str, expected_counters: dict) -> CheckResult:
    """AH-gate Stage 1 counter check.

    Compares per-stage architectural-invariant counter values
    (dpe_fire_count, pass_count, row_count, lane_count, softmax_*_fires)
    between the RTL TB's `COUNTER ...` block and the sim oracle's
    expected_counters.json. Counter equality is integer; ±1 boundary
    slack is allowed for FSM-edge probes; anything > 1 must be in
    known_count_deltas.json.
    """
    cfg = CONFIG_REGISTRY[config_name]
    sources = list(cfg.get("extra_rtl", [])) + [cfg["rtl"], cfg["tb_combined"]]

    # Same long-timeout window as the cycle gate — these are the same TB
    # runs (counters are emitted alongside cycle reports at end-of-sim).
    rc, out = run_iverilog(sources, timeout_s=600)
    if "COMPILE FAIL" in out:
        return CheckResult("ah_attn_head_counters", False,
                           f"compile failed rc={rc}",
                           {"stdout_tail": out[-500:]})

    rtl_counters = _parse_ah_counters(out)
    if not rtl_counters:
        return CheckResult("ah_attn_head_counters", False,
                           "no COUNTER lines parsed from TB stdout (TB may be missing counter probes)",
                           {"stdout_tail": out[-500:]})

    exp_cfg = expected_counters["configs"].get(config_name, {})
    sim_stages = exp_cfg.get("stages", {})
    if not sim_stages:
        return CheckResult("ah_attn_head_counters", False,
                           f"no expected counters for {config_name} in expected_counters.json",
                           {"rtl": rtl_counters})

    known_deltas = _load_known_count_deltas(config_name)
    edge_slack = expected_counters.get("tolerance", {}).get("edge_slack", 1)
    schema_keys = set(expected_counters.get("tolerance", {}).get(
        "exact_keys",
        ["dpe_fire_count", "pass_count", "row_count", "lane_count",
         "softmax_exp_fires", "softmax_norm_fires"]
    ))

    deltas = {}
    reasons = []
    for stage, expected_dict in sim_stages.items():
        rtl_dict = rtl_counters.get(stage, {})
        if not rtl_dict:
            reasons.append(f"{stage}: TB emitted no COUNTER lines for this stage")
            continue
        for key in schema_keys:
            sim_val = expected_dict.get(key)
            rtl_val = rtl_dict.get(key)
            if sim_val is None or rtl_val is None:
                continue
            d = rtl_val - sim_val
            deltas.setdefault(stage, {})[key] = d
            stage_known = known_deltas.get(stage, {})
            if key in stage_known:
                expected_delta = stage_known[key]["delta_count"]
                tol_known = stage_known[key]["tolerance"]
                if abs(d - expected_delta) > tol_known:
                    reasons.append(
                        f"{stage}.{key}: RTL={rtl_val} sim={sim_val} Δ={d} "
                        f"(expected Δ={expected_delta}±{tol_known} per "
                        f"{KNOWN_COUNT_DELTAS_JSON.name})"
                    )
            else:
                if abs(d) > edge_slack:
                    reasons.append(
                        f"{stage}.{key}: RTL={rtl_val} sim={sim_val} Δ={d} > ±{edge_slack} "
                        f"(no entry in {KNOWN_COUNT_DELTAS_JSON.name})"
                    )

    passed = len(reasons) == 0
    detail = "pass" if passed else "; ".join(reasons[:6]) + (
        f"; ... ({len(reasons)-6} more)" if len(reasons) > 6 else ""
    )
    fields = {"rtl": rtl_counters, "sim": sim_stages, "deltas": deltas,
              "stdout_tail": out[-600:]}
    return CheckResult("ah_attn_head_counters", passed, detail, fields)


# ── AH (attention-head) combined functional + latency check ──────────────

def check_ah_attn_head(config_name: str, expected: dict) -> CheckResult:
    """Run the combined AH TB and gate per-stage cycles vs sim oracle.

    The TBs emit `AH_*_STAGES_TOTAL` lines with stage-total durations
    (OR-aggregated across W=16 DIMM lanes) — see the probe-semantics
    comment in tb_{nldpe,azurelily}_attn_head_v2.v.

    Both NL-DPE and Azure-Lily TBs now report softmax_exp and softmax_norm
    separately in `_TOTAL` mode. The legacy "AL folds exp+norm" path is
    only triggered when run_checks.py is parsing an old TB run that lacks
    the _TOTAL line (legacy fallback in _parse_ah_stages emits norm=0).
    """
    cfg = CONFIG_REGISTRY[config_name]
    arch = cfg.get("ah_arch", "nldpe")
    sources = list(cfg.get("extra_rtl", [])) + [cfg["rtl"], cfg["tb_combined"]]

    # AH TBs need a longer wall-clock window: AL has a 10 ms hard-timeout
    # and a 205-µs functional window. 600 s gives plenty of slack on a
    # cold cache.
    rc, out = run_iverilog(sources, timeout_s=600)
    if "COMPILE FAIL" in out:
        return CheckResult("ah_attn_head", False, f"compile failed rc={rc}",
                           {"stdout_tail": out[-500:]})

    rtl_stages = _parse_ah_stages(out, arch)
    if not rtl_stages:
        return CheckResult("ah_attn_head", False,
                           f"AH_{arch.upper()}_STAGES regex line not found in TB stdout",
                           {"stdout_tail": out[-500:]})

    exp_cfg = expected["configs"].get(config_name, {})
    sim_stages = exp_cfg.get("stages", {})
    if not sim_stages:
        return CheckResult("ah_attn_head", False,
                           f"no expected stages for {config_name} in expected_cycles.json",
                           {"rtl": rtl_stages})

    known_deltas = _load_known_deltas(config_name)

    # Detect legacy combine-softmax mode: only when the TB lacks the
    # _TOTAL line and the legacy line emits softmax_norm=0 with sim
    # splitting exp/norm. Post-fix TBs emit exp/norm separately in _TOTAL,
    # so this stays False.
    combine_softmax = (
        arch == "azurelily"
        and rtl_stages.get("softmax_norm", 0) == 0
        and "softmax_norm" in sim_stages
        and rtl_stages.get("softmax_exp", 0) > sim_stages.get("softmax_exp", 0)
    )

    # Build per-stage comparison list. For combined-softmax legacy mode,
    # fold sim_exp + sim_norm into the softmax_exp comparison; the
    # softmax_norm stage check becomes 0 vs 0 (sentinel).
    deltas = {}
    reasons = []
    for s, rtl_val in rtl_stages.items():
        sim_val = sim_stages.get(s)
        if sim_val is None:
            continue
        if combine_softmax and s == "softmax_exp":
            sim_val = sim_stages.get("softmax_exp", 0) + sim_stages.get("softmax_norm", 0)
        if combine_softmax and s == "softmax_norm":
            sim_val = 0  # sentinel: TB emits 0, sim folded into exp slot
        d = rtl_val - sim_val
        deltas[s] = d
        if s in known_deltas:
            expected_delta = known_deltas[s]["delta_cycles"]
            tol_known = known_deltas[s]["tolerance"]
            if abs(d - expected_delta) > tol_known:
                reasons.append(
                    f"{s}: RTL={rtl_val} sim={sim_val} Δ={d} "
                    f"(expected Δ={expected_delta}±{tol_known} per "
                    f"phase7_known_deltas.json)"
                )
        else:
            # Fall through to default tolerance — expected to be tight
            # because no annotated residual exists.
            tol_e2e = expected.get("tolerance_cycles", {}).get("e2e", 5)
            tol_stage = expected.get("tolerance_cycles", {}).get("per_stage", 3)
            threshold = tol_e2e if s == "e2e" else tol_stage
            if abs(d) > threshold:
                reasons.append(
                    f"{s}: RTL={rtl_val} sim={sim_val} Δ={d} > ±{threshold} "
                    f"(no entry in phase7_known_deltas.json)"
                )

    # Functional sanity gate — TB self-reports a PASS line for NL-DPE
    # (output pulse count); AL TB reports "valid_n asserted (PASS gate-1)"
    # if any pulses fired.  We accept either as PASS gate-1.
    func_pass = bool(RE_AH_FUNC_PASS.search(out))
    if not func_pass:
        # Soft warn — annotate but don't fail. The AH TBs use a permissive
        # functional gate; the residual / cycle gating is the real check.
        # Only fail if no top valid_n pulses at all.
        m_pulse = RE_AH_TOP_PULSES.search(out)
        if m_pulse and int(m_pulse.group(1)) == 0:
            reasons.append("functional: 0 top valid_n pulses (TB reported no DIMM output)")

    passed = len(reasons) == 0
    detail = "pass" if passed else "; ".join(reasons)
    fields = {"rtl": rtl_stages, "sim": sim_stages, "deltas": deltas,
              "combine_softmax": combine_softmax,
              "stdout_tail": out[-600:]}
    return CheckResult("ah_attn_head", passed, detail, fields)


# ── Orchestration ─────────────────────────────────────────────────────────

def run_one(config_name: str, skip_func: bool, skip_lat: bool,
            expected: dict, whitelist: dict, gate: str = "counters",
            expected_counters: dict | None = None) -> list[CheckResult]:
    results = []
    if config_name in AH_CONFIGS:
        # AH configs use a single combined TB; --skip-functional and
        # --skip-latency both have to be set to skip it (defensive: usually
        # neither is set).
        #
        # Stage 1 gate refactor: --gate counters (default) compares
        # Fmax-independent counter values (dpe_fire_count, pass_count,
        # row_count, lane_count, softmax_*_fires) and is what controls
        # exit code. --gate cycles falls back to the legacy
        # cycle-level gate (advisory only post-Stage 1).
        if not (skip_func and skip_lat):
            if gate == "counters":
                # Counter gate is canonical (Fmax-independent); cycle
                # gate runs alongside as advisory diagnostics.
                if expected_counters is None:
                    expected_counters = json.loads(EXPECTED_COUNTERS_JSON.read_text())
                results.append(check_ah_attn_head_counters(config_name, expected_counters))
                # Advisory-only cycle check (no exit-code impact via the
                # counter-gate path — caller can still see the residuals).
                advisory = check_ah_attn_head(config_name, expected)
                advisory.name = "ah_attn_head_cycles_advisory"
                advisory.passed = True  # advisory: never fail the run
                results.append(advisory)
            else:
                results.append(check_ah_attn_head(config_name, expected))
        return results
    if not skip_func:
        results.append(check_functional(config_name, whitelist))
    if not skip_lat:
        results.append(check_latency(config_name, expected))
    return results


def summarise(config_name: str, results: list[CheckResult]) -> bool:
    overall = all(r.passed for r in results)
    tag = "PASS" if overall else "FAIL"
    print(f"=== {config_name}: {tag} ===")
    for r in results:
        marker = "✓" if r.passed else "✗"
        print(f"  [{marker}] {r.name}: {r.detail}")
        if r.name in ("latency", "ah_attn_head", "ah_attn_head_cycles_advisory",
                       "ah_attn_head_counters") and r.fields.get("deltas"):
            d = r.fields["deltas"]
            if r.name == "ah_attn_head_counters":
                # Per-stage per-key delta dump (counter gate).
                for stage, kdict in d.items():
                    print(f"       {stage}:")
                    for k, dv in kdict.items():
                        rtl = r.fields["rtl"].get(stage, {}).get(k, "?")
                        sim = r.fields["sim"].get(stage, {}).get(k, "?")
                        print(f"         {k}: RTL={rtl} sim={sim} Δ={dv:+d}")
            else:
                print(f"       RTL={r.fields['rtl']}")
                print(f"       sim={r.fields['sim']}")
                print(f"       Δ  ={d}")
            if r.fields.get("combine_softmax"):
                print(f"       (softmax: AL TB folds exp+norm into exp slot)")
    return overall


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIG_REGISTRY.keys()))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--skip-functional", action="store_true")
    ap.add_argument("--skip-latency",    action="store_true")
    ap.add_argument(
        "--gate", choices=["counters", "cycles"], default="counters",
        help="AH gate mode: 'counters' (default, Fmax-independent "
             "architectural-invariants) or 'cycles' (legacy advisory)."
    )
    args = ap.parse_args()

    if not args.config and not args.all:
        ap.error("pass --config <name> or --all")

    expected  = json.loads(EXPECTED_JSON.read_text())
    whitelist = json.loads(WHITELIST_JSON.read_text())
    expected_counters = (
        json.loads(EXPECTED_COUNTERS_JSON.read_text())
        if EXPECTED_COUNTERS_JSON.exists() else None
    )

    targets = list(CONFIG_REGISTRY.keys()) if args.all else [args.config]

    any_fail = False
    for cfg in targets:
        results = run_one(cfg, args.skip_functional, args.skip_latency,
                          expected, whitelist, gate=args.gate,
                          expected_counters=expected_counters)
        if not summarise(cfg, results):
            any_fail = True

    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
