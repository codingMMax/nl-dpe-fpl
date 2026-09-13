"""Ad-hoc verification for Config.patch() and Config.from_dict().

Run from repo root:
    python azurelily/IMC/test_config_runtime.py
"""
import json
import os
import subprocess
import sys

# Make `imc_core` importable regardless of CWD.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

from imc_core.config import Config

NL_DPE_JSON = os.path.join(HERE, "configs", "nl_dpe.json")


def main():
    # ---------------------------------------------------------------
    # Test 1: original JSON loading still works
    # ---------------------------------------------------------------
    cfg = Config(NL_DPE_JSON)
    assert cfg.rows == 256, f"expected rows=256, got {cfg.rows}"
    assert cfg.cols == 256, f"expected cols=256, got {cfg.cols}"
    assert cfg.imc == "NL-DPE"
    assert cfg.core_freq_MHz == 1000
    # NL-DPE has scale_with_geometry default False; t_conv_ns = 1 * 1.0 = 1.0
    assert abs(cfg.t_conv_ns - 1.0) < 1e-9, f"t_conv_ns={cfg.t_conv_ns}"
    print("Test 1 PASS: JSON loading backward compatible.")

    # ---------------------------------------------------------------
    # Test 2: patch updates fields and recomputes derived
    # ---------------------------------------------------------------
    old_core_cycle_ns = cfg.core_cycle_ns
    cfg.patch(core_freq_MHz=1500)
    assert cfg.core_freq_MHz == 1500
    assert cfg.core_cycle_ns != old_core_cycle_ns
    assert abs(cfg.core_cycle_ns - 1.5) < 1e-6, (
        f"expected core_cycle_ns=1.5, got {cfg.core_cycle_ns}"
    )
    # Derived t_*_ns should also have updated.
    assert abs(cfg.t_conv_ns - 1.5) < 1e-6, f"t_conv_ns={cfg.t_conv_ns}"
    assert abs(cfg.t_analoge_ns - 1.5) < 1e-6, f"t_analoge_ns={cfg.t_analoge_ns}"
    assert abs(cfg.t_digital_ns - 1.5) < 1e-6, f"t_digital_ns={cfg.t_digital_ns}"
    print("Test 2 PASS: patch() updates and recomputes derived attrs.")

    # ---------------------------------------------------------------
    # Test 2b: patch geometry and verify recomputation paths
    # ---------------------------------------------------------------
    cfg_b = Config(NL_DPE_JSON)
    cfg_b.patch(rows=512, cols=128)
    assert cfg_b.rows == 512
    assert cfg_b.cols == 128
    # scale_with_geometry is False for nl_dpe.json, so e_analoge_pj should
    # equal the base value (3.89), unchanged.
    assert abs(cfg_b.e_analoge_pj - 3.89) < 1e-9, f"e_analoge_pj={cfg_b.e_analoge_pj}"
    # Now flip scale_with_geometry on and confirm e_analoge_pj scales by cols.
    cfg_b.patch(scale_with_geometry=True)
    assert abs(cfg_b.e_analoge_pj - 3.89 * 128) < 1e-9, (
        f"scaled e_analoge_pj={cfg_b.e_analoge_pj}"
    )
    print("Test 2b PASS: patch() honors scale_with_geometry recompute path.")

    # ---------------------------------------------------------------
    # Test 3: patch with unknown attr raises AttributeError
    # ---------------------------------------------------------------
    raised = False
    try:
        cfg.patch(nonexistent_field=42)
    except AttributeError:
        raised = True
    assert raised, "patch() should raise AttributeError for unknown attr"
    print("Test 3 PASS: unknown attr raises AttributeError.")

    # ---------------------------------------------------------------
    # Test 4: from_dict construction works
    # ---------------------------------------------------------------
    with open(NL_DPE_JSON, "r") as f:
        d = json.load(f)
    cfg2 = Config.from_dict(d)
    assert cfg2.rows == 256
    assert cfg2.cols == 256
    assert cfg2.imc == "NL-DPE"
    assert cfg2.core_freq_MHz == 1000
    # Verify a derived attr matches the JSON-loaded version exactly.
    cfg_ref = Config(NL_DPE_JSON)
    assert abs(cfg2.t_conv_ns - cfg_ref.t_conv_ns) < 1e-12
    assert abs(cfg2.act_energy_pj_per_op - cfg_ref.act_energy_pj_per_op) < 1e-12
    print("Test 4 PASS: from_dict() constructs equivalent Config.")

    # ---------------------------------------------------------------
    # Test 4b: __init__ rejects both / neither input
    # ---------------------------------------------------------------
    raised = False
    try:
        Config()  # neither
    except ValueError:
        raised = True
    assert raised, "Config() with no args should raise ValueError"

    raised = False
    try:
        Config(NL_DPE_JSON, data_dict=d)  # both
    except ValueError:
        raised = True
    assert raised, "Config() with both args should raise ValueError"
    print("Test 4b PASS: __init__ enforces exactly-one-input.")

    # ---------------------------------------------------------------
    # Test 5: re-run the existing sanity check harness
    # ---------------------------------------------------------------
    result = subprocess.run(
        ["python", "azurelily/IMC/test.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("---- stdout ----")
        print(result.stdout)
        print("---- stderr ----")
        print(result.stderr)
    assert result.returncode == 0, f"test.py exited {result.returncode}"
    print("Test 5 PASS: existing sanity harness exit 0.")

    print("\nAll Config runtime override tests PASS")


if __name__ == "__main__":
    main()
