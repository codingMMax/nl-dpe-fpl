# DPE behavior model + FC top + TB build/run harness.
#
# Usage examples:
#   make help                                          # show all targets and variables
#   make tb-vmm-nldpe                                  # NL-DPE VMM, defaults (R=256 C=256 P=8)
#   make tb-vmm-al R=1024 C=128                        # AL VMM with custom geometry
#   make tb-vmm-nldpe PRECISION=4                      # NL-DPE INT4
#   make tb-acam                                       # NL-DPE ACAM exp mode
#   make tb-dsp-mac K=128                              # AL DSP-MAC, K=128
#   make tb-fc-nldpe M=1 K=128 N=128                   # fc_top NL-DPE single-fire
#   make tb-fc-nldpe M=4 K=256 N=256 ACTIVATION=0      # fc_top NL-DPE batched
#   make tb-fc-al M=1 K=128 N=128 ACTIVATION=1         # fc_top AL with ReLU
#   make fc-smoke                                      # full Stage 1A FC sweep
#   make all                                           # all four primitive TBs at defaults
#   make regen                                         # regenerate dpe_*.v from configs
#   make smoke                                         # full primitive smoke sweep
#   make clean                                         # remove tmp binaries
#
# Variables (override on command line, e.g. `make tb-vmm-nldpe R=1024 PRECISION=16`):

# DPE geometry / DPE-axiom (per-arch defaults below; override these as needed)
R              ?=
C              ?=
BUF            ?=
PRECISION      ?= 8
PIPELINE_DEPTH ?= 3   # legacy single-knob default (pre-Task #86)
ACAM_CYCLES    ?= 0   # legacy single-knob default (pre-Task #86)

# DSP-MAC knobs
K              ?= 64
DSP_WIDTH      ?= 4

# FC top-level knobs
M              ?= 1
N              ?= 128
ACTIVATION     ?= 0  # 0 = raw GEMM, 1 = FC + ReLU

# Per-arch defaults
NLDPE_R   ?= 256
NLDPE_C   ?= 256
NLDPE_BUF ?= 40
# Per-arch CCYC decomposition (Task #86):
#   CCYC = PRECISION + (PIPELINE_DEPTH - 1) + ACAM_CYCLES
# NL-DPE     : 2-stage pipeline (MAC, Acc) + 1-cycle ACAM read-out
#              → PD=2, AC=1  → CCYC = P + 2
# Azure-Lily : 3-stage pipeline (MAC, ADC, SA), no ACAM
#              → PD=3, AC=0  → CCYC = P + 2
# The per-arch defaults below are used when the user does NOT pass
# PIPELINE_DEPTH / ACAM_CYCLES on the make CLI. When PIPELINE_DEPTH
# is overridden on the CLI, the override takes priority (legacy
# single-knob compat).
NLDPE_PIPELINE_DEPTH ?= 2
NLDPE_ACAM_CYCLES    ?= 1
AL_PIPELINE_DEPTH    ?= 3
AL_ACAM_CYCLES       ?= 0

AL_R   ?= 512
AL_C   ?= 128
AL_BUF ?= 16

DSP_BUF ?= 16

# Optional: dump VCD waveform if WAVE=1
WAVE ?= 0

# ── paths ─────────────────────────────────────────────────────────────────
REPO   := $(shell git rev-parse --show-toplevel 2>/dev/null || echo $(CURDIR)/..)
RTL    := $(REPO)/fc_verification/rtl
TB_DIR := $(REPO)/fc_verification
BIN    := $(REPO)/fc_verification/build
GEN    := $(REPO)/nl_dpe

# Resolve effective param values (per-arch fallback when R/C/BUF not given)
NLDPE_R_EFF   := $(if $(R),$(R),$(NLDPE_R))
NLDPE_C_EFF   := $(if $(C),$(C),$(NLDPE_C))
NLDPE_BUF_EFF := $(if $(BUF),$(BUF),$(NLDPE_BUF))

AL_R_EFF   := $(if $(R),$(R),$(AL_R))
AL_C_EFF   := $(if $(C),$(C),$(AL_C))
AL_BUF_EFF := $(if $(BUF),$(BUF),$(AL_BUF))

DSP_BUF_EFF := $(if $(BUF),$(BUF),$(DSP_BUF))

# Per-arch CCYC decomposition (Task #86): when the user passes
# PIPELINE_DEPTH on the CLI, treat it as the LEGACY single-knob
# (CCYC = P + PD - 1) and force ACAM_CYCLES=0 unless ACAM_CYCLES
# is ALSO passed. This preserves pre-Task-#86 behaviour when callers
# write `make tb-vmm-nldpe PIPELINE_DEPTH=3` (legacy intent: CCYC=10).
# When neither override is given, use per-arch decomposition.
ifeq ($(origin PIPELINE_DEPTH),command line)
  # User passed PIPELINE_DEPTH explicitly: legacy single-knob mode.
  NLDPE_PIPELINE_DEPTH_EFF := $(PIPELINE_DEPTH)
  AL_PIPELINE_DEPTH_EFF    := $(PIPELINE_DEPTH)
  ifeq ($(origin ACAM_CYCLES),command line)
    NLDPE_ACAM_CYCLES_EFF := $(ACAM_CYCLES)
    AL_ACAM_CYCLES_EFF    := $(ACAM_CYCLES)
  else
    # Legacy single-knob: ACAM_CYCLES=0 by default.
    NLDPE_ACAM_CYCLES_EFF := 0
    AL_ACAM_CYCLES_EFF    := 0
  endif
else ifeq ($(origin ACAM_CYCLES),command line)
  # User passed ACAM_CYCLES alone: keep per-arch PIPELINE_DEPTH, override ACAM.
  NLDPE_PIPELINE_DEPTH_EFF := $(NLDPE_PIPELINE_DEPTH)
  AL_PIPELINE_DEPTH_EFF    := $(AL_PIPELINE_DEPTH)
  NLDPE_ACAM_CYCLES_EFF    := $(ACAM_CYCLES)
  AL_ACAM_CYCLES_EFF       := $(ACAM_CYCLES)
else
  # Neither overridden: per-arch defaults (Task #86 principled mode).
  NLDPE_PIPELINE_DEPTH_EFF := $(NLDPE_PIPELINE_DEPTH)
  AL_PIPELINE_DEPTH_EFF    := $(AL_PIPELINE_DEPTH)
  NLDPE_ACAM_CYCLES_EFF    := $(NLDPE_ACAM_CYCLES)
  AL_ACAM_CYCLES_EFF       := $(AL_ACAM_CYCLES)
endif

# Common iverilog flags
IV_FLAGS := -g2005

# Optional VCD waveform dump (`make tb-vmm-nldpe WAVE=1`)
ifeq ($(WAVE),1)
VCD_DEFINES := -DWAVE=1
else
VCD_DEFINES :=
endif

# Per-arch DPE TB define sets (Task #86: per-arch PIPELINE_DEPTH /
# ACAM_CYCLES decomposition).
NLDPE_DPE_DEFS = -DR_TB=$(1) -DC_TB=$(2) -DBUF_TB=$(3) \
                 -DPRECISION_TB=$(PRECISION) \
                 -DPIPELINE_DEPTH_TB=$(NLDPE_PIPELINE_DEPTH_EFF) \
                 -DACAM_CYCLES_TB=$(NLDPE_ACAM_CYCLES_EFF)

AL_DPE_DEFS    = -DR_TB=$(1) -DC_TB=$(2) -DBUF_TB=$(3) \
                 -DPRECISION_TB=$(PRECISION) \
                 -DPIPELINE_DEPTH_TB=$(AL_PIPELINE_DEPTH_EFF) \
                 -DACAM_CYCLES_TB=$(AL_ACAM_CYCLES_EFF)

# DSP-MAC define set (no PIPELINE_DEPTH / ACAM_CYCLES — DSP-MAC's
# CCYC = ceil(K / DSP_WIDTH) is K-driven, not bit-serial-pipeline driven).
DSP_DEFS = -DK_TB=$(K) -DBUF_TB=$(DSP_BUF_EFF) -DDSP_WIDTH_TB=$(DSP_WIDTH)

# FC top-level define set: shared between NL-DPE and AL flavours.
# FC's K knob defaults to K (shared with DSP-MAC) but the user is
# expected to pass it explicitly; same for N. M defaults to 1.
FC_DEFS_NLDPE = -DARCH_NLDPE -DFC_TOP_ARCH_NLDPE \
                -DM_TB=$(M) -DK_TB=$(K) -DN_TB=$(N) \
                -DR_TB=$(NLDPE_R_EFF) -DC_TB=$(NLDPE_C_EFF) -DBUF_TB=$(NLDPE_BUF_EFF) \
                -DPRECISION_TB=$(PRECISION) \
                -DPIPELINE_DEPTH_TB=$(NLDPE_PIPELINE_DEPTH_EFF) \
                -DACAM_CYCLES_TB=$(NLDPE_ACAM_CYCLES_EFF) \
                -DACTIVATION_TB=$(ACTIVATION)

FC_DEFS_AL    = -DARCH_AL \
                -DM_TB=$(M) -DK_TB=$(K) -DN_TB=$(N) \
                -DR_TB=$(AL_R_EFF) -DC_TB=$(AL_C_EFF) -DBUF_TB=$(AL_BUF_EFF) \
                -DPRECISION_TB=$(PRECISION) \
                -DPIPELINE_DEPTH_TB=$(AL_PIPELINE_DEPTH_EFF) \
                -DACAM_CYCLES_TB=$(AL_ACAM_CYCLES_EFF) \
                -DACTIVATION_TB=$(ACTIVATION)

.PHONY: help all clean regen smoke tb-vmm-nldpe tb-vmm-al tb-acam tb-dsp-mac \
        tb-fc-nldpe tb-fc-al fc-smoke tb-faithful-nldpe faithful-vectors faithful-smoke \
        tb-faithful-al faithful-vectors-al faithful-smoke-al

# ── default target ────────────────────────────────────────────────────────
help:
	@echo "DPE behavior model + FC top + TB harness"
	@echo ""
	@echo "Primitive TB targets:"
	@echo "  make tb-vmm-nldpe   NL-DPE VMM TB (override: R, C, BUF, PRECISION, PIPELINE_DEPTH)"
	@echo "  make tb-vmm-al      Azure-Lily VMM TB (same overrides)"
	@echo "  make tb-acam        NL-DPE ACAM exp mode TB"
	@echo "  make tb-dsp-mac     AL DSP-MAC TB (override: K, DSP_WIDTH, BUF)"
	@echo "  make all            run all 4 TBs at defaults"
	@echo ""
	@echo "FC top-level TB targets (Stage 1A: V=1, H=1):"
	@echo "  make tb-fc-nldpe    fc_top + NL-DPE primitive (override: M, K, N, ACTIVATION)"
	@echo "  make tb-fc-al       fc_top + Azure-Lily primitive (same overrides)"
	@echo "  make fc-smoke       full Stage 1A FC sweep via run_fc_smoke.py"
	@echo ""
	@echo "Faithful NL-DPE primitive (Task #91) -- emergent cycle count:"
	@echo "  make tb-faithful-nldpe TEST_MODE=N    Run faithful TB at TEST_MODE in 1..7"
	@echo "  make faithful-vectors                  Regenerate .mem test vectors from numpy oracle"
	@echo "  make faithful-smoke                    Run all 7 TEST_MODE cases sequentially"
	@echo ""
	@echo "Faithful Azure-Lily primitive (Task #92) -- emergent cycle count (3-stage, no ACAM):"
	@echo "  make tb-faithful-al TEST_MODE=N        Run AL faithful TB at TEST_MODE in 1..5"
	@echo "  make faithful-vectors-al               Regenerate .mem test vectors from AL numpy oracle"
	@echo "  make faithful-smoke-al                 Run all 5 TEST_MODE cases sequentially"
	@echo ""
	@echo "Misc:"
	@echo "  make regen          regenerate dpe_*.v from per-arch JSON configs"
	@echo "  make smoke          full primitive smoke sweep via run_dpe_smoke.py"
	@echo "  make clean          remove fc_verification/build/"
	@echo ""
	@echo "Variables:"
	@echo "  R, C, BUF                 override DPE geometry (per-arch defaults if unset)"
	@echo "  PRECISION                 INT precision (default 8)"
	@echo "  PIPELINE_DEPTH            (Task #86 legacy single-knob; if not"
	@echo "                              passed, per-arch defaults apply)"
	@echo "  ACAM_CYCLES               (Task #86 legacy single-knob; if not"
	@echo "                              passed, per-arch defaults apply)"
	@echo "  Per-arch defaults (Task #86):"
	@echo "    NLDPE_PIPELINE_DEPTH=2, NLDPE_ACAM_CYCLES=1   (CCYC = P + 2)"
	@echo "    AL_PIPELINE_DEPTH=3,    AL_ACAM_CYCLES=0      (CCYC = P + 2)"
	@echo "  K, DSP_WIDTH              DSP-MAC parameters (defaults 64, 4)"
	@echo "  M, K, N, ACTIVATION       FC matmul shape + activation (defaults 1, 64, 128, 0)"
	@echo "  WAVE=1                    dump VCD to build/<tb>.vcd"
	@echo ""
	@echo "Examples:"
	@echo "  make tb-vmm-nldpe                            # defaults (R=256 C=256 P=8)"
	@echo "  make tb-vmm-nldpe R=1024 C=128 PRECISION=4   # override"
	@echo "  make tb-vmm-al BUF=40                        # AL with NL-style buf"
	@echo "  make tb-dsp-mac K=128                        # bigger K"
	@echo "  make tb-fc-nldpe M=1 K=128 N=128             # bert_qkv_proj on NL-DPE"
	@echo "  make tb-fc-nldpe M=4 K=256 N=256             # batched GEMM on NL-DPE"
	@echo "  make tb-fc-al M=1 K=128 N=128 ACTIVATION=1   # AL with ReLU"

$(BIN):
	@mkdir -p $@

# ── NL-DPE VMM ────────────────────────────────────────────────────────────
tb-vmm-nldpe: | $(BIN)
	@echo "[make] tb-vmm-nldpe  R=$(NLDPE_R_EFF) C=$(NLDPE_C_EFF) BUF=$(NLDPE_BUF_EFF) PRECISION=$(PRECISION) PIPELINE_DEPTH=$(NLDPE_PIPELINE_DEPTH_EFF) ACAM_CYCLES=$(NLDPE_ACAM_CYCLES_EFF)"
	iverilog $(IV_FLAGS) -DARCH_NLDPE $(call NLDPE_DPE_DEFS,$(NLDPE_R_EFF),$(NLDPE_C_EFF),$(NLDPE_BUF_EFF)) $(VCD_DEFINES) \
	  -o $(BIN)/tb_vmm_nldpe \
	  $(RTL)/dpe_nldpe.v $(TB_DIR)/tb_dpe_vmm.v
	vvp $(BIN)/tb_vmm_nldpe

# ── Azure-Lily VMM ────────────────────────────────────────────────────────
tb-vmm-al: | $(BIN)
	@echo "[make] tb-vmm-al     R=$(AL_R_EFF) C=$(AL_C_EFF) BUF=$(AL_BUF_EFF) PRECISION=$(PRECISION) PIPELINE_DEPTH=$(AL_PIPELINE_DEPTH_EFF) ACAM_CYCLES=$(AL_ACAM_CYCLES_EFF)"
	iverilog $(IV_FLAGS) -DARCH_AL $(call AL_DPE_DEFS,$(AL_R_EFF),$(AL_C_EFF),$(AL_BUF_EFF)) $(VCD_DEFINES) \
	  -o $(BIN)/tb_vmm_al \
	  $(RTL)/dpe_azurelily.v $(TB_DIR)/tb_dpe_vmm.v
	vvp $(BIN)/tb_vmm_al

# ── NL-DPE ACAM exp mode ──────────────────────────────────────────────────
tb-acam: | $(BIN)
	@echo "[make] tb-acam       R=$(NLDPE_R_EFF) C=$(NLDPE_C_EFF) BUF=$(NLDPE_BUF_EFF) PRECISION=$(PRECISION) PIPELINE_DEPTH=$(NLDPE_PIPELINE_DEPTH_EFF) ACAM_CYCLES=$(NLDPE_ACAM_CYCLES_EFF)"
	iverilog $(IV_FLAGS) $(call NLDPE_DPE_DEFS,$(NLDPE_R_EFF),$(NLDPE_C_EFF),$(NLDPE_BUF_EFF)) $(VCD_DEFINES) \
	  -o $(BIN)/tb_acam \
	  $(RTL)/dpe_nldpe.v $(TB_DIR)/tb_dpe_acam.v
	vvp $(BIN)/tb_acam

# ── AL DSP-MAC ────────────────────────────────────────────────────────────
tb-dsp-mac: | $(BIN)
	@echo "[make] tb-dsp-mac    K=$(K) BUF=$(DSP_BUF_EFF) DSP_WIDTH=$(DSP_WIDTH)"
	iverilog $(IV_FLAGS) $(DSP_DEFS) $(VCD_DEFINES) \
	  -o $(BIN)/tb_dsp_mac \
	  $(RTL)/dsp_mac.v $(TB_DIR)/tb_dsp_mac.v
	vvp $(BIN)/tb_dsp_mac

# ── FC top NL-DPE ─────────────────────────────────────────────────────────
# Task #97: fc_top now binds the faithful primitive (dpe_nldpe_faithful.v),
# which models the silicon-faithful single-substrate slice with LOAD-gated
# cadence (T_steady = LCYC + PRECISION). The lazy primitive (dpe_nldpe.v)
# is kept for the primitive smoke harness (run_dpe_smoke.py) only.
tb-fc-nldpe: | $(BIN)
	@echo "[make] tb-fc-nldpe   M=$(M) K=$(K) N=$(N) ACTIVATION=$(ACTIVATION)"
	@echo "[make]               R=$(NLDPE_R_EFF) C=$(NLDPE_C_EFF) BUF=$(NLDPE_BUF_EFF) PRECISION=$(PRECISION)"
	iverilog $(IV_FLAGS) $(FC_DEFS_NLDPE) $(VCD_DEFINES) \
	  -o $(BIN)/tb_fc_nldpe \
	  $(RTL)/dpe_nldpe_faithful.v $(RTL)/fc_top.v $(TB_DIR)/tb_fc.v
	vvp $(BIN)/tb_fc_nldpe

# ── FC top Azure-Lily ─────────────────────────────────────────────────────
# Task #97: faithful AL primitive (dpe_azurelily_faithful.v).
tb-fc-al: | $(BIN)
	@echo "[make] tb-fc-al      M=$(M) K=$(K) N=$(N) ACTIVATION=$(ACTIVATION)"
	@echo "[make]               R=$(AL_R_EFF) C=$(AL_C_EFF) BUF=$(AL_BUF_EFF) PRECISION=$(PRECISION)"
	iverilog $(IV_FLAGS) $(FC_DEFS_AL) $(VCD_DEFINES) \
	  -o $(BIN)/tb_fc_al \
	  $(RTL)/dpe_azurelily_faithful.v $(RTL)/fc_top.v $(TB_DIR)/tb_fc.v
	vvp $(BIN)/tb_fc_al

# ── full FC smoke sweep ───────────────────────────────────────────────────
fc-smoke:
	python3 $(REPO)/fc_verification/run_fc_smoke.py

# ── all four at defaults ──────────────────────────────────────────────────
all: tb-vmm-nldpe tb-vmm-al tb-acam tb-dsp-mac

# ── regenerate behavior model .v files from JSON configs ──────────────────
regen:
	@echo "[make] regen dpe_*.v"
	python3 $(GEN)/gen_dpe_stub.py
	python3 $(GEN)/gen_dsp_mac.py

# ── full smoke sweep (delegates to run_dpe_smoke.py) ──────────────────────
smoke:
	python3 $(REPO)/fc_verification/run_dpe_smoke.py

# ── Faithful NL-DPE primitive (Task #91) ─────────────────────────────────
# TEST_MODE selects the test case (1..7). The numpy oracle pre-generates
# .mem files under fc_verification/oracles/test_vectors/ -- call
# `make faithful-vectors` first if you've edited the oracle.
TEST_MODE ?= 1
VECT_DIR  := $(REPO)/fc_verification/oracles/test_vectors/

tb-faithful-nldpe: | $(BIN)
	@echo "[make] tb-faithful-nldpe TEST_MODE=$(TEST_MODE)"
	iverilog $(IV_FLAGS) -DTEST_MODE=$(TEST_MODE) -DVECT_DIR=\"$(VECT_DIR)\" $(VCD_DEFINES) \
	  -o $(BIN)/tb_faithful_nldpe \
	  $(RTL)/dpe_nldpe_faithful.v $(TB_DIR)/tb_dpe_nldpe_faithful.v
	vvp $(BIN)/tb_faithful_nldpe

# Regenerate test vectors from the numpy oracle. Run after editing
# fc_verification/oracles/nldpe_mac_oracle.py.
faithful-vectors:
	@echo "[make] regen faithful test vectors via numpy oracle"
	python3 $(REPO)/fc_verification/oracles/nldpe_mac_oracle.py --gen-test-vectors

# Run all 7 TEST_MODE cases (sequential smoke gate for Task #91).
faithful-smoke: | $(BIN)
	@echo "[make] faithful-smoke (7/7 TEST_MODE sweep)"
	@for tm in 1 2 3 4 5 6 7; do \
	    $(MAKE) -s tb-faithful-nldpe TEST_MODE=$$tm || exit 1; \
	done

# ── Faithful Azure-Lily primitive (Task #92) ─────────────────────────────
# TEST_MODE selects the test case (1..5). The numpy oracle pre-generates
# .mem files under fc_verification/oracles/test_vectors_al/ -- call
# `make faithful-vectors-al` first if you've edited the oracle.
VECT_DIR_AL := $(REPO)/fc_verification/oracles/test_vectors_al/

tb-faithful-al: | $(BIN)
	@echo "[make] tb-faithful-al TEST_MODE=$(TEST_MODE)"
	iverilog $(IV_FLAGS) -DTEST_MODE=$(TEST_MODE) -DVECT_DIR=\"$(VECT_DIR_AL)\" $(VCD_DEFINES) \
	  -o $(BIN)/tb_faithful_al \
	  $(RTL)/dpe_azurelily_faithful.v $(TB_DIR)/tb_dpe_azurelily_faithful.v
	vvp $(BIN)/tb_faithful_al

# Regenerate AL faithful test vectors from the numpy oracle. Run after
# editing fc_verification/oracles/azurelily_mac_oracle.py.
faithful-vectors-al:
	@echo "[make] regen faithful AL test vectors via numpy oracle"
	python3 $(REPO)/fc_verification/oracles/azurelily_mac_oracle.py --gen-test-vectors

# Run all 5 TEST_MODE cases (sequential smoke gate for Task #92).
faithful-smoke-al: | $(BIN)
	@echo "[make] faithful-smoke-al (5/5 TEST_MODE sweep)"
	@for tm in 1 2 3 4 5; do \
	    $(MAKE) -s tb-faithful-al TEST_MODE=$$tm || exit 1; \
	done

# ── clean ─────────────────────────────────────────────────────────────────
clean:
	rm -rf $(BIN)
