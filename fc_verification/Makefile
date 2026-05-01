# DPE behavior model + TB build/run harness.
#
# Usage examples:
#   make help                                          # show all targets and variables
#   make tb-vmm-nldpe                                  # NL-DPE VMM, defaults (R=256 C=256 P=8)
#   make tb-vmm-al R=1024 C=128                        # AL VMM with custom geometry
#   make tb-vmm-nldpe PRECISION=4                      # NL-DPE INT4
#   make tb-acam                                       # NL-DPE ACAM exp mode
#   make tb-dsp-mac K=128                              # AL DSP-MAC, K=128
#   make all                                           # all four TBs at defaults
#   make regen                                         # regenerate dpe_*.v from configs
#   make smoke                                         # full smoke sweep via run_dpe_smoke.py
#   make clean                                         # remove tmp binaries
#
# Variables (override on command line, e.g. `make tb-vmm-nldpe R=1024 PRECISION=16`):

# DPE geometry / DPE-axiom (per-arch defaults below; override these as needed)
R              ?=
C              ?=
BUF            ?=
PRECISION      ?= 8
PIPELINE_DEPTH ?= 3

# DSP-MAC knobs
K              ?= 64
DSP_WIDTH      ?= 4

# Per-arch defaults
NLDPE_R   ?= 256
NLDPE_C   ?= 256
NLDPE_BUF ?= 40

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

# Common iverilog flags
IV_FLAGS := -g2005

# Optional VCD waveform dump (`make tb-vmm-nldpe WAVE=1`)
ifeq ($(WAVE),1)
VCD_DEFINES := -DWAVE=1
else
VCD_DEFINES :=
endif

# Common define set for DPE TBs
DPE_DEFS = -DR_TB=$(1) -DC_TB=$(2) -DBUF_TB=$(3) -DPRECISION_TB=$(PRECISION) \
           -DPIPELINE_DEPTH_TB=$(PIPELINE_DEPTH)

# DSP-MAC define set
DSP_DEFS = -DK_TB=$(K) -DBUF_TB=$(DSP_BUF_EFF) -DDSP_WIDTH_TB=$(DSP_WIDTH)

.PHONY: help all clean regen smoke tb-vmm-nldpe tb-vmm-al tb-acam tb-dsp-mac

# ── default target ────────────────────────────────────────────────────────
help:
	@echo "DPE behavior model + TB harness"
	@echo ""
	@echo "Targets:"
	@echo "  make tb-vmm-nldpe   NL-DPE VMM TB (override: R, C, BUF, PRECISION, PIPELINE_DEPTH)"
	@echo "  make tb-vmm-al      Azure-Lily VMM TB (same overrides)"
	@echo "  make tb-acam        NL-DPE ACAM exp mode TB"
	@echo "  make tb-dsp-mac     AL DSP-MAC TB (override: K, DSP_WIDTH, BUF)"
	@echo "  make all            run all 4 TBs at defaults"
	@echo "  make regen          regenerate dpe_*.v from per-arch JSON configs"
	@echo "  make smoke          full smoke sweep via run_dpe_smoke.py"
	@echo "  make clean          remove fc_verification/build/"
	@echo ""
	@echo "Variables:"
	@echo "  R, C, BUF                 override DPE geometry (per-arch defaults if unset)"
	@echo "  PRECISION                 INT precision (default 8)"
	@echo "  PIPELINE_DEPTH            bit-serial pipeline depth (default 3)"
	@echo "  K, DSP_WIDTH              DSP-MAC parameters (defaults 64, 4)"
	@echo "  WAVE=1                    dump VCD to build/<tb>.vcd"
	@echo ""
	@echo "Examples:"
	@echo "  make tb-vmm-nldpe                            # defaults (R=256 C=256 P=8)"
	@echo "  make tb-vmm-nldpe R=1024 C=128 PRECISION=4   # override"
	@echo "  make tb-vmm-al BUF=40                        # AL with NL-style buf"
	@echo "  make tb-dsp-mac K=128                        # bigger K"

$(BIN):
	@mkdir -p $@

# ── NL-DPE VMM ────────────────────────────────────────────────────────────
tb-vmm-nldpe: | $(BIN)
	@echo "[make] tb-vmm-nldpe  R=$(NLDPE_R_EFF) C=$(NLDPE_C_EFF) BUF=$(NLDPE_BUF_EFF) PRECISION=$(PRECISION) PIPELINE_DEPTH=$(PIPELINE_DEPTH)"
	iverilog $(IV_FLAGS) -DARCH_NLDPE $(call DPE_DEFS,$(NLDPE_R_EFF),$(NLDPE_C_EFF),$(NLDPE_BUF_EFF)) $(VCD_DEFINES) \
	  -o $(BIN)/tb_vmm_nldpe \
	  $(RTL)/dpe_nldpe.v $(TB_DIR)/tb_dpe_vmm.v
	vvp $(BIN)/tb_vmm_nldpe

# ── Azure-Lily VMM ────────────────────────────────────────────────────────
tb-vmm-al: | $(BIN)
	@echo "[make] tb-vmm-al     R=$(AL_R_EFF) C=$(AL_C_EFF) BUF=$(AL_BUF_EFF) PRECISION=$(PRECISION) PIPELINE_DEPTH=$(PIPELINE_DEPTH)"
	iverilog $(IV_FLAGS) -DARCH_AL $(call DPE_DEFS,$(AL_R_EFF),$(AL_C_EFF),$(AL_BUF_EFF)) $(VCD_DEFINES) \
	  -o $(BIN)/tb_vmm_al \
	  $(RTL)/dpe_azurelily.v $(TB_DIR)/tb_dpe_vmm.v
	vvp $(BIN)/tb_vmm_al

# ── NL-DPE ACAM exp mode ──────────────────────────────────────────────────
tb-acam: | $(BIN)
	@echo "[make] tb-acam       R=$(NLDPE_R_EFF) C=$(NLDPE_C_EFF) BUF=$(NLDPE_BUF_EFF) PRECISION=$(PRECISION) PIPELINE_DEPTH=$(PIPELINE_DEPTH)"
	iverilog $(IV_FLAGS) $(call DPE_DEFS,$(NLDPE_R_EFF),$(NLDPE_C_EFF),$(NLDPE_BUF_EFF)) $(VCD_DEFINES) \
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

# ── clean ─────────────────────────────────────────────────────────────────
clean:
	rm -rf $(BIN)
