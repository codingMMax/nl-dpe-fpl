# Load environment variables from .env if available
-include .env

# Default VTR path (change if needed)
DEFAULT_VTR_PATH := /home/gajjar/VTR/vtr-verilog-to-routing
VTR_ROOT ?= $(DEFAULT_VTR_PATH)

# Design and architecture files
#DESIGN := dpe_one_layer.v
#DESIGN := lenet_1_channel.v
DESIGN := tanh_LUT_comb.v
#DESIGN := vgg11_1_channel.v
#DESIGN := resnet_1_channel.v
#DESIGN := resnet_1_channel_small.v
#DESIGN := conv_layer_stacked_dpes_conv3.v
#DESIGN := conv_layer_single_dpe.v
#DESIGN := cnn_one_layer.v
#DESIGN := xbar_module.v
#ARCH := fixed_grid_65x65.xml
#ARCH := k6FracN10LB_mem20K_complexDSP_customSB_22nm.xml
#ARCH := azure_lily_22nm_with_dpe_550x550.xml
ARCH := azure_lily_22nm_with_dpe_auto.xml

ROUTE_CHAN_WIDTH := 300
SDC_FILE := clock.sdc
VTR_RUN := $(VTR_ROOT)/vtr_flow/scripts/run_vtr_flow.py
OUTPUT_DIR ?= $(abspath tanh_auto) 

# --------- Make Targets ---------
.PHONY: help
help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Available targets:"
	@echo "  help         - Show this help message."
	@echo "  setup        - Configure VTR_ROOT and activate venv."
	@echo "  run          - Run the VTR flow with specified design and architecture."
	@echo "  clean        - Delete generated output directory."

# Setup VTR_ROOT and activate venv
.PHONY: setup
setup:
	@echo "Setting up VTR environment..."
	echo "VTR_ROOT=$(DEFAULT_VTR_PATH)" > .env
	. $(DEFAULT_VTR_PATH)/.venv/bin/activate && \
	echo "Activated virtual environment."

# Run the VTR flow
.PHONY: run
run: check-vtr
	@echo "Running VTR flow with VTR_ROOT=$(VTR_ROOT)"
	. $(VTR_ROOT)/.venv/bin/activate && \
	$(VTR_RUN) $(DESIGN) $(ARCH) --route_chan_width $(ROUTE_CHAN_WIDTH) -temp_dir $(OUTPUT_DIR)
	#$(VTR_RUN) $(DESIGN) $(ARCH) --route_chan_width $(ROUTE_CHAN_WIDTH) -temp_dir $(OUTPUT_DIR) --sdc_file $(SDC_FILE)

# Validate VTR install path
.PHONY: check-vtr
check-vtr:
	@if [ ! -f "$(VTR_RUN)" ]; then \
		echo "Error: VTR script not found at $(VTR_RUN)"; \
		echo "Please ensure VTR_ROOT is set correctly or run 'make setup'."; \
		exit 1; \
	fi

# Remove output files
.PHONY: clean
clean:
	@echo "Cleaning output directory..."
	rm -rf $(OUTPUT_DIR)
