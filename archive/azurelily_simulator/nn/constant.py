CLK_LAT = 3.3

DPE_ROWS = 512
DPE_COLS = 128

NUM_ADDS = 16
NUM_MAXPOOLS = 16
NUM_ACTS = 16

BIT_WIDTH = 8                          # bit width of one element (we assume inputs, outputs and weights all have the same bit width)
BYTES_PER_ELEM = BIT_WIDTH // 8        # number of bytes of each element (we assume inputs, outputs and weights all have the same bit width)

PHIT_SIZE = 16                        # bits that can be read/write at the same time to/from buffer/SRAM
ELEMS_PER_MV = PHIT_SIZE / BIT_WIDTH  # number of elements that can be read/write at the same time to/from buffer

PHIT_SIZE_ADD = 16                            # bits that can be read/write at the same time to/from buffer/SRAM
ELEMS_PER_MV_ADD = PHIT_SIZE_ADD / BIT_WIDTH  # number of elements that can be read/write at the same time to/from buffer

COLUMNS_PER_ADC = 16  # number of columns that share the same ADC

##########################################################
# Latency, Unit: ns

BUFFER_LAT = 1     # for reading entire external buffer to internal buffer

DPE_LAT = 1        # for one input bit

ADC_LAT = 1.09     # for one input bit for one DPE column

SRAM_LAT = CLK_LAT

MAXPOOL_LAT = CLK_LAT

ACT_LAT = CLK_LAT

SUM_LAT = CLK_LAT

MAC_LAT = CLK_LAT

MUL_LAT = CLK_LAT

##########################################################
# Energy, Unit: nJ

def SRAM_READ_ENERGY(N):        # read N bytes
    return N * 4.95*1e-6
def SRAM_WRITE_ENERGY(N):       # write N bytes
    return N * 4.95*1e-6

def BUFFER_READ_ENERGY(N):      # read N bytes
    return N * 4.95*1e-6
def BUFFER_WRITE_ENERGY(N):     # write N bytes
    return N * 4.95*1e-6

ADC_ENERGY = 0.00233            # for one input bit for one DPE column

MAXPOOL_ENERGY = 793.1801*1e-6/3   # for one compute per unit

ACT_ENERGY = 453.2458*1e-6         # for one compute per unit

SUM_ENERGY = 84.98358*1e-6          # for one compute per unit

MAC_ENERGY = 908.7433*1e-6            # for one compute per unit

MUL_ENERGY = 908.7433*1e-6            # for one compute per unit

##########################################################
# Event type

EVENT_NEW_DATA = 0      # event to read data from SRAM
EVENT_COMPUT = 1        # event to compute (DPE for conv2d and linear, pooling unit for maxpool)
EVENT_SUM = 2           # event to sum across multiple DPEs vertically
EVENT_ACT = 3           # event to compute activate (only for conv2d and linear)
EVENT_WRITE = 4         # event to write data back into SRAM
EVENT_WRITE_DONE = 5    # event to emulate the finish of writing back to SRAM