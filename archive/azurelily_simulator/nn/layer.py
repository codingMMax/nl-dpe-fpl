import math
import numpy as np

import nn.constant as C
from nn.utils import *


class Layer:

    def __init__(
        self,                #    conv2d       linear      maxpool       residual
        in_channels: int,    #      int         int          int            int
        out_channels: int,   #      int         int          int            int
        kernel_size: int,    #      int          1           int             1
        stride: int,         #      int          1           int             1
        padding: int,        #      int          0           int             0
        name: str,           # name of this layer, used for printing log
        type: str,           # type of this layer, choose from 'conv2d', 'linear', 'maxpool', and 'residual'
        has_act: bool,       # whether there is an activaion in this layer, only available for 'conv2d' and 'linear'
        num_computes: int,   # number of DPE replicates, only valid for conv2d/linear layers
        num_inputs: int,     # number of input images to process, i.e., the batch size. This is used to simulate the pipeline latency
        debug: bool,         # set to True to print out debugging information
        energy_stats,        # statistics
    ):
        '''
            Constructor
        '''
        # Basic parameters
        self.in_channels = in_channels    # number of input channels
        self.out_channels = out_channels  # number of ouput channels (also the number of filters)
        self.kernel_size = kernel_size    # height and width diemension of the kernel for conv/linear/pooling
        self.stride = stride              # stride size
        self.padding = padding            # padding size

        # Parameters derived from input, set by set_input() function
        self.input_height = 0   # height of input
        self.input_width = 0    # width of input
        self.output_height = 0  # height of output
        self.output_width = 0   # width of output
        self.is_first = False   # True if this layer is the first layer in the model
        self.ready = None       # will be a table, one entry for each input element. Initially set to -1, means that the input is not available
                                # 0: means the element is ready for computing the first input; 1: means the element is ready for computing the second input; etc.

        # Custum parameters
        self.name = name                 # name of this layer, used for printing log
        self.type = type                 # type of this layer, choose from 'conv2d', 'linear', 'maxpool' and 'residual
        self.has_act = has_act           # whether there is an activaion in this layer, only available for 'conv2d' and 'linear'
        self.next_layer = None           # chain to the next layer
        self.Q_next_layer = None         # chain to the next layer, only used if this layer is uses as the linear layer to generate Q
        self.K_next_layer = None         # chain to the next layer, only used if this layer is uses as the linear layer to generate K
        self.V_next_layer = None         # chain to the next layer, only used if this layer is uses as the linear layer to generate V
        self.read_available_time = 0     # the cycle at which the input SRAM is idle for a read request
        self.compute_available_time = 0  # the cycle at which the DPE is idle for new computation
        self.sum_available_time = 0      # the cycle at which the adders are idle for new computation
        self.act_available_time = 0      # the cycle at which the activation unit is idle for new computation
        self.write_available_time = 0    # the cycle at which the output SRAM is idle for a write request
        self.event_queue = []            # even queue
        self.done = False                # True if this layers has finished all computations
        self.stop_fetching = False       # stop fetching new data from input SRAM, only for the frist layer because new EVENT_NEW_DATA events will be consistently added
        self.num_computes = num_computes # number of DPE replicates, only valid for conv2d/linear layers
        self.num_inputs = num_inputs     # number of input images to process, i.e., the batch size. This is used to simulate the pipeline latency
        self.cur_input = 0               # which image is currently being computed in this layer, start from 0
        self.next_out_h = 0              # next output neuron in the height dimension that need to be computed
        self.next_out_w = 0              # next output neuron in the width dimension that need to be computed
        self.debug = debug               # set to True to print out debugging information
        self.energy_stats = energy_stats # energy statistics

        # Checking, the linear layer is a special case of conv2d layer, the residual layer does not change input size
        if self.type == "linear":
            assert self.kernel_size == 1 and self.stride == 1 and self.padding == 0
        elif self.type == "residual":
            assert self.kernel_size == 1 and self.stride == 1 and self.padding == 0

    def set_input(self, input_height, input_width, input_channels, is_first=True):
        '''
            Set the input size of this layer, compute the output size, initialize the ready flag of input neurons
        '''
        if self.type == "linear":
            # Set input size
            self.input_height = 1
            self.input_width = 1
            assert self.in_channels == input_height*input_width*input_channels

            # Compute output size.
            self.output_height = 1
            self.output_width = 1
        elif self.type == "conv2d" or self.type == "maxpool":
            # Set input size
            self.input_height = input_height
            self.input_width = input_width
            assert self.in_channels == input_channels

            # Compute output size.
            self.output_height = (input_height - self.kernel_size + 2 * self.padding) // self.stride + 1
            self.output_width = (input_width - self.kernel_size + 2 * self.padding) // self.stride + 1
        elif self.type == "residual":
            # Set input size
            self.input_height = input_height
            self.input_width = input_width
            assert self.in_channels == input_channels

            # Compute output size.
            self.output_height = input_height
            self.output_width = input_width
            assert self.out_channels == input_channels
        else:
            assert False, "Unsupported layer type"

        # Each input neuron has a ready flag, for the first layer all inputs are already ready
        self.is_first = is_first
        if self.is_first:
            self.ready = np.full((input_height*input_width*self.in_channels), 0)
        else:
            self.ready = np.full((input_height*input_width*self.in_channels), -1)

    def set_next_layer(self, next_layer):
        '''
            Chain up the the next layer.
        '''
        self.next_layer = next_layer
        self.next_layer.set_input(self.output_height, self.output_width, self.out_channels, False)

    def Q_set_next_layer(self, next_layer):
        '''
            Chain up the the next layer.
        '''
        self.Q_next_layer = next_layer

    def K_set_next_layer(self, next_layer):
        '''
            Chain up the the next layer.
        '''
        self.K_next_layer = next_layer

    def V_set_next_layer(self, next_layer):
        '''
            Chain up the the next layer.
        '''
        self.V_next_layer = next_layer

    def set_ready(self, in_addr):
        '''
            Set one input element ready
        '''
        self.ready[in_addr] += 1

    def check_ready(self, out_h, out_w):
        '''
            Check if all input elements are ready to compute one specific output position (across all output channels)
            If yes, return the addresses of all the input elements
            If no, return None
        '''
        input_start_h = out_h*self.stride - self.padding
        input_start_w = out_w*self.stride - self.padding
        addresses = []
        for h in range(self.kernel_size):
            input_h = input_start_h + h
            if input_h < 0 or input_h > self.input_height-1:
                continue

            for w in range(self.kernel_size):
                input_w = input_start_w + w
                if input_w < 0 or input_w > self.input_width-1:
                    continue

                for c in range(self.in_channels):
                    addr = input_h * self.input_width * self.in_channels + input_w * self.in_channels + c
                    if self.ready[addr] < self.cur_input:
                        return None
                    addresses.append(addr)
        return addresses

    def get_next_output(self, out_h, out_w):
        '''
            Return the next output neuron based on current (out_h, out_w)
        '''
        if out_w < self.output_width - 1:
            return out_h, out_w + 1
        elif out_h < self.output_height - 1:
            return out_h + 1, 0
        else:
            return 0, 0

    def add_event(self, event_type, event_time, out_h=None, out_w=None, num_computes=1):
        '''
            Add a new event into the event queue
        '''
        event = {"EVENT_TYPE": event_type, "EVENT_TIME": event_time, "OUT_H": out_h, "OUT_W": out_w, "NUM_COMPUTES": num_computes}
        self.event_queue.append(event)
        self.event_queue = sorted(self.event_queue, key=lambda d: d['EVENT_TIME'])

    def read(self, addresses):
        '''
            Read all the input elements from the input SRAM to the external buffer
        '''
        for addr in addresses:
            self.log(f"Reading address [{addr}]")

        num_elems = len(addresses)  # total number of elements we need to read
        num_positions = num_elems // self.in_channels  # how many different positions (W and H), we will try to read all channels of the same position simultaneously
        num_reads_per_pos = divide_up(self.in_channels, C.ELEMS_PER_MV)  # for a single position, the number of reads depends on phit size
        num_reads = num_reads_per_pos * num_positions  # total number of reads
        latency = C.SRAM_LAT * num_reads  # latency depends on the number of reads
        self.log(f"Read latency: {latency} ns \n")

        self.energy_stats["sram_read_energy"] += C.SRAM_READ_ENERGY(num_elems * C.BYTES_PER_ELEM)
        self.energy_stats["external_buffer_write_energy"] += C.BUFFER_WRITE_ENERGY(num_elems * C.BYTES_PER_ELEM)

        return latency

    def compute(self, num_computes):
        '''
            Compute using DPE for conv/linear layer, compute pool using pooling unit
        '''
        self.log(f"Computing")

        if self.type == "conv2d" or self.type == "linear":
            lat_external_buf_to_internal_buf = C.BUFFER_LAT  # move data from external buffer to internal buffer, this moves the entire data at once (no pipeline), only happens once
            lat_adc = C.ADC_LAT * C.COLUMNS_PER_ADC  # each ADC is shared by COLUMNS_PER_ADC columns in a DPE, this is the latency of one ADC to convert all COLUMNS_PER_ADC columns
            lat_dpe_adc_pipeline = max(C.DPE_LAT, lat_adc) * (C.BIT_WIDTH - 1) + C.DPE_LAT + lat_adc  # the is the latency of computing 8 input bits using DPE-ADC pipeline
            latency = lat_external_buf_to_internal_buf + lat_dpe_adc_pipeline
            cycles = math.ceil(latency / C.CLK_LAT)
            latency = cycles * C.CLK_LAT
            self.log(f"Compute latency: {latency} ns \n")

            one_kernel_size = self.kernel_size * self.kernel_size * self.in_channels  # total size of one kernel
            num_dpe_vert = divide_up(one_kernel_size, C.DPE_ROWS)  # how many DPEs does one kernel needs vertically
            self.energy_stats["external_buffer_read_energy"] += C.BUFFER_READ_ENERGY(one_kernel_size * C.BYTES_PER_ELEM) * num_computes
            self.energy_stats["internal_buffer_write_energy"] += C.BUFFER_WRITE_ENERGY(one_kernel_size * C.BYTES_PER_ELEM) * num_computes
            self.energy_stats["adc_energy"] += (C.ADC_ENERGY * self.out_channels * C.BIT_WIDTH) * num_dpe_vert * num_computes
            self.energy_stats["output_buffer_write_energy"] += C.BUFFER_WRITE_ENERGY(self.out_channels * C.BYTES_PER_ELEM) * num_computes
        elif self.type == "maxpool":
            latency = 0
            # read from SRAM
            num_reads = divide_up(self.kernel_size * self.kernel_size * self.in_channels, C.ELEMS_PER_MV) * num_computes  # total number reads from SRAM
            latency1 = C.SRAM_LAT * num_reads
            # perform maxpool
            pool_size = next_power_of_2(self.kernel_size * self.kernel_size)  # round to the next nearest power of 2
            num_pool_stages = int(math.log2(pool_size))  # number of stages of compares for one output
            lat_pool_one_output = num_pool_stages * C.MAXPOOL_LAT  # latency spent on comparisons to get one output, assuming we have just enough comparitors for each stage
            num_pools_in_serial = divide_up(self.out_channels, C.NUM_MAXPOOLS) * num_computes  # this is the number of outputs we need to compute in serial
            latency2 = lat_pool_one_output * num_pools_in_serial
            self.log(f"Compute latency: {latency} ns \n")
            # write to SRAM
            num_writes = divide_up(self.out_channels, C.ELEMS_PER_MV) * num_computes  # total number reads from SRAM
            latency3 = C.SRAM_LAT * num_writes
            latency = max(latency1, latency2, latency3)

            self.energy_stats["sram_read_energy"] += C.SRAM_WRITE_ENERGY(self.kernel_size * self.kernel_size * self.in_channels * C.BYTES_PER_ELEM) * num_computes
            num_compares = sum(2**i for i in range(num_pool_stages)) * self.out_channels * num_computes  # total number of compares
            self.energy_stats["maxpool_energy"] += num_compares * C.MAXPOOL_ENERGY
            self.energy_stats["sram_write_energy"] += C.SRAM_WRITE_ENERGY(self.out_channels * C.BYTES_PER_ELEM) * num_computes
        else:
            assert False, "Unsupported layer type"

        return latency

    def sum(self, num_computes):
        '''
            Sum up the output from multiple DPEs
        '''
        if self.type == "residual":
            self.log(f"Residual")

            latency = 0
            # perform add
            latency1 = C.SUM_LAT * num_computes
            # write to SRAM
            latency2 = C.SRAM_LAT * num_computes
            latency = max(latency1, latency2)

            self.energy_stats["sum_energy"] += C.SUM_ENERGY * num_computes
            self.energy_stats["sram_write_energy"] += C.SRAM_WRITE_ENERGY(C.BYTES_PER_ELEM) * num_computes
        elif self.type == "conv2d" or self.type == "linear":
            self.log(f"Summing")

            one_kernel_size = self.kernel_size * self.kernel_size * self.in_channels  # total size of one kernel
            num_dpe_vert = divide_up(one_kernel_size, C.DPE_ROWS)  # how many DPEs does one kernel needs vertically
            num_sum_stages = int(math.log2(next_power_of_2(num_dpe_vert)))  # number of stages of adds
            lat_sum_one_output = num_sum_stages * C.SUM_LAT  # latency spent on sums to get one output, assuming we have just enough comparitors for each stage
            num_sums_in_serial = divide_up(self.out_channels, C.NUM_ADDS) * num_computes  # this is the number of outputs we need to compute in serial
            latency = lat_sum_one_output * num_sums_in_serial
            self.log(f"Sum latency: {latency} ns \n")

            num_adds = sum(2**i for i in range(num_sum_stages)) * self.out_channels * num_computes   # total number of adds
            self.energy_stats["sum_energy"] += num_adds * C.SUM_ENERGY
        else:
            assert False, "Unsupported layer type"

        return latency

    def act(self, num_computes):
        '''
            Compute activation
        '''
        self.log(f"Activation")

        latency = 0
        # read from SRAM
        num_reads = divide_up(self.out_channels, C.ELEMS_PER_MV) * num_computes  # total number reads from SRAM
        latency1 = C.SRAM_LAT * num_reads
        # perform activation
        num_acts_in_serial = divide_up(self.out_channels, C.NUM_ACTS) * num_computes  # this is the number of outputs we need to compute in serial
        latency2 = C.ACT_LAT * num_acts_in_serial
        # write to SRAM
        num_writes = divide_up(self.out_channels, C.ELEMS_PER_MV) * num_computes  # total number reads from SRAM
        latency3 = C.SRAM_LAT * num_writes
        latency = max(latency1, latency2, latency3)
        self.log(f"Activation latency: {latency} ns \n")

        self.energy_stats["sram_read_energy"] += C.SRAM_WRITE_ENERGY(self.out_channels * C.BYTES_PER_ELEM) * num_computes
        self.energy_stats["act_energy"] += self.out_channels * C.ACT_ENERGY * num_computes
        self.energy_stats["sram_write_energy"] += C.SRAM_WRITE_ENERGY(self.out_channels * C.BYTES_PER_ELEM) * num_computes

        return latency

    def write(self, next_out_h, next_out_w, num_computes):
        '''
            Write the output at the specific position (across all output channels) into output SRAM
        '''
        w = next_out_w
        h = next_out_h
        for _ in range(num_computes):
            for c in range(self.out_channels):
                write_addr = h * self.output_width*self.out_channels + w*self.out_channels + c
                self.log(f"Writing address [{write_addr}]")
            h, w = self.get_next_output(h, w)

        num_writes = divide_up(self.out_channels, C.ELEMS_PER_MV) * num_computes  # total number of writes
        latency = C.SRAM_LAT * num_writes  # we may compute may output positions if we have duplicates of computing units
        self.log(f"Write latency: {latency} ns \n")

        self.energy_stats["output_buffer_read_energy"] += C.BUFFER_READ_ENERGY(self.out_channels * C.BYTES_PER_ELEM) * num_computes
        self.energy_stats["sram_write_energy"] += C.SRAM_WRITE_ENERGY(self.out_channels * C.BYTES_PER_ELEM) * num_computes

        return latency

    def log(self, message):
        '''
            Log debugging information
        '''
        if self.debug:
            print(self.name, message)

    def update(self, cur_time):
        """
            Args:
                cur_time: Current time.

            Returns: -1: This layer has finished all computing
                     -2: This layer has just finished one new input
                     inf: Nothing to update, and this layer is not finished yet
                     Non-negative: The nearest event time in future
        """
        self.log(f"------------------------------")

        ###################################################################################

        if self.done:
            self.event_queue.clear()
            self.log(f"This layer has finished \n")
            return -1

        ###################################################################################

        ret_latency = float("inf")
        idx = 0

        # Process pending events
        while len(self.event_queue[idx:]) > 0:
            # Get next event from queue
            e = self.event_queue[idx]

            # Break loop if the event is a future event
            if e["EVENT_TIME"] > cur_time:
                if e["EVENT_TIME"] < ret_latency:
                    ret_latency = e["EVENT_TIME"]
                break

            # --------------------------------------------------------------
            # Try to load data for next computation
            # --------------------------------------------------------------
            if e["EVENT_TYPE"] == C.EVENT_NEW_DATA:
                latency = 0

                # Stop fetching new data, this check is only for the frist layer because new EVENT_NEW_DATA events will be consistently added
                if self.stop_fetching:
                    self.event_queue.pop(idx)
                    continue

                # Check if SRAM is free, if not, will try again when until it is free
                if self.read_available_time > cur_time:
                    ret_latency = min(self.read_available_time, ret_latency)
                    idx += 1
                    continue

                # GREAT, we can read, delete this event
                self.event_queue.pop(idx)

                # Check if all needed inputs are ready, if not, will try again later when new data is ready
                num_computes = 0              # how many actual output positions need to compute for this round
                addresses = set()             # all addresses to read from input SRAM
                success = True                # remains True if read is successful and can go to next state
                next_out_h = self.next_out_h  # index to indicate the output position to compute next
                next_out_w = self.next_out_w  # index to indicate the output position to compute next
                # This loop exits in one of the following three cases:
                #   1. If any input element is not available yet, success=False
                #   2. In case of the last few output positions, maybe less than self.num_computes output positions will be computed
                #   3. All input elements are available for all self.num_computes output positions
                while num_computes < self.num_computes:
                    addr = self.check_ready(next_out_h, next_out_w)
                    if addr is None:
                        success = False
                        break
                    addresses = addresses | set(addr)
                    num_computes += 1

                    # Update to start fetching for next output, also check if the current input has finished
                    next_out_h, next_out_w = self.get_next_output(next_out_h, next_out_w)
                    if next_out_h == next_out_w == 0:
                        self.cur_input += 1
                        if self.cur_input < self.num_inputs:
                            if self.is_first:
                                self.ready += 1
                        else:
                            self.stop_fetching = True
                        break

                # If not all input elements are ready to fetch, just ignore what we did in the above while loop
                if not success:
                    continue

                # Read data from SRAM
                latency += self.read(addresses)
                self.read_available_time = cur_time + latency

                # Transit to next state
                if self.type == "residual":
                    self.add_event(C.EVENT_SUM, cur_time+latency, self.next_out_h, self.next_out_w, num_computes)
                else:
                    self.add_event(C.EVENT_COMPUT, cur_time+latency, self.next_out_h, self.next_out_w, num_computes)

                # Update next output neuron
                self.next_out_w = next_out_w
                self.next_out_h = next_out_h

            # --------------------------------------------------------------
            # Try to start the next computation
            # --------------------------------------------------------------
            elif e["EVENT_TYPE"] == C.EVENT_COMPUT:
                latency = 0

                # Check if compute unit is free, if not, will try again until it is free
                if self.compute_available_time > cur_time:
                    ret_latency = min(self.compute_available_time, ret_latency)
                    idx += 1
                    continue

                # GREAT, we can compute, delete this event
                next_out_h = e["OUT_H"]
                next_out_w = e["OUT_W"]
                num_computes = e['NUM_COMPUTES']
                self.event_queue.pop(idx)

                # For first layer, whenever the compute starts, input buffer can start recieving new data
                if self.is_first:
                    self.add_event(C.EVENT_NEW_DATA, cur_time)

                # Compute
                latency += self.compute(num_computes)
                self.compute_available_time = cur_time + latency

                # Transit to next state
                if self.type == "conv2d" or self.type == "linear":
                    if self.kernel_size * self.kernel_size * self.in_channels > C.DPE_ROWS:
                        self.add_event(C.EVENT_SUM, cur_time+latency, next_out_h, next_out_w, num_computes)
                    elif self.has_act:
                        self.add_event(C.EVENT_ACT, cur_time+latency, next_out_h, next_out_w, num_computes)
                    else:
                        self.add_event(C.EVENT_WRITE, cur_time+latency, next_out_h, next_out_w, num_computes)
                else:
                    self.add_event(C.EVENT_WRITE, cur_time+latency, next_out_h, next_out_w, num_computes)

            # --------------------------------------------------------------
            # Sum outputs from multiple DPEs
            # --------------------------------------------------------------
            elif e["EVENT_TYPE"] == C.EVENT_SUM:
                latency = 0

                # Check if activation unit is free, if not, will try again until it is free
                if self.sum_available_time > cur_time:
                    ret_latency = min(self.sum_available_time, ret_latency)
                    idx += 1
                    continue

                # GREAT, we can sum, delete this event
                next_out_h = e["OUT_H"]
                next_out_w = e["OUT_W"]
                num_computes = e['NUM_COMPUTES']
                self.event_queue.pop(idx)

                # Sum
                latency += self.sum(num_computes)
                self.sum_available_time = cur_time + latency

                # Transit to next state
                if self.has_act:
                    self.add_event(C.EVENT_ACT, cur_time+latency, next_out_h, next_out_w, num_computes)
                else:
                    self.add_event(C.EVENT_WRITE, cur_time+latency, next_out_h, next_out_w, num_computes)

            # --------------------------------------------------------------
            # Try to start activation
            # --------------------------------------------------------------
            elif e["EVENT_TYPE"] == C.EVENT_ACT:
                latency = 0

                # Check if activation unit is free, if not, will try again until it is free
                if self.act_available_time > cur_time:
                    ret_latency = min(self.act_available_time, ret_latency)
                    idx += 1
                    continue

                # GREAT, we can activate, delete this event
                next_out_h = e["OUT_H"]
                next_out_w = e["OUT_W"]
                num_computes = e['NUM_COMPUTES']
                self.event_queue.pop(idx)

                # Activation
                latency += self.act(num_computes)
                self.act_available_time = cur_time + latency

                # Transit to next state
                self.add_event(C.EVENT_WRITE, cur_time+latency, next_out_h, next_out_w, num_computes)

            # --------------------------------------------------------------
            # Write result to memory
            # --------------------------------------------------------------
            elif e["EVENT_TYPE"] == C.EVENT_WRITE:
                latency = 0

                # Check if SRAM is free, if not, will try again until it is free
                if self.write_available_time > cur_time:
                    ret_latency = min(self.write_available_time, ret_latency)
                    idx += 1
                    continue

                # GREAT, we can write, delete this event
                next_out_h = e["OUT_H"]
                next_out_w = e["OUT_W"]
                num_computes = e['NUM_COMPUTES']
                self.event_queue.pop(idx)

                # Write result into output memory
                latency += self.write(next_out_h, next_out_w, num_computes)
                self.write_available_time = cur_time + latency

                # Transit to next state
                self.add_event(C.EVENT_WRITE_DONE, cur_time+latency, next_out_h, next_out_w, num_computes)

            # --------------------------------------------------------------
            # Write done, infor next layer
            # --------------------------------------------------------------
            elif e["EVENT_TYPE"] == C.EVENT_WRITE_DONE:
                # This event is just to emulate the write latency, so this event can always be served
                next_out_h = e["OUT_H"]
                next_out_w = e["OUT_W"]
                num_computes = e['NUM_COMPUTES']
                self.event_queue.pop(idx)

                for _ in range(num_computes):
                    # Inform next layer
                    if self.next_layer is not None:
                        for c in range(self.out_channels):
                            self.next_layer.set_ready(next_out_h*self.output_width*self.out_channels + next_out_w*self.out_channels + c)
                        self.next_layer.add_event(C.EVENT_NEW_DATA, cur_time)
                        # for neuron at the right and bottom border, add additional events padding
                        if next_out_w == self.output_width - 1:
                            for i in range(self.next_layer.padding):
                                self.next_layer.add_event(C.EVENT_NEW_DATA, cur_time)
                            if next_out_h == self.output_height - 1:
                                for i in range((self.output_width+self.next_layer.padding)*self.next_layer.padding):
                                    self.next_layer.add_event(C.EVENT_NEW_DATA, cur_time)
                    elif self.Q_next_layer is not None:
                        for c in range(self.out_channels):
                            self.Q_next_layer.set_Q_ready(next_out_h*self.output_width*self.out_channels + next_out_w*self.out_channels + c)
                        self.Q_next_layer.add_event(C.EVENT_NEW_DATA, cur_time)
                    elif self.K_next_layer is not None:
                        for c in range(self.out_channels):
                            self.K_next_layer.set_K_ready(next_out_h*self.output_width*self.out_channels + next_out_w*self.out_channels + c)
                        self.K_next_layer.add_event(C.EVENT_NEW_DATA, cur_time)
                    elif self.V_next_layer is not None:
                        for c in range(self.out_channels):
                            self.V_next_layer.set_V_ready(next_out_h*self.output_width*self.out_channels + next_out_w*self.out_channels + c)
                        self.V_next_layer.add_event(C.EVENT_NEW_DATA, cur_time)

                    # Update to next finished output, also check if the current input has finished
                    next_out_h, next_out_w = self.get_next_output(next_out_h, next_out_w)
                    if next_out_h == next_out_w == 0:
                        if self.cur_input == self.num_inputs and len(self.event_queue) == 0:
                            self.done = True
                        return -2

        return ret_latency
