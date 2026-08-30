import math
import numpy as np

import nn.constant as C
from nn.utils import *
from nn import Layer

class Linear_Layer(Layer):

    def __init__(
        self,
        in_channels: int,         # num input features
        out_channels: int,        # num output features
        out_sram_phit_size: int,  # phit size of output SRAM
        name: str,                # name of this layer, used for printing log
        num_inputs: int,          # number of input images to process, i.e., the batch size. This is used to simulate the pipeline latency
        debug: bool,              # set to True to print out debugging information
        energy_stats,             # statistics
    ):
        '''
            Constructor
        '''
        self.in_channels = in_channels               # number of input channels
        self.out_channels = out_channels             # number of ouput channels (also the number of filters)
        self.out_sram_phit_size= out_sram_phit_size  # phit size of output SRAM
        self.name = name                             # name of this layer, used for printing log
        self.num_inputs = num_inputs                 # number of input images to process, i.e., the batch size. This is used to simulate the pipeline latency
        self.debug = debug                           # set to True to print out debugging information
        self.energy_stats = energy_stats             # energy statistics
        self.type = "linear"
        self.ready = np.full((self.in_channels), -1)  # will be a table, one entry for each input element. Initially set to -1, means that the input is not available
                                                      # 0: means the element is ready for computing the first input; 1: means the element is ready for computing the second input; etc.
        self.is_first = False            # True if this layer is the first layer in the model
        self.stop_fetching = False       # stop fetching new data from input SRAM, only for the frist layer because new EVENT_NEW_DATA events will be consistently added
        self.cur_input = 0               # which image is currently being computed in this layer, start from 0

        self.next_layer = None           # chain to the next layer
        self.Q_next_layer = None         # chain to the next layer, only used if this layer is uses as the linear layer to generate Q
        self.K_next_layer = None         # chain to the next layer, only used if this layer is uses as the linear layer to generate K
        self.V_next_layer = None         # chain to the next layer, only used if this layer is uses as the linear layer to generate V
        self.done = False                # True if this layers has finished all computations
        self.event_queue = []            # even queue

        self.read_available_time = 0     # the cycle at which the input SRAM is idle for a read request
        self.compute_available_time = 0  # the cycle at which the DPE is idle for new computation
        self.sum_available_time = 0      # the cycle at which the adders are idle for new computation
        self.write_available_time = 0    # the cycle at which the output SRAM is idle for a write request

    def set_first(self):
        '''
            For the first layer all inputs are already ready
        '''
        self.is_first = True
        self.ready *= 0

    def set_next_layer(self, next_layer):
        '''
            Chain up the the next layer.
        '''
        self.next_layer = next_layer

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
            Set one input element ready, -1 means not ready
        '''
        assert in_addr < self.in_channels
        self.ready[in_addr] += 1

    def check_ready(self):
        '''
            Check if all input elements are ready to compute one specific output position (across all output channels)
            If yes, return the addresses of all the input elements
            If no, return None
        '''
        addresses = []
        for c in range(self.in_channels):
            if self.ready[c] < self.cur_input:
                return None
            addresses.append(c)
        return addresses

    def add_event(self, event_type, event_time):
        '''
            Add a new event into the event queue
        '''
        event = {"EVENT_TYPE": event_type, "EVENT_TIME": event_time}
        self.event_queue.append(event)
        self.event_queue = sorted(self.event_queue, key=lambda d: d['EVENT_TIME'])

    def read(self, addresses):
        '''
            Read all the input elements from the input SRAM to the external buffer
        '''
        latency = C.SRAM_LAT

        self.energy_stats["sram_read_energy"] += C.SRAM_READ_ENERGY(len(addresses) * C.BYTES_PER_ELEM)
        self.energy_stats["external_buffer_write_energy"] += C.BUFFER_WRITE_ENERGY(len(addresses) * C.BYTES_PER_ELEM)

        return latency

    def compute(self):
        '''
            Compute using DPE
        '''
        lat_external_buf_to_internal_buf = C.BUFFER_LAT  # move data from external buffer to internal buffer, this moves the entire data at once (no pipeline), only happens once
        lat_adc = C.ADC_LAT * C.COLUMNS_PER_ADC  # each ADC is shared by COLUMNS_PER_ADC columns in a DPE, this is the latency of one ADC to convert all COLUMNS_PER_ADC columns
        lat_dpe_adc_pipeline = max(C.DPE_LAT, lat_adc) * (C.BIT_WIDTH - 1) + C.DPE_LAT + lat_adc  # the is the latency of computing 8 input bits using DPE-ADC pipeline
        latency = lat_external_buf_to_internal_buf + lat_dpe_adc_pipeline
        cycles = math.ceil(latency / C.CLK_LAT)
        latency = cycles * C.CLK_LAT

        num_dpe_vert = divide_up(self.in_channels, C.DPE_ROWS)  # how many DPEs does one kernel needs vertically
        self.energy_stats["external_buffer_read_energy"] += C.BUFFER_READ_ENERGY(self.in_channels * C.BYTES_PER_ELEM)
        self.energy_stats["internal_buffer_write_energy"] += C.BUFFER_WRITE_ENERGY(self.in_channels * C.BYTES_PER_ELEM)
        self.energy_stats["adc_energy"] += (C.ADC_ENERGY * self.out_channels * C.BIT_WIDTH) * num_dpe_vert
        self.energy_stats["output_buffer_write_energy"] += C.BUFFER_WRITE_ENERGY(self.out_channels * C.BYTES_PER_ELEM)

        return latency

    def sum(self):
        '''
            Sum up the output from multiple DPEs
        '''
        num_dpe_vert = divide_up(self.in_channels, C.DPE_ROWS)  # how many DPEs does one kernel needs vertically
        num_sum_stages = int(math.log2(next_power_of_2(num_dpe_vert)))  # number of stages of adds
        latency = num_sum_stages * C.SUM_LAT  # latency spent on sums to get one output, assuming we have just enough comparitors for each stage

        num_adds = sum(2**i for i in range(num_sum_stages)) * self.out_channels
        self.energy_stats["sum_energy"] += num_adds * C.SUM_ENERGY

        return latency

    def write(self):
        '''
            Write the output at the specific position (across all output channels) into output SRAM
        '''
        num_writes = divide_up(self.out_channels, self.out_sram_phit_size//C.BIT_WIDTH)
        latency = C.SRAM_LAT * num_writes

        self.energy_stats["output_buffer_read_energy"] += C.BUFFER_READ_ENERGY(self.out_channels * C.BYTES_PER_ELEM)
        self.energy_stats["sram_write_energy"] += C.SRAM_WRITE_ENERGY(self.out_channels * C.BYTES_PER_ELEM)

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
                addresses = self.check_ready()
                if addresses is None:
                    continue

                # Update to start fetching for next output, also check if the current input has finished
                self.cur_input += 1
                if self.cur_input < self.num_inputs:
                    if self.is_first:
                        self.ready += 1
                else:
                    self.stop_fetching = True

                # Read data from SRAM
                latency += self.read(addresses)
                self.read_available_time = cur_time + latency

                # Transit to next state
                self.add_event(C.EVENT_COMPUT, cur_time+latency)

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
                self.event_queue.pop(idx)

                # For first layer, whenever the compute starts, input buffer can start recieving new data
                if self.is_first:
                    self.add_event(C.EVENT_NEW_DATA, cur_time)

                # Compute
                latency += self.compute()
                self.compute_available_time = cur_time + latency

                # Transit to next state
                if self.in_channels > C.DPE_ROWS:
                    self.add_event(C.EVENT_SUM, cur_time+latency)
                else:
                    self.add_event(C.EVENT_WRITE, cur_time+latency)

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
                self.event_queue.pop(idx)

                # Sum
                latency += self.sum()
                self.sum_available_time = cur_time + latency

                # Transit to next state
                self.add_event(C.EVENT_WRITE, cur_time+latency)

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
                self.event_queue.pop(idx)

                # Write result into output memory
                latency += self.write()
                self.write_available_time = cur_time + latency

                # Transit to next state
                self.add_event(C.EVENT_WRITE_DONE, cur_time+latency)

            # --------------------------------------------------------------
            # Write done, infor next layer
            # --------------------------------------------------------------
            elif e["EVENT_TYPE"] == C.EVENT_WRITE_DONE:
                # This event is just to emulate the write latency, so this event can always be served
                self.event_queue.pop(idx)

                # Inform next layer
                if self.next_layer is not None:
                    for c in range(self.out_channels):
                        self.next_layer.set_ready(c)
                    self.next_layer.add_event(C.EVENT_NEW_DATA, cur_time)
                elif self.Q_next_layer is not None:
                    for c in range(self.out_channels):
                        self.Q_next_layer.set_Q_ready(c)
                    self.Q_next_layer.add_event(C.EVENT_NEW_DATA, cur_time)
                elif self.K_next_layer is not None:
                    for c in range(self.out_channels):
                        self.K_next_layer.set_K_ready(c)
                    self.K_next_layer.add_event(C.EVENT_NEW_DATA, cur_time)
                elif self.V_next_layer is not None:
                    for c in range(self.out_channels):
                        self.V_next_layer.set_V_ready(c)
                    self.V_next_layer.add_event(C.EVENT_NEW_DATA, cur_time)

                # Update to next finished output, also check if the current input has finished
                if self.cur_input == self.num_inputs and len(self.event_queue) == 0:
                    self.done = True
                return -2

        return ret_latency
