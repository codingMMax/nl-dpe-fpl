import math
import numpy as np

import nn.constant as C
from nn.utils import *
from nn import Layer

class MAC_SV_Layer(Layer):

    def __init__(
        self,
        d: int,                 # attention head dim
        N: int,                 # sequence length, number of tokens
        num_macs: int,          # number of parallel MAC units, actually it is d*num_macs
        V_sram_phit_size: int,  # phit size of SRAM for V
        name: str,              # name of this layer, used for printing log
        debug: bool,            # print debugging information
        energy_stats,           # statistics
    ):
        '''
            Constructor
        '''
        self.d = d
        self.N = N
        self.num_macs = num_macs
        self.V_sram_phit_size = V_sram_phit_size
        self.name = name
        self.debug = debug
        self.energy_stats = energy_stats
        self.type = "mac_sv"
        self.S_ready = np.full((N), -1)      # store the readiness of one row of S
        self.V_ready = np.full((N * d), -1)  # store the readiness of the whole V, once an element becomes ready, it will keep ready
        self.V_ready_index = 0               # because the elements are always generated in order from previous layer, we use an index to keep track

        self.tok_index = 0   # which token of S is currently used (range: 0 -- N-1)
        self.start_dim = 0   # the start of embedding within that S (range: 0 -- N-1)

        self.next_layer = None         # must None, because this is the last layer in attention
        self.num_finished_compute = 0  # to record whether this layer has finsihed all computations, in total N * ceil(N / num_macs)
        self.done = False              # set True if all computations finishes
        self.event_queue = []          # Event queue

        self.read_available_time = 0     # the cycle at which the input SRAM is idle for a read request
        self.compute_available_time = 0  # the cycle at which the MAC units are idle for new computation
        self.write_available_time = 0    # the cycle at which the output SRAM is idle for a write request

    def set_next_layer(self, next_layer):
        '''
            Chain up the the next layer.
        '''
        self.next_layer = next_layer

    def set_S_ready(self, in_addr):
        '''
            Set one input element of S ready, -1 means not ready
        '''
        assert in_addr < self.N
        self.S_ready[in_addr] += 1

    def set_V_ready(self, in_addr):
        '''
            Set one input element of V ready, -1 means not ready
        '''
        assert self.V_ready_index < self.N * self.d
        self.V_ready[self.V_ready_index] += 1  # because the elements are always generated in order from previous layer, we use an index to keep track, not in_addr
        self.V_ready_index += 1

    def check_ready(self, tok_index, start_dim):
        '''
            Check if all input elements are ready to start computation
            If yes, return the addresses of all the input elements
            If no, return None
        '''
        S_addresses = []
        V_addresses = []
        for i in range(self.num_macs):
            if self.S_ready[start_dim + i] == -1:
                return None
            S_addresses.append(start_dim + i)
            for j in range(self.d):
                if self.V_ready[(start_dim + i)*self.d + j] == -1:
                    return None
                V_addresses.append((start_dim + i)*self.d + j)
        return S_addresses, V_addresses

    def add_event(self, event_type, event_time):
        '''
            Add a new event into the event queue
        '''
        event = {"EVENT_TYPE": event_type, "EVENT_TIME": event_time}
        self.event_queue.append(event)
        self.event_queue = sorted(self.event_queue, key=lambda d: d['EVENT_TIME'])

    def read(self, S_addresses, V_addresses):
        '''
            Read all the input elements from the input SRAM
        '''
        # read S

        S_latency = C.SRAM_LAT

        self.energy_stats["sram_read_energy"] += C.SRAM_READ_ENERGY(len(S_addresses) * C.BYTES_PER_ELEM)

        # read V

        num_reads = divide_up(4*self.d, self.V_sram_phit_size//C.BIT_WIDTH)
        V_latency = C.SRAM_LAT * num_reads

        self.energy_stats["sram_read_energy"] += C.SRAM_READ_ENERGY(len(V_addresses) * C.BYTES_PER_ELEM)

        return max(S_latency, V_latency)

    def compute(self):
        '''
            element-wise add and compute activation (exp) on one row of Q and one row of K
        '''
        latency = C.MAC_LAT

        self.energy_stats["mac_energy"] += C.MAC_ENERGY * self.num_macs * self.d

        return latency

    def write(self):
        '''
            Write the element-wise output into output SRAM
        '''
        latency = C.SRAM_LAT

        self.energy_stats["sram_write_energy"] += C.SRAM_WRITE_ENERGY(self.d * C.BYTES_PER_ELEM)

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

                # Check if SRAM is free, if not, will try again when until it is free
                if self.read_available_time > cur_time:
                    ret_latency = min(self.read_available_time, ret_latency)
                    idx += 1
                    continue

                # GREAT, we can read, delete this event
                self.event_queue.pop(idx)

                # Check if all needed inputs are ready, if not, will try again later when new data is ready
                addresses = self.check_ready(self.tok_index, self.start_dim)
                if addresses is None:
                    idx += 1
                    continue
                S_addresses, V_addresses = addresses
                self.start_dim += self.num_macs
                if self.start_dim >= self.N:
                    self.start_dim = 0
                    self.tok_index += 1
                    if self.tok_index >= self.N:
                        self.tok_index = 0
                if self.tok_index > 0 or self.start_dim > 0:
                    self.add_event(C.EVENT_NEW_DATA, cur_time+1)

                # Read data from SRAM
                latency += self.read(S_addresses, V_addresses)
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

                # Compute
                latency += self.compute()
                self.compute_available_time = cur_time + latency

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
                assert self.next_layer is None, "This layer can not have next layer"

                # check if this layer has just finished all computation
                self.num_finished_compute += 1
                if self.num_finished_compute == self.N * divide_up(self.N, self.num_macs):
                    self.done = True
                return -2

        return ret_latency
