import math
import numpy as np


import nn.constant as C
from nn.utils import *
from nn import Layer

class MAC_QK_Layer(Layer):

    def __init__(
        self,
        d: int,                 # attention head dim
        N: int,                 # sequence length, number of tokens
        num_macs: int,          # number of parallel MAC units, actually it is N*num_macs
        Q_sram_phit_size: int,  # phit size of SRAM for V
        K_sram_phit_size: int,  # phit size of SRAM for V
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
        self.Q_sram_phit_size = Q_sram_phit_size
        self.K_sram_phit_size = K_sram_phit_size
        self.name = name
        self.debug = debug
        self.energy_stats = energy_stats
        self.type = "mac_qk"

        self.Q_ready = np.full((N * d), -1)  # store the readiness of the whole Q, once an element becomes ready, it will keep ready
        self.K_ready = np.full((N * d), -1)  # store the readiness of the whole K, once an element becomes ready, it will keep ready
        self.Q_ready_index = 0               # because the elements are always generated in order from previous layer, we use an index to keep track
        self.K_ready_index = 0               # because the elements are always generated in order from previous layer, we use an index to keep track

        self.tok_index = 0   # which token of Q is currently used (range: 0 -- N-1)
        self.start_dim = 0  # the start of embedding within that token (range: 0 -- d-1)

        self.next_layer = None         # must be Softmax_Exp_Layer
        self.num_finished_compute = 0  # to record whether this layer has finsihed all computations, in total N * ceil(d / num_macs)
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

    def set_Q_ready(self, in_addr):
        '''
            Set one input element of Q ready, -1 means not ready
        '''
        assert self.Q_ready_index < self.N * self.d
        self.Q_ready[self.Q_ready_index] += 1  # because the elements are always generated in order from previous layer, we use an index to keep track, not in_addr
        self.Q_ready_index += 1

    def set_K_ready(self, in_addr):
        '''
            Set one input element of K ready, -1 means not ready
        '''
        assert self.K_ready_index < self.N * self.d
        self.K_ready[self.K_ready_index] += 1  # because the elements are always generated in order from previous layer, we use an index to keep track, not in_addr
        self.K_ready_index += 1

    def check_ready(self, tok_index, start_dim):
        '''
            Check if all input elements are ready to start computation
            If yes, return the addresses of all the input elements
            If no, return None
        '''
        Q_addresses = []
        K_addresses = []
        for i in range(self.num_macs):
            if self.Q_ready[tok_index*self.d + start_dim + i] == -1:
                return None
            Q_addresses.append(tok_index*self.d + start_dim + i)
            for j in range(self.N):
                if self.K_ready[j*self.d + start_dim + i] == -1:
                    return None
                K_addresses.append(j*self.d + start_dim + i)
        return Q_addresses, K_addresses

    def add_event(self, event_type, event_time):
        '''
            Add a new event into the event queue
        '''
        event = {"EVENT_TYPE": event_type, "EVENT_TIME": event_time}
        self.event_queue.append(event)
        self.event_queue = sorted(self.event_queue, key=lambda d: d['EVENT_TIME'])

    def read(self, Q_addresses, K_addresses):
        '''
            Read all the input elements from the input SRAM
        '''
        # read Q

        num_reads = divide_up(4, self.Q_sram_phit_size//C.BIT_WIDTH)
        Q_latency = C.SRAM_LAT * num_reads

        self.energy_stats["sram_read_energy"] += C.SRAM_READ_ENERGY(len(Q_addresses) * C.BYTES_PER_ELEM)

        # read K

        num_reads = divide_up(self.N * 4, self.K_sram_phit_size//C.BIT_WIDTH)
        K_latency = C.SRAM_LAT * num_reads

        self.energy_stats["sram_read_energy"] += C.SRAM_READ_ENERGY(len(K_addresses) * C.BYTES_PER_ELEM)

        return max(Q_latency, K_latency)

    def compute(self):
        '''
            MAC
        '''
        latency = C.MAC_LAT

        self.energy_stats["mac_energy"] += C.MAC_ENERGY * self.num_macs * self.N

        return latency

    def write(self):
        '''
            Write the element-wise output into output SRAM
        '''
        latency = C.SRAM_LAT

        self.energy_stats["sram_write_energy"] += C.SRAM_WRITE_ENERGY(self.N * C.BYTES_PER_ELEM)

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
                Q_addresses, K_addresses = addresses
                self.start_dim += self.num_macs
                if self.start_dim >= self.d:
                    self.start_dim = 0
                    self.tok_index += 1
                    if self.tok_index >= self.N:
                        self.tok_index = 0
                if self.tok_index > 0 or self.start_dim > 0:
                    self.add_event(C.EVENT_NEW_DATA, cur_time+1)

                # Read data from SRAM
                latency += self.read(Q_addresses, K_addresses)
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
                if self.next_layer is not None:
                    if (self.num_finished_compute + 1) % divide_up(self.d, self.num_macs) == 0:
                        for i in range(self.N):
                            self.next_layer.set_ready(i)
                        self.next_layer.add_event(C.EVENT_NEW_DATA, cur_time)

                # check if this layer has just finished all computation
                self.num_finished_compute += 1
                if self.num_finished_compute == self.N * divide_up(self.d, self.num_macs):
                    self.done = True
                return -2

        return ret_latency
