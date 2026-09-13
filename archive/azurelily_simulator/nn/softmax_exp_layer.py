import math
import numpy as np

import nn.constant as C
from nn.utils import *
from nn import Layer

class Softmax_Exp_Layer(Layer):

    def __init__(
        self,
        d: int,        # attention head dim
        N: int,        # sequence length, number of tokens
        name: str,     # name of this layer, used for printing log
        debug: bool,   # print debugging information
        energy_stats,  # statistics
    ):
        '''
            Constructor
        '''
        self.d = d
        self.N = N
        self.name = name
        self.debug = debug
        self.energy_stats = energy_stats
        self.type = "softmax_exp"

        self.ready = np.full((N), -1)  # store the readiness

        self.ready_index = 0   # (range: 0 -- N-1)

        self.next_layer = None         # must be Softmax_Norm_Layer
        self.num_finished_compute = 0  # to record whether this layer has finsihed all computations, in total N
        self.done = False              # set True if all computations finishes
        self.event_queue = []          # Event queue

        self.read_available_time = 0     # the cycle at which the input SRAM is idle for a read request
        self.compute_available_time = 0  # the cycle at which the exp units are idle for new computation
        self.write_available_time = 0    # the cycle at which the output SRAM is idle for a write request

    def set_next_layer(self, next_layer):
        '''
            Chain up the the next layer.
        '''
        self.next_layer = next_layer

    def set_ready(self, in_addr):
        '''
            Set one input element ready, -1 means not ready
        '''
        assert in_addr < self.N
        self.ready[in_addr] += 1

    def check_ready(self, ready_index):
        '''
            Check if all input elements are ready to start computation
            If yes, return the addresses of all the input elements
            If no, return None
        '''
        addresses = []
        for i in range(self.N):
            if self.ready[i] < ready_index:
                return None
            addresses.append(i)
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
            Read all the input elements from the input SRAM
        '''
        latency = C.SRAM_LAT

        self.energy_stats["sram_read_energy"] += C.SRAM_READ_ENERGY(len(addresses) * C.BYTES_PER_ELEM)

        return latency

    def compute(self):
        '''
            Exp
        '''
        latency = C.ACT_LAT

        self.energy_stats["act_energy"] += C.ACT_ENERGY * self.N

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
                addresses = self.check_ready(self.ready_index)
                if addresses is None:
                    idx += 1
                    continue
                self.ready_index += 1

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
                    for i in range(self.N):
                        self.next_layer.set_ready(i)
                    self.next_layer.add_event(C.EVENT_NEW_DATA, cur_time)

                # check if this layer has just finished all computation
                self.num_finished_compute += 1
                if self.num_finished_compute == self.N:
                    self.done = True
                return -2

        return ret_latency
