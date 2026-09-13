import argparse

import nn.constant as C

from models.test import test_model
from models.lenet import lenet_model
from models.resnet import resnet_model
from models.vggnet import vgg_model
from models.single_layer import single_layer_model_full, single_layer_model_small, single_layer_model_half
from models.attention import attention_model


# All supported models
model_list = {
    "test":                 test_model,
    "lenet":                lenet_model,
    "resnet":               resnet_model,
    "vgg":                  vgg_model,
    "single_layer_full":    single_layer_model_full,
    "single_layer_small":   single_layer_model_small,
    "single_layer_half":    single_layer_model_half,
    "attention":            attention_model,
}

# Command line arguments
parser = argparse.ArgumentParser(description="Simulator for AzureLily")
parser.add_argument("--model", type=str, help="Model type name.",)
parser.add_argument("--seq_length", type=int, default=128, help="Sequence length, i.e., number of tokens (only used for attention).",)
parser.add_argument("--head_dim", type=int, default=128, help="Dimension of attention head (only used for attention).",)
parser.add_argument("--num_computes", type=int, default=1, help="Number of DPE replicates.",)
parser.add_argument("--num_inputs", type=int, default=1, help="Number of inputs.",)
parser.add_argument("--num_adds", type=int, default=16, help="Number of adders.",)
parser.add_argument("--num_maxpools", type=int, default=16, help="Number of maxpool.",)
parser.add_argument("--num_acts", type=int, default=16, help="Number of activations.",)
parser.add_argument("--phit_size", type=int, default=16, help="Bits that can be read/write at the same time to/from buffer/SRAM.",)
parser.add_argument("--phit_size_add", type=int, default=16, help="Bits that can be read/write at the same time to/from buffer/SRAM.",)
parser.add_argument('--debug', action='store_true', help='Print cycle information during running',)
args = parser.parse_args()

# Update parameters
setattr(C, "NUM_ADDS", args.num_adds)
setattr(C, "NUM_MAXPOOLS", args.num_maxpools)
setattr(C, "NUM_ACTS", args.num_acts)
setattr(C, "PHIT_SIZE", args.phit_size)
setattr(C, "ELEMS_PER_MV", args.phit_size / C.BIT_WIDTH)
setattr(C, "PHIT_SIZE_ADD", args.phit_size_add)
setattr(C, "ELEMS_PER_MV_ADD", args.phit_size_add / C.BIT_WIDTH)

# Energy statistics
energy_stats = {
    "sram_read_energy" : 0,
    "sram_write_energy" : 0,
    "external_buffer_write_energy" : 0,
    "external_buffer_read_energy" : 0,
    "internal_buffer_write_energy" : 0,
    "output_buffer_write_energy" : 0,
    "output_buffer_read_energy" : 0,
    "adc_energy" : 0,
    "maxpool_energy" : 0,
    "act_energy" : 0,
    "sum_energy" : 0,
    "mac_energy" : 0,
    "mul_energy" : 0,
}

# Every layer of the target model
all_layers, num_finishes = model_list[args.model](args.num_computes, args.num_inputs, args.seq_length, args.head_dim, args.debug, energy_stats)


###################################################################################
# DO NOT MODIFY

cur_time = 0
finish_times = []
while True:
    if args.debug:
        print("=========================================================")
        print("Cycle:", cur_time, "ns \n")

    finish = False
    next_time = float("inf")
    for i, layer in enumerate(all_layers):
        t = layer.update(cur_time)
        if t == -1:
            continue
        elif t == -2:
            finish_times.append(cur_time)
            if len(finish_times) == num_finishes:
                finish = True
                break
        elif t != float("inf"):
             next_time = min(t, next_time)

    if finish:
        break
    elif next_time != float("inf"):
        cur_time = next_time
    else:
        cur_time += 1

# DO NOT MODIFY
###################################################################################
    
print("#####################################################")
print("Total latency:", cur_time, "ns")
if args.num_inputs > 1:
    saturated_lat = 0
    for i in range(1, len(finish_times)):
        saturated_lat += finish_times[i] - finish_times[i-1]
    saturated_lat /= len(finish_times) - 1
    throughput = 1e9 / saturated_lat
    print("Pipeline latency:", saturated_lat, "ns")
    print("Throughput:", throughput, "ips")

print("#####################################################")
total_energy = 0
for k, v in energy_stats.items():
    print(k, v)
    total_energy += v
print("-----------------------------")
print("Total energy:", total_energy, "nJ")
