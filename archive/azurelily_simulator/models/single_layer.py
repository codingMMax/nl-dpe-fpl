import nn
import nn.constant as C


def single_layer_model_full(num_computes, num_inputs, seq_length, head_dim, debug, energy_stats):
    conv1 = nn.Layer(
        in_channels=32,
        out_channels=128,
        kernel_size=4,
        stride=1,
        padding=0,
        name="conv1",
        type="conv2d",
        has_act = True,
        num_computes = num_computes,
        num_inputs = num_inputs,
        debug = debug,
        energy_stats = energy_stats,
    )

    conv1.set_input(32, 32, 32)

    all_layers = []
    all_layers.append(conv1)

    all_layers[0].add_event(C.EVENT_NEW_DATA, 0)

    num_finishes = num_inputs * len(all_layers)

    return all_layers, num_finishes


def single_layer_model_small(num_computes, num_inputs, seq_length, head_dim, debug, energy_stats):
    conv1 = nn.Layer(
        in_channels=1,
        out_channels=128,
        kernel_size=2,
        stride=1,
        padding=0,
        name="conv1",
        type="conv2d",
        has_act = True,
        num_computes = num_computes,
        num_inputs = num_inputs,
        debug = debug,
        energy_stats = energy_stats,
    )

    conv1.set_input(32, 32, 1)

    all_layers = []
    all_layers.append(conv1)

    all_layers[0].add_event(C.EVENT_NEW_DATA, 0)

    num_finishes = num_inputs * len(all_layers)

    return all_layers, num_finishes


def single_layer_model_half(num_computes, num_inputs, seq_length, head_dim, debug, energy_stats):
    conv1 = nn.Layer(
        in_channels=16,
        out_channels=128,
        kernel_size=4,
        stride=1,
        padding=0,
        name="conv1",
        type="conv2d",
        has_act = True,
        num_computes = num_computes,
        num_inputs = num_inputs,
        debug = debug,
        energy_stats = energy_stats,
    )

    conv1.set_input(32, 32, 16)

    all_layers = []
    all_layers.append(conv1)

    all_layers[0].add_event(C.EVENT_NEW_DATA, 0)

    num_finishes = num_inputs * len(all_layers)

    return all_layers, num_finishes
