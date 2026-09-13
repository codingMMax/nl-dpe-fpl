import nn
import nn.constant as C


def lenet_model(num_computes, num_inputs, seq_length, head_dim, debug, energy_stats):
    conv1 = nn.Layer(
        in_channels=1,
        out_channels=6,
        kernel_size=5,
        stride=1,
        padding=2,
        name="conv1",
        type="conv2d",
        has_act = True,
        num_computes = num_computes,
        num_inputs = num_inputs,
        debug = debug,
        energy_stats = energy_stats,
    )

    pool1 = nn.Layer(
        in_channels=6,
        out_channels=6,
        kernel_size=2,
        stride=2,
        padding=0,
        name="pool1",
        type="maxpool",
        has_act = False,
        num_computes = num_computes,
        num_inputs = num_inputs,
        debug = debug,
        energy_stats = energy_stats,
    )

    conv2 = nn.Layer(
        in_channels=6,
        out_channels=16,
        kernel_size=5,
        stride=1,
        padding=0,
        name="conv2",
        type="conv2d",
        has_act = True,
        num_computes = num_computes,
        num_inputs = num_inputs,
        debug = debug,
        energy_stats = energy_stats,
    )

    pool2 = nn.Layer(
        in_channels=16,
        out_channels=16,
        kernel_size=2,
        stride=2,
        padding=0,
        name="pool2",
        type="maxpool",
        has_act = False,
        num_computes = num_computes,
        num_inputs = num_inputs,
        debug = debug,
        energy_stats = energy_stats,
    )

    full1 = nn.Layer(
        in_channels=16*5*5,
        out_channels=120,
        kernel_size=1,
        stride=1,
        padding=0,
        name="full1",
        type="linear",
        has_act = True,
        num_computes = num_computes,
        num_inputs = num_inputs,
        debug = debug,
        energy_stats = energy_stats,
    )

    full2 = nn.Layer(
        in_channels=120,
        out_channels=84,
        kernel_size=1,
        stride=1,
        padding=0,
        name="full2",
        type="linear",
        has_act = True,
        num_computes = num_computes,
        num_inputs = num_inputs,
        debug = debug,
        energy_stats = energy_stats,
    )

    full3 = nn.Layer(
        in_channels=84,
        out_channels=10,
        kernel_size=1,
        stride=1,
        padding=0,
        name="full3",
        type="linear",
        has_act = False,
        num_computes = num_computes,
        num_inputs = num_inputs,
        debug = debug,
        energy_stats = energy_stats,
    )

    conv1.set_input(28, 28, 1)
    conv1.set_next_layer(pool1)
    pool1.set_next_layer(conv2)
    conv2.set_next_layer(pool2)
    pool2.set_next_layer(full1)
    full1.set_next_layer(full2)
    full2.set_next_layer(full3)

    all_layers = []
    all_layers.append(conv1)
    all_layers.append(pool1)
    all_layers.append(conv2)
    all_layers.append(pool2)
    all_layers.append(full1)
    all_layers.append(full2)
    all_layers.append(full3)

    all_layers[0].add_event(C.EVENT_NEW_DATA, 0)

    num_finishes = num_inputs * len(all_layers)

    return all_layers, num_finishes