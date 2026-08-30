import nn
import nn.constant as C


def test_model(num_computes, num_inputs, seq_length, head_dim, debug, energy_stats):
    conv1 = nn.Layer(
        in_channels=2,
        out_channels=2,
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

    pool1 = nn.Layer(
        in_channels=2,
        out_channels=2,
        kernel_size=2,
        stride=1,
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
        in_channels=2,
        out_channels=2,
        kernel_size=2,
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

    full1 = nn.Layer(
        in_channels=2,
        out_channels=2,
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

    conv1.set_input(4, 4, 2)
    conv1.set_next_layer(pool1)
    pool1.set_next_layer(conv2)
    conv2.set_next_layer(full1)

    all_layers = []
    all_layers.append(conv1)
    all_layers.append(pool1)
    all_layers.append(conv2)
    all_layers.append(full1)

    all_layers[0].add_event(C.EVENT_NEW_DATA, 0)

    num_finishes = num_inputs * len(all_layers)

    return all_layers, num_finishes