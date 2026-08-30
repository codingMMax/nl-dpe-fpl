import nn
import nn.constant as C


def operator_model(num_computes, num_inputs, seq_length, head_dim, debug, energy_stats):
    conv1 = nn.Layer(
        in_channels=3,
        out_channels=8,
        kernel_size=3,
        stride=1,
        padding=1,
        name="conv1",
        type="conv2d",
        has_act=True,
        num_computes=num_computes,
        num_inputs=num_inputs,
        debug=debug,
        energy_stats=energy_stats,
    )

    pool1 = nn.Layer(
        in_channels=8,
        out_channels=8,
        kernel_size=2,
        stride=2,
        padding=0,
        name="pool1",
        type="maxpool",
        has_act=False,
        num_computes=num_computes,
        num_inputs=num_inputs,
        debug=debug,
        energy_stats=energy_stats,
    )

    full1 = nn.Layer(
        in_channels=8 * 16 * 16,
        out_channels=10,
        kernel_size=1,
        stride=1,
        padding=0,
        name="full1",
        type="linear",
        has_act=False,
        num_computes=num_computes,
        num_inputs=num_inputs,
        debug=debug,
        energy_stats=energy_stats,
    )

    conv1.set_input(32, 32, 3)
    conv1.set_next_layer(pool1)
    pool1.set_next_layer(full1)

    all_layers = [conv1, pool1, full1]
    all_layers[0].add_event(C.EVENT_NEW_DATA, 0)

    num_finishes = num_inputs * len(all_layers)
    return all_layers, num_finishes
