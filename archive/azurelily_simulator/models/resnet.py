import nn
import nn.constant as C


def resnet_model(num_computes, num_inputs, seq_length, head_dim, debug, energy_stats):
    conv1 = nn.Layer(
        in_channels=3,
        out_channels=56,
        kernel_size=3,
        stride=1,
        padding=1,
        name="conv1",
        type="conv2d",
        has_act = True,
        num_computes = num_computes,
        num_inputs = num_inputs,
        debug = debug,
        energy_stats = energy_stats,
    )

    conv2 = nn.Layer(
        in_channels=56,
        out_channels=112,
        kernel_size=3,
        stride=1,
        padding=1,
        name="conv2",
        type="conv2d",
        has_act = True,
        num_computes = num_computes,
        num_inputs = num_inputs,
        debug = debug,
        energy_stats = energy_stats,
    )

    pool1 = nn.Layer(
        in_channels=112,
        out_channels=112,
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

    conv3 = nn.Layer(
        in_channels=112,
        out_channels=112,
        kernel_size=3,
        stride=1,
        padding=1,
        name="conv3",
        type="conv2d",
        has_act = True,
        num_computes = num_computes,
        num_inputs = num_inputs,
        debug = debug,
        energy_stats = energy_stats,
    )

    conv4 = nn.Layer(
        in_channels=112,
        out_channels=112,
        kernel_size=3,
        stride=1,
        padding=1,
        name="conv4",
        type="conv2d",
        has_act = True,
        num_computes = num_computes,
        num_inputs = num_inputs,
        debug = debug,
        energy_stats = energy_stats,
    )

    res1 = nn.Layer(
        in_channels=112,
        out_channels=112,
        kernel_size=1,
        stride=1,
        padding=0,
        name="res1",
        type="residual",
        has_act = False,
        num_computes = num_computes,
        num_inputs = num_inputs,
        debug = debug,
        energy_stats = energy_stats,
    )

    conv5 = nn.Layer(
        in_channels=112,
        out_channels=224,
        kernel_size=3,
        stride=1,
        padding=1,
        name="conv5",
        type="conv2d",
        has_act = True,
        num_computes = num_computes,
        num_inputs = num_inputs,
        debug = debug,
        energy_stats = energy_stats,
    )

    pool2 = nn.Layer(
        in_channels=224,
        out_channels=224,
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

    conv6 = nn.Layer(
        in_channels=224,
        out_channels=224,
        kernel_size=3,
        stride=1,
        padding=1,
        name="conv6",
        type="conv2d",
        has_act = True,
        num_computes = num_computes,
        num_inputs = num_inputs,
        debug = debug,
        energy_stats = energy_stats,
    )

    pool3 = nn.Layer(
        in_channels=224,
        out_channels=224,
        kernel_size=2,
        stride=2,
        padding=0,
        name="pool3",
        type="maxpool",
        has_act = False,
        num_computes = num_computes,
        num_inputs = num_inputs,
        debug = debug,
        energy_stats = energy_stats,
    )

    conv7 = nn.Layer(
        in_channels=224,
        out_channels=224,
        kernel_size=3,
        stride=1,
        padding=1,
        name="conv7",
        type="conv2d",
        has_act = True,
        num_computes = num_computes,
        num_inputs = num_inputs,
        debug = debug,
        energy_stats = energy_stats,
    )

    conv8 = nn.Layer(
        in_channels=224,
        out_channels=224,
        kernel_size=3,
        stride=1,
        padding=1,
        name="conv8",
        type="conv2d",
        has_act = True,
        num_computes = num_computes,
        num_inputs = num_inputs,
        debug = debug,
        energy_stats = energy_stats,
    )

    res2 = nn.Layer(
        in_channels=224,
        out_channels=224,
        kernel_size=1,
        stride=1,
        padding=0,
        name="res2",
        type="residual",
        has_act = False,
        num_computes = num_computes,
        num_inputs = num_inputs,
        debug = debug,
        energy_stats = energy_stats,
    )

    pool4 = nn.Layer(
        in_channels=224,
        out_channels=224,
        kernel_size=4,
        stride=4,
        padding=0,
        name="pool4",
        type="maxpool",
        has_act = False,
        num_computes = num_computes,
        num_inputs = num_inputs,
        debug = debug,
        energy_stats = energy_stats,
    )

    full1 = nn.Layer(
        in_channels=224,
        out_channels=10,
        kernel_size=1,
        stride=1,
        padding=0,
        name="full1",
        type="linear",
        has_act = False,
        num_computes = num_computes,
        num_inputs = num_inputs,
        debug = debug,
        energy_stats = energy_stats,
    )

    conv1.set_input(32, 32, 3)
    conv1.set_next_layer(conv2)
    conv2.set_next_layer(pool1)
    pool1.set_next_layer(conv3)
    conv3.set_next_layer(conv4)
    conv4.set_next_layer(res1)
    res1.set_next_layer(conv5)
    conv5.set_next_layer(pool2)
    pool2.set_next_layer(conv6)
    conv6.set_next_layer(pool3)
    pool3.set_next_layer(conv7)
    conv7.set_next_layer(conv8)
    conv8.set_next_layer(res2)
    res2.set_next_layer(pool4)
    pool4.set_next_layer(full1)

    all_layers = []
    all_layers.append(conv1)
    all_layers.append(conv2)
    all_layers.append(pool1)
    all_layers.append(conv3)
    all_layers.append(conv4)
    all_layers.append(res1)
    all_layers.append(conv5)
    all_layers.append(pool2)
    all_layers.append(conv6)
    all_layers.append(pool3)
    all_layers.append(conv7)
    all_layers.append(conv8)
    all_layers.append(res2)
    all_layers.append(pool4)
    all_layers.append(full1)

    all_layers[0].add_event(C.EVENT_NEW_DATA, 0)

    num_finishes = num_inputs * len(all_layers)

    return all_layers, num_finishes