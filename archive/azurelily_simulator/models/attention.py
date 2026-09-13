import nn
import nn.constant as C
from nn.utils import *


def attention_model(num_computes, num_inputs, seq_length, head_dim, debug, energy_stats,
                    M=1):
    # ``M`` (batch / row dim) is accepted for signature parity with the
    # extended fc_model so --batch can be forwarded uniformly; attention
    # derives its own row dimension from seq_length so a non-default M is
    # currently a no-op (it does not change layer wiring).  Keep M=1 as
    # default to preserve byte-identical output for the regression path.
    assert num_inputs==1, "Currently attention only support 1 input"
    _ = M  # reserved for future per-batch attention scheduling

    linear_Q = nn.Linear_Layer(
        in_channels=head_dim,
        out_channels=head_dim,
        out_sram_phit_size=32,
        name="linear_Q",
        num_inputs = seq_length,
        debug = debug,
        energy_stats = energy_stats,
    )

    linear_K = nn.Linear_Layer(
        in_channels=head_dim,
        out_channels=head_dim,
        out_sram_phit_size=32,
        name="linear_K",
        num_inputs = seq_length,
        debug = debug,
        energy_stats = energy_stats,
    )

    linear_V = nn.Linear_Layer(
        in_channels=head_dim,
        out_channels=head_dim,
        out_sram_phit_size=8,
        name="linear_V",
        num_inputs = seq_length,
        debug = debug,
        energy_stats = energy_stats,
    )

    mac_qk = nn.MAC_QK_Layer(
        d=head_dim,
        N=seq_length,
        num_macs=4,
        Q_sram_phit_size=32,
        K_sram_phit_size=seq_length*32,
        name="mac_qk",
        debug=debug,
        energy_stats = energy_stats,
    )

    exp = nn.Softmax_Exp_Layer(
        d=head_dim,
        N=seq_length,
        name="exp",
        debug=debug,
        energy_stats = energy_stats,
    )

    norm = nn.Softmax_Norm_Layer(
        d=head_dim,
        N=seq_length,
        name="norm",
        debug=debug,
        energy_stats = energy_stats,
    )

    mac_sv = nn.MAC_SV_Layer(
        d=head_dim,
        N=seq_length,
        num_macs=4,
        V_sram_phit_size=8*head_dim,
        name="mac_sv",
        debug=debug,
        energy_stats = energy_stats,
    )

    linear_Q.set_first()
    linear_K.set_first()
    linear_V.set_first()
    linear_Q.Q_set_next_layer(mac_qk)
    linear_K.K_set_next_layer(mac_qk)
    mac_qk.set_next_layer(exp)
    exp.set_next_layer(norm)
    norm.set_next_layer(mac_sv)
    linear_V.V_set_next_layer(mac_sv)

    all_layers = []
    all_layers.append(linear_Q)
    all_layers.append(linear_K)
    all_layers.append(linear_V)
    all_layers.append(mac_qk)
    all_layers.append(exp)
    all_layers.append(norm)
    all_layers.append(mac_sv)

    all_layers[0].add_event(C.EVENT_NEW_DATA, 0)
    all_layers[1].add_event(C.EVENT_NEW_DATA, 0)
    all_layers[2].add_event(C.EVENT_NEW_DATA, 0)

    #              linear_K     linear_V      linear_Q                 mac_qk                   exp           norm                   mac_sv
    num_finishes = seq_length + seq_length + seq_length + seq_length*divide_up(head_dim, 4) + seq_length + seq_length + seq_length*divide_up(seq_length, 4)

    return all_layers, num_finishes
