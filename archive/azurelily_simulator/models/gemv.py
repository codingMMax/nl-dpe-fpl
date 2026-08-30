"""GEMV model for IMC simulator: y = Wx (weight-persistent matrix-vector multiply).

Maps GEMV to a single linear layer:
  - in_channels  = K (input vector dimension = crossbar rows consumed)
  - out_channels = N (output vector dimension = crossbar cols consumed)
  - kernel_size  = 1
  - input: 1x1xK (single vector)

5 GEMV shapes for DSE (paper_questions.md), named gemv_{M}_{K}_{N}:
  gemv_1_64_64:     M=1, K=64,   N=64
  gemv_1_512_128:   M=1, K=512,  N=128
  gemv_1_2048_256:  M=1, K=2048, N=256
  gemv_1_256_512:   M=1, K=256,  N=512
  gemv_1_512_512:   M=1, K=512,  N=512
"""

import nn
import nn.constant as C

GEMV_SHAPES = {
    "1_64_64":     (64,   64),
    "1_512_128":   (512,  128),
    "1_2048_256":  (2048, 256),
    "1_256_512":   (256,  512),
    "1_512_512":   (512,  512),
}


def gemv_model(num_computes, num_inputs, seq_length, head_dim, debug, energy_stats,
               K=64, N=64):
    """Create a single-layer GEMV model.

    K and N are always taken from seq_length and head_dim (passed via CLI).
    """
    K = seq_length
    N = head_dim

    layer = nn.Layer(
        in_channels=K,
        out_channels=N,
        kernel_size=1,
        stride=1,
        padding=0,
        name=f"gemv_1_{K}_{N}",
        type="linear",
        has_act=False,        # GEMV has no activation (ACAM mode depends on tiling)
        num_computes=num_computes,
        num_inputs=num_inputs,
        debug=debug,
        energy_stats=energy_stats,
    )

    layer.set_input(1, 1, K)

    all_layers = [layer]
    all_layers[0].add_event(C.EVENT_NEW_DATA, 0)

    num_finishes = num_inputs * len(all_layers)
    return all_layers, num_finishes
