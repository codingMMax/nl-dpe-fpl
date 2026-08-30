"""FC+activation model for IMC simulator: y = act(Wx).

Identical to gemv_model (azurelily/models/gemv.py) except has_act=True.

The simulator's _run_linear (scheduler.py) handles the V=1 vs V>1 branching:
  V=1 (K <= crossbar rows): analoge_nonlinear_check() returns True
      → ACAM absorbs activation, no CLB energy or latency added
  V>1 (K > crossbar rows): analoge_nonlinear_check() returns False
      → fpga.activation(M, N) called, adds CLB energy and latency

The optional ``M`` argument threads a batch / GEMM row dimension through to
the scheduler.  When M=1 (default) behaviour is byte-identical to the
pre-batch call path; when M>1 the FC layer's ``num_inputs`` is set to M so
that ``_run_linear`` picks it up and calls ``run_gemm(M, K, N)`` / the
``gemm_log`` formula with the correct batch factor.

Usage:
    python azurelily/IMC/test.py --model fc --imc_file IMC/configs/nl_dpe.json \\
        --seq_length 64 --head_dim 64 [--batch 128]
"""

import nn
import nn.constant as C


def fc_model(num_computes, num_inputs, seq_length, head_dim, debug, energy_stats,
             M=1):
    """Single FC+activation layer model.

    K = seq_length (input dimension / crossbar rows consumed)
    N = head_dim   (output dimension / crossbar cols consumed)
    M = batch / GEMM row dimension (default 1 = original single-input path)
    """
    K = seq_length
    N = head_dim

    # Preserve M=1 default (byte-identical to the pre-batch path).  When
    # M>1 we thread the batch dimension through the layer's num_inputs so
    # the scheduler's _run_linear sees M = layer.num_inputs = M.
    effective_num_inputs = M if M > 1 else num_inputs

    layer = nn.Layer(
        in_channels=K,
        out_channels=N,
        kernel_size=1,
        stride=1,
        padding=0,
        name=f"fc_{K}_{N}",
        type="linear",
        has_act=True,          # only difference from gemv_model
        num_computes=num_computes,
        num_inputs=effective_num_inputs,
        debug=debug,
        energy_stats=energy_stats,
    )

    layer.set_input(1, 1, K)
    all_layers = [layer]
    all_layers[0].add_event(C.EVENT_NEW_DATA, 0)
    return all_layers, effective_num_inputs * len(all_layers)
