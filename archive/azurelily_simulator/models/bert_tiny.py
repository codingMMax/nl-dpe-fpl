"""BERT-Tiny model for IMC simulator.

Architecture (from TinyBERT / prajjwal1/bert-tiny):
  - 2 transformer layers, 2 attention heads
  - d_model=128, d_head=64, d_ff=512
  - GELU activation in FFN (modeled same as existing activation: ACAM if V=1, CLB if V>1)
  - LayerNorm after attention and FFN sub-layers
  - Embedding: token (30522) + position (512) + segment (2) + LayerNorm

Usage:
    python azurelily/IMC/test.py --model bert_tiny --imc_file IMC/configs/nl_dpe.json \\
        --seq_length 1024
"""

import nn
import nn.constant as C
from nn.utils import divide_up


# BERT-Tiny constants
D_MODEL = 128
NUM_HEADS = 2
D_HEAD = D_MODEL // NUM_HEADS   # 64
D_FF = 512
NUM_LAYERS = 2
VOCAB_SIZE = 30522
MAX_POS_EMBEDDINGS = 512


def bert_tiny_model(num_computes, num_inputs, seq_length, head_dim, debug, energy_stats):
    """Build BERT-Tiny model for the IMC analytical simulator.

    Parameters
    ----------
    seq_length : int
        Input sequence length (default 1024 from CLI).
    head_dim : int
        Ignored (d_model=128 is fixed for BERT-Tiny).

    Returns
    -------
    model_desc : dict
        Structured model description for run_bert_model() traversal.
    num_finishes : int
        Placeholder (not used by analytical scheduler path).
    """
    S = seq_length  # shorthand

    # ── Embedding ────────────────────────────────────────────────────
    embedding = nn.Embedding_Layer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        max_seq_len=MAX_POS_EMBEDDINGS,
        seq_len=S,
        name="embedding",
        debug=debug,
        energy_stats=energy_stats,
    )

    embed_ln = nn.LayerNorm_Layer(
        normalized_shape=D_MODEL,
        seq_len=S,
        name="embed_ln",
        debug=debug,
        energy_stats=energy_stats,
    )

    # ── Transformer Blocks ──────────────────────────────────────────
    blocks = []
    for blk in range(NUM_LAYERS):
        prefix = f"block{blk}"

        # --- Q / K / V projections (full d_model → d_model) ---
        q_proj = nn.Linear_Layer(
            in_channels=D_MODEL,
            out_channels=D_MODEL,
            out_sram_phit_size=32,
            name=f"{prefix}_Q",
            num_inputs=S,
            debug=debug,
            energy_stats=energy_stats,
        )

        k_proj = nn.Linear_Layer(
            in_channels=D_MODEL,
            out_channels=D_MODEL,
            out_sram_phit_size=32,
            name=f"{prefix}_K",
            num_inputs=S,
            debug=debug,
            energy_stats=energy_stats,
        )

        v_proj = nn.Linear_Layer(
            in_channels=D_MODEL,
            out_channels=D_MODEL,
            out_sram_phit_size=8,
            name=f"{prefix}_V",
            num_inputs=S,
            debug=debug,
            energy_stats=energy_stats,
        )

        # --- Per-head attention (single head, d_head=64) ---
        # We create layers for ONE head; run_bert_model scales energy × num_heads.
        mac_qk = nn.MAC_QK_Layer(
            d=D_HEAD,
            N=S,
            num_macs=4,
            Q_sram_phit_size=32,
            K_sram_phit_size=S * 32,
            name=f"{prefix}_head_mac_qk",
            debug=debug,
            energy_stats=energy_stats,
        )

        # Softmax over attention scores: (S × S) matrix
        # d=S because each row has S elements to softmax over
        softmax_exp = nn.Softmax_Exp_Layer(
            d=S,
            N=S,
            name=f"{prefix}_head_softmax_exp",
            debug=debug,
            energy_stats=energy_stats,
        )

        softmax_norm = nn.Softmax_Norm_Layer(
            d=S,
            N=S,
            name=f"{prefix}_head_softmax_norm",
            debug=debug,
            energy_stats=energy_stats,
        )

        mac_sv = nn.MAC_SV_Layer(
            d=D_HEAD,
            N=S,
            num_macs=4,
            V_sram_phit_size=8 * D_HEAD,
            name=f"{prefix}_head_mac_sv",
            debug=debug,
            energy_stats=energy_stats,
        )

        # --- Output projection ---
        o_proj = nn.Layer(
            in_channels=D_MODEL,
            out_channels=D_MODEL,
            kernel_size=1,
            stride=1,
            padding=0,
            name=f"{prefix}_O_proj",
            type="linear",
            has_act=False,
            num_computes=num_computes,
            num_inputs=S,
            debug=debug,
            energy_stats=energy_stats,
        )
        o_proj.set_input(1, 1, D_MODEL)

        # --- Residual + LayerNorm (post-attention) ---
        attn_residual = nn.Layer(
            in_channels=D_MODEL,
            out_channels=D_MODEL,
            kernel_size=1,
            stride=1,
            padding=0,
            name=f"{prefix}_attn_residual",
            type="residual",
            has_act=False,
            num_computes=num_computes,
            num_inputs=S,
            debug=debug,
            energy_stats=energy_stats,
        )
        attn_residual.set_input(1, 1, D_MODEL)

        attn_ln = nn.LayerNorm_Layer(
            normalized_shape=D_MODEL,
            seq_len=S,
            name=f"{prefix}_attn_ln",
            debug=debug,
            energy_stats=energy_stats,
        )

        # --- FFN ---
        ffn1 = nn.Layer(
            in_channels=D_MODEL,
            out_channels=D_FF,
            kernel_size=1,
            stride=1,
            padding=0,
            name=f"{prefix}_ffn1",
            type="linear",
            has_act=True,      # GELU — same cost model as existing activation
            num_computes=num_computes,
            num_inputs=S,
            debug=debug,
            energy_stats=energy_stats,
        )
        ffn1.set_input(1, 1, D_MODEL)

        ffn2 = nn.Layer(
            in_channels=D_FF,
            out_channels=D_MODEL,
            kernel_size=1,
            stride=1,
            padding=0,
            name=f"{prefix}_ffn2",
            type="linear",
            has_act=False,
            num_computes=num_computes,
            num_inputs=S,
            debug=debug,
            energy_stats=energy_stats,
        )
        ffn2.set_input(1, 1, D_FF)

        # --- Residual + LayerNorm (post-FFN) ---
        ffn_residual = nn.Layer(
            in_channels=D_MODEL,
            out_channels=D_MODEL,
            kernel_size=1,
            stride=1,
            padding=0,
            name=f"{prefix}_ffn_residual",
            type="residual",
            has_act=False,
            num_computes=num_computes,
            num_inputs=S,
            debug=debug,
            energy_stats=energy_stats,
        )
        ffn_residual.set_input(1, 1, D_MODEL)

        ffn_ln = nn.LayerNorm_Layer(
            normalized_shape=D_MODEL,
            seq_len=S,
            name=f"{prefix}_ffn_ln",
            debug=debug,
            energy_stats=energy_stats,
        )

        blocks.append({
            "qkv_proj": [q_proj, k_proj, v_proj],
            "head_attention": [mac_qk, softmax_exp, softmax_norm, mac_sv],
            "num_heads": NUM_HEADS,
            "post_attn": [o_proj, attn_residual, attn_ln],
            "ffn": [ffn1, ffn2, ffn_residual, ffn_ln],
        })

    model_desc = {
        "embedding": [embedding, embed_ln],
        "blocks": blocks,
    }

    return model_desc, 0
