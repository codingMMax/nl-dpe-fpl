from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMC_DIR = ROOT / "IMC"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(IMC_DIR) not in sys.path:
    sys.path.insert(0, str(IMC_DIR))

import nn.constant as C
from nn.linear_layer import Linear_Layer
from models.lenet import lenet_model
from models.resnet import resnet_model
from models.vggnet import vgg_model
from models.single_layer import single_layer_model_full, single_layer_model_small, single_layer_model_half
from models.attention import attention_model


MODEL_LIST = {
    "lenet":                lenet_model,
    "resnet":               resnet_model,
    "vgg":                  vgg_model,
    "single_layer_full":    single_layer_model_full,
    "single_layer_small":   single_layer_model_small,
    "single_layer_half":    single_layer_model_half,
    "attention":            attention_model,
}


def build_energy_stats():
    return {
        "sram_read_energy": 0,
        "sram_write_energy": 0,
        "external_buffer_write_energy": 0,
        "external_buffer_read_energy": 0,
        "internal_buffer_write_energy": 0,
        "output_buffer_write_energy": 0,
        "output_buffer_read_energy": 0,
        "adc_energy": 0,
        "maxpool_energy": 0,
        "act_energy": 0,
        "sum_energy": 0,
        "mac_energy": 0,
        "mul_energy": 0,
    }


def update_constants(num_adds, num_maxpools, num_acts, phit_size, phit_size_add):
    setattr(C, "NUM_ADDS", num_adds)
    setattr(C, "NUM_MAXPOOLS", num_maxpools)
    setattr(C, "NUM_ACTS", num_acts)
    setattr(C, "PHIT_SIZE", phit_size)
    setattr(C, "ELEMS_PER_MV", phit_size / C.BIT_WIDTH)
    setattr(C, "PHIT_SIZE_ADD", phit_size_add)
    setattr(C, "ELEMS_PER_MV_ADD", phit_size_add / C.BIT_WIDTH)


def iter_layers(model_name, layers):
    if model_name != "attention":
        layer = layers[0]
        while layer is not None:
            yield layer
            layer = layer.next_layer
        return

    layer = layers[0]
    while layer is not None:
        yield layer
        if isinstance(layer, Linear_Layer):
            Q_next = layer.Q_next_layer
            K_next = layer.K_next_layer
            layer = Q_next if Q_next else K_next
        else:
            layer = layer.next_layer


def run_old_simulator(model_name, args):
    energy_stats = build_energy_stats()
    all_layers, num_finishes = MODEL_LIST[model_name](
        args.num_computes,
        args.num_inputs,
        args.seq_length,
        args.head_dim,
        args.debug,
        energy_stats,
    )

    cur_time = 0
    finish_times = []
    while True:
        if args.debug:
            print("=========================================================")
            print("Cycle:", cur_time, "ns \n")

        finish = False
        next_time = float("inf")
        for layer in all_layers:
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

    total_energy = sum(energy_stats.values())
    return {
        "total_latency": cur_time,
        "total_energy": total_energy,
        "energy_stats": energy_stats,
    }


def resource_usage(layer, cfg, imc_core, fpga):
    imc_tiles = 0
    dsp_used = 0
    clb_used = 0

    if layer.type in ("linear", "conv2d"):
        if layer.type == "conv2d":
            K = layer.kernel_size * layer.kernel_size * layer.in_channels
            M = layer.output_height * layer.output_width * layer.num_inputs
            N = layer.out_channels
        else:
            K = layer.in_channels
            M = layer.num_inputs
            N = layer.out_channels
        k_tile = math.ceil(K / cfg.rows)
        n_tile = math.ceil(N / cfg.cols)
        imc_tiles = k_tile * n_tile
        if layer.has_act and not imc_core.analoge_nonlinear_check(M, K, N):
            luts_per_act = 32
            luts_per_clb = 8
            clb_per_act = math.ceil(luts_per_act / luts_per_clb)
            act_parallelism = 4
            clb_used = act_parallelism * clb_per_act
        return imc_tiles, dsp_used, clb_used

    if layer.type in ("mac_qk", "mac_sv"):
        if cfg.imc == "NL-DPE":
            K = layer.d
            dsp_used = min(cfg.total_dsp, K)
            exp_clbs = _exp_clb_used(K)
            sum_clbs = max(0, K - 1)
            clb_used = max(exp_clbs, sum_clbs)
        else:
            dsp_used = cfg.total_dsp
        return imc_tiles, dsp_used, clb_used

    if layer.type == "softmax_exp":
        clb_used = _exp_clb_used(layer.d)
        return imc_tiles, dsp_used, clb_used

    if layer.type == "softmax_norm":
        sum_clbs = max(0, layer.d - 1)
        inv_clbs = _inv_clb_used(layer.d)
        clb_used = max(sum_clbs, inv_clbs)
        if cfg.total_dsp > 0:
            dsp_used = min(cfg.total_dsp, layer.d)
        return imc_tiles, dsp_used, clb_used

    if layer.type == "maxpool":
        window_elements = layer.kernel_size * layer.kernel_size
        clb_used = math.ceil(window_elements / 2)
        return imc_tiles, dsp_used, clb_used

    if layer.type == "residual":
        dsp_per_lane = math.ceil((layer.output_height * layer.output_width * layer.num_computes - 1) / (4 - 1))
        total_required = dsp_per_lane * layer.out_channels
        dsp_used = min(cfg.total_dsp, total_required)
        return imc_tiles, dsp_used, clb_used

    return imc_tiles, dsp_used, clb_used


def _exp_clb_used(vec_length):
    INPUT_WIDTH = 8
    LUT_PER_CLB = 8
    OUTPUT_WIDTH = 16
    LUT_WIDTH = 6
    total_store_values = 2 ** INPUT_WIDTH
    total_store_bits = OUTPUT_WIDTH * total_store_values
    total_luts = math.ceil(total_store_bits / (2 ** LUT_WIDTH))
    return math.ceil(total_luts / LUT_PER_CLB)


def _inv_clb_used(vec_length):
    INV_INPUT_WIDTH = 8
    INV_OUTPUT_WIDTH = 16
    LUT_WIDTH = 6
    LUT_PER_CLB = 8
    total_store_values = 2 ** INV_INPUT_WIDTH
    total_store_bits = INV_OUTPUT_WIDTH * total_store_values
    total_luts = math.ceil(total_store_bits / (2 ** LUT_WIDTH))
    return math.ceil(total_luts / LUT_PER_CLB)


def run_new_simulator(model_name, imc_file, args, simulator_cls):
    energy_stats = build_energy_stats()
    imc = simulator_cls(imc_file)
    all_layers, _ = MODEL_LIST[model_name](
        args.num_computes,
        args.num_inputs,
        args.seq_length,
        args.head_dim,
        args.debug,
        energy_stats,
    )

    records = []
    for layer in iter_layers(model_name, all_layers):
        lat, energy = imc.run_layer(layer)
        layer_res = imc.resource_layer.get(layer.name, {})
        imc_tiles = layer_res.get("imc_tiles", 0.0)
        dsp_used = layer_res.get("dsp_used", 0.0)
        clb_used = layer_res.get("clb_used", 0.0)
        records.append({
            "layer": layer.name,
            "type": layer.type,
            "latency": lat,
            "energy": energy,
            "imc_tiles": imc_tiles,
            "dsp_used": dsp_used,
            "clb_used": clb_used,
            "bram_read_cnt_access": layer_res.get("bram_read_cnt_access", 0.0),
            "bram_write_cnt_access": layer_res.get("bram_write_cnt_access", 0.0),
        })

    imc.finalize_latency_stats()
    return {
        "records": records,
        "energy_stats": dict(imc.energy_stats),
        "energy_breakdown": dict(imc.energy_breakdown),
        "latency_stats": dict(imc.latency_stats),
        "latency_raw": dict(imc.latency_raw),
        "total_energy": sum(imc.energy_stats.values()),
        "total_latency": sum(imc.latency_stats.values()),
        "cfg": imc.cfg,
        "resource_total": dict(imc.resource_total),
        "resource_peak": dict(imc.resource_peak),
        "resource_layer": dict(imc.resource_layer),
    }


def _resource_total(records):
    return {
        "imc_tiles": sum(r["imc_tiles"] for r in records),
        "dsp_used": sum(r["dsp_used"] for r in records),
        "clb_used": sum(r["clb_used"] for r in records),
    }


def format_markdown_table(headers, rows):
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def format_text_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    lines = [fmt.format(*headers)]
    lines.append(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        lines.append(fmt.format(*row))
    return "\n".join(lines)
