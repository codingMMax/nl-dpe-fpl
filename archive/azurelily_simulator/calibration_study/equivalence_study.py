from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMC_DIR = ROOT / "IMC"
IMC_NEW_DIR = ROOT / "IMC_new"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(IMC_DIR) not in sys.path:
    sys.path.insert(0, str(IMC_DIR))
if str(IMC_NEW_DIR) not in sys.path:
    sys.path.insert(0, str(IMC_NEW_DIR))

import nn
import nn.constant as C
from calibration_study.operator_model import operator_model
from models.lenet import lenet_model
from models.resnet import resnet_model

from IMC.simulator import IMC as IMCNew
from IMC.imc_core.config import Config
from IMC.scheduler_stats import common as sim_common
from IMC.scheduler_stats.pipeline_profiler import PipelineProfiler


OPERATOR_ORDER = ("conv2d", "linear", "maxpool")


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


def _accumulate_energy_stats(dst, src):
    for k, v in src.items():
        dst[k] = dst.get(k, 0.0) + v


def update_constants(args):
    setattr(C, "NUM_ADDS", args.num_adds)
    setattr(C, "NUM_MAXPOOLS", args.num_maxpools)
    setattr(C, "NUM_ACTS", args.num_acts)
    setattr(C, "PHIT_SIZE", args.phit_size)
    setattr(C, "ELEMS_PER_MV", args.phit_size / C.BIT_WIDTH)
    setattr(C, "PHIT_SIZE_ADD", args.phit_size_add)
    setattr(C, "ELEMS_PER_MV_ADD", args.phit_size_add / C.BIT_WIDTH)


def build_operator_layers(args):
    energy_stats = build_energy_stats()
    layers, _ = operator_model(
        args.num_computes,
        args.num_inputs,
        args.seq_length,
        args.head_dim,
        args.debug,
        energy_stats,
    )
    return layers


def build_model_layers(model_fn, args):
    energy_stats = build_energy_stats()
    layers, num_finishes = model_fn(
        args.num_computes,
        args.num_inputs,
        args.seq_length,
        args.head_dim,
        args.debug,
        energy_stats,
    )
    return layers, num_finishes, energy_stats


def run_old_model(layers, num_finishes):
    cur_time = 0.0
    finish_times = []
    while True:
        finish = False
        next_time = float("inf")
        for layer in layers:
            t = layer.update(cur_time)
            if t == -1:
                continue
            if t == -2:
                finish_times.append(cur_time)
                if len(finish_times) == num_finishes:
                    finish = True
                    break
            elif t != float("inf"):
                next_time = min(t, next_time)

        if finish:
            break
        if next_time != float("inf"):
            cur_time = next_time
        else:
            cur_time += 1
    return cur_time


def clone_layer(template, energy_stats, debug):
    layer = nn.Layer(
        in_channels=template.in_channels,
        out_channels=template.out_channels,
        kernel_size=template.kernel_size,
        stride=template.stride,
        padding=template.padding,
        name=template.name,
        type=template.type,
        has_act=template.has_act,
        num_computes=template.num_computes,
        num_inputs=template.num_inputs,
        debug=debug,
        energy_stats=energy_stats,
    )
    layer.set_input(template.input_height, template.input_width, template.in_channels, True)
    return layer


def run_old_layer(template, args):
    energy_stats = build_energy_stats()
    layer = clone_layer(template, energy_stats, args.debug)
    layer.add_event(C.EVENT_NEW_DATA, 0)

    cur_time = 0.0
    finish_count = 0
    while True:
        t = layer.update(cur_time)
        if t == -2:
            finish_count += 1
            if finish_count == layer.num_inputs:
                break
            continue
        if t == -1:
            break
        if t != float("inf"):
            cur_time = t
        else:
            cur_time += 1

    return {
        "energy_stats": energy_stats,
        "total_latency": cur_time,
    }


def run_new_layer(template, args):
    imc = IMCNew(args.imc_file)
    layer = clone_layer(template, build_energy_stats(), args.debug)
    imc.run_layer(layer)
    imc.finalize_latency_stats()
    return {
        "energy_breakdown": dict(imc.energy_breakdown),
        "total_energy": sum(imc.energy_stats.values()),
        "total_latency": sum(imc.latency_stats.values()),
        "cfg": imc.cfg,
    }


def energy_buckets_old(energy_stats, unit_scale=1.0):
    memory = (
        energy_stats.get("sram_read_energy", 0)
        + energy_stats.get("sram_write_energy", 0)
        + energy_stats.get("external_buffer_write_energy", 0)
        + energy_stats.get("external_buffer_read_energy", 0)
        + energy_stats.get("internal_buffer_write_energy", 0)
        + energy_stats.get("output_buffer_write_energy", 0)
        + energy_stats.get("output_buffer_read_energy", 0)
    )
    core = energy_stats.get("adc_energy", 0) + energy_stats.get("sum_energy", 0)
    peripheral = (
        energy_stats.get("mac_energy", 0)
        + energy_stats.get("mul_energy", 0)
        + energy_stats.get("act_energy", 0)
        + energy_stats.get("maxpool_energy", 0)
    )
    memory *= unit_scale
    core *= unit_scale
    peripheral *= unit_scale
    return _bucket(memory, core, peripheral)


def energy_buckets_new(energy_breakdown, unit_scale=1.0):
    memory = energy_breakdown.get("sram_read", 0) + energy_breakdown.get("sram_write", 0)
    core = (
        energy_breakdown.get("imc_vmm", 0)
        + energy_breakdown.get("imc_conversion", 0)
        + energy_breakdown.get("imc_digital_post", 0)
    )
    peripheral = (
        energy_breakdown.get("dsp_gemm", 0)
        + energy_breakdown.get("dsp_add", 0)
        + energy_breakdown.get("clb_add", 0)
        + energy_breakdown.get("clb_exp", 0)
        + energy_breakdown.get("clb_norm_sum", 0)
        + energy_breakdown.get("clb_norm_inv", 0)
        + energy_breakdown.get("mul", 0)
        + energy_breakdown.get("clb_reduction", 0)
        + energy_breakdown.get("clb_compare", 0)
        + energy_breakdown.get("fpga_activation", 0)
    )
    memory *= unit_scale
    core *= unit_scale
    peripheral *= unit_scale
    return _bucket(memory, core, peripheral)


def pct_breakdown(bucket):
    total = bucket["total"] if bucket["total"] else 1.0
    return {
        "memory": bucket["memory"] / total,
        "core": bucket["core"] / total,
        "peripheral": bucket["peripheral"] / total,
        "total": total,
    }


def new_latency_breakdown(layers, cfg):
    memory = 0.0
    core = 0.0
    peripheral = 0.0

    def _assign_compute_time(total_ns, core_ns, periph_ns):
        nonlocal core, peripheral
        if total_ns <= 0:
            return
        if cfg.pipelinable:
            if periph_ns >= core_ns:
                peripheral += total_ns
            else:
                core += total_ns
        else:
            denom = max(1e-12, core_ns + periph_ns)
            core += total_ns * (core_ns / denom)
            peripheral += total_ns * (periph_ns / denom)

    for layer in layers:
        if layer.type in ("conv2d", "linear"):
            if layer.type == "conv2d":
                K = layer.kernel_size * layer.kernel_size * layer.in_channels
                M = layer.output_height * layer.output_width * layer.num_inputs
                N = layer.out_channels
            else:
                K = layer.in_channels
                M = layer.num_inputs
                N = layer.out_channels

            k_tile = math.ceil(K / cfg.rows)

            t_read_row = _bram_latency(cfg, K)
            t_write_row = _bram_latency(cfg, N)
            t_core_row = (
                _latency_per_vmm(cfg)
                + _latency_per_conv(cfg)
                + _latency_per_digital(cfg)
            )
            t_periph_row = _clb_reduction_latency(cfg, k_tile)
            t_compute_row = max(t_core_row, t_periph_row) if cfg.pipelinable else (t_core_row + t_periph_row)

            t_steady = max(t_read_row, t_compute_row, t_write_row)
            n_repeat = max(0, M - 1)

            memory += t_read_row + t_write_row
            _assign_compute_time(t_compute_row, t_core_row, t_periph_row)

            if n_repeat > 0:
                repeated = n_repeat * t_steady
                if t_steady == t_read_row or t_steady == t_write_row:
                    memory += repeated
                else:
                    _assign_compute_time(repeated, t_core_row, t_periph_row)

            if layer.has_act:
                if not _analoge_nonlinear_check(cfg, M, K, N):
                    if cfg.pipelinable:
                        t_act_row = _fpga_activation_row_latency(cfg, N)
                        peripheral += t_act_row + max(0.0, t_act_row - t_steady) * max(0, M - 1)
                    else:
                        peripheral += _fpga_activation_latency(cfg, M, N)

        elif layer.type == "maxpool":
            window_elements = layer.kernel_size * layer.kernel_size
            output_positions = layer.output_height * layer.output_width * layer.num_inputs
            t_read = _bram_latency(cfg, window_elements * layer.in_channels)
            t_write = _bram_latency(cfg, layer.out_channels)
            compare_levels = math.ceil(math.log2(window_elements))
            compare_parallel = max(1, min(layer.out_channels, cfg.total_clb))
            compare_batches = math.ceil(layer.out_channels / compare_parallel)
            t_compare = compare_levels * compare_batches * (1e9 / (cfg.freq * 1e6))
            t_steady = max(t_read, t_compare, t_write)
            memory += t_read + t_write
            peripheral += t_compare
            if output_positions > 1:
                repeated = (output_positions - 1) * t_steady
                if t_steady == t_compare:
                    peripheral += repeated
                else:
                    memory += repeated
        elif layer.type == "residual":
            output_positions = layer.output_height * layer.output_width * layer.num_inputs
            out_channels = layer.out_channels
            bytes_total = output_positions * out_channels
            t_read = _bram_latency(cfg, bytes_total)
            t_write = _bram_latency(cfg, bytes_total)
            t_compute = sim_common.cycles_to_ns(output_positions, cfg.freq)
            memory += t_read + t_write
            peripheral += t_compute

    return _bucket(memory, core, peripheral)


def old_latency_breakdown(layers, cfg=None, bandwidth_mode=False):
    memory = 0.0
    core = 0.0
    peripheral = 0.0

    for layer in layers:
        if layer.type in ("conv2d", "linear"):
            if layer.type == "conv2d":
                K = layer.kernel_size * layer.kernel_size * layer.in_channels
                M = layer.output_height * layer.output_width * layer.num_inputs
                N = layer.out_channels
            else:
                K = layer.in_channels
                M = layer.num_inputs
                N = layer.out_channels
            output_positions = layer.output_height * layer.output_width * layer.num_inputs
            num_computes = layer.num_computes
            num_events = math.ceil(output_positions / max(1, num_computes))

            num_reads_per_pos = math.ceil(layer.in_channels / C.ELEMS_PER_MV)
            num_positions = layer.kernel_size * layer.kernel_size
            read_per_event = C.SRAM_LAT * num_reads_per_pos * num_positions * num_computes

            lat_external_buf = C.BUFFER_LAT
            lat_adc = C.ADC_LAT * C.COLUMNS_PER_ADC
            lat_dpe_adc_pipeline = max(C.DPE_LAT, lat_adc) * (C.BIT_WIDTH - 1) + C.DPE_LAT + lat_adc
            compute_per_event = _ceil_to_clk(lat_external_buf + lat_dpe_adc_pipeline)

            one_kernel_size = layer.kernel_size * layer.kernel_size * layer.in_channels
            num_dpe_vert = math.ceil(one_kernel_size / C.DPE_ROWS)
            num_sum_stages = int(math.log2(_next_power_of_2(num_dpe_vert)))
            lat_sum_one_output = num_sum_stages * C.SUM_LAT
            num_sums_in_serial = math.ceil(layer.out_channels / C.NUM_ADDS) * num_computes
            sum_per_event = lat_sum_one_output * num_sums_in_serial

            write_per_event = C.SRAM_LAT * math.ceil(layer.out_channels / C.ELEMS_PER_MV) * num_computes

            read_stage = read_per_event
            write_stage = write_per_event
            if bandwidth_mode and cfg is not None:
                read_stage = _bram_latency(cfg, M * K) / max(1, num_events)
                write_stage = _bram_latency(cfg, M * N) / max(1, num_events)

            core_stage = compute_per_event + sum_per_event
            act_stage = 0.0

            if layer.has_act:
                act_read = C.SRAM_LAT * math.ceil(layer.out_channels / C.ELEMS_PER_MV) * num_computes
                act_compute = C.ACT_LAT * math.ceil(layer.out_channels / C.NUM_ACTS) * num_computes
                act_write = C.SRAM_LAT * math.ceil(layer.out_channels / C.ELEMS_PER_MV) * num_computes
                if bandwidth_mode and cfg is not None:
                    act_read = _bram_latency(cfg, M * N) / max(1, num_events)
                    act_write = _bram_latency(cfg, M * N) / max(1, num_events)
                act_stage = max(act_read, act_compute, act_write)

            t_steady = max(read_stage, core_stage, act_stage, write_stage)

            memory += read_stage + write_stage
            core += core_stage
            peripheral += act_stage
            if num_events > 1:
                if t_steady == core_stage:
                    core += (num_events - 1) * t_steady
                elif t_steady == act_stage:
                    peripheral += (num_events - 1) * t_steady
                else:
                    memory += (num_events - 1) * t_steady

        elif layer.type == "maxpool":
            output_positions = layer.output_height * layer.output_width * layer.num_inputs
            num_computes = layer.num_computes
            num_events = math.ceil(output_positions / max(1, num_computes))

            num_reads = math.ceil(layer.kernel_size * layer.kernel_size * layer.in_channels / C.ELEMS_PER_MV) * num_computes
            lat_read = C.SRAM_LAT * num_reads
            pool_size = _next_power_of_2(layer.kernel_size * layer.kernel_size)
            num_pool_stages = int(math.log2(pool_size))
            lat_pool_one_output = num_pool_stages * C.MAXPOOL_LAT
            num_pools_in_serial = math.ceil(layer.out_channels / C.NUM_MAXPOOLS) * num_computes
            lat_compare = lat_pool_one_output * num_pools_in_serial
            num_writes = math.ceil(layer.out_channels / C.ELEMS_PER_MV) * num_computes
            lat_write = C.SRAM_LAT * num_writes
            read_stage = lat_read
            write_stage = lat_write
            compare_stage = lat_compare
            if bandwidth_mode and cfg is not None:
                total_reads = layer.kernel_size * layer.kernel_size * layer.in_channels * output_positions
                total_writes = layer.out_channels * output_positions
                read_stage = _bram_latency(cfg, total_reads) / max(1, num_events)
                write_stage = _bram_latency(cfg, total_writes) / max(1, num_events)

            t_steady = max(read_stage, compare_stage, write_stage)
            memory += read_stage + write_stage
            peripheral += compare_stage
            if num_events > 1:
                repeated = (num_events - 1) * t_steady
                if t_steady == compare_stage:
                    peripheral += repeated
                else:
                    memory += repeated
        elif layer.type == "residual":
            output_positions = layer.output_height * layer.output_width * layer.num_inputs
            num_computes = layer.num_computes
            num_events = math.ceil(output_positions / max(1, num_computes))

            read_stage = C.SRAM_LAT * math.ceil(layer.in_channels / C.ELEMS_PER_MV) * num_computes
            write_stage = C.SRAM_LAT * math.ceil(layer.out_channels / C.ELEMS_PER_MV) * num_computes
            compute_stage = max(C.SUM_LAT * num_computes, C.SRAM_LAT * num_computes)

            if bandwidth_mode and cfg is not None:
                M = layer.output_height * layer.output_width * layer.num_inputs
                N = layer.out_channels
                read_stage = _bram_latency(cfg, M * N) / max(1, num_events)
                write_stage = _bram_latency(cfg, M * N) / max(1, num_events)

            t_steady = max(read_stage, compute_stage, write_stage)
            memory += read_stage + write_stage
            peripheral += compute_stage
            if num_events > 1:
                repeated = (num_events - 1) * t_steady
                if t_steady == compute_stage:
                    peripheral += repeated
                else:
                    memory += repeated

    return _bucket(memory, core, peripheral)


def _bucket(memory, core, peripheral):
    total = memory + core + peripheral
    return {
        "memory": memory,
        "core": core,
        "peripheral": peripheral,
        "total": total,
    }


def _bram_latency(cfg, bytes):
    bytes_per_access = max(
        1e-9, math.floor(cfg.bram_width / 8) * getattr(cfg, "mem_bw_utilization", 1.0)
    )
    num_access = math.ceil(bytes / bytes_per_access)
    if cfg.bram_mode == cfg.SP:
        num_cycle = num_access
    elif cfg.bram_mode == cfg.TDP:
        num_cycle = math.ceil(num_access / 2)
    else:
        num_cycle = num_access
    return num_cycle * (1e9 / (cfg.freq * 1e6))


def _clb_reduction_latency(cfg, reductions):
    if reductions <= 1:
        return 0.0
    levels = math.ceil(math.log2(reductions))
    return levels * (1e9 / (cfg.freq * 1e6))


def _latency_per_vmm(cfg):
    k_vmm, _, _ = cfg._get_arch_factors()
    return k_vmm * cfg.t_analoge_ns


def _latency_per_conv(cfg):
    _, k_conv, _ = cfg._get_arch_factors()
    return k_conv * cfg.t_conv_ns


def _latency_per_digital(cfg):
    _, _, k_digital = cfg._get_arch_factors()
    return k_digital * cfg.t_digital_ns


def _analoge_nonlinear_check(cfg, M, K, N):
    k_tile = math.ceil(K / cfg.rows)
    if cfg.imc == "NL-DPE" and k_tile > 4:
        return False
    return cfg.analoge_nonlinear_support


def _fpga_activation_latency(cfg, M, N):
    luts_per_act = 32
    luts_per_clb = 8
    clb_per_act = max(1, math.ceil(luts_per_act / luts_per_clb))
    max_units_by_clb = max(1, cfg.total_clb // clb_per_act)
    act_units = max(1, min(getattr(cfg, "act_units", 16), max_units_by_clb))
    cycles_per_op = max(1, int(getattr(cfg, "act_cycles_per_op", 1)))
    act_cycles = math.ceil((M * N) / act_units) * cycles_per_op
    return act_cycles * (1e9 / (cfg.freq * 1e6))


def _fpga_activation_row_latency(cfg, N):
    luts_per_act = 32
    luts_per_clb = 8
    clb_per_act = max(1, math.ceil(luts_per_act / luts_per_clb))
    max_units_by_clb = max(1, cfg.total_clb // clb_per_act)
    act_units = max(1, min(getattr(cfg, "act_units", 16), max_units_by_clb))
    cycles_per_op = max(1, int(getattr(cfg, "act_cycles_per_op", 1)))
    row_cycles = math.ceil(N / act_units) * cycles_per_op
    return row_cycles * (1e9 / (cfg.freq * 1e6))


def _next_power_of_2(x):
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _ceil_to_clk(lat):
    cycles = math.ceil(lat / C.CLK_LAT)
    return cycles * C.CLK_LAT


def l1_diff_pct(a, b):
    return abs(a["memory"] - b["memory"]) + abs(a["core"] - b["core"]) + abs(a["peripheral"] - b["peripheral"])


def format_md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def aggregate_operator(layers, args, old_scale, new_scale):
    new_energy_bucket = _bucket(0.0, 0.0, 0.0)
    old_energy_bucket = _bucket(0.0, 0.0, 0.0)
    new_lat_bucket = _bucket(0.0, 0.0, 0.0)
    old_lat_bucket = _bucket(0.0, 0.0, 0.0)
    new_total_energy = 0.0
    old_total_energy = 0.0
    new_total_latency = 0.0
    old_total_latency = 0.0
    cfg = None

    for layer in layers:
        old_res = run_old_layer(layer, args)
        new_res = run_new_layer(layer, args)
        cfg = new_res["cfg"]

        old_bucket = energy_buckets_old(old_res["energy_stats"], unit_scale=old_scale)
        new_bucket = energy_buckets_new(new_res["energy_breakdown"], unit_scale=new_scale)
        old_lat = old_latency_breakdown([layer], cfg=cfg, bandwidth_mode=args.bandwidth_mode)
        new_lat = new_latency_breakdown([layer], cfg)

        old_energy_bucket["memory"] += old_bucket["memory"]
        old_energy_bucket["core"] += old_bucket["core"]
        old_energy_bucket["peripheral"] += old_bucket["peripheral"]
        new_energy_bucket["memory"] += new_bucket["memory"]
        new_energy_bucket["core"] += new_bucket["core"]
        new_energy_bucket["peripheral"] += new_bucket["peripheral"]

        old_lat_bucket["memory"] += old_lat["memory"]
        old_lat_bucket["core"] += old_lat["core"]
        old_lat_bucket["peripheral"] += old_lat["peripheral"]
        new_lat_bucket["memory"] += new_lat["memory"]
        new_lat_bucket["core"] += new_lat["core"]
        new_lat_bucket["peripheral"] += new_lat["peripheral"]

        old_total_energy += sum(old_res["energy_stats"].values()) * old_scale
        new_total_energy += new_res["total_energy"] * new_scale
        old_total_latency += old_res["total_latency"]
        new_total_latency += new_res["total_latency"]

    old_energy_bucket = _bucket(old_energy_bucket["memory"], old_energy_bucket["core"], old_energy_bucket["peripheral"])
    new_energy_bucket = _bucket(new_energy_bucket["memory"], new_energy_bucket["core"], new_energy_bucket["peripheral"])
    old_lat_bucket = _bucket(old_lat_bucket["memory"], old_lat_bucket["core"], old_lat_bucket["peripheral"])
    new_lat_bucket = _bucket(new_lat_bucket["memory"], new_lat_bucket["core"], new_lat_bucket["peripheral"])

    return {
        "old_energy": old_energy_bucket,
        "new_energy": new_energy_bucket,
        "old_latency": old_lat_bucket,
        "new_latency": new_lat_bucket,
        "old_total_energy": old_total_energy,
        "new_total_energy": new_total_energy,
        "old_total_latency": old_total_latency,
        "new_total_latency": new_total_latency,
        "old_latency_proxy_total": old_lat_bucket["total"],
        "cfg": cfg,
    }


def _event_driven_conv_linear_latency(layer, cfg=None, bandwidth_mode=False):
    output_positions = layer.output_height * layer.output_width * layer.num_inputs
    num_computes = layer.num_computes
    num_events = math.ceil(output_positions / max(1, num_computes))

    num_reads_per_pos = math.ceil(layer.in_channels / C.ELEMS_PER_MV)
    num_positions = layer.kernel_size * layer.kernel_size
    read_per_event = C.SRAM_LAT * num_reads_per_pos * num_positions * num_computes
    read_total = read_per_event * num_events

    lat_external_buf = C.BUFFER_LAT
    lat_adc = C.ADC_LAT * C.COLUMNS_PER_ADC
    lat_dpe_adc_pipeline = max(C.DPE_LAT, lat_adc) * (C.BIT_WIDTH - 1) + C.DPE_LAT + lat_adc
    adc_per_event = _ceil_to_clk(lat_external_buf + lat_dpe_adc_pipeline)
    adc_total = adc_per_event * num_events

    one_kernel_size = layer.kernel_size * layer.kernel_size * layer.in_channels
    num_dpe_vert = math.ceil(one_kernel_size / C.DPE_ROWS)
    num_sum_stages = int(math.log2(_next_power_of_2(num_dpe_vert)))
    lat_sum_one_output = num_sum_stages * C.SUM_LAT
    num_sums_in_serial = math.ceil(layer.out_channels / C.NUM_ADDS) * num_computes
    sum_per_event = lat_sum_one_output * num_sums_in_serial
    sum_total = sum_per_event * num_events

    write_per_event = C.SRAM_LAT * math.ceil(layer.out_channels / C.ELEMS_PER_MV) * num_computes
    write_total = write_per_event * num_events

    act_total = 0.0
    act_read_total = 0.0
    act_write_total = 0.0
    act_compute_total = 0.0
    if layer.has_act:
        act_read = C.SRAM_LAT * math.ceil(layer.out_channels / C.ELEMS_PER_MV) * num_computes
        act_compute = C.ACT_LAT * math.ceil(layer.out_channels / C.NUM_ACTS) * num_computes
        act_write = C.SRAM_LAT * math.ceil(layer.out_channels / C.ELEMS_PER_MV) * num_computes
        act_read_total = act_read * num_events
        act_write_total = act_write * num_events
        act_compute_total = act_compute * num_events
        act_total = max(act_read, act_compute, act_write) * num_events

    if bandwidth_mode and cfg is not None:
        if layer.type == "conv2d":
            K = layer.kernel_size * layer.kernel_size * layer.in_channels
            M = layer.output_height * layer.output_width * layer.num_inputs
            N = layer.out_channels
        else:
            K = layer.in_channels
            M = layer.num_inputs
            N = layer.out_channels
        read_total = _bram_latency(cfg, M * K)
        write_total = _bram_latency(cfg, M * N)
        if layer.has_act:
            act_read_total = _bram_latency(cfg, M * N)
            act_write_total = _bram_latency(cfg, M * N)
            act_total = max(act_read_total, act_compute_total, act_write_total)

    total = read_total + write_total + adc_total + sum_total + act_total
    return {
        "read": read_total,
        "write": write_total,
        "adc": adc_total,
        "sum": sum_total,
        "act": act_total,
        "act_read": act_read_total,
        "act_write": act_write_total,
        "act_compute": act_compute_total,
        "num_events": num_events,
        "total": total,
    }


def _imc_conv_linear_latency(layer, cfg):
    if layer.type == "conv2d":
        K = layer.kernel_size * layer.kernel_size * layer.in_channels
        M = layer.output_height * layer.output_width * layer.num_inputs
        N = layer.out_channels
    else:
        K = layer.in_channels
        M = layer.num_inputs
        N = layer.out_channels

    k_tile = math.ceil(K / cfg.rows)
    t_read = _bram_latency(cfg, M * K)
    t_write = _bram_latency(cfg, M * N)
    t_reduc = _clb_reduction_latency(cfg, k_tile)
    t_vmm = M * _latency_per_vmm(cfg)
    t_conv = M * _latency_per_conv(cfg)
    t_digital = M * _latency_per_digital(cfg)

    t_act = 0.0
    if layer.has_act and not _analoge_nonlinear_check(cfg, M, K, N):
        if cfg.pipelinable:
            t_act_row = _fpga_activation_row_latency(cfg, N)
            t_read_row = _bram_latency(cfg, K)
            t_write_row = _bram_latency(cfg, N)
            t_compute_row = max(
                _latency_per_vmm(cfg),
                _latency_per_conv(cfg),
                _latency_per_digital(cfg),
                _clb_reduction_latency(cfg, k_tile),
            )
            t_steady = max(t_read_row, t_compute_row, t_write_row)
            t_act = t_act_row + max(0.0, t_act_row - t_steady) * max(0, M - 1)
        else:
            t_act = _fpga_activation_latency(cfg, M, N)

    total = t_read + t_write + t_vmm + t_conv + t_digital + t_reduc + t_act
    return {
        "read": t_read,
        "write": t_write,
        "vmm": t_vmm,
        "adc": t_conv,
        "digital": t_digital,
        "sum": t_reduc,
        "act": t_act,
        "total": total,
    }


def _required_upstream_outputs_before_launch(layer):
    if layer.type in ("conv2d", "maxpool"):
        # Conservative receptive-field dependency estimate.
        required_rows = max(1, layer.kernel_size - layer.padding)
        outputs_per_row = max(1, layer.input_width * layer.num_inputs)
        return max(1, required_rows * outputs_per_row)
    return 1


def _new_layer_timing_profile(layer, cfg):
    if layer.type in ("conv2d", "linear"):
        if layer.type == "conv2d":
            K = layer.kernel_size * layer.kernel_size * layer.in_channels
            M = layer.output_height * layer.output_width * layer.num_inputs
            N = layer.out_channels
        else:
            K = layer.in_channels
            M = layer.num_inputs
            N = layer.out_channels

        k_tile = math.ceil(K / cfg.rows)
        t_read_row = _bram_latency(cfg, K)
        t_write_row = _bram_latency(cfg, N)
        t_vmm_row = _latency_per_vmm(cfg)
        t_conv_row = _latency_per_conv(cfg)
        t_digital_row = _latency_per_digital(cfg)
        t_reduce_row = _clb_reduction_latency(cfg, k_tile)
        if cfg.pipelinable:
            t_compute_row = max(t_vmm_row, t_conv_row, t_digital_row, t_reduce_row)
        else:
            t_compute_row = t_vmm_row + t_conv_row + t_digital_row + t_reduce_row

        t_fill = t_read_row + t_compute_row + t_write_row
        t_steady = max(t_read_row, t_compute_row, t_write_row)
        total = t_fill + max(0, M - 1) * t_steady

        act_row = 0.0
        if layer.has_act and not _analoge_nonlinear_check(cfg, M, K, N):
            if cfg.pipelinable:
                act_row = _fpga_activation_row_latency(cfg, N)
                total += act_row + max(0.0, act_row - t_steady) * max(0, M - 1)
                t_fill += act_row
                t_steady = max(t_steady, act_row)
            else:
                act_total = _fpga_activation_latency(cfg, M, N)
                total += act_total
                t_fill += act_total
                t_steady = total

        return {
            "layer": layer.name,
            "layer_obj": layer,
            "type": layer.type,
            "total_latency": total,
            "first_output_ns": t_fill,
            "steady_ns": t_steady,
            "events": M,
            "required_upstream_outputs": _required_upstream_outputs_before_launch(layer),
            "read_row_ns": t_read_row,
            "compute_row_ns": t_compute_row,
            "write_row_ns": t_write_row,
            "act_row_ns": act_row,
        }

    if layer.type == "maxpool":
        window_elements = layer.kernel_size * layer.kernel_size
        output_positions = layer.output_height * layer.output_width * layer.num_inputs
        reads_per_output = window_elements * layer.in_channels
        writes_per_output = layer.out_channels
        t_read = _bram_latency(cfg, reads_per_output)
        t_write = _bram_latency(cfg, writes_per_output)
        compare_levels = math.ceil(math.log2(window_elements))
        compare_parallel = max(1, min(layer.out_channels, cfg.total_clb))
        compare_batches = math.ceil(layer.out_channels / compare_parallel)
        t_compare = compare_levels * compare_batches * (1e9 / (cfg.freq * 1e6))
        t_fill = t_read + t_compare + t_write
        t_steady = max(t_read, t_compare, t_write)
        total = t_fill + max(0, output_positions - 1) * t_steady
        return {
            "layer": layer.name,
            "layer_obj": layer,
            "type": layer.type,
            "total_latency": total,
            "first_output_ns": t_fill,
            "steady_ns": t_steady,
            "events": output_positions,
            "required_upstream_outputs": _required_upstream_outputs_before_launch(layer),
            "read_row_ns": t_read,
            "compute_row_ns": t_compare,
            "write_row_ns": t_write,
            "act_row_ns": 0.0,
        }

    if layer.type == "residual":
        output_positions = layer.output_height * layer.output_width * layer.num_inputs
        out_channels = layer.out_channels
        bytes_total = output_positions * out_channels
        t_read = _bram_latency(cfg, bytes_total)
        t_compute = sim_common.cycles_to_ns(output_positions, cfg.freq)
        t_write = _bram_latency(cfg, bytes_total)
        total = t_read + t_compute + t_write
        return {
            "layer": layer.name,
            "layer_obj": layer,
            "type": layer.type,
            "total_latency": total,
            "first_output_ns": total,
            "steady_ns": total,
            "events": 1,
            "required_upstream_outputs": 1,
            "read_row_ns": t_read,
            "compute_row_ns": t_compute,
            "write_row_ns": t_write,
            "act_row_ns": 0.0,
        }

    return {
        "layer": layer.name,
        "layer_obj": layer,
        "type": layer.type,
        "total_latency": 0.0,
        "first_output_ns": 0.0,
        "steady_ns": 0.0,
        "events": 1,
        "required_upstream_outputs": 1,
        "read_row_ns": 0.0,
        "compute_row_ns": 0.0,
        "write_row_ns": 0.0,
        "act_row_ns": 0.0,
    }


def _simulate_inter_layer_timeline(profiles):
    profiler = PipelineProfiler()
    for p in profiles:
        profiler.record(
            p["layer_obj"],
            p["total_latency"],
            timing={
                "first_output_ns": p["first_output_ns"],
                "steady": p["steady_ns"],
                "events": p["events"],
                "required_upstream_outputs": p["required_upstream_outputs"],
            },
        )
    trace = profiler.trace()
    trace_by_layer = {entry["layer"]: entry for entry in trace}
    timeline = []
    for p in profiles:
        entry = trace_by_layer.get(p["layer"], {})
        timeline.append(
            {
                **p,
                "start_ns": entry.get("start_ns", 0.0),
                "first_output_t_ns": entry.get("first_output_t_ns", 0.0),
                "finish_ns": entry.get("finish_ns", 0.0),
                "critical_end_ns": entry.get("critical_end_ns", 0.0),
                "critical_contribution_ns": entry.get("critical_contribution_ns", 0.0),
                "starvation_total_ns": entry.get("backpressure_add_ns", 0.0),
                "effective_total_ns": entry.get("effective_total_ns", p["total_latency"]),
            }
        )
    return timeline


def maxpool_latency_audit(layer, cfg, bandwidth_mode=False):
    window_elements = layer.kernel_size * layer.kernel_size
    output_positions = layer.output_height * layer.output_width * layer.num_inputs
    num_events = math.ceil(output_positions / max(1, layer.num_computes))

    num_reads = math.ceil(layer.kernel_size * layer.kernel_size * layer.in_channels / C.ELEMS_PER_MV) * layer.num_computes
    lat_read = C.SRAM_LAT * num_reads
    pool_size = _next_power_of_2(layer.kernel_size * layer.kernel_size)
    num_pool_stages = int(math.log2(pool_size))
    lat_pool_one_output = num_pool_stages * C.MAXPOOL_LAT
    num_pools_in_serial = math.ceil(layer.out_channels / C.NUM_MAXPOOLS) * layer.num_computes
    lat_compare = lat_pool_one_output * num_pools_in_serial
    num_writes = math.ceil(layer.out_channels / C.ELEMS_PER_MV) * layer.num_computes
    lat_write = C.SRAM_LAT * num_writes

    old_read_stage = lat_read
    old_write_stage = lat_write
    old_compare_stage = lat_compare
    old_steady = max(old_read_stage, old_compare_stage, old_write_stage)
    old_read_total = old_read_stage
    old_write_total = old_write_stage
    old_compare_total = old_compare_stage
    if num_events > 1:
        old_repeated = (num_events - 1) * old_steady
        if old_steady == old_compare_stage:
            old_compare_total += old_repeated
        else:
            old_read_total += old_repeated
    old_total = old_read_total + old_write_total + old_compare_total

    total_reads = window_elements * layer.in_channels * output_positions
    total_writes = layer.out_channels * output_positions
    t_read_stage = _bram_latency(cfg, total_reads) / max(1, num_events)
    t_write_stage = _bram_latency(cfg, total_writes) / max(1, num_events)
    compare_levels = math.ceil(math.log2(window_elements))
    compare_parallel = max(1, min(layer.out_channels, cfg.total_clb))
    compare_batches = math.ceil(layer.out_channels / compare_parallel)
    t_compare_stage = (compare_levels * compare_batches) * (1e9 / (cfg.freq * 1e6))
    new_steady = max(t_read_stage, t_compare_stage, t_write_stage)
    t_read = t_read_stage
    t_write = t_write_stage
    t_compare = t_compare_stage
    if output_positions > 1:
        repeated = (output_positions - 1) * new_steady
        if new_steady == t_compare_stage:
            t_compare += repeated
        else:
            t_read += repeated
    new_total = t_read + t_write + t_compare

    if bandwidth_mode:
        old_read_stage = _bram_latency(cfg, total_reads) / max(1, num_events)
        old_write_stage = _bram_latency(cfg, total_writes) / max(1, num_events)
        old_steady = max(old_read_stage, old_compare_stage, old_write_stage)
        old_read_total = old_read_stage
        old_write_total = old_write_stage
        old_compare_total = old_compare_stage
        if num_events > 1:
            old_repeated = (num_events - 1) * old_steady
            if old_steady == old_compare_stage:
                old_compare_total += old_repeated
            else:
                old_read_total += old_repeated
        old_total = old_read_total + old_write_total + old_compare_total

    return {
        "window_elements": window_elements,
        "output_positions": output_positions,
        "num_events": num_events,
        "old_read": old_read_total,
        "old_write": old_write_total,
        "old_compare": old_compare_total,
        "old_total": old_total,
        "new_read": t_read,
        "new_write": t_write,
        "new_compare": t_compare,
        "new_total": new_total,
    }


def linear_latency_audit(layer, cfg, bandwidth_mode=False):
    old_parts = _event_driven_conv_linear_latency(layer, cfg=cfg, bandwidth_mode=bandwidth_mode)
    new_parts = _imc_conv_linear_latency(layer, cfg)
    return {**old_parts, **{f"new_{k}": v for k, v in new_parts.items()}}


def conv2d_latency_audit(layer, cfg, bandwidth_mode=False):
    old_parts = _event_driven_conv_linear_latency(layer, cfg=cfg, bandwidth_mode=bandwidth_mode)
    new_parts = _imc_conv_linear_latency(layer, cfg)
    return {**old_parts, **{f"new_{k}": v for k, v in new_parts.items()}}


def render_operator_summary(lines, op, layer_names, res, energy_unit_label):
    new_energy_pct = pct_breakdown(res["new_energy"])
    old_energy_pct = pct_breakdown(res["old_energy"])
    new_lat_pct = pct_breakdown(res["new_latency"])
    old_lat_pct = pct_breakdown(res["old_latency"])

    lines.append(f"## {op.upper()} (layers: {layer_names})")
    lines.append("")


def render_end_to_end(lines, title, old_res, new_res, energy_unit_label):
    lines.append(f"## {title}")
    lines.append("")
    lines.append(f"### Totals (energy={energy_unit_label}, latency=ns)")
    lines.append(
        format_md_table(
            ["Metric", "IMC_new", "Event‑Driven", "Ratio (new/old)"],
            [
                [
                    f"Total Energy ({energy_unit_label})",
                    f"{new_res['total_energy']:.2f}",
                    f"{old_res['total_energy']:.2f}",
                    f"{new_res['total_energy']/old_res['total_energy'] if old_res['total_energy'] else 0:.2f}",
                ],
                [
                    "Total Latency (ns)",
                    f"{new_res['total_latency']:.2f}",
                    f"{old_res['total_latency']:.2f}",
                    f"{new_res['total_latency']/old_res['total_latency'] if old_res['total_latency'] else 0:.2f}",
                ],
                [
                    "Event‑Driven breakdown vs Event‑Driven total (ns)",
                    f"{old_res['latency_proxy_total']:.2f}",
                    f"{old_res['total_latency']:.2f}",
                    f"{old_res['latency_proxy_total']/old_res['total_latency'] if old_res['total_latency'] else 0:.2f}",
                ],
            ],
        )
    )
    lines.append("")
    lines.append(f"### Energy Breakdown (percent of total, unit={energy_unit_label})")
    rows = [
        ["memory", f"{new_res['energy_pct']['memory']*100:.2f}%", f"{old_res['energy_pct']['memory']*100:.2f}%", f"{(new_res['energy_pct']['memory']-old_res['energy_pct']['memory'])*100:.2f}%"],
        ["core", f"{new_res['energy_pct']['core']*100:.2f}%", f"{old_res['energy_pct']['core']*100:.2f}%", f"{(new_res['energy_pct']['core']-old_res['energy_pct']['core'])*100:.2f}%"],
        ["peripheral", f"{new_res['energy_pct']['peripheral']*100:.2f}%", f"{old_res['energy_pct']['peripheral']*100:.2f}%", f"{(new_res['energy_pct']['peripheral']-old_res['energy_pct']['peripheral'])*100:.2f}%"],
    ]
    lines.append(format_md_table(["Bucket", "IMC_new", "Event‑Driven", "Delta"], rows))
    lines.append("")
    lines.append("### Latency Breakdown (percent of total, proxy for event‑driven)")
    rows = [
        ["memory", f"{new_res['latency_pct']['memory']*100:.2f}%", f"{old_res['latency_pct']['memory']*100:.2f}%", f"{(new_res['latency_pct']['memory']-old_res['latency_pct']['memory'])*100:.2f}%"],
        ["core", f"{new_res['latency_pct']['core']*100:.2f}%", f"{old_res['latency_pct']['core']*100:.2f}%", f"{(new_res['latency_pct']['core']-old_res['latency_pct']['core'])*100:.2f}%"],
        ["peripheral", f"{new_res['latency_pct']['peripheral']*100:.2f}%", f"{old_res['latency_pct']['peripheral']*100:.2f}%", f"{(new_res['latency_pct']['peripheral']-old_res['latency_pct']['peripheral'])*100:.2f}%"],
    ]
    lines.append(format_md_table(["Bucket", "IMC_new", "Event‑Driven (proxy)", "Delta"], rows))
    lines.append("")


def render_pipeline_tutorial(lines, args):
    cfg = Config(args.imc_file)
    layers, _, _ = build_model_layers(resnet_model, args)
    focus_names = ("conv1", "conv2", "pool1", "conv3")
    focus_layers = [layer for layer in layers if layer.name in focus_names]
    if len(focus_layers) < 2:
        return

    profiles = [_new_layer_timing_profile(layer, cfg) for layer in focus_layers]
    timeline = _simulate_inter_layer_timeline(profiles)

    lines.append("## Cross‑Layer Pipeline Strategy Guide")
    lines.append("")
    lines.append(
        "High-level design: each layer is reduced to a timing tuple "
        "(first_output_ns, steady_ns, events, required_upstream_outputs). "
        "The scheduler then computes a critical-path overlap timeline."
    )
    lines.append("")
    lines.append("Control parameters and how they are computed:")
    lines.append("- `first_output_ns`: time from layer start to first produced output token.")
    lines.append("- `steady_ns`: per-token interval after pipeline fill.")
    lines.append("- `events`: number of produced output tokens for the layer.")
    lines.append(
        "- `required_upstream_outputs`: conservative dependency count before downstream launch; "
        "for conv/maxpool = max(1, kernel_size - padding) * input_width * num_inputs."
    )
    lines.append("- Token readiness recurrence (streaming layers):")
    lines.append(
        "  T_i(1) = T_{i-1}(R_i(1)) + first_output_i, "
        "T_i(n) = max(T_i(n-1) + steady_i, T_{i-1}(R_i(n)) + first_output_i)."
    )
    lines.append("- `R_i(n)` is upstream token index needed for downstream token `n` from window geometry.")
    lines.append("- `backpressure_add` in tables is the accumulated dependency-wait beyond local steady cadence.")
    lines.append("")
    lines.append("### ResNet timestamp walkthrough (conv1 → conv2 → pool1 → conv3)")
    lines.append("")

    ts_rows = []
    for idx, item in enumerate(timeline):
        if idx == 0:
            launch_formula = "0"
        else:
            launch_formula = f"T_prev(R= {item['required_upstream_outputs']})"
        ts_rows.append([f"t{idx * 3}", f"launch {item['layer']}", launch_formula, f"{item['start_ns']:.2f}"])
        ts_rows.append(
            [
                f"t{idx * 3 + 1}",
                f"{item['layer']} first output ready",
                f"{item['start_ns']:.2f} + {item['first_output_ns']:.2f}",
                f"{item['first_output_t_ns']:.2f}",
            ]
        )
        ts_rows.append(
            [
                f"t{idx * 3 + 2}",
                f"{item['layer']} drained",
                f"{item['start_ns']:.2f} + {item['effective_total_ns']:.2f}",
                f"{item['finish_ns']:.2f}",
            ]
        )
    lines.append(format_md_table(["Time", "Event", "Formula (ns)", "Timestamp (ns)"], ts_rows))
    lines.append("")

    lines.append("Per-layer pipeline terms (same units and definitions as scheduler):")
    rows = []
    for item in timeline:
        rows.append(
            [
                item["layer"],
                item["type"],
                item["events"],
                item["required_upstream_outputs"],
                f"{item['first_output_ns']:.2f}",
                f"{item['steady_ns']:.2f}",
                f"{item['total_latency']:.2f}",
                f"{item['starvation_total_ns']:.2f}",
                f"{item['effective_total_ns']:.2f}",
                f"{item['critical_contribution_ns']:.2f}",
            ]
        )
    lines.append(
        format_md_table(
            [
                "Layer",
                "Type",
                "Events",
                "Required Upstream Outputs",
                "First Output (ns)",
                "Steady (ns)",
                "Base Total (ns)",
                "Backpressure Add (ns)",
                "Effective Total (ns)",
                "Critical Contribution (ns)",
            ],
            rows,
        )
    )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- Larger `required_upstream_outputs` delays downstream launch beyond previous first output.")
    lines.append("- `Backpressure Add` appears when upstream steady-state is slower than downstream demand.")
    lines.append("- Critical-path contribution can be smaller than layer effective total when overlap is high.")
    lines.append("")


def render_model_layerwise_latency(
    lines,
    args,
    title,
    model_fn,
    old_scale,
    new_scale,
    energy_unit_label,
    old_e2e=None,
    new_e2e=None,
    old_e2e_energy=None,
    new_e2e_energy=None,
    old_layer_energy_stats=None,
    old_total_mode="exact",
):
    layers, _, _ = build_model_layers(model_fn, args)
    lines.append(f"## {title} Layer‑Wise Latency and Energy Breakdown")
    lines.append("")
    lines.append("Event‑Driven totals are from per‑layer event simulation. IMC_new totals are per‑layer IMC simulation.")
    lines.append("Breakdown buckets use the same proxy model as elsewhere in this report.")
    if old_total_mode != "exact":
        lines.append("Event‑Driven per-layer total latency uses proxy model for runtime efficiency.")
    if args.bandwidth_mode:
        lines.append("Event‑driven read/write latency uses BRAM bandwidth model (option A).")
    lines.append("")
    rows = []
    energy_rows = []
    old_layer_sum = 0.0
    new_layer_sum = 0.0
    old_energy_layer_sum = 0.0
    new_energy_layer_sum = 0.0
    for layer in layers:
        new_res = run_new_layer(layer, args)
        old_layer_res = None
        old_layer_energy = None
        if old_layer_energy_stats is not None:
            old_layer_energy = old_layer_energy_stats.get(layer.name)

        old_bucket = old_latency_breakdown([layer], cfg=new_res["cfg"], bandwidth_mode=args.bandwidth_mode)
        new_bucket = new_latency_breakdown([layer], new_res["cfg"])
        if old_total_mode == "exact":
            old_layer_res = run_old_layer(layer, args)
            old_total = old_layer_res["total_latency"]
        else:
            old_total = old_bucket["total"]

        if old_layer_energy is None:
            if old_layer_res is None:
                old_layer_res = run_old_layer(layer, args)
            old_layer_energy = old_layer_res["energy_stats"]

        old_energy_bucket = energy_buckets_old(old_layer_energy, unit_scale=old_scale)
        new_energy_bucket = energy_buckets_new(new_res["energy_breakdown"], unit_scale=new_scale)
        old_energy_total = sum(old_layer_energy.values()) * old_scale
        new_energy_total = new_res["total_energy"] * new_scale

        old_layer_sum += old_total
        new_layer_sum += new_res["total_latency"]
        old_energy_layer_sum += old_energy_total
        new_energy_layer_sum += new_energy_total

        rows.append(
            [
                layer.name,
                layer.type,
                f"{old_total:.2f}",
                f"{old_bucket['memory']:.2f}",
                f"{old_bucket['core']:.2f}",
                f"{old_bucket['peripheral']:.2f}",
                f"{new_res['total_latency']:.2f}",
                f"{new_bucket['memory']:.2f}",
                f"{new_bucket['core']:.2f}",
                f"{new_bucket['peripheral']:.2f}",
                f"{new_res['total_latency']/old_total if old_total else 0:.2f}",
            ]
        )
        energy_rows.append(
            [
                layer.name,
                layer.type,
                f"{old_energy_total:.2f}",
                f"{old_energy_bucket['memory']:.2f}",
                f"{old_energy_bucket['core']:.2f}",
                f"{old_energy_bucket['peripheral']:.2f}",
                f"{new_energy_total:.2f}",
                f"{new_energy_bucket['memory']:.2f}",
                f"{new_energy_bucket['core']:.2f}",
                f"{new_energy_bucket['peripheral']:.2f}",
                f"{new_energy_total/old_energy_total if old_energy_total else 0:.2f}",
            ]
        )

    lines.append(f"### {title} Layer‑Wise Latency Breakdown (ns)")
    lines.append("")
    lines.append(
        format_md_table(
            [
                "Layer",
                "Type",
                "Event‑Driven Total (ns)",
                "Event‑Mem (ns)",
                "Event‑Core (ns)",
                "Event‑Periph (ns)",
                "IMC_new Total (ns)",
                "IMC‑Mem (ns)",
                "IMC‑Core (ns)",
                "IMC‑Periph (ns)",
                "Ratio (new/old)",
            ],
            rows,
        )
    )
    lines.append("")

    lines.append(f"### {title} Layer‑Wise Energy Breakdown ({energy_unit_label})")
    lines.append("")
    lines.append(
        format_md_table(
            [
                "Layer",
                "Type",
                f"Event‑Driven Total ({energy_unit_label})",
                f"Event‑Mem ({energy_unit_label})",
                f"Event‑Core ({energy_unit_label})",
                f"Event‑Periph ({energy_unit_label})",
                f"IMC_new Total ({energy_unit_label})",
                f"IMC‑Mem ({energy_unit_label})",
                f"IMC‑Core ({energy_unit_label})",
                f"IMC‑Periph ({energy_unit_label})",
                "Ratio (new/old)",
            ],
            energy_rows,
        )
    )
    lines.append("")

    if old_e2e is None or new_e2e is None:
        old_layers, old_finishes, _ = build_model_layers(model_fn, args)
        old_e2e = run_old_model(old_layers, old_finishes)
        new_layers, _, _ = build_model_layers(model_fn, args)
        imc = IMCNew(args.imc_file)
        for layer in new_layers:
            imc.run_layer(layer)
        imc.finalize_latency_stats()
        new_e2e = sum(imc.latency_stats.values())

    if old_e2e_energy is None or new_e2e_energy is None:
        old_res, new_res = compute_end_to_end_result(model_fn, args, old_scale, new_scale)
        old_e2e_energy = old_res["total_energy"]
        new_e2e_energy = new_res["total_energy"]

    lines.append(f"### {title} Sanity Check (Layer‑Sum vs End‑to‑End)")
    lines.append(
        format_md_table(
            ["Metric", f"Event‑Driven (ns / {energy_unit_label})", f"IMC_new (ns / {energy_unit_label})"],
            [
                ["Layer-wise isolated sum", f"{old_layer_sum:.2f}", f"{new_layer_sum:.2f}"],
                ["End-to-end total", f"{old_e2e:.2f}", f"{new_e2e:.2f}"],
                [
                    "Overlap factor (isolated/e2e)",
                    f"{old_layer_sum / old_e2e if old_e2e else 0:.2f}",
                    f"{new_layer_sum / new_e2e if new_e2e else 0:.2f}",
                ],
                [f"Layer-wise energy sum ({energy_unit_label})", f"{old_energy_layer_sum:.2f}", f"{new_energy_layer_sum:.2f}"],
                [f"End-to-end energy ({energy_unit_label})", f"{old_e2e_energy:.2f}", f"{new_e2e_energy:.2f}"],
                [
                    "Energy ratio (layer-sum/e2e)",
                    f"{old_energy_layer_sum / old_e2e_energy if old_e2e_energy else 0:.2f}",
                    f"{new_energy_layer_sum / new_e2e_energy if new_e2e_energy else 0:.2f}",
                ],
            ],
        )
    )
    lines.append("")


def render_lenet_layerwise_latency(lines, args):
    render_model_layerwise_latency(lines, args, "LeNet", lenet_model, old_scale=1e3, new_scale=1.0, energy_unit_label="pJ")


def render_resnet_layerwise_latency(lines, args):
    render_model_layerwise_latency(lines, args, "ResNet", resnet_model, old_scale=1e3, new_scale=1.0, energy_unit_label="pJ")


def compute_end_to_end_result(model_fn, args, old_scale, new_scale):
    old_layers, old_finishes, _ = build_model_layers(model_fn, args)
    for layer in old_layers:
        layer.energy_stats = build_energy_stats()
    old_total_latency = run_old_model(old_layers, old_finishes)
    old_layer_energy_stats = {}
    old_energy_stats = build_energy_stats()
    for layer in old_layers:
        layer_stats = dict(layer.energy_stats)
        old_layer_energy_stats[layer.name] = layer_stats
        _accumulate_energy_stats(old_energy_stats, layer_stats)

    old_total_energy = sum(old_energy_stats.values()) * old_scale
    old_energy_bucket = energy_buckets_old(old_energy_stats, unit_scale=old_scale)

    new_layers, _, _ = build_model_layers(model_fn, args)
    imc = IMCNew(args.imc_file)
    for layer in new_layers:
        imc.run_layer(layer)
    imc.finalize_latency_stats()
    new_total_energy = sum(imc.energy_stats.values()) * new_scale
    new_total_latency = sum(imc.latency_stats.values())
    new_energy_bucket = energy_buckets_new(imc.energy_breakdown, unit_scale=new_scale)

    old_latency_bucket = old_latency_breakdown(old_layers, cfg=imc.cfg, bandwidth_mode=args.bandwidth_mode)
    new_latency_bucket = new_latency_breakdown(new_layers, imc.cfg)

    old_res = {
        "total_energy": old_total_energy,
        "total_latency": old_total_latency,
        "energy_pct": pct_breakdown(old_energy_bucket),
        "latency_pct": pct_breakdown(old_latency_bucket),
        "latency_proxy_total": old_latency_bucket["total"],
        "layer_energy_stats": old_layer_energy_stats,
    }
    new_res = {
        "total_energy": new_total_energy,
        "total_latency": new_total_latency,
        "energy_pct": pct_breakdown(new_energy_bucket),
        "latency_pct": pct_breakdown(new_latency_bucket),
    }
    return old_res, new_res



def main():
    parser = argparse.ArgumentParser(description="Operator-level equivalence study (custom operator-only model)")
    parser.add_argument("--imc_file", type=str, default="IMC/configs/azure_lily.json")
    parser.add_argument("--out", type=str, default="calibration_study/equivalence_report.md")
    parser.add_argument("--energy_unit", type=str, default="pJ", choices=["pJ", "nJ"])
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--num_computes", type=int, default=1)
    parser.add_argument("--num_inputs", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=128)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--num_adds", type=int, default=16)
    parser.add_argument("--num_maxpools", type=int, default=16)
    parser.add_argument("--num_acts", type=int, default=16)
    parser.add_argument("--phit_size", type=int, default=16)
    parser.add_argument("--phit_size_add", type=int, default=16)
    parser.add_argument("--bandwidth_mode", action="store_true", help="Use BRAM bandwidth model for event-driven read/write latencies")
    parser.add_argument(
        "--models",
        type=str,
        default="lenet",
        choices=["lenet", "resnet", "both"],
        help="Select end-to-end/layer-wise model scope for this report",
    )
    args = parser.parse_args()

    sim_common.DEBUG = args.debug
    try:
        import scheduler_stats.common as bare_sim_common

        bare_sim_common.DEBUG = args.debug
    except Exception:
        pass
    update_constants(args)

    # Old simulator energy is in nJ (see nn/constant.py). New simulator reports pJ.
    if args.energy_unit == "pJ":
        old_scale = 1e3
        new_scale = 1.0
        energy_unit_label = "pJ"
    else:
        old_scale = 1.0
        new_scale = 1e-3
        energy_unit_label = "nJ"

    layers = build_operator_layers(args)
    by_type = {op: [] for op in OPERATOR_ORDER}
    for layer in layers:
        if layer.type in by_type:
            by_type[layer.type].append(layer)

    lines = []
    lines.append("# IMC_new vs Event‑Driven Simulator — Operator‑Level Study (Operator‑Only Model)")
    lines.append("")
    lines.append("This report compares operator‑level energy/latency breakdowns.")
    lines.append("Event‑driven latency breakdown is a proxy (non‑overlapped analytic estimate).")
    if args.bandwidth_mode:
        lines.append("Event‑driven read/write latency is converted to BRAM bandwidth model (option A).")
    else:
        lines.append("Event‑driven read/write latency uses per‑event SRAM model (default).")
    lines.append("")
    cfg = Config(args.imc_file)
    event_bytes_per_access = C.ELEMS_PER_MV * C.BYTES_PER_ELEM
    imc_raw_bytes_per_access = max(1, math.floor(cfg.bram_width / 8))
    imc_bytes_per_access = imc_raw_bytes_per_access * getattr(cfg, "mem_bw_utilization", 1.0)
    event_bw = event_bytes_per_access / C.SRAM_LAT if C.SRAM_LAT else 0.0
    imc_cycle_ns = 1e9 / (cfg.freq * 1e6)
    imc_bw = imc_bytes_per_access / imc_cycle_ns if imc_cycle_ns else 0.0

    lines.append("Inter-layer pipeline/parallelism assumptions (IMC_new):")
    lines.append("- Layer order is the model order; no global reordering across different layer names.")
    lines.append("- Layer timing profile is (first_output_ns, steady_ns, events).")
    lines.append("- Conv/maxpool dependencies are converted to required upstream token indices R_i(n) from receptive-field geometry.")
    lines.append("- Streaming timeline uses token recurrence: T_i(n)=max(T_i(n-1)+steady_i, T_{i-1}(R_i(n))+first_output_i).")
    lines.append("- Effective layer latency on timeline includes dependency-wait (backpressure_add); critical path accumulates max-overlap only.")
    lines.append("- Attention special case: linear_Q/linear_K/linear_V are grouped as one parallel block via max latency.")
    lines.append("Intra-layer stage assumptions (IMC_new):")
    lines.append("- Conv/linear and maxpool use first_output + (events-1)*steady pipelines.")
    lines.append("- IMC core stage pipeline uses max(vmm, adc, digital, reduction) per output when cfg.pipelinable is true.")
    lines.append("- Activation is practical (finite throughput): act_units parallel ops, act_cycles_per_op per op, configurable in JSON.")
    lines.append("- Memory model assumes enough BRAM banks to sustain configured effective width (bram_width * mem_bw_utilization).")
    lines.append("Assumptions (Event‑Driven reference):")
    lines.append("- Energy units are nJ; constants are from nn/constant.py.")
    lines.append("- Runtime latency is event-driven with explicit readiness/resource checks; breakdown here is analytic proxy.")
    lines.append("- Reads/writes use ELEMS_PER_MV and NUM_* resources to compute event counts.")
    lines.append("- Activation/reduction/maxpool are explicit per-event stages, then globally overlapped by scheduler.")
    lines.append("Expected behavior:")
    lines.append("- Memory‑bound layers (small M) show latency ratios driven by effective port width/bandwidth.")
    lines.append("- Compute‑bound layers (large M) converge as steady‑state compute dominates fill cost.")
    lines.append(
        f"Effective mem access: event‑driven uses ELEMS_PER_MV={C.ELEMS_PER_MV:.0f} and BYTES_PER_ELEM={C.BYTES_PER_ELEM}, "
        f"bytes/access={event_bytes_per_access:.0f} with SRAM_LAT={C.SRAM_LAT}ns → BW={event_bw:.3f} bytes/ns; "
        f"IMC_new uses BRAM width={cfg.bram_width}b with mem_bw_utilization={getattr(cfg, 'mem_bw_utilization', 1.0):.3f} "
        f"→ effective {imc_bytes_per_access:.3f} bytes/access (raw {imc_raw_bytes_per_access}), "
        f"cycle={imc_cycle_ns:.3f}ns → BW={imc_bw:.3f} bytes/ns."
    )
    lines.append(
        f"Configured activation knobs: act_units={getattr(cfg, 'act_units', 16)}, "
        f"act_cycles_per_op={getattr(cfg, 'act_cycles_per_op', 1)}."
    )
    lines.append("")
    lines.append("Execution orchestration (IMC_new):")
    lines.append("- Scheduler maps conv/linear to IMC core GEMM, and maxpool/activation/softmax to FPGA peripherals.")
    lines.append("- MemoryModel provides BRAM read/write latency and energy for all mapped operators.")
    lines.append("- Stats tracks per-layer energy and raw latency, while PipelineProfiler tracks overlap-aware critical path.")
    lines.append("- End-to-end latency is the critical path from PipelineProfiler; raw sum is reported separately for sanity checks.")
    lines.append("")
    if args.models in ("resnet", "both"):
        render_pipeline_tutorial(lines, args)

    for op in OPERATOR_ORDER:
        op_layers = by_type.get(op, [])
        if not op_layers:
            continue

        layer_names = ", ".join(layer.name for layer in op_layers)
        res = aggregate_operator(op_layers, args, old_scale, new_scale)
        render_operator_summary(lines, op, layer_names, res, energy_unit_label)

        if op == "maxpool":
            for layer in op_layers:
                audit = maxpool_latency_audit(layer, res["cfg"], bandwidth_mode=args.bandwidth_mode)
                lines.append(f"### Maxpool Latency Audit — {layer.name}")
                lines.append("Counts and scaling assumptions:")
                rows = [
                    ["output_positions (count)", audit["output_positions"], audit["output_positions"]],
                    ["num_events (events)", audit["num_events"], "n/a"],
                    ["window_elements (count)", audit["window_elements"], audit["window_elements"]],
                ]
                lines.append(format_md_table(["Metric", "Event‑Driven", "IMC_new"], rows))
                lines.append("")
                lines.append("Latency contributions (ns):")
                rows = [
                    ["read_latency", f"{audit['old_read']:.2f}", f"{audit['new_read']:.2f}"],
                    ["write_latency", f"{audit['old_write']:.2f}", f"{audit['new_write']:.2f}"],
                    ["compare_latency", f"{audit['old_compare']:.2f}", f"{audit['new_compare']:.2f}"],
                    ["total_latency", f"{audit['old_total']:.2f}", f"{audit['new_total']:.2f}"],
                ]
                lines.append(format_md_table(["Metric", "Event‑Driven (proxy)", "IMC_new"], rows))
                lines.append(
                    "Note: read/write gaps often reflect different effective port widths "
                    "(event‑driven uses ELEMS_PER_MV; IMC_new uses BRAM width)."
                )
                lines.append("")

    selected_models = []
    if args.models in ("lenet", "both"):
        selected_models.append(("LeNet", lenet_model))
    if args.models in ("resnet", "both"):
        selected_models.append(("ResNet", resnet_model))

    model_results = {}
    for model_name, model_fn in selected_models:
        model_results[model_name] = compute_end_to_end_result(model_fn, args, old_scale, new_scale)

    if "LeNet" in model_results:
        render_model_layerwise_latency(
            lines,
            args,
            "LeNet",
            lenet_model,
            old_scale,
            new_scale,
            energy_unit_label,
            old_e2e=model_results["LeNet"][0]["total_latency"],
            new_e2e=model_results["LeNet"][1]["total_latency"],
            old_e2e_energy=model_results["LeNet"][0]["total_energy"],
            new_e2e_energy=model_results["LeNet"][1]["total_energy"],
            old_layer_energy_stats=model_results["LeNet"][0].get("layer_energy_stats"),
            old_total_mode="exact",
        )
    if "ResNet" in model_results:
        render_model_layerwise_latency(
            lines,
            args,
            "ResNet",
            resnet_model,
            old_scale,
            new_scale,
            energy_unit_label,
            old_e2e=model_results["ResNet"][0]["total_latency"],
            new_e2e=model_results["ResNet"][1]["total_latency"],
            old_e2e_energy=model_results["ResNet"][0]["total_energy"],
            new_e2e_energy=model_results["ResNet"][1]["total_energy"],
            old_layer_energy_stats=model_results["ResNet"][0].get("layer_energy_stats"),
            old_total_mode="proxy",
        )

    lines.append("# End‑to‑End Comparison")
    lines.append("")
    for model_name in [name for name, _ in selected_models]:
        old_res, new_res = model_results[model_name]
        render_end_to_end(lines, model_name, old_res, new_res, energy_unit_label)

    out_path = Path(args.out)
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
