from __future__ import annotations

import argparse
from pathlib import Path

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMC_DIR = ROOT / "IMC"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(IMC_DIR) not in sys.path:
    sys.path.insert(0, str(IMC_DIR))

from simulator import IMC as IMCNew
import scheduler_stats.common as sim_common

from report_utils import (
    MODEL_LIST,
    update_constants,
    run_new_simulator,
    run_old_simulator,
    format_markdown_table,
    format_text_table,
)


def export_tables(args):
    sim_common.DEBUG = args.debug
    update_constants(args.num_adds, args.num_maxpools, args.num_acts, args.phit_size, args.phit_size_add)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    imc_files = [p.strip() for p in args.imc_files.split(",") if p.strip()]

    lines = []
    lines.append("# Simulator Comparison Tables")
    lines.append("")

    lines.append("## 1. Layer-wise latency/energy/resource usage (new simulator, aligned by layer)")
    for model in models:
        if model not in MODEL_LIST:
            continue
        lines.append(f"### {model.upper()}")
        results = {}
        layer_order = []
        layer_type = {}
        for imc_file in imc_files:
            result = run_new_simulator(model, imc_file, args, IMCNew)
            results[imc_file] = result
            if not layer_order:
                for r in result["records"]:
                    layer_order.append(r["layer"])
                    layer_type[r["layer"]] = r["type"]

        tags = [Path(p).stem for p in imc_files]
        headers = ["Layer", "Type"]
        for tag in tags:
            headers.append(f"Latency {tag} (ns)")
        for tag in tags:
            headers.append(f"Energy {tag} (pJ)")
        for tag in tags:
            headers.append(f"IMC tiles {tag}")
        for tag in tags:
            headers.append(f"DSP used {tag}")
        for tag in tags:
            headers.append(f"CLB used {tag}")

        rows = []
        for layer_name in layer_order:
            row = [layer_name, layer_type.get(layer_name, "-")]
            per_imc = []
            for imc_file in imc_files:
                rec_map = {r["layer"]: r for r in results[imc_file]["records"]}
                per_imc.append(rec_map.get(layer_name))

            for rec in per_imc:
                row.append(f"{rec['latency']:.2f}" if rec else "-")
            for rec in per_imc:
                row.append(f"{rec['energy']:.2f}" if rec else "-")
            for rec in per_imc:
                row.append(rec["imc_tiles"] if rec else "-")
            for rec in per_imc:
                row.append(rec["dsp_used"] if rec else "-")
            for rec in per_imc:
                row.append(rec["clb_used"] if rec else "-")

            rows.append(row)

        total_row = ["TOTAL", "-"]
        for imc_file in imc_files:
            res = results[imc_file]
            total_row.append(f"{sum(r['latency'] for r in res['records']):.2f}")
        for imc_file in imc_files:
            res = results[imc_file]
            total_row.append(f"{sum(r['energy'] for r in res['records']):.2f}")
        for imc_file in imc_files:
            res = results[imc_file]
            total_row.append(res["resource_total"]["imc_tiles"])
        for imc_file in imc_files:
            res = results[imc_file]
            total_row.append(res["resource_total"]["dsp_used"])
        for imc_file in imc_files:
            res = results[imc_file]
            total_row.append(res["resource_total"]["clb_used"])
        rows.append(total_row)

        lines.append(format_markdown_table(headers, rows))
        lines.append("")

    lines.append("## 2. Total latency/energy/resource usage (Azure-Lily: new vs old)")
    rows = []
    for model in models:
        new_res = run_new_simulator(model, "IMC/configs/azure_lily.json", args, IMCNew)
        old_res = run_old_simulator(model, args)
        rows.append([
            model,
            "new",
            f"{new_res['total_latency']:.2f}",
            f"{new_res['total_energy']:.2f}",
            new_res["resource_total"]["imc_tiles"],
            new_res["resource_total"]["dsp_used"],
            new_res["resource_total"]["clb_used"],
        ])
        rows.append([
            model,
            "old",
            f"{old_res['total_latency']:.2f}",
            f"{old_res['total_energy']:.2f}",
            "N/A",
            "N/A",
            "N/A",
        ])
        rows.append([
            model,
            "delta (new-old)",
            f"{new_res['total_latency'] - old_res['total_latency']:.2f}",
            f"{new_res['total_energy'] - old_res['total_energy']:.2f}",
            "-",
            "-",
            "-",
        ])

    lines.append(
        format_markdown_table(
            ["Model", "Simulator", "Total Latency (ns)", "Total Energy (pJ)", "IMC tiles", "DSP used", "CLB used"],
            rows,
        )
    )
    lines.append("")

    lines.append("## 3. Quick overview of differences (new vs old)")
    lines.append("- New simulator separates IMC core and FPGA fabric; old simulator models a monolithic DPE pipeline.")
    lines.append("- New simulator uses explicit IMC/FPGA scheduling per layer; old simulator does not model heterogeneous mapping.")
    lines.append("- New simulator exposes resource usage (IMC tiles/DSPs/CLBs) and critical-path latency; old simulator only reports energy totals and event-based latency.")
    lines.append("- New simulator tracks energy breakdowns aligned with hardware blocks; old simulator tracks per-op energy based on abstract constants.")
    lines.append("")

    out_path = Path(args.out)
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")


def print_tables(args):
    sim_common.DEBUG = args.debug
    update_constants(args.num_adds, args.num_maxpools, args.num_acts, args.phit_size, args.phit_size_add)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for model in models:
        if model not in MODEL_LIST:
            continue
        result = run_new_simulator(model, args.imc_file, args, IMCNew)
        rows = []
        for r in result["records"]:
            rows.append([
                r["layer"],
                r["type"],
                f"{r['latency']:.2f}",
                f"{r['energy']:.2f}",
            ])
        rows.append([
            "TOTAL",
            "-",
            f"{sum(r['latency'] for r in result['records']):.2f}",
            f"{sum(r['energy'] for r in result['records']):.2f}",
        ])
        print(f"\n=== {model.upper()} ({args.imc_file}) ===")
        print(format_text_table(["Layer", "Type", "Latency (ns)", "Energy (pJ)"], rows))


def sanity_check(args):
    import imc_legacy

    from report_utils import run_new_simulator as run_new
    from report_utils import run_old_simulator as run_old

    sim_common.DEBUG = args.debug
    imc_legacy.DEBUG = args.debug
    update_constants(args.num_adds, args.num_maxpools, args.num_acts, args.phit_size, args.phit_size_add)

    new = run_new(args.model, args.imc_file, args, IMCNew)
    old = run_old(args.model, args)

    print("New total latency:", f"{new['total_latency']:.2f}")
    print("Old total latency:", f"{old['total_latency']:.2f}")
    print("New total energy:", f"{new['total_energy']:.2f}")
    print("Old total energy:", f"{old['total_energy']:.2f}")


def build_parser():
    parser = argparse.ArgumentParser(description="IMC_new tools")
    sub = parser.add_subparsers(dest="cmd", required=True)

    export_p = sub.add_parser("export_tables", help="Export comparison tables to markdown")
    export_p.add_argument("--out", type=str, default="IMC_new/comparison_tables.md")
    export_p.add_argument("--models", type=str, default="lenet,resnet")
    export_p.add_argument("--imc_files", type=str, default="IMC/configs/azure_lily.json,IMC/configs/nl_dpe.json,IMC/configs/sram_cha.json")
    export_p.add_argument("--debug", action="store_true")
    export_p.add_argument("--num_computes", type=int, default=1)
    export_p.add_argument("--num_inputs", type=int, default=1)
    export_p.add_argument("--seq_length", type=int, default=128)
    export_p.add_argument("--head_dim", type=int, default=128)
    export_p.add_argument("--num_adds", type=int, default=16)
    export_p.add_argument("--num_maxpools", type=int, default=16)
    export_p.add_argument("--num_acts", type=int, default=16)
    export_p.add_argument("--phit_size", type=int, default=16)
    export_p.add_argument("--phit_size_add", type=int, default=16)

    print_p = sub.add_parser("print_tables", help="Print layerwise energy/latency tables")
    print_p.add_argument("--models", type=str, default="lenet,resnet")
    print_p.add_argument("--imc_file", type=str, default="IMC/configs/azure_lily.json")
    print_p.add_argument("--debug", action="store_true")
    print_p.add_argument("--num_computes", type=int, default=1)
    print_p.add_argument("--num_inputs", type=int, default=1)
    print_p.add_argument("--seq_length", type=int, default=128)
    print_p.add_argument("--head_dim", type=int, default=128)
    print_p.add_argument("--num_adds", type=int, default=16)
    print_p.add_argument("--num_maxpools", type=int, default=16)
    print_p.add_argument("--num_acts", type=int, default=16)
    print_p.add_argument("--phit_size", type=int, default=16)
    print_p.add_argument("--phit_size_add", type=int, default=16)

    sanity_p = sub.add_parser("sanity_check", help="Compare new vs old totals")
    sanity_p.add_argument("--model", type=str, required=True)
    sanity_p.add_argument("--imc_file", type=str, default="IMC/configs/azure_lily.json")
    sanity_p.add_argument("--debug", action="store_true")
    sanity_p.add_argument("--num_computes", type=int, default=1)
    sanity_p.add_argument("--num_inputs", type=int, default=1)
    sanity_p.add_argument("--seq_length", type=int, default=128)
    sanity_p.add_argument("--head_dim", type=int, default=128)
    sanity_p.add_argument("--num_adds", type=int, default=16)
    sanity_p.add_argument("--num_maxpools", type=int, default=16)
    sanity_p.add_argument("--num_acts", type=int, default=16)
    sanity_p.add_argument("--phit_size", type=int, default=16)
    sanity_p.add_argument("--phit_size_add", type=int, default=16)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "export_tables":
        export_tables(args)
    elif args.cmd == "print_tables":
        print_tables(args)
    elif args.cmd == "sanity_check":
        sanity_check(args)


if __name__ == "__main__":
    main()
