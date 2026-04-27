import argparse
import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))
sys.path.insert(0, SRC_DIR)

from plot_three_window_results import build_three_window_plot
from plot_individual_windows import plot_all_individual
from real_data_engine import RealDataConfig, run_real_data_inversion


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run three 10 km inversions and stitch them into one 30 km result."
    )
    parser.add_argument(
        "--csv",
        default=os.path.join(PROJECT_DIR, "real_data", "real_testing_data.csv"),
    )
    parser.add_argument(
        "--results-dir",
        default=os.path.join(PROJECT_DIR, "results"),
    )
    parser.add_argument("--trend-degree", type=int, default=2)
    parser.add_argument("--maxiter", type=int, default=45)
    parser.add_argument("--popsize", type=int, default=9)
    parser.add_argument("--mc", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--radio-weight",
        type=float,
        default=1.0,
        help="Higher values improve K/U/Th fit in multi-window runs; gravity still has weight 2.0.",
    )
    parser.add_argument(
        "--reverse",
        default="auto",
        choices=["auto", "false", "true"],
        help="Whether each fixed window may be mirrored to match the model family.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast smoke-test settings. Use full defaults for final results.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    maxiter = args.maxiter
    popsize = args.popsize
    mc = args.mc
    if args.quick:
        maxiter = min(maxiter, 3)
        popsize = min(popsize, 4)
        mc = min(mc, 2)

    os.makedirs(args.results_dir, exist_ok=True)
    windows = [(0.0, "window_00_10"), (10.0, "window_10_20"), (20.0, "window_20_30")]
    for idx, (start, name) in enumerate(windows, start=1):
        print("\n" + "#" * 78)
        print(f"# THREE-WINDOW RUN: window {idx}/3 ({start:.0f}-{start + 10:.0f} km)")
        print("#" * 78)
        config = RealDataConfig(
            csv_path=args.csv,
            results_dir=os.path.join(args.results_dir, name),
            window_start_km=str(start),
            window_length_km=10.0,
            trend_degree=args.trend_degree,
            seed=args.seed + idx * 101,
            maxiter=maxiter,
            popsize=popsize,
            mc_samples=mc,
            radio_weight=args.radio_weight,
            fixed_window_reverse=args.reverse,
        )
        run_real_data_inversion(config)

    build_three_window_plot(args.results_dir)
    plot_all_individual(args.results_dir)


if __name__ == "__main__":
    main()
