import argparse
import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))
sys.path.insert(0, SRC_DIR)

from real_data_engine import RealDataConfig, run_real_data_inversion


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run calibrated real-data inversion for qgeo_v3."
    )
    parser.add_argument(
        "--csv",
        default=os.path.join(PROJECT_DIR, "real_data", "real_testing_data.csv"),
        help="Input CSV with dist_km, G_mGal, U_ppm, Th_ppm, K_pct columns.",
    )
    parser.add_argument(
        "--results-dir",
        default=os.path.join(PROJECT_DIR, "results"),
        help="Directory for NPZ, JSON, and plots.",
    )
    parser.add_argument(
        "--window-start",
        default="auto",
        help='Window start in km, or "auto" to scan the full profile.',
    )
    parser.add_argument("--window-km", type=float, default=10.0)
    parser.add_argument("--trend-degree", type=int, default=2)
    parser.add_argument("--maxiter", type=int, default=45)
    parser.add_argument("--popsize", type=int, default=9)
    parser.add_argument("--mc", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--radio-weight", type=float, default=0.35)
    parser.add_argument(
        "--no-reverse",
        action="store_true",
        help="Disable reversed-profile candidates during auto-window scoring.",
    )
    parser.add_argument(
        "--fixed-polarity",
        action="store_true",
        help="Disallow negative calibration scales for gravity and radiometrics.",
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
        maxiter = min(maxiter, 4)
        popsize = min(popsize, 5)
        mc = min(mc, 3)

    config = RealDataConfig(
        csv_path=args.csv,
        results_dir=args.results_dir,
        window_start_km=args.window_start,
        window_length_km=args.window_km,
        trend_degree=args.trend_degree,
        seed=args.seed,
        maxiter=maxiter,
        popsize=popsize,
        mc_samples=mc,
        radio_weight=args.radio_weight,
        allow_reverse=not args.no_reverse,
        allow_negative_gravity_scale=not args.fixed_polarity,
        allow_negative_radio_scale=not args.fixed_polarity,
    )
    run_real_data_inversion(config)


if __name__ == "__main__":
    main()
