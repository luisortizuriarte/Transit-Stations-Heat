#!/usr/bin/env python3
"""
main.py
=======
Command-line interface for the Transit Station Heat Exposure Analysis pipeline.

Usage Examples:
---------------
# Run full pipeline for all cities (NYC, Chicago, DC):
python main.py

# Run only walkshed network routing for Chicago and DC:
python main.py --cities Chicago DC --stage walksheds

# Run only zonal temperature extraction on existing walksheds:
python main.py --cities NYC --stage zonal

# Custom pedestrian velocity (1.3 m/s) and walking threshold (15 minutes):
python main.py --walk-speed 1.30 --walk-time 15
"""

import argparse
import sys
from transit_heat import TransitHeatPipeline
from transit_heat.config import DEFAULT_CITIES, DEFAULT_WALK_SPEED_MPS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transit Station Heat Exposure Analysis Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--cities", "-c",
        nargs="+",
        default=DEFAULT_CITIES,
        help="Metropolitan transit networks to process (e.g. NYC Chicago DC)"
    )

    parser.add_argument(
        "--stage", "-s",
        choices=["all", "walksheds", "zonal", "thermal"],
        default="all",
        help="Pipeline processing stage to execute"
    )

    parser.add_argument(
        "--walk-speed",
        type=float,
        default=DEFAULT_WALK_SPEED_MPS,
        help="Standardized adult pedestrian walking velocity in meters/second"
    )

    parser.add_argument(
        "--walk-time",
        type=int,
        default=10,
        help="Transit access walking time threshold in minutes"
    )

    parser.add_argument(
        "--cache-folder",
        default="cache",
        help="Directory folder for caching OpenStreetMap network graphs"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("Transit Station Heat Exposure Analysis Pipeline")
    print(f"Target Cities   : {', '.join(args.cities)}")
    print(f"Execution Stage : {args.stage}")
    print(f"Walking Speed   : {args.walk_speed} m/s")
    print(f"Travel Duration : {args.walk_time} minutes ({args.walk_speed * args.walk_time * 60:.1f} m reach)")
    print("=" * 70)

    stages = ['walksheds', 'zonal'] if args.stage == 'all' else [args.stage]

    pipeline = TransitHeatPipeline(
        cities=args.cities,
        walk_speed_mps=args.walk_speed,
        trip_time_seconds=args.walk_time * 60
    )

    pipeline.run(stages=stages)

    print("\n" + "=" * 70)
    print("Pipeline execution completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
