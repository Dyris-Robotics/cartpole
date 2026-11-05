"""Test a trained agent on starts it has never been judged on.

Run from cartpole/:  python scripts/evaluate.py runs/2026-10-28_1052_ppo_first
"""

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from cartpole_rl.evaluation import evaluate
from cartpole_rl.factory import make_env
from cartpole_rl.interfaces.mujoco_cartpole import MujocoCartPole
from cartpole_rl.utils.configs import load_config

FIRST_SEED = 5000  # train.py tests on 1000-1019, so start well away from those


def main() -> None:
    """Evaluate the best agent of a run and save the results next to it."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained agent on fresh seeds."
    )
    parser.add_argument(
        "run_dir", type=Path, help="run folder, e.g. runs/2026-10-28_1052_ppo_first"
    )
    parser.add_argument(
        "--episodes", type=int, default=100, help="number of test episodes"
    )
    parser.add_argument(
        "--gear",
        type=float,
        default=None,
        help="test with this fixed gear instead of the run's config",
    )
    parser.add_argument(
        "--push",
        type=float,
        default=None,
        help="knock the pole every 3 s by up to this many rad/s",
    )

    args = parser.parse_args()

    configs = load_config(args.run_dir / "config.yaml")
    if args.gear is not None:
        configs["randomizations"] = {
            "cart_gear": {"gear_min": args.gear, "gear_max": args.gear}
        }  # exactly this gear
    if args.push is not None:
        configs["disturbances"] = {
            "pole_kick": {"kick_angular_velocity": args.push, "every_seconds": 3.0}
        }

    cartpole_env = make_env(MujocoCartPole(**(configs.get("system") or {})), configs)

    model = PPO.load(args.run_dir / "best.zip", device="cpu")

    # One episode at a time, so we keep each score and can report the worst one.
    scores, off_rail = [], 0
    for seed in range(FIRST_SEED, FIRST_SEED + args.episodes):
        score, crashed = evaluate(model, cartpole_env, [seed])
        scores.append(score)
        off_rail += crashed

    results = {
        "episodes": args.episodes,
        "mean": float(np.mean(scores)),
        "worst": float(np.min(scores)),
        "worst_seed": FIRST_SEED + int(np.argmin(scores)),
        "off_rail": int(off_rail),
    }
    print(
        f"{results['episodes']} episodes: mean {results['mean']:.1f}, "
        f"worst {results['worst']:.1f} (seed {results['worst_seed']}), "
        f"off rail {results['off_rail']}/{results['episodes']}"
    )

    name = "eval"
    if args.gear is not None:
        name += f"_gear{args.gear:g}"
    if args.push is not None:
        name += f"_push{args.push:g}"
    out_file = args.run_dir / f"{name}.json"

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved in {out_file}")


if __name__ == "__main__":
    main()
