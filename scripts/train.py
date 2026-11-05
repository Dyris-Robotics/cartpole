"""Train PPO on the cart-pole.

Every TEST_EVERY timesteps, test the agent and keep it if it is the best so far.

Run from cartpole/:  python scripts/train.py [--config configs/env.yaml]
"""

import argparse
import shutil
import time
from pathlib import Path

from stable_baselines3 import PPO

from cartpole_rl.evaluation import evaluate
from cartpole_rl.factory import make_env
from cartpole_rl.interfaces.mujoco_cartpole import MujocoCartPole
from cartpole_rl.utils.configs import load_config
from cartpole_rl.utils.run_dir import make_run_dir

RUN_NAME = "ppo_first"
TOTAL_TIMESTEPS = 500_000  # about 1,000 episodes of 10 s
TEST_EVERY = 20_000  # pause and test the brain this often
TEST_SEEDS = range(1000, 1020)  # the same 20 test episodes every time


def main() -> None:
    """Train, test regularly, and save the best agent in a new run folder."""
    # Configs Load
    parser = argparse.ArgumentParser(description="Train PPO on the cart-pole.")
    parser.add_argument("--config", type=Path, default=Path("configs/env.yaml"))
    args = parser.parse_args()
    configs = load_config(args.config)

    # Run Save Location
    run_dir = make_run_dir(RUN_NAME)
    shutil.copy(args.config, run_dir / "config.yaml")

    # Cartpole Env initialization
    train_env = make_env(
        MujocoCartPole(**(configs.get("system") or {})), configs
    )  # PPO practises here
    test_env = make_env(
        MujocoCartPole(**(configs.get("system") or {})), configs
    )  # we test here

    # Model Creation
    model = PPO(
        "MlpPolicy", train_env, seed=0, device="cpu", tensorboard_log=str(run_dir)
    )

    # Other
    best = float("-inf")
    start = time.time()

    print(" timesteps   test score   off rail   best so far   minutes")

    while model.num_timesteps < TOTAL_TIMESTEPS:
        model.learn(TEST_EVERY, reset_num_timesteps=False)

        score, crashes = evaluate(model, test_env, TEST_SEEDS)

        model.logger.record("test/score", score)
        model.logger.record("test/off_rail", crashes)
        model.logger.dump(model.num_timesteps)

        note = ""
        if score > best:
            best = score
            model.save(run_dir / "best.zip")
            note = "  <- saved"

        minutes = (time.time() - start) / 60
        print(
            f" {model.num_timesteps:9d}    {score:7.1f}      "
            f"{crashes:2d}/{len(TEST_SEEDS)}     {best:7.1f}      "
            f"{minutes:5.1f}{note}"
        )

    print(f"\nBest test score {best:.1f}, saved in {run_dir / 'best.zip'}")


if __name__ == "__main__":
    main()
