"""Watch a trained agent in the MuJoCo viewer.

Run from cartpole/:  python scripts/play.py runs/2026-10-28_1052_ppo_first
"""

import argparse
import time
from pathlib import Path

import mujoco.viewer
from stable_baselines3 import PPO

from cartpole_rl.factory import make_env
from cartpole_rl.interfaces.mujoco_cartpole import MujocoCartPole
from cartpole_rl.utils.configs import load_config


def main() -> None:
    """Play the best agent of a run, one episode after another, until closed."""
    parser = argparse.ArgumentParser(
        description="Watch a trained agent in Mujoco viewer."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="run folder, e.g. runs/2026-10-28_1052_ppo_first (uses its best.zip)",
    )
    args = parser.parse_args()

    configs = load_config(args.run_dir / "config.yaml")
    system = MujocoCartPole(**(configs.get("system") or {}))
    cartpole_env = make_env(system, configs)

    model = PPO.load(args.run_dir / "best.zip", device="cpu")

    viewer = mujoco.viewer.launch_passive(system.model, system.data)

    seed = 1000
    while viewer.is_running():  # one pass = one episode
        obs, info = cartpole_env.reset(seed=seed)
        seed += 1  # a different start each episode
        while viewer.is_running():  # one pass = one timestep
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, reward_parts = cartpole_env.step(action)
            viewer.sync()
            time.sleep(cartpole_env.control_period)
            if terminated or truncated:
                break  # leaves the inner loop only


if __name__ == "__main__":
    main()
