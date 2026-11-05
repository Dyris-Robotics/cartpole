from collections.abc import Iterable

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm

from cartpole_rl.envs.environment import Environment


def evaluate(
    model: BaseAlgorithm, env: Environment, seeds: Iterable[int]
) -> tuple[float, int]:
    """Play one episode per seed with no exploring.

    Returns:
        The average score, and the number of episodes that ended off the rail.
    """
    scores, crashes = [], 0
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        total = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        scores.append(total)
        crashes += int(terminated)
    return float(np.mean(scores)), crashes
