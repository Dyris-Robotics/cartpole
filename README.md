<div align="center">
  <p align="center">
    <img src="docs/images/cartpole_banner.png" alt="Cart-Pole Swing-Up" width="100%">
  </p>
</div>

<div style="display: flex; justify-content: center; gap: 10px;">
  <p>
    <img src="https://img.shields.io/badge/MuJoCo-3.12-blue" alt="MuJoCo 3.12">
  </p>
  <p>
    <img src="https://img.shields.io/badge/Gymnasium-1.3-blue" alt="Gymnasium 1.3">
  </p>
  <p>
    <img src="https://img.shields.io/badge/Stable--Baselines3-PPO-blue" alt="Stable-Baselines3 PPO">
  </p>
  <p>
    <img src="https://img.shields.io/badge/PyTorch-CPU-blue?logo=pytorch" alt="PyTorch">
  </p>
</div>

---

## About The Project

`CARTPOLE` is a reinforcement learning project that teaches an agent to swing a pole up from its hanging position and keep it balanced upright, by moving a cart along a rail. It is designed from the ground up as a clean, structured and reusable foundation for training robot controllers in simulation and taking them to real hardware.

<div align="center">
  <img src="docs/images/pole_hangs.png" alt="Cart-Pole Robot" width="80%">
</div>

The cart-pole is a small problem, but it already contains everything that makes robot learning hard, a part that cannot be driven directly, a motor with limited strength, a rail that ends, sensors that are never perfect, and a real robot that never quite behaves like its simulation.

<div align="center">
  <p align="center">
    <img src="docs/images/replay_run.gif" alt="Read Documentation" width="100%">
  </p>
</div>

Rather than being a one-off training script, `CARTPOLE` is built using real-world software engineering practices. It features an environment that talks to the robot only through an interface, so the same code runs on the MuJoCo simulation and on a real robot. Randomizations and disturbances plug in through registries, every setting lives in a configuration file, and the documentation covers each part. Together, these provide a **robust foundation for anyone looking to build, study, or extend a sim-to-real reinforcement learning pipeline.**

---

## Documentation

> [!IMPORTANT]
>
> `CARTPOLE DOCUMENTATION`: https://cartpole-rl.netlify.app/
>

<div align="center">
  <p align="center">
    <img src="docs/images/read_it.gif" alt="Read Documentation" width="70%">
  </p>
</div>

---

### Development

To contribute, install the development tools (Ruff, Pyright, pre-commit) and enable the git hooks:

```bash
pip install -e ".[dev]"
pre-commit install            # format and check on every commit
pre-commit run --all-files    # run all checks once on the whole repo
```
