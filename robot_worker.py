"""Shared MuJoCo evaluation worker for all RE_* experiments.

Subclasses ariel's MuJoCoWorkerBase so instances are picklable and safe to
pass to multiprocessing.Pool.imap.  The base class handles thread-count
clamping and XML loading; this module adds the CMA-ES controller loop.
"""

from dataclasses import dataclass
from multiprocessing import get_context

import mujoco as mj
import nevergrad as ng
import numpy as np
import torch

from ariel.simulation.mujoco_worker import EvalConfig, MuJoCoWorkerBase


# ---------------------------------------------------------------------------
# Minimal feedforward ANN controller
# ---------------------------------------------------------------------------


class _Network:
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        self.W1 = torch.randn(hidden_size, input_size)
        self.b1 = torch.zeros(hidden_size)
        self.W2 = torch.randn(output_size, hidden_size)
        self.b2 = torch.zeros(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.W1 @ x + self.b1)
        return torch.tanh(self.W2 @ x + self.b2)

    def parameters(self) -> list[torch.Tensor]:
        return [self.W1, self.b1, self.W2, self.b2]


def _fill_parameters(net: _Network, params: np.ndarray) -> None:
    offset = 0
    for p in net.parameters():
        numel = p.numel()
        p.data = torch.tensor(
            params[offset : offset + numel], dtype=torch.float32
        ).reshape(p.shape)
        offset += numel


def _get_state(data: mj.MjData, target: tuple[float, ...]) -> np.ndarray:
    pos = data.qpos[:3].copy()
    vec = np.array(target) - pos
    dist = np.linalg.norm(vec) + 1e-6
    return np.concatenate([data.qpos.copy(), data.qvel.copy(), vec / dist])


# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------


@dataclass
class LocomotionConfig(EvalConfig):
    """Extends EvalConfig with CMA-ES hyper-parameters."""

    cma_generations: int = 20
    cma_pop_size: int = 5
    hidden_size: int = 16


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class RobotWorker(MuJoCoWorkerBase):
    """Evaluates a robot on a targeted-locomotion task.

    Fitness = final Euclidean distance to the target after CMA-ES controller
    optimisation (lower is better).  Returns a distance-based penalty for
    robots with fewer than two actuators.
    """

    def __call__(self, args: tuple[str, LocomotionConfig]) -> float:
        try:
            return super().__call__(args)
        except Exception:
            _, config = args
            penalty = float(
                np.linalg.norm(
                    np.array(config.target_position) - np.array(config.spawn_position)
                )
            ) * 10
            return penalty

    def evaluate(
        self,
        model: mj.MjModel,
        data: mj.MjData,
        config: LocomotionConfig,
    ) -> float:
        if model.nu < 2:
            return float(
                np.linalg.norm(
                    np.array(config.target_position) - np.array(config.spawn_position)
                )
            )

        rng = np.random.default_rng(config.seed)
        input_size = len(_get_state(data, config.target_position))
        net = _Network(input_size, config.hidden_size, model.nu)
        num_vars = sum(p.numel() for p in net.parameters())

        opt = ng.optimizers.CMA(
            parametrization=num_vars,
            budget=config.cma_generations * config.cma_pop_size,
        )

        min_fitness = float("inf")

        for _ in range(config.cma_generations):
            candidates = [opt.ask() for _ in range(config.cma_pop_size)]
            for cand in candidates:
                _fill_parameters(net, cand.value)
                mj.mj_resetData(model, data)

                # 3-second warmup with random controls
                data.ctrl[:] = rng.normal(scale=0.1, size=model.nu)
                mj.mj_step(model, data, nstep=300)

                # Compensate for warmup displacement so required travel stays constant
                displacement = data.qpos[:3].copy()
                target = tuple(np.array(config.target_position) + displacement)

                def cb(m, d) -> None:
                    state = _get_state(d, target)
                    ctrl = net.forward(torch.tensor(state, dtype=torch.float32))
                    d.ctrl[:] = ctrl.detach().numpy()[: m.nu]

                mj.set_mjcb_control(cb)
                mj.mj_step(model, data, nstep=1_000)
                mj.set_mjcb_control(None)

                fitness = float(np.linalg.norm(np.array(target) - data.qpos[:3]))
                opt.tell(cand, fitness)
                min_fitness = min(min_fitness, fitness)

        return min_fitness


# ---------------------------------------------------------------------------
# Pool helper (used by evaluate_pop in all RE_* scripts)
# ---------------------------------------------------------------------------


def run_pool(
    eval_args: list[tuple[str, LocomotionConfig]],
    num_workers: int,
) -> list[float]:
    """Run eval_args through a spawn-context process pool."""
    worker = RobotWorker()
    ctx = get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        return list(pool.imap(worker, eval_args, chunksize=1))
