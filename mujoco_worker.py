"""MuJoCo worker for parallel robot evaluation.

Contains the simulation and controller optimization logic to be run in
separate processes. Sets PyTorch threads to 1 to prevent oversubscription.
"""

import os

# Limit OpenMP threads before importing numpy/pytorch
os.environ["OMP_NUM_THREADS"] = "1"

from collections.abc import Callable
from dataclasses import dataclass

import mujoco as mj
import nevergrad as ng
import numpy as np
import torch


@dataclass
class EvalConfig:
    """Configuration passed to each worker."""

    cma_generations: int
    cma_pop_size: int
    spawn_position: tuple[float, float, float]
    target_position: tuple[float, float, float]
    seed: int | None = None


class Network:
    """Simple feedforward controller (ANN)."""

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int
    ) -> None:
        self.W1 = torch.randn(hidden_size, input_size)
        self.b1 = torch.zeros(hidden_size)
        self.W2 = torch.randn(output_size, hidden_size)
        self.b2 = torch.zeros(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.W1 @ x + self.b1)
        return torch.tanh(self.W2 @ x + self.b2)

    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2]


def fill_parameters(net: Network, params: np.ndarray) -> None:
    """Fill network parameters from flat array."""
    offset = 0
    for p in net.parameters():
        numel = p.numel()
        p.data = torch.tensor(
            params[offset : offset + numel],
            dtype=torch.float32,
        ).reshape(p.shape)
        offset += numel


def get_robot_state(
    data: mj.MjData,
    target_position: tuple[float, ...],
) -> np.ndarray:
    """Extract state vector for controller."""
    pos = data.qpos[:3].copy()
    target_vec = np.array(target_position) - pos
    dist = np.linalg.norm(target_vec) + 1e-6
    target_vec /= dist
    return np.concatenate([data.qpos.copy(), data.qvel.copy(), target_vec])


class Tracker:
    """Minimal tracker implementation."""

    def __init__(
        self,
        name_to_bind: str,
        observable_attributes: list,
        quiet: bool = True,
    ) -> None:
        self.quiet = quiet

    def setup(self, spec, data: mj.MjData) -> None:
        pass


class Controller:
    """Neural network controller interface."""

    def __init__(
        self,
        controller_callback_function: Callable,
        tracker: Tracker,
    ) -> None:
        self.callback = controller_callback_function
        self.tracker = tracker

    def set_control(
        self,
        m: mj.MjModel,
        d: mj.MjData,
        target_position: tuple[float, ...],
    ) -> None:
        state = get_robot_state(d, target_position)
        state_t = torch.tensor(state, dtype=torch.float32)
        control = self.callback(state_t).detach().numpy()
        if len(control) != m.nu:
            control = np.resize(control, m.nu)
        d.ctrl[:] = control


def evaluate_individual(args: tuple[str, EvalConfig]) -> float:
    """Evaluate a single robot using CMA-ES controller optimization.

    Args:
        args: Tuple of (xml_string, config) where xml_string is the full MuJoCo XML (world + robot) and config contains evaluation params.

    Returns
    -------
        Best fitness (distance to target, lower is better).
    """
    # Critical: Limit PyTorch threads in subprocess
    torch.set_num_threads(1)

    xml_string, config = args

    # Load model from XML
    try:
        model = mj.MjModel.from_xml_string(xml_string)
        data = mj.MjData(model)
    except Exception:
        # Penalty for invalid morphology
        return float(
            np.linalg.norm(
                np.array(config.target_position)
                - np.array(config.spawn_position),
            )
            * 10,
        )

    # Early exit for immobile robots
    if model.nu < 2:
        return float(
            np.linalg.norm(
                np.array(config.target_position)
                - np.array(config.spawn_position),
            ),
        )

    # Setup controller
    input_size = len(
        get_robot_state(data, target_position=config.target_position),
    )
    net = Network(input_size=input_size, hidden_size=16, output_size=model.nu)
    num_vars = sum(p.numel() for p in net.parameters())

    # CMA-ES setup
    rng = np.random.default_rng(config.seed)
    opt = ng.optimizers.CMA(
        parametrization=num_vars,
        budget=config.cma_generations * config.cma_pop_size,
    )

    controller = Controller(
        controller_callback_function=net.forward,
        tracker=Tracker("core", ["xpos"], quiet=True),
    )

    min_fitness = float("inf")

    # Optimization loop
    for _ in range(config.cma_generations):
        candidates = [opt.ask() for _ in range(config.cma_pop_size)]

        for cand in candidates:
            fill_parameters(net, cand.value)
            mj.mj_resetData(model, data)

            # Warmup (3s)
            data.ctrl[:] = rng.normal(scale=0.1, size=model.nu)
            mj.mj_step(model, data, nstep=300)

            # Adjust target based on displacement
            displacement = data.qpos[:3].copy()
            adjusted_target = tuple(
                np.array(config.target_position) + displacement,
            )

            # Controlled simulation (10s)
            def cb(m, d) -> None:
                controller.set_control(m, d, target_position=adjusted_target)

            mj.set_mjcb_control(cb)
            mj.mj_step(model, data, nstep=1000)
            mj.set_mjcb_control(None)

            # Fitness
            final_pos = data.qpos[:3].copy()
            fitness = float(
                np.linalg.norm(np.array(adjusted_target) - final_pos),
            )
            opt.tell(cand, fitness)
            min_fitness = min(min_fitness, fitness)

    return min_fitness
