"""MuJoCo worker for parallel robot evaluation.

Standalone worker for parallel MuJoCo evaluation using Nevergrad CMA-ES.
Fully importable, no shared state, zero-order hold control.

Designed to be called via multiprocessing.Pool.imap from RE_async.py:
    evaluate_individual((xml_string, EvalConfig)) -> float
where the returned float is the best (lowest) distance-to-target fitness
found over the inner CMA-ES loop.
"""

import os

# Limit OpenMP threads BEFORE importing numpy/torch in workers
os.environ.setdefault("OMP_NUM_THREADS", "1")

from dataclasses import dataclass

import mujoco as mj
import nevergrad as ng
import numpy as np
import torch
from torch import nn


@dataclass
class EvalConfig:
    """Configuration passed to each worker."""

    cma_generations: int
    cma_pop_size: int
    spawn_position: tuple[float, float, float]
    target_position: tuple[float, float, float]
    seed: int | None = None


class ANN(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 32,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.hidden_activation = nn.ELU()
        self.output_activation = nn.Tanh()

        for p in self.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def forward(self, state: torch.Tensor) -> np.ndarray:
        x = self.hidden_activation(self.fc1(state))
        x = self.hidden_activation(self.fc2(x))
        x = self.output_activation(self.fc_out(x)) * (torch.pi / 2)
        return x.detach().numpy()

    def __call__(self, model: mj.MjModel, data: mj.MjData) -> None:
        # Observation: orientation quaternion + phase clock
        qpos = data.qpos[3:7].copy() if data.qpos.size >= 7 else np.zeros(4)
        phase_inputs = np.array(
            [
                2.0 * np.sin(data.time * 2.0 * np.pi),
                2.0 * np.cos(data.time * 2.0 * np.pi),
            ],
            dtype=np.float32,
        )
        obs = torch.tensor(
            np.concatenate([qpos, phase_inputs]).astype(np.float32),
            dtype=torch.float32,
        )

        with torch.no_grad():
            action = self.forward(obs)

        # Zero-order hold
        if action.shape[0] != model.nu:
            action = np.resize(action, model.nu)
        data.ctrl[:] = action


def fill_parameters(net: ANN, flat_weights: np.ndarray) -> None:
    """Inject a flat weight vector into the network's parameters."""
    flat_weights = np.ascontiguousarray(flat_weights, dtype=np.float32)
    with torch.no_grad():
        idx = 0
        for param in net.parameters():
            numel = param.numel()
            param.copy_(
                torch.from_numpy(flat_weights[idx : idx + numel]).view_as(param),
            )
            idx += numel


def evaluate_individual(args: tuple[str, EvalConfig]) -> float:
    """Evaluate a single robot using CMA-ES controller optimization.

    Args:
        args: (xml_string, EvalConfig) — full MuJoCo XML (world + robot) and
            evaluation parameters.

    Returns
    -------
        Best fitness (distance to target, lower is better) found across the
        inner CMA-ES loop.
    """
    # Critical: prevent PyTorch oversubscription in subprocess
    torch.set_num_threads(1)

    xml_string, config = args

    spawn = np.array(config.spawn_position)
    target = np.array(config.target_position)
    base_dist = float(np.linalg.norm(target - spawn))

    # Load model from XML
    try:
        model = mj.MjModel.from_xml_string(xml_string)
        data = mj.MjData(model)
    except Exception:
        # Penalty for invalid morphology
        return base_dist * 10.0

    # Early exit for immobile robots
    if model.nu < 2:
        return base_dist

    # Network sized to robot's actuator count
    obs_dim = 6  # 4 (quaternion) + 2 (phase)
    action_dim = model.nu
    net = ANN(input_size=obs_dim, output_size=action_dim)
    num_vars = sum(p.numel() for p in net.parameters())

    # CMA-ES setup (mirrors RE_sync's ParametrizedCMA usage)
    rng = np.random.default_rng(config.seed)
    initial_weights = rng.uniform(-0.5, 0.5, size=(num_vars,))
    param = ng.p.Array(init=initial_weights)
    param.set_mutation(sigma=0.075)

    cma_config = ng.optimizers.ParametrizedCMA()
    opt = cma_config(
        parametrization=param,
        budget=config.cma_generations * config.cma_pop_size,
    )

    min_fitness = float("inf")

    for _ in range(config.cma_generations):
        candidates = [opt.ask() for _ in range(config.cma_pop_size)]

        for cand in candidates:
            fill_parameters(net, np.asarray(cand.value))
            mj.mj_resetData(model, data)

            # Warmup (~3s) — let the body settle / "flop"
            data.ctrl[:] = rng.normal(scale=0.1, size=model.nu)
            mj.mj_step(model, data, nstep=300)

            # Re-anchor target relative to settled position
            displacement = data.qpos[:3].copy()
            adjusted_target = target + displacement

            # Controlled rollout (~10s)
            mj.set_mjcb_control(lambda m, d: net(m, d))
            mj.mj_step(model, data, nstep=1000)
            mj.set_mjcb_control(None)

            # Distance-to-target fitness (lower is better)
            final_pos = data.qpos[:3].copy()
            fitness = float(np.linalg.norm(adjusted_target - final_pos))
            opt.tell(cand, fitness)
            min_fitness = min(min_fitness, fitness)

    return min_fitness
