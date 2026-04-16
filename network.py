""""Netowrk definition for the RE_sync experiment."""

import torch
import torch.nn as nn
import numpy as np

from nn_utils import get_robot_state


class Network(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fch = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.hidden_activation = nn.ELU()
        self.output_activation = nn.Tanh()

        for p in self.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def forward(self, model, data, target_position=(2, 0.0, 0.1)):
        robot_state = get_robot_state(data, target_position=target_position)
        state = torch.tensor(robot_state.astype(np.float32))
        x = self.hidden_activation(self.fc1(state))
        x = self.hidden_activation(self.fch(x))
        x = self.output_activation(self.fc_out(x)) * (torch.pi / 2)
        return x.detach().numpy()
