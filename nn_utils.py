import torch
from torch import nn
import numpy as np
import mujoco as mj

def calculate_alpha(data: mj.MjData,
                    target_position: np.ndarray,
                    threshold: float = 1e-8,
                    ) -> float:
    """Calculate the angle alpha between robot and target.

    Parameters
    ----------
    data : mujoco.MjData
        The MuJoCo data of the robot.
    target_position : np.ndarray
        The target position in world coordinates.
    threshold : float, optional
        A small value to avoid division by zero, by default 1e-8.

    Returns
    -------
    float
        The angle alpha in radians.
    """
    robot_position = data.geom("robot1_core").xpos[:2]
    vector_to_target = target_position[:2] - robot_position

    dist = np.linalg.norm(vector_to_target)
    if dist < threshold:
        return 0.0
    to_target = vector_to_target / dist

    xmat = data.geom("robot1_core").xmat.reshape(3, 3)
    world_forward_2d = (xmat @ [0.0, 1.0, 0.0])[:2]
    wf_norm = np.linalg.norm(world_forward_2d)

    if wf_norm < threshold:
        return 0.0

    world_forward_2d /= wf_norm

    dot = np.dot(world_forward_2d, to_target)
    cross = (
            world_forward_2d[0] * to_target[1]
              - world_forward_2d[1] * to_target[0]
              )
    return float(np.arctan2(cross, dot))

# Currently Completed
def get_robot_state(data:mj.MjData, 
                    target_position:tuple[float, float, float] = (2, 0.0, 0.1),
                    ) -> np.ndarray:
    """
    Extracts the robot state EXCLUDING global position.
    Processes quaternion to be consistent (scaled by sign of w).
    """

    # 1. Get Quaternion (w, x, y, z) - Index 3 to 7
    quat = data.qpos[3:7].copy()
    
    # 2. Scale/Normalize Quaternion
    # If w is negative, negate the whole quaternion.
    if quat[0] < 0:
        quat = -quat
        
    # 3. Use only the Imaginary parts (x, y, z)
    quat_imag = quat[1:] 
    
    # 4. Get Hinge Joints (Index 7 onwards)
    joints = data.qpos[7:]

    # Using both sin and cos gives the network a smooth, circular sense of time
    phase_inputs = [
        np.sin(data.time * 2.0 * np.pi), 
        np.cos(data.time * 2.0 * np.pi)
    ]

    alpha = calculate_alpha(data, target_position=np.array(target_position))
    
    return np.concatenate([quat_imag, joints, [alpha], phase_inputs])

@torch.no_grad()
def fill_parameters(net: nn.Module, vector: torch.Tensor):
    """Fill the parameters of a torch module (net) from a 1-D vector.

    No gradient information is kept.

    The vector's length must be exactly the same with the number
    of parameters of the PyTorch module.

    Args:
        net: The torch module whose parameter values will be filled.
        vector: A 1-D torch tensor which stores the parameter values.
    """
    address = 0
    for p in net.parameters():
        d = p.data.view(-1)
        n = len(d)
        d[:] = torch.as_tensor(vector[address : address + n], device=d.device)
        address += n

    if address != len(vector):
        raise IndexError("The parameter vector is larger than expected")
