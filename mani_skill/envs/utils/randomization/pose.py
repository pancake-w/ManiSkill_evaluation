import numpy as np
import torch

from transforms3d.euler import euler2quat
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)
from mani_skill.utils.structs.types import Device


def random_quaternions(
    n: int,
    device: Device = None,
    lock_x: bool = False,
    lock_y: bool = False,
    lock_z: bool = False,
    bounds=(0, np.pi * 2),
):
    """
    Generates random quaternions by generating random euler angles uniformly, with each of
    the X, Y, Z angles ranging from bounds[0] to bounds[1] radians. Can optionally
    choose to fix X, Y, and/or Z euler angles to 0 via lock_x, lock_y, lock_z arguments
    """
    dist = bounds[1] - bounds[0]
    xyz_angles = torch.rand((n, 3), device=device) * (dist) + bounds[0]
    if lock_x:
        xyz_angles[:, 0] *= 0
    if lock_y:
        xyz_angles[:, 1] *= 0
    if lock_z:
        xyz_angles[:, 2] *= 0
    return matrix_to_quaternion(euler_angles_to_matrix(xyz_angles, convention="XYZ"))

def generate_random_pos(xy_center, fixed_z_value, half_edge_length_x, half_edge_length_y):
    """
    Generate a random position within a rectangular region in the XY plane at a fixed Z height.

    Parameters:
    xy_center (array, tuple or list of float): The (x, y) coordinates of the center of the rectangle.
    z_value (float): The fixed Z-coordinate of the generated position.
    half_edge_length_x (float): Half of the rectangle's edge length along the X-axis.
    half_edge_length_y (float): Half of the rectangle's edge length along the Y-axis.

    Returns:
    np.ndarray: A NumPy array of shape (3,) representing the (x, y, z) coordinates of the random position.
    """
    random_x = np.random.uniform(xy_center[0] - half_edge_length_x, xy_center[0] + half_edge_length_x)
    random_y = np.random.uniform(xy_center[1] - half_edge_length_y, xy_center[1] + half_edge_length_y)
    return np.array([random_x, random_y, fixed_z_value])

def generate_random_pos_batch(xy_center, fixed_z_value, half_edge_length_x, half_edge_length_y, batch_size):
    random_x = np.random.uniform(
        xy_center[0] - half_edge_length_x, xy_center[0] + half_edge_length_x, size=(batch_size,)
    )
    random_y = np.random.uniform(
        xy_center[1] - half_edge_length_y, xy_center[1] + half_edge_length_y, size=(batch_size,)
    )
    z = np.full((batch_size,), fixed_z_value)
    return np.stack([random_x, random_y, z], axis=1)  # (b, 3)

def get_objs_random_pose_batch(xy_center, half_edge_length_x, half_edge_length_y, z_value,
                               extents_x, extents_y, quats, b, threshold_scale=1.0):
    xy_center = np.array(xy_center)
    half_edge_length_x = np.array(half_edge_length_x)
    half_edge_length_y = np.array(half_edge_length_y)
    z_value = np.array(z_value)
    extents_x = np.array(extents_x)
    extents_y = np.array(extents_y)

    if np.linalg.norm([half_edge_length_x[0], half_edge_length_y[0]]) >= np.linalg.norm([half_edge_length_x[1], half_edge_length_y[1]]):
        first_idx, second_idx = 1, 0  
    else:
        first_idx, second_idx = 0, 1

    half_extents_x = extents_x / 2.0
    half_extents_y = extents_y / 2.0

    # threshold = threshold_scale * (max(half_extents_x) + max(half_extents_y)) # 规则长方体/正方体
    threshold = threshold_scale * np.linalg.norm([2 * max(half_extents_x), 2 * max(half_extents_y)]) # 物体的形状不规则
    # threshold = threshold_scale * np.mean([half_extents_x[0] + half_extents_x[1], half_extents_y[0] + half_extents_y[1]]) # 物体大小相差比较大

    # Generate batch positions for first object
    first_obj_xyz = generate_random_pos_batch(
        xy_center[first_idx], z_value[first_idx],
        half_edge_length_x[first_idx], half_edge_length_y[first_idx], b
    )

    second_obj_xyz = np.zeros((b, 3))
    n_candidates = 500
    for i in range(b):
        candidates = generate_random_pos_batch(
            xy_center[second_idx], z_value[second_idx],
            half_edge_length_x[second_idx], half_edge_length_y[second_idx],
            n_candidates
        )
        distances = np.linalg.norm(candidates[:, :2] - first_obj_xyz[i, :2], axis=1)

        valid_idx = np.where(distances >= threshold)[0]
        if len(valid_idx) > 0:
            second_obj_xyz[i] = candidates[valid_idx[0]]
        else:
            max_idx = np.argmax(distances)
            second_obj_xyz[i] = candidates[max_idx]


    # Reorder to source/target
    if first_idx == 0:
        source_obj_xyz = first_obj_xyz
        target_obj_xyz = second_obj_xyz
    else:
        source_obj_xyz = second_obj_xyz
        target_obj_xyz = first_obj_xyz

    # Batch quats
    source_obj_quat = np.array([
        q if q is not None else euler2quat(0, 0, np.random.uniform(-np.pi, np.pi), "sxyz")
        for q in [quats[0]] * b
    ])
    target_obj_quat = np.array([
        q if q is not None else euler2quat(0, 0, np.random.uniform(-np.pi, np.pi), "sxyz")
        for q in [quats[1]] * b
    ])

    return (
        torch.from_numpy(source_obj_xyz).float(),
        torch.from_numpy(source_obj_quat).float(),
        torch.from_numpy(target_obj_xyz).float(),
        torch.from_numpy(target_obj_quat).float(),
    )


def get_single_obj_grid_pose_batch(xy_center, half_edge_length_x, half_edge_length_y, z_value, step_size, batch_size, episode_idx=None):
    xy_center = np.array(xy_center)
    x_range = np.array([-half_edge_length_x, half_edge_length_x])
    y_range = np.array([-half_edge_length_y, half_edge_length_y])

    # x_vals = np.random.uniform(x_range[0]+xy_center[0], x_range[1]+xy_center[0], 110)
    # y_vals = np.random.uniform(y_range[0]+xy_center[1], y_range[1]+xy_center[1], 110)

    x_vals = np.arange(x_range[0] + xy_center[0], x_range[1] + xy_center[0] + 1e-6, step_size)
    y_vals = np.arange(y_range[0] + xy_center[1], y_range[1] + xy_center[1] + 1e-6, step_size)

    X, Y = np.meshgrid(x_vals, y_vals)
    grid_positions = np.vstack([X.ravel(), Y.ravel()]).T  # (N, 2)

    total_positions = grid_positions.shape[0]
    if isinstance(episode_idx, int):
        pos_ids = (np.arange(batch_size) + episode_idx) % total_positions
    else:
        pos_ids = np.random.choice(total_positions, size=batch_size, replace=False)

    xy_batch = grid_positions[pos_ids]
    z_batch = np.zeros((batch_size, 1)) + z_value
    xyz_batch = np.hstack([xy_batch, z_batch])  # (b, 3)

    # default unit quaternion [1, 0, 0, 0] for all
    quat_batch = np.tile(np.array([1, 0, 0, 0]), (batch_size, 1))  # (b, 4)

    return xyz_batch, quat_batch