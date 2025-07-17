import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np


def rotation_matrix_from_vectors(a, b):
    """ Return the rotation matrix that aligns vector a to vector b """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)

    if np.isclose(c, 1.0):
        return np.eye(3)

    if np.isclose(c, -1.0):
        # 180-degree rotation using Householder reflection
        # Find any orthogonal vector to 'a'
        axis = np.zeros(3)
        if not np.isclose(a[0], 0.0):
            axis = np.array([-a[1], a[0], 0.0])
        else:
            axis = np.array([0.0, -a[2], a[1]])
        axis /= np.linalg.norm(axis)

        # Rodrigues formula with theta = pi:
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        return np.eye(3) + 2 * K @ K  # Since sin(pi)=0, (1-cos(pi))=2

    s = np.linalg.norm(v)
    kmat = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    R = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
    return R


def transform_trajectory(tra, v1, v2):
    """
    Rotate and scale the trajectory 'tra' so that:
    - tra[0] maps to v1
    - tra[-1] maps to v2
    """
    # Original vectors
    src_vec = tra[-1] - tra[0]
    tgt_vec = v2 - v1

    # Compute rotation
    R = rotation_matrix_from_vectors(src_vec, tgt_vec)

    # Compute scale
    scale = np.linalg.norm(tgt_vec) / np.linalg.norm(src_vec)

    # Apply rotation and scale to the centered trajectory
    centered_tra = tra - tra[0]  # move to origin
    rotated_scaled_tra = (centered_tra @ R.T) * scale

    # Translate to v1
    transformed_tra = rotated_scaled_tra + v1

    # Diagnostics
    print("Start error:", np.linalg.norm(transformed_tra[0] - v1))
    print("End error:  ", np.linalg.norm(transformed_tra[-1] - v2))
    return transformed_tra


def interpolate_traj(traj, t):
    """
    Linearly interpolate position along a trajectory for t in [0, 1].

    Parameters:
    - traj: (N, 3) numpy array of positions
    - t: scalar in [0, 1]

    Returns:
    - interpolated 3D position
    """
    traj = np.asarray(traj)
    N = len(traj)
    t = np.clip(t, 0.0, 1.0)

    # Determine segment index
    idx_float = t * (N - 1)
    idx = int(np.floor(idx_float))
    alpha = idx_float - idx

    if idx >= N - 1:
        return traj[-1]  # if t == 1.0 exactly

    # Linear interpolation between traj[idx] and traj[idx + 1]
    return (1 - alpha) * traj[idx] + alpha * traj[idx + 1]
# --- Example usage ---

# Sample trajectory (spiral)
# t = np.linspace(0, 1, 100)
# tra = np.stack([
#     t,
#     t,
#     np.sin(2 * np.pi * t)
# ], axis=1)

start1 = [0.28667097385077456, -0.48847984909271797, 0.5940973561794143 + 0.022]
mid1 = [0.28667097385077456 - 0.25, -0.48847984909271797, 0.5940973561794143 + 0.022 + 0.15]
end1 = [0.28667097385077456 - 0.5, -0.48847984909271797, 0.5940973561794143 + 0.022]

tra_pre = np.array([np.linspace(start1[i], start1[i], 20) for i in range(3)]).T
tra_go = np.array([np.linspace(start1[i], mid1[i], 100) for i in range(3)]).T
tra_go2 = np.array([np.linspace(mid1[i], end1[i], 100) for i in range(3)]).T

tra= np.vstack((tra_pre, tra_go, tra_go2))

# Define v1 and v2
v1 = np.array([5, 0, 0])
v2 = np.array([1, 0, 2])

# Transform
transformed_tra = transform_trajectory(tra, v1, v2)
point = interpolate_traj(transformed_tra, t=0.7)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Original
ax.plot(tra[:, 0], tra[:, 1], tra[:, 2], label='Original', linewidth=2)

# Transformed
ax.plot(transformed_tra[:, 0], transformed_tra[:, 1], transformed_tra[:, 2], label='Transformed', linewidth=2)

# Start and End markers
ax.scatter(*v1, color='green', s=50, label='v1 (start)')
ax.scatter(*v2, color='red', s=50, label='v2 (end)')
ax.scatter(*point, color='black', s=50, label='point (end)')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
ax.set_title('Trajectory Alignment (Rotation + Scale)')

plt.show()
