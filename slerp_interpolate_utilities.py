import numpy as np
from scipy.spatial.transform import Rotation as R

def quat_log(q):
    """Quaternion logarithm map: q -> R^3"""
    q = np.asarray(q)
    vec = q[:3]
    w = q[3]
    norm_vec = np.linalg.norm(vec)

    if norm_vec < 1e-8:
        return np.zeros(3)
    theta = np.arccos(np.clip(w, -1.0, 1.0))
    return theta * vec / norm_vec

def quat_exp(v):
    """Quaternion exponential map: R^3 -> unit quaternion"""
    v = np.asarray(v)
    theta = np.linalg.norm(v)
    if theta < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = v / theta
    return np.concatenate([np.sin(theta) * axis, [np.cos(theta)]])

def quat_conjugate(q):
    """Quaternion conjugate: q*"""
    return np.array([-q[0], -q[1], -q[2], q[3]])

def quat_multiply(q1, q2):
    """Hamilton product: q1 ⊗ q2"""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    return np.array([x, y, z, w])

def slerp_projection(q0, q1, q4):
    """Find t such that SLERP(q0, q1, t) is closest to q4"""
    q0 = np.asarray(q0)
    q1 = np.asarray(q1)
    q4 = np.asarray(q4)

    # Ensure all are unit quaternions
    q0 /= np.linalg.norm(q0)
    q1 /= np.linalg.norm(q1)
    q4 /= np.linalg.norm(q4)

    # Relative quaternions
    q01 = quat_multiply(quat_conjugate(q0), q1)
    q04 = quat_multiply(quat_conjugate(q0), q4)

    # Log maps to tangent space
    v = quat_log(q01)
    v4 = quat_log(q04)

    # Projection in tangent space
    if np.linalg.norm(v) < 1e-8:
        t = 0.0
    else:
        t = np.dot(v, v4) / np.dot(v, v)

    # Interpolated quaternion
    q3_relative = quat_exp(t * v)
    q3 = quat_multiply(q0, q3_relative)
    return t, q3


def slerp_geodesic(q0, q1, t):
    """
    SLERP interpolation (or extrapolation) between q0 and q1 for scalar or array t.

    Parameters:
    - q0, q1: unit quaternions [x, y, z, w]
    - t: scalar or np.ndarray of interpolation factors

    Returns:
    - q_interp: shape (4,) if t is scalar, or (N, 4) if t is array-like
    """
    q0 = np.asarray(q0)
    q1 = np.asarray(q1)
    t = np.asarray(t)

    # Ensure unit quaternions
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    # Flip to ensure shortest arc
    if np.dot(q0, q1) < 0:
        q1 = -q1

    q01 = quat_multiply(quat_conjugate(q0), q1)
    v = quat_log(q01)

    def single_interp(scalar_t):
        q_delta = quat_exp(scalar_t * v)
        return quat_multiply(q0, q_delta)

    if t.ndim == 0:
        return single_interp(t)
    else:
        return np.array([single_interp(ti) for ti in t])

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