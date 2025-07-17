import numpy as np
from scipy.spatial.transform import Rotation as R
from slerp_interpolate_utilities import slerp_projection,slerp_geodesic

# Define unit quaternions (x, y, z, w)
q0 = R.from_euler('xyz', [0, 0, 0]).as_quat()
q1 = R.from_euler('xyz', [0, np.pi/2, np.pi/2]).as_quat()
q4 = R.from_euler('xyz', [0, np.pi/4, np.pi/4]).as_quat()

t_star, q3 = slerp_projection(q0, q1, q4)
print(f"Best t: {t_star:.4f}")
print(f"Interpolated quaternion q3: {q3}")
print("q4:",q4,type(q4))

# q3_new=slerp_geodesic(q0,q1,[0.2,0.4])
# print(f"Interpolated quaternion q3: {q3_new}")

# Check how close q3 is to q4
dot = np.abs(np.dot(q3, q4))
angle_error = 2 * np.arccos(np.clip(dot, -1.0, 1.0))
print(f"Angular error (rad): {angle_error:.6f}")
print(f"Angular error (degree): {angle_error/np.pi*180:.6f}")