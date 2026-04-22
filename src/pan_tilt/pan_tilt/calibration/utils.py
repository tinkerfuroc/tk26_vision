import numpy as np
from scipy.spatial.transform import Rotation

def pose_to_matrix(translation, quaternion):
    """Convert translation + quaternion to 4x4 matrix."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    T[:3, 3] = translation
    return T

def matrix_to_pose(T):
    """Convert 4x4 matrix to (translation, quaternion)."""
    trans = T[:3, 3].tolist()
    quat = Rotation.from_matrix(T[:3, :3]).as_quat().tolist()
    return trans, quat
