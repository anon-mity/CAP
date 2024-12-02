import numpy as np
import torch
import math
from scipy.spatial.transform import Rotation as sciR



def uniform_resample_index_np(pc, n_sample, batch=False):
    if batch == True:
        raise NotImplementedError('resample in batch is not implemented')
    n_point = pc.shape[0]
    if n_point >= n_sample:
        # downsample
        idx = np.random.choice(n_point, n_sample, replace=False)
    else:
        # upsample
        idx = np.random.choice(n_point, n_sample-n_point, replace=True)
        idx = np.concatenate((np.arange(n_point), idx), axis=0)
    return idx

def uniform_resample_np(pc, n_sample, label=None, batch=False):
    if batch == True:
        raise NotImplementedError('resample in batch is not implemented')
    idx = uniform_resample_index_np(pc, n_sample, batch)
    if label is None:
        return idx, pc[idx]
    else:
        return idx, pc[idx], label[idx]




def R_from_euler_np(angles):
    '''
    angles: [(b, )3]
    '''
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(angles[0]), -math.sin(angles[0]) ],
                    [0,         math.sin(angles[0]), math.cos(angles[0])  ]
                    ])
    R_y = np.array([[math.cos(angles[1]),    0,      math.sin(angles[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(angles[1]),   0,      math.cos(angles[1])  ]
                    ])

    R_z = np.array([[math.cos(angles[2]),    -math.sin(angles[2]),    0],
                    [math.sin(angles[2]),    math.cos(angles[2]),     0],
                    [0,                     0,                      1]
                    ])
    return np.dot(R_z, np.dot( R_y, R_x ))

def rotate_point_cloud(data, R = None, max_degree = None):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original point clouds
        R:
          3x3 array, optional Rotation matrix used to rotate the input
        max_degree:
          float, optional maximum DEGREE to randomly generate rotation
        Return:
          Nx3 array, rotated point clouds
    """
    # rotated_data = np.zeros(data.shape, dtype=np.float32)
    if R is not None:
      rotation_angle = R
    elif max_degree is not None:
      rotation_angle = np.random.randint(0, max_degree, 3) * np.pi / 180.0
    else:
      rotation_angle = sciR.random().as_matrix() if R is None else R

    if isinstance(rotation_angle, list) or  rotation_angle.ndim == 1:
      rotation_matrix = R_from_euler_np(rotation_angle)
    else:
      assert rotation_angle.shape[0] >= 3 and rotation_angle.shape[1] >= 3
      rotation_matrix = rotation_angle[:3, :3]

    if data is None:
      return None, rotation_matrix
    rotated_data = np.dot(data, rotation_matrix)

    return rotated_data, rotation_matrix   # return [N, 3],


def random_rotation(pc):
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.rand() * np.pi
    A = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * A + (1 - np.cos(angle)) * np.dot(A, A)
    original_axes = np.eye(3)
    random_R = np.dot(original_axes, R)
    trans_pc = np.dot(pc, R)
    return trans_pc, random_R