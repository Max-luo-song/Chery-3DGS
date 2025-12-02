import numpy as np

def project_numpy(xyz, K, RT, H, W):
    xyz_cam = np.dot(xyz, RT[:3, :3].T) + RT[:3, 3:].T
    valid_depth = xyz_cam[:, 2] > 0
    xyz_pixel = np.dot(xyz_cam, K.T)
    xyz_pixel = xyz_pixel[:, :2] / xyz_pixel[:, 2:]
    valid_x = np.logical_and(xyz_pixel[:, 0] >= 0, xyz_pixel[:, 0] < W)
    valid_y = np.logical_and(xyz_pixel[:, 1] >= 0, xyz_pixel[:, 1] < H)
    valid_pixel = np.logical_and(valid_x, valid_y)
    mask = np.logical_and(valid_depth, valid_pixel)
    return xyz_pixel, mask

