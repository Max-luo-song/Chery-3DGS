import cv2
import numpy as np

cache_index = {}


def distort_image(
    image,
    intrinsics,
    kb_coeffs,
    camera_model="pinhole",  # 'pinhole' or 'fisheye'
    new_size=None,
    enable_cache=True,
):
    h, w = image.shape[:2]
    if new_size is None:
        new_size = (w, h)
    new_w, new_h = new_size

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    # cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    cx, cy = new_w / 2.0, new_h / 2.0

    if camera_model == "pinhole":
        if len(kb_coeffs) < 8:
            kb_coeffs = np.concatenate([kb_coeffs, np.zeros(8 - len(kb_coeffs))])
        kb_coeffs = kb_coeffs[:8]
    elif camera_model == "fisheye":
        if len(kb_coeffs) < 4:
            raise ValueError("Fisheye needs at least 4 coeffs (k1~k4)")
        kb_coeffs = kb_coeffs[:4]

    cache_key = (h, w, new_w, new_h, tuple(kb_coeffs), camera_model)

    if enable_cache and cache_key in cache_index:
        mapx, mapy = cache_index[cache_key]
    else:
        x_out, y_out = np.meshgrid(np.arange(new_w), np.arange(new_h))

        x_norm = (x_out - cx) / fx
        y_norm = (y_out - cy) / fy

        r_sq = x_norm**2 + y_norm**2
        r = np.sqrt(r_sq + 1e-10)

        if camera_model == "pinhole":
            k1, k2, p1, p2, k3, k4, k5, k6 = kb_coeffs

            # 径向畸变
            radial_factor = (
                1
                + k1 * r_sq
                + k2 * r_sq**2
                + k3 * r_sq**3
                + k4 * r_sq**4
                + k5 * r_sq**5
                + k6 * r_sq**6
            )
            x_rad = x_norm * radial_factor
            y_rad = y_norm * radial_factor
            
            # 切向畸变
            x_tan = 2 * p1 * x_rad* y_rad+ p2 * (r_sq + 2 * x_rad**2)
            y_tan = p1 * (r_sq + 2 * y_rad**2) + 2 * p2 * x_rad* y_rad
            
            x_src = x_rad + x_tan
            y_src = y_rad + y_tan

        elif camera_model == "fisheye":
            k1, k2, k3, k4 = kb_coeffs
            p1, p2 = 0.0, 0.0  # fisheye model typically does not use tangential distortion

            theta = r
            theta_poly = 1 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6 + k4 * theta**8
            theta_src = np.where(theta_poly > 1e-8, theta / theta_poly, theta)
            scale = np.where(theta > 1e-8, theta_src / theta, 1)
            x_src = x_norm * scale + 2 * p1 * x_norm * y_norm + p2 * (r_sq + 2 * x_norm**2)
            y_src = y_norm * scale + p1 * (r_sq + 2 * y_norm**2) + 2 * p2 * x_norm * y_norm

        # 转换回像素坐标
        mapx = x_src * fx + cx
        mapy = y_src * fy + cy

        if enable_cache:
            cache_index[cache_key] = (mapx, mapy)

    # 设置边界为黑色
    dst = cv2.remap(
        image,
        mapx.astype(np.float32),
        mapy.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return dst
