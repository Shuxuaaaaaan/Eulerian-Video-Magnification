import numpy as np
import cupy as cp


gaussian_kernel_np = (
    np.array(
        [
            [1,  4,  6,  4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1,  4,  6,  4, 1]
        ]
    )
    / 256
)

gaussian_kernel = cp.asarray(gaussian_kernel_np, dtype=cp.float64)


yiq_from_rgb_np = (
    np.array(
            [
                [0.29900000,  0.58700000,  0.11400000],
                [0.59590059, -0.27455667, -0.32134392],
                [0.21153661, -0.52273617,  0.31119955]
            ]
        )
    ).astype(np.float32)

rgb_from_yiq_np = np.linalg.inv(yiq_from_rgb_np)

yiq_from_rgb = cp.asarray(yiq_from_rgb_np)
rgb_from_yiq = cp.asarray(rgb_from_yiq_np)
