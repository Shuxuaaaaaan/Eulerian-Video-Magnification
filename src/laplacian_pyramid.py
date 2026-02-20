import numpy as np
import tqdm
from concurrent.futures import ProcessPoolExecutor
from scipy.signal import butter

import accelerator
from processing import pyrDown, pyrUp, rgb2yiq


def generateLaplacianPyramid(image, kernel, level):
    laplacian_pyramid = []
    prev_image = image.copy()

    for _ in range(level):
        downsampled_image = pyrDown(image=prev_image, kernel=kernel)
        upsampled_image = pyrUp(image=downsampled_image,
                                kernel=kernel,
                                dst_shape=prev_image.shape[:2])
        laplacian_pyramid.append(prev_image - upsampled_image)
        prev_image = downsampled_image

    return laplacian_pyramid


def _worker_generate_laplacian(args):
    """Worker for parallel Laplacian pyramid generation (CPU mode)."""
    image_rgb, kernel, level = args
    return generateLaplacianPyramid(
        image=rgb2yiq(image_rgb),
        kernel=kernel,
        level=level
    )


def _build_pyramid_object_array(laplacian_pyramids):
    """Build a 2D numpy object array of shape (num_frames, num_levels).

    Each element result[i, j] is an individual array (numpy or cupy),
    NOT a Python list. This enables proper indexing like pyramids[i, lvl]
    and avoids implicit CuPy→NumPy conversion errors.
    """
    n = len(laplacian_pyramids)
    num_levels = len(laplacian_pyramids[0])
    result = np.empty((n, num_levels), dtype='object')
    for i in range(n):
        for j in range(num_levels):
            result[i, j] = laplacian_pyramids[i][j]
    return result


def getLaplacianPyramids(images, kernel, level):
    threads = accelerator.get_threads()
    backend = accelerator.get_backend()

    if backend == "cpu" and threads > 1:
        # CPU multiprocessing
        args_list = [(images[i], kernel, level) for i in range(len(images))]

        with ProcessPoolExecutor(max_workers=threads) as executor:
            laplacian_pyramids = list(tqdm.tqdm(
                executor.map(_worker_generate_laplacian, args_list),
                total=len(args_list),
                ascii=True,
                desc="Laplacian Pyramids Generation (CPU parallel)"
            ))
    else:
        laplacian_pyramids = []
        desc = "Laplacian Pyramids Generation"
        if backend == "cuda":
            desc += " (CUDA)"

        for image in tqdm.tqdm(images,
                               ascii=True,
                               desc=desc):
            laplacian_pyramid = generateLaplacianPyramid(
                                        image=rgb2yiq(image),
                                        kernel=kernel,
                                        level=level
                            )
            laplacian_pyramids.append(laplacian_pyramid)

    return _build_pyramid_object_array(laplacian_pyramids)


def filterLaplacianPyramids(pyramids,
                            level,
                            fps,
                            freq_range,
                            alpha,
                            lambda_cutoff,
                            attenuation):
    """Filter Laplacian pyramids with IIR temporal bandpass.

    pyramids: 2D numpy object array, shape (num_frames, num_levels)
              each element is a (H, W, 3) array (numpy or cupy).
    """
    xp = accelerator.get_xp()
    num_frames = pyramids.shape[0]
    num_levels = pyramids.shape[1]

    # Allocate output container — 2D numpy object array (same shape)
    filtered_pyramids = np.empty_like(pyramids)

    delta = lambda_cutoff / (8 * (1 + alpha))

    # Butter coefficients computed on CPU (just scalars)
    b_low, a_low = butter(1, freq_range[0], btype='low', output='ba', fs=fps)
    b_high, a_high = butter(1, freq_range[1], btype='low', output='ba', fs=fps)

    # Initialize per-level lowpass/highpass state
    # These are lists of arrays, one per pyramid level
    lowpass = [pyramids[0, lvl].copy() for lvl in range(num_levels)]
    highpass = [pyramids[0, lvl].copy() for lvl in range(num_levels)]

    # Copy first frame
    for lvl in range(num_levels):
        filtered_pyramids[0, lvl] = pyramids[0, lvl].copy()

    desc = "Laplacian Pyramids Filtering"
    if accelerator.get_backend() == "cuda":
        desc += " (CUDA)"

    for i in tqdm.tqdm(range(1, num_frames),
                       ascii=True,
                       desc=desc):

        for lvl in range(num_levels):
            # IIR temporal filter — per level, per frame
            lowpass[lvl] = (
                -a_low[1] * lowpass[lvl]
                + b_low[0] * pyramids[i, lvl]
                + b_low[1] * pyramids[i - 1, lvl]
            ) / a_low[0]

            highpass[lvl] = (
                -a_high[1] * highpass[lvl]
                + b_high[0] * pyramids[i, lvl]
                + b_high[1] * pyramids[i - 1, lvl]
            ) / a_high[0]

            filtered_pyramids[i, lvl] = highpass[lvl] - lowpass[lvl]

        # Apply alpha / attenuation to levels 1..level-2
        for lvl in range(1, level - 1):
            (height, width, _) = filtered_pyramids[i, lvl].shape
            lambd = ((height ** 2) + (width ** 2)) ** 0.5
            new_alpha = (lambd / (8 * delta)) - 1

            filtered_pyramids[i, lvl] = filtered_pyramids[i, lvl] * min(alpha, new_alpha)
            filtered_pyramids[i, lvl][:, :, 1:] *= attenuation

    return filtered_pyramids
