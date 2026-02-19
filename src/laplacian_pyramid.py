import numpy as np
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.signal import butter

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


def _generate_laplacian_pyramid_worker(args):
    """Worker function for parallel Laplacian pyramid generation."""
    index, image, kernel, level = args
    result = generateLaplacianPyramid(
                image=rgb2yiq(image),
                kernel=kernel,
                level=level
             )
    return index, result


def getLaplacianPyramids(images, kernel, level, max_workers=None):
    num_frames = len(images)
    laplacian_pyramids = [None] * num_frames

    tasks = [(i, images[i], kernel, level) for i in range(num_frames)]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_generate_laplacian_pyramid_worker, t): t[0]
                   for t in tasks}

        with tqdm.tqdm(total=num_frames, ascii=True,
                       desc="Laplacian Pyramids Generation") as pbar:
            for future in as_completed(futures):
                idx, result = future.result()
                laplacian_pyramids[idx] = result
                pbar.update(1)

    return np.asarray(laplacian_pyramids, dtype='object')


def filterLaplacianPyramids(pyramids,
                            level,
                            fps,
                            freq_range,
                            alpha,
                            lambda_cutoff,
                            attenuation):

    filtered_pyramids = np.zeros_like(pyramids)
    delta = lambda_cutoff / (8 * (1 + alpha))
    b_low, a_low = butter(1, freq_range[0], btype='low', output='ba', fs=fps)
    b_high, a_high = butter(1, freq_range[1], btype='low', output='ba', fs=fps)

    lowpass = pyramids[0]
    highpass = pyramids[0]
    filtered_pyramids[0] = pyramids[0]

    for i in tqdm.tqdm(range(1, pyramids.shape[0]),
                       ascii=True,
                       desc="Laplacian Pyramids Filtering"):

        lowpass = (-a_low[1] * lowpass
                   + b_low[0] * pyramids[i]
                   + b_low[1] * pyramids[i - 1]) / a_low[0]
        highpass = (-a_high[1] * highpass
                    + b_high[0] * pyramids[i]
                    + b_high[1] * pyramids[i - 1]) / a_high[0]

        filtered_pyramids[i] = highpass - lowpass

        for lvl in range(1, level - 1):
            (height, width, _) = filtered_pyramids[i, lvl].shape
            lambd = ((height ** 2) + (width ** 2)) ** 0.5
            new_alpha = (lambd / (8 * delta)) - 1

            filtered_pyramids[i, lvl] *= min(alpha, new_alpha)
            filtered_pyramids[i, lvl][:, :, 1:] *= attenuation

    return filtered_pyramids
