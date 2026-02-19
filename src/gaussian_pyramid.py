import numpy as np
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from processing import idealTemporalBandpassFilter, pyrDown, pyrUp, rgb2yiq


def generateGaussianPyramid(image, kernel, level):
    image_shape = [image.shape[:2]]
    downsampled_image = image.copy()

    for _ in range(level):
        downsampled_image = pyrDown(image=downsampled_image, kernel=kernel)
        image_shape.append(downsampled_image.shape[:2])

    gaussian_pyramid = downsampled_image
    for curr_level in range(level):
        gaussian_pyramid = pyrUp(
                            image=gaussian_pyramid,
                            kernel=kernel,
                            dst_shape=image_shape[level - curr_level - 1]
                        )

    return gaussian_pyramid


def _generate_gaussian_pyramid_worker(args):
    """Worker function for parallel Gaussian pyramid generation."""
    index, image, kernel, level = args
    result = generateGaussianPyramid(
                image=rgb2yiq(image),
                kernel=kernel,
                level=level
             )
    return index, result


def getGaussianPyramids(images, kernel, level, max_workers=None):
    gaussian_pyramids = np.zeros_like(images, dtype=np.float32)
    num_frames = images.shape[0]

    tasks = [(i, images[i], kernel, level) for i in range(num_frames)]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_generate_gaussian_pyramid_worker, t): t[0]
                   for t in tasks}

        with tqdm.tqdm(total=num_frames, ascii=True,
                       desc='Gaussian Pyramids Generation') as pbar:
            for future in as_completed(futures):
                idx, result = future.result()
                gaussian_pyramids[idx] = result
                pbar.update(1)

    return gaussian_pyramids


def filterGaussianPyramids(pyramids,
                           fps,
                           freq_range,
                           alpha,
                           attenuation):

    filtered_pyramids = idealTemporalBandpassFilter(
                            images=pyramids,
                            fps=fps,
                            freq_range=freq_range
                        ).astype(np.float32)

    filtered_pyramids *= alpha
    filtered_pyramids[:, :, :, 1:] *= attenuation

    return filtered_pyramids
