import numpy as np
import tqdm
from concurrent.futures import ProcessPoolExecutor

import accelerator
from processing import idealTemporalBandpassFilter, pyrDown, pyrUp, rgb2yiq


def generateGaussianPyramid(image, kernel, level):
    xp = accelerator.get_xp()
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


def _worker_generate_gaussian(args):
    """Worker for parallel Gaussian pyramid generation (CPU mode)."""
    image_rgb, kernel, level = args
    return generateGaussianPyramid(
        image=rgb2yiq(image_rgb),
        kernel=kernel,
        level=level
    )


def getGaussianPyramids(images, kernel, level):
    xp = accelerator.get_xp()
    threads = accelerator.get_threads()
    backend = accelerator.get_backend()

    if backend == "cpu" and threads > 1:
        # CPU multiprocessing
        gaussian_pyramids = np.zeros_like(images, dtype=np.float32)
        args_list = [(images[i], kernel, level) for i in range(images.shape[0])]

        with ProcessPoolExecutor(max_workers=threads) as executor:
            results = list(tqdm.tqdm(
                executor.map(_worker_generate_gaussian, args_list),
                total=len(args_list),
                ascii=True,
                desc='Gaussian Pyramids Generation (CPU parallel)'
            ))

        for i, result in enumerate(results):
            gaussian_pyramids[i] = result
    else:
        # Serial (single-thread CPU or CUDA)
        gaussian_pyramids = xp.zeros_like(images, dtype=xp.float32)
        desc = 'Gaussian Pyramids Generation'
        if backend == "cuda":
            desc += " (CUDA)"

        for i in tqdm.tqdm(range(images.shape[0]),
                           ascii=True,
                           desc=desc):
            gaussian_pyramids[i] = generateGaussianPyramid(
                                        image=rgb2yiq(images[i]),
                                        kernel=kernel,
                                        level=level
                            )

    return gaussian_pyramids


def filterGaussianPyramids(pyramids,
                           fps,
                           freq_range,
                           alpha,
                           attenuation):
    xp = accelerator.get_xp()

    filtered_pyramids = idealTemporalBandpassFilter(
                            images=pyramids,
                            fps=fps,
                            freq_range=freq_range
                        ).astype(xp.float32)

    filtered_pyramids *= alpha
    filtered_pyramids[:, :, :, 1:] *= attenuation

    return filtered_pyramids
