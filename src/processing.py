import os

import cv2
import numpy as np
import tqdm
from concurrent.futures import ProcessPoolExecutor

import accelerator
from constants import rgb_from_yiq, yiq_from_rgb


def loadVideo(video_path):
    image_sequence = []
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)

    while video.isOpened():
        ret, frame = video.read()

        if ret is False:
            break

        image_sequence.append(frame[:, :, ::-1])

    video.release()

    return np.asarray(image_sequence), fps


def rgb2yiq(rgb_image):
    xp = accelerator.get_xp()
    image = rgb_image.astype(xp.float32)
    yiq_mat = accelerator.to_device(yiq_from_rgb)
    return image @ yiq_mat.T


def yiq2rgb(yiq_image):
    xp = accelerator.get_xp()
    image = yiq_image.astype(xp.float32)
    rgb_mat = accelerator.to_device(rgb_from_yiq)
    return image @ rgb_mat.T


def pyrDown(image, kernel):
    kernel_dev = accelerator.to_device(kernel)
    return accelerator.convolve2d(image, kernel_dev)[::2, ::2]


def pyrUp(image, kernel, dst_shape=None):
    xp = accelerator.get_xp()
    src_h, src_w = image.shape[0], image.shape[1]

    # Compute destination dimensions (same logic as before)
    dst_height = src_h * 2 - 1
    dst_width = src_w * 2 - 1

    if dst_shape is not None:
        dst_height = dst_shape[0]
        dst_width = dst_shape[1]

    # Zero-interleaved upsampling: place pixels at even indices
    if image.ndim == 3:
        upsampled_image = xp.zeros((dst_height, dst_width, image.shape[2]),
                                   dtype=image.dtype)
    else:
        upsampled_image = xp.zeros((dst_height, dst_width), dtype=image.dtype)

    upsampled_image[::2, ::2] = image[:((dst_height + 1) // 2),
                                       :((dst_width + 1) // 2)]

    kernel_dev = accelerator.to_device(kernel)
    return accelerator.convolve2d(upsampled_image, 4 * kernel_dev)


def idealTemporalBandpassFilter(images,
                                fps,
                                freq_range,
                                axis=0):
    xp = accelerator.get_xp()

    fft = xp.fft.fft(images, axis=axis)
    frequencies = xp.fft.fftfreq(images.shape[0], d=1.0/fps)

    low = int((xp.abs(frequencies - freq_range[0])).argmin())
    high = int((xp.abs(frequencies - freq_range[1])).argmin())

    fft[:low] = 0
    fft[high:] = 0

    return xp.fft.ifft(fft, axis=0).real


def reconstructGaussianImage(image, pyramid):
    reconstructed_image = rgb2yiq(image) + pyramid
    reconstructed_image = yiq2rgb(reconstructed_image)
    xp = accelerator.get_xp()
    reconstructed_image = xp.clip(reconstructed_image, 0, 255)

    return reconstructed_image.astype(xp.uint8)


def reconstructLaplacianImage(image, pyramid, kernel):
    xp = accelerator.get_xp()
    reconstructed_image = rgb2yiq(image)

    for level in range(1, pyramid.shape[0] - 1):
        tmp = pyramid[level]
        for curr_level in range(level):
            tmp = pyrUp(tmp, kernel, pyramid[level - curr_level - 1].shape[:2])
        reconstructed_image += tmp.astype(xp.float32)

    reconstructed_image = yiq2rgb(reconstructed_image)
    reconstructed_image = xp.clip(reconstructed_image, 0, 255)

    return reconstructed_image.astype(xp.uint8)


# ---------------------------------------------------------------------------
# Worker functions for CPU multiprocessing
# ---------------------------------------------------------------------------


def _worker_reconstruct_gaussian(args):
    """Worker for parallel Gaussian reconstruction (CPU mode)."""
    image, pyramid = args
    return reconstructGaussianImage(image, pyramid)


def _worker_reconstruct_laplacian(args):
    """Worker for parallel Laplacian reconstruction (CPU mode)."""
    image, pyramid, kernel = args
    return reconstructLaplacianImage(image, pyramid, kernel)


# ---------------------------------------------------------------------------
# Output video generation
# ---------------------------------------------------------------------------


def getGaussianOutputVideo(original_images, filtered_images):
    xp = accelerator.get_xp()
    threads = accelerator.get_threads()
    backend = accelerator.get_backend()

    if backend == "cpu" and threads > 1:
        # CPU multiprocessing
        video = np.zeros_like(original_images)
        args_list = [(original_images[i], filtered_images[i])
                     for i in range(filtered_images.shape[0])]

        with ProcessPoolExecutor(max_workers=threads) as executor:
            results = list(tqdm.tqdm(
                executor.map(_worker_reconstruct_gaussian, args_list),
                total=len(args_list),
                ascii=True,
                desc="Video Reconstruction (CPU parallel)"
            ))

        for i, frame in enumerate(results):
            video[i] = frame
    else:
        # Serial (single-thread CPU or CUDA)
        video = xp.zeros_like(original_images)
        desc = "Video Reconstruction"
        if backend == "cuda":
            desc += " (CUDA)"

        for i in tqdm.tqdm(range(filtered_images.shape[0]),
                           ascii=True,
                           desc=desc):
            video[i] = reconstructGaussianImage(
                image=original_images[i],
                pyramid=filtered_images[i]
            )

    return video


def getLaplacianOutputVideo(original_images, filtered_images, kernel):
    xp = accelerator.get_xp()
    threads = accelerator.get_threads()
    backend = accelerator.get_backend()

    if backend == "cpu" and threads > 1:
        video = np.zeros_like(original_images)
        args_list = [(original_images[i], filtered_images[i], kernel)
                     for i in range(original_images.shape[0])]

        with ProcessPoolExecutor(max_workers=threads) as executor:
            results = list(tqdm.tqdm(
                executor.map(_worker_reconstruct_laplacian, args_list),
                total=len(args_list),
                ascii=True,
                desc="Video Reconstruction (CPU parallel)"
            ))

        for i, frame in enumerate(results):
            video[i] = frame
    else:
        video = xp.zeros_like(original_images)
        desc = "Video Reconstruction"
        if backend == "cuda":
            desc += " (CUDA)"

        for i in tqdm.tqdm(range(original_images.shape[0]),
                           ascii=True,
                           desc=desc):
            video[i] = reconstructLaplacianImage(
                image=original_images[i],
                pyramid=filtered_images[i],
                kernel=kernel
            )

    return video


# ---------------------------------------------------------------------------
# Save video — direct MP4 output
# ---------------------------------------------------------------------------


def saveVideo(video, saving_path, fps):
    """Save the output video. Always outputs MP4 format.

    If saving_path ends with .avi, it will be changed to .mp4.
    """
    # Ensure we save as MP4
    base, ext = os.path.splitext(saving_path)
    mp4_path = base + ".mp4"

    video_host = accelerator.to_host(video) if accelerator.get_backend() == "cuda" else video

    (height, width) = video_host[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))

    for i in tqdm.tqdm(range(len(video_host)), ascii=True, desc="Saving Video"):
        writer.write(video_host[i][:, :, ::-1])

    writer.release()
    print(f"\nVideo saved to: {mp4_path}")

    return mp4_path

