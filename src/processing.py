import cv2
import cupy as cp
import numpy as np
import tqdm
from cupyx.scipy.ndimage import convolve

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

    # Load as NumPy then transfer to GPU in one batch
    images_np = np.asarray(image_sequence)
    return cp.asarray(images_np), fps


def rgb2yiq(rgb_image):
    image = rgb_image.astype(cp.float32)
    return image @ yiq_from_rgb.T


def yiq2rgb(yiq_image):
    image = yiq_image.astype(cp.float32)
    return image @ rgb_from_yiq.T


def pyrDown(image, kernel):
    # Apply 2D convolution per channel on GPU, then downsample
    if image.ndim == 3:
        filtered = cp.empty_like(image)
        for c in range(image.shape[2]):
            filtered[:, :, c] = convolve(image[:, :, c], kernel)
    else:
        filtered = convolve(image, kernel)
    return filtered[::2, ::2]


def pyrUp(image, kernel, dst_shape=None):
    dst_height = image.shape[0] + 1
    dst_width = image.shape[1] + 1

    if dst_shape is not None:
        dst_height -= (dst_shape[0] % image.shape[0] != 0)
        dst_width -= (dst_shape[1] % image.shape[1] != 0)

    # Zero-insertion upsampling on GPU
    h, w = image.shape[:2]
    if image.ndim == 3:
        out_h = h + dst_height - 1
        out_w = w + dst_width - 1
        upsampled_image = cp.zeros((out_h, out_w, image.shape[2]), dtype=image.dtype)
        upsampled_image[::2, ::2, :] = image
    else:
        out_h = h + dst_height - 1
        out_w = w + dst_width - 1
        upsampled_image = cp.zeros((out_h, out_w), dtype=image.dtype)
        upsampled_image[::2, ::2] = image

    # Apply filter with 4x gain
    kernel4 = 4 * kernel
    if upsampled_image.ndim == 3:
        filtered = cp.empty_like(upsampled_image)
        for c in range(upsampled_image.shape[2]):
            filtered[:, :, c] = convolve(upsampled_image[:, :, c], kernel4)
    else:
        filtered = convolve(upsampled_image, kernel4)

    return filtered


def idealTemporalBandpassFilter(images, fps, freq_range, axis=0):
    fft = cp.fft.fft(images, axis=axis)
    frequencies = cp.fft.fftfreq(images.shape[0], d=1.0 / fps)

    low = int(cp.abs(frequencies - freq_range[0]).argmin())
    high = int(cp.abs(frequencies - freq_range[1]).argmin())

    fft[:low] = 0
    fft[high:] = 0

    return cp.fft.ifft(fft, axis=0).real


def reconstructGaussianImage(image, pyramid):
    reconstructed_image = rgb2yiq(image) + pyramid
    reconstructed_image = yiq2rgb(reconstructed_image)
    reconstructed_image = cp.clip(reconstructed_image, 0, 255)

    return reconstructed_image.astype(cp.uint8)


def reconstructLaplacianImage(image, pyramid, kernel):
    """pyramid is a list of CuPy arrays (one per level)."""
    reconstructed_image = rgb2yiq(image)
    n_levels = len(pyramid)

    for level in range(1, n_levels - 1):
        tmp = pyramid[level]
        for curr_level in range(level):
            tmp = pyrUp(tmp, kernel, pyramid[level - curr_level - 1].shape[:2])
        reconstructed_image += tmp.astype(cp.float32)

    reconstructed_image = yiq2rgb(reconstructed_image)
    reconstructed_image = cp.clip(reconstructed_image, 0, 255)

    return reconstructed_image.astype(cp.uint8)


def getGaussianOutputVideo(original_images, filtered_images):
    video = cp.zeros_like(original_images)

    for i in tqdm.tqdm(range(filtered_images.shape[0]),
                       ascii=True,
                       desc="Video Reconstruction (GPU)"):

        video[i] = reconstructGaussianImage(
                    image=original_images[i],
                    pyramid=filtered_images[i]
                )

    return cp.asnumpy(video)


def getLaplacianOutputVideo(original_images, filtered_images, kernel):
    """filtered_images is a list of lists of CuPy arrays."""
    n_frames = len(filtered_images)
    # Pre-allocate on CPU
    first_frame = reconstructLaplacianImage(
        image=original_images[0],
        pyramid=filtered_images[0],
        kernel=kernel
    )
    video_np = np.zeros((n_frames,) + cp.asnumpy(first_frame).shape, dtype=np.uint8)
    video_np[0] = cp.asnumpy(first_frame)

    for i in tqdm.tqdm(range(1, n_frames),
                       ascii=True,
                       desc="Video Reconstruction (GPU)"):

        frame = reconstructLaplacianImage(
                    image=original_images[i],
                    pyramid=filtered_images[i],
                    kernel=kernel
                )
        video_np[i] = cp.asnumpy(frame)

    return video_np


def saveVideo(video, saving_path, fps):
    (height, width) = video[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(saving_path, fourcc, fps, (width, height))

    for i in tqdm.tqdm(range(len(video)), ascii=True, desc="Saving Video"):
        frame = video[i]
        if isinstance(frame, cp.ndarray):
            frame = cp.asnumpy(frame)
        writer.write(frame[:, :, ::-1])

    writer.release()
