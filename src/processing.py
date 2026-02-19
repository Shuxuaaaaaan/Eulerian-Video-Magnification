import cv2
import numpy as np
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    image = rgb_image.astype(np.float32)
    return image @ yiq_from_rgb.T


def yiq2rgb(yiq_image):
    image = yiq_image.astype(np.float32)
    return image @ rgb_from_yiq.T


def pyrDown(image, kernel):
    return cv2.filter2D(image, -1, kernel)[::2, ::2]


def pyrUp(image, kernel, dst_shape=None):
    dst_height = image.shape[0] + 1
    dst_width = image.shape[1] + 1

    if dst_shape is not None:
        dst_height -= (dst_shape[0] % image.shape[0] != 0)
        dst_width -= (dst_shape[1] % image.shape[1] != 0)

    height_indexes = np.arange(1, dst_height)
    width_indexes = np.arange(1, dst_width)

    upsampled_image = np.insert(image, height_indexes, 0, axis=0)
    upsampled_image = np.insert(upsampled_image, width_indexes, 0, axis=1)

    return cv2.filter2D(upsampled_image, -1, 4 * kernel)


def idealTemporalBandpassFilter(images,
                                fps,
                                freq_range,
                                axis=0):

    fft = np.fft.fft(images, axis=axis)
    frequencies = np.fft.fftfreq(images.shape[0], d=1.0/fps)

    low = (np.abs(frequencies - freq_range[0])).argmin()
    high = (np.abs(frequencies - freq_range[1])).argmin()

    fft[:low] = 0
    fft[high:] = 0

    return np.fft.ifft(fft, axis=0).real


def reconstructGaussianImage(image, pyramid):
    reconstructed_image = rgb2yiq(image) + pyramid
    reconstructed_image = yiq2rgb(reconstructed_image)
    reconstructed_image = np.clip(reconstructed_image, 0, 255)

    return reconstructed_image.astype(np.uint8)


def reconstructLaplacianImage(image, pyramid, kernel):
    reconstructed_image = rgb2yiq(image)

    for level in range(1, pyramid.shape[0] - 1):
        tmp = pyramid[level]
        for curr_level in range(level):
            tmp = pyrUp(tmp, kernel, pyramid[level - curr_level - 1].shape[:2])
        reconstructed_image += tmp.astype(np.float32)

    reconstructed_image = yiq2rgb(reconstructed_image)
    reconstructed_image = np.clip(reconstructed_image, 0, 255)

    return reconstructed_image.astype(np.uint8)


def _reconstruct_gaussian_worker(args):
    """Worker function for parallel Gaussian video reconstruction."""
    index, image, pyramid = args
    result = reconstructGaussianImage(image=image, pyramid=pyramid)
    return index, result


def _reconstruct_laplacian_worker(args):
    """Worker function for parallel Laplacian video reconstruction."""
    index, image, pyramid, kernel = args
    result = reconstructLaplacianImage(image=image, pyramid=pyramid, kernel=kernel)
    return index, result


def getGaussianOutputVideo(original_images, filtered_images, max_workers=None):
    video = np.zeros_like(original_images)
    num_frames = filtered_images.shape[0]

    tasks = [(i, original_images[i], filtered_images[i])
             for i in range(num_frames)]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_reconstruct_gaussian_worker, t): t[0]
                   for t in tasks}

        with tqdm.tqdm(total=num_frames, ascii=True,
                       desc="Video Reconstruction") as pbar:
            for future in as_completed(futures):
                idx, result = future.result()
                video[idx] = result
                pbar.update(1)

    return video


def getLaplacianOutputVideo(original_images, filtered_images, kernel,
                            max_workers=None):
    video = np.zeros_like(original_images)
    num_frames = original_images.shape[0]

    tasks = [(i, original_images[i], filtered_images[i], kernel)
             for i in range(num_frames)]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_reconstruct_laplacian_worker, t): t[0]
                   for t in tasks}

        with tqdm.tqdm(total=num_frames, ascii=True,
                       desc="Video Reconstruction") as pbar:
            for future in as_completed(futures):
                idx, result = future.result()
                video[idx] = result
                pbar.update(1)

    return video


def saveVideo(video, saving_path, fps):
    (height, width) = video[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(saving_path, fourcc, fps, (width, height))

    for i in tqdm.tqdm(range(len(video)), ascii=True, desc="Saving Video"):
        writer.write(video[i][:, :, ::-1])

    writer.release()
