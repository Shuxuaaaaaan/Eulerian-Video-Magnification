import cupy as cp
import tqdm
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


def getLaplacianPyramids(images, kernel, level):
    """Returns list of list of CuPy arrays (ragged structure, no np.asarray)."""
    laplacian_pyramids = []

    for image in tqdm.tqdm(images,
                           ascii=True,
                           desc="Laplacian Pyramids Generation (GPU)"):

        laplacian_pyramid = generateLaplacianPyramid(
                                    image=rgb2yiq(image),
                                    kernel=kernel,
                                    level=level
                        )
        laplacian_pyramids.append(laplacian_pyramid)

    return laplacian_pyramids


def _deep_copy_pyramid(pyramids):
    """Deep copy a list-of-lists of CuPy arrays."""
    return [[lvl.copy() for lvl in frame] for frame in pyramids]


def _subtract_pyramids(a, b):
    """Element-wise subtract two list-of-lists of CuPy arrays."""
    return [[a_lvl - b_lvl for a_lvl, b_lvl in zip(a_frame, b_frame)]
            for a_frame, b_frame in zip(a, b)]


def _scale_add_pyramids(coeff, a, b_coeff, b, c_coeff, c):
    """Compute (coeff * a + b_coeff * b + c_coeff * c) element-wise for pyramid lists."""
    result = []
    for a_frame, b_frame, c_frame in zip(a, b, c):
        frame = []
        for a_lvl, b_lvl, c_lvl in zip(a_frame, b_frame, c_frame):
            frame.append(coeff * a_lvl + b_coeff * b_lvl + c_coeff * c_lvl)
        result.append(frame)
    return result


def filterLaplacianPyramids(pyramids,
                            level,
                            fps,
                            freq_range,
                            alpha,
                            lambda_cutoff,
                            attenuation):

    n_frames = len(pyramids)
    # Initialize filtered_pyramids as zeros with same shapes
    filtered_pyramids = [[cp.zeros_like(lvl) for lvl in frame] for frame in pyramids]
    delta = lambda_cutoff / (8 * (1 + alpha))

    # Butter filter coefficients computed on CPU (just scalars)
    b_low, a_low = butter(1, freq_range[0], btype='low', output='ba', fs=fps)
    b_high, a_high = butter(1, freq_range[1], btype='low', output='ba', fs=fps)

    # lowpass/highpass are lists of CuPy arrays (one per pyramid level)
    lowpass = [lvl.copy() for lvl in pyramids[0]]
    highpass = [lvl.copy() for lvl in pyramids[0]]
    filtered_pyramids[0] = [lvl.copy() for lvl in pyramids[0]]

    for i in tqdm.tqdm(range(1, n_frames),
                       ascii=True,
                       desc="Laplacian Pyramids Filtering (GPU)"):

        # Update lowpass and highpass per level
        new_lowpass = []
        new_highpass = []
        for lvl_idx in range(len(pyramids[i])):
            lp = (-a_low[1] * lowpass[lvl_idx]
                  + b_low[0] * pyramids[i][lvl_idx]
                  + b_low[1] * pyramids[i - 1][lvl_idx]) / a_low[0]
            new_lowpass.append(lp)

            hp = (-a_high[1] * highpass[lvl_idx]
                  + b_high[0] * pyramids[i][lvl_idx]
                  + b_high[1] * pyramids[i - 1][lvl_idx]) / a_high[0]
            new_highpass.append(hp)

        lowpass = new_lowpass
        highpass = new_highpass

        # filtered = highpass - lowpass
        filtered_pyramids[i] = [hp - lp for hp, lp in zip(highpass, lowpass)]

        for lvl in range(1, level - 1):
            if lvl < len(filtered_pyramids[i]):
                (height, width, _) = filtered_pyramids[i][lvl].shape
                lambd = ((height ** 2) + (width ** 2)) ** 0.5
                new_alpha = (lambd / (8 * delta)) - 1

                filtered_pyramids[i][lvl] *= min(alpha, new_alpha)
                filtered_pyramids[i][lvl][:, :, 1:] *= attenuation

    return filtered_pyramids
