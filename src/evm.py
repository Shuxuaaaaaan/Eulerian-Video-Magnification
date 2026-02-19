import argparse
import os

import cupy as cp

from constants import gaussian_kernel
from gaussian_pyramid import filterGaussianPyramids, getGaussianPyramids
from laplacian_pyramid import filterLaplacianPyramids, getLaplacianPyramids
from processing import (getGaussianOutputVideo, getLaplacianOutputVideo,
                        loadVideo, saveVideo)


def print_cuda_info():
    """Print CUDA device information."""
    device = cp.cuda.Device(0)
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    name = props['name'].decode('utf-8') if isinstance(props['name'], bytes) else props['name']
    total_mem = props['totalGlobalMem'] / (1024 ** 3)
    print(f"[CUDA] Device: {name}")
    print(f"[CUDA] Total Memory: {total_mem:.1f} GB")
    print(f"[CUDA] Compute Capability: {props['major']}.{props['minor']}")
    print()


def gaussian_evm(images,
                 fps,
                 kernel,
                 level,
                 alpha,
                 freq_range,
                 attenuation):

    gaussian_pyramids = getGaussianPyramids(
                            images=images,
                            kernel=kernel,
                            level=level
                    )

    print("Gaussian Pyramids Filtering (GPU)...")
    filtered_pyramids = filterGaussianPyramids(
                            pyramids=gaussian_pyramids,
                            fps=fps,
                            freq_range=freq_range,
                            alpha=alpha,
                            attenuation=attenuation
                        )
    print("Finished!")

    output_video = getGaussianOutputVideo(
                        original_images=images,
                        filtered_images=filtered_pyramids
                )

    return output_video


def laplacian_evm(images,
                  fps,
                  kernel,
                  level,
                  alpha,
                  lambda_cutoff,
                  freq_range,
                  attenuation):

    laplacian_pyramids = getLaplacianPyramids(
                                images=images,
                                kernel=kernel,
                                level=level
                    )

    filtered_pyramids = filterLaplacianPyramids(
                            pyramids=laplacian_pyramids,
                            fps=fps,
                            freq_range=freq_range,
                            alpha=alpha,
                            attenuation=attenuation,
                            lambda_cutoff=lambda_cutoff,
                            level=level
                    )

    output_video = getLaplacianOutputVideo(
                            original_images=images,
                            filtered_images=filtered_pyramids,
                            kernel=kernel
                )

    return output_video


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Eulerian Video Magnification (CUDA Accelerated)"
    )

    parser.add_argument(
        "--video_path",
        "-v",
        type=str,
        help="Path to the video to be used",
        required=True
    )

    parser.add_argument(
        "--level",
        "-l",
        type=int,
        help="Number of level of the Gaussian/Laplacian Pyramid",
        required=False,
        default=4
    )

    parser.add_argument(
        "--alpha",
        "-a",
        type=int,
        help="Amplification factor",
        required=False,
        default=100
    )

    parser.add_argument(
        "--lambda_cutoff",
        "-lc",
        type=int,
        help="λ cutoff for Laplacian EVM",
        required=False,
        default=1000
    )

    parser.add_argument(
        "--low_omega",
        "-lo",
        type=float,
        help="Minimum allowed frequency",
        required=False,
        default=0.833
    )

    parser.add_argument(
        "--high_omega",
        "-ho",
        type=float,
        help="Maximum allowed frequency",
        required=False,
        default=1
    )

    parser.add_argument(
        "--saving_path",
        "-s",
        type=str,
        help="Saving path of the magnified video",
        required=True
    )

    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        help="Type of pyramids to use (gaussian or laplacian)",
        choices=['gaussian', 'laplacian'],
        required=False,
        default='gaussian'
    )

    parser.add_argument(
        "--attenuation",
        "-at",
        type=float,
        help="Attenuation factor for I and Q channel post filtering",
        required=False,
        default=1
    )

    args = parser.parse_args()

    # Print CUDA device info
    print_cuda_info()

    kwargs = {}
    kwargs['kernel'] = gaussian_kernel
    kwargs['level'] = args.level
    kwargs['alpha'] = args.alpha
    kwargs['freq_range'] = [args.low_omega, args.high_omega]
    kwargs['attenuation'] = args.attenuation
    mode = args.mode
    video_path = args.video_path

    assert os.path.exists(video_path), f"Video {video_path} not found :("

    print("Loading video...")
    images, fps = loadVideo(video_path=video_path)
    print(f"Video loaded: {images.shape[0]} frames, {images.shape[1]}x{images.shape[2]}, FPS={fps}")
    print(f"GPU memory allocated: {cp.get_default_memory_pool().used_bytes() / 1024**2:.1f} MB")
    print()

    kwargs['images'] = images
    kwargs['fps'] = fps

    if mode == 'gaussian':
        output_video = gaussian_evm(**kwargs)
    else:
        kwargs['lambda_cutoff'] = args.lambda_cutoff
        output_video = laplacian_evm(**kwargs)

    saveVideo(video=output_video, saving_path=args.saving_path, fps=fps)

    # Free GPU memory
    cp.get_default_memory_pool().free_all_blocks()
    print("Done! GPU memory released.")
