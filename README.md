# Eulerian Video Magnification

```
███████╗██╗   ██╗███╗   ███╗
██╔════╝██║   ██║████╗ ████║
█████╗  ██║   ██║██╔████╔██║
██╔══╝  ╚██╗ ██╔╝██║╚██╔╝██║
███████╗ ╚████╔╝ ██║ ╚═╝ ██║
╚══════╝  ╚═══╝  ╚═╝     ╚═╝
```

Eulerian Video Magnification for colors and motions magnification.  
Supports **CPU multithreading** and **CUDA GPU acceleration**.

基于欧拉视频放大算法的颜色与运动放大工具。  
支持 **CPU 多线程** 与 **CUDA GPU 加速**。

## Requirements / 环境要求

- Python >= 3.9
- [UV](https://docs.astral.sh/uv/) — Package manager / 包管理器
- CUDA Toolkit 12.x — Optional, for GPU acceleration / 可选，用于 GPU 加速

## Installation / 安装

```bash
# CPU only / 仅 CPU
uv sync

# With CUDA support / 包含 CUDA 支持
uv sync --extra cuda
```

## Usage / 使用方法

```bash
uv run python src/evm.py -v <video_path> -s <saving_path> [options]
```

### Arguments / 参数

| Argument / 参数 | Short / 缩写 | Description / 说明 | Default / 默认值 |
|----------|-------|-------------|---------|
| `--video_path` | `-v` | Input video path / 输入视频路径 | *required / 必填* |
| `--saving_path` | `-s` | Output path (auto .mp4) / 输出路径（自动转 .mp4） | *required / 必填* |
| `--mode` | `-m` | `gaussian` or `laplacian` / 金字塔类型 | `gaussian` |
| `--level` | `-l` | Pyramid levels / 金字塔层数 | `4` |
| `--alpha` | `-a` | Amplification factor / 放大系数 | `100` |
| `--lambda_cutoff` | `-lc` | λ cutoff (Laplacian) / λ 截止值 | `1000` |
| `--low_omega` | `-lo` | Min frequency / 最低频率 | `0.833` |
| `--high_omega` | `-ho` | Max frequency / 最高频率 | `1` |
| `--attenuation` | `-at` | I/Q channel attenuation / I/Q 通道衰减 | `1` |
| `--accel` | `-acc` | Acceleration: `cpu` or `cuda` / 加速方式 | `cpu` |
| `--threads` | `-t` | CPU workers (CPU mode only) / CPU 进程数 | `1` |

### Examples / 示例

```bash
# CPU single thread (default) / CPU 单线程（默认）
uv run python src/evm.py -v data/resources/face1.mp4 -s data/results/face1.mp4

# CPU 8 threads / CPU 8 线程
uv run python src/evm.py -v data/resources/face1.mp4 -s data/results/face1.mp4 --accel cpu --threads 8

# CUDA GPU acceleration / CUDA GPU 加速
uv run python src/evm.py -v data/resources/face1.mp4 -s data/results/face1.mp4 --accel cuda

# Laplacian mode (motion magnification) / 拉普拉斯模式（运动放大）
uv run python src/evm.py -v data/resources/baby.mp4 -s data/results/baby.mp4 -m laplacian -l 4 -a 15 -lc 16 -at 0.1 -lo 0.4 -ho 3 --accel cuda
```

## Sample Configurations / 样例配置

The sample files are located in the `/data` folder; `/resources` contains the original videos, and `/results` contains the processed videos. 
`timetest.txt` records and compares the differences in execution time for different hardware acceleration methods.


样列文件位于`/data`文件夹内，`/resources`为原始视频，`/results`为处理后视频。
`timetest.txt`记录并比较了不同硬件加速方式的运行耗时差异。

### Color Magnification (Gaussian) / 颜色放大（高斯）

| Video / 视频 | Level / 层数 | Alpha | Attenuation / 衰减 | Low ω | High ω |
|----------|-------|-------|-------------|-------|--------|
| face1    | 4     | 50    | 1           | 0.833 | 1      |
| face2    | 6     | 50    | 1           | 0.833 | 1      |

### Motion Magnification (Laplacian) / 运动放大（拉普拉斯）

| Video / 视频 | Level / 层数 | Alpha | λ Cutoff | Attenuation / 衰减 | Low ω | High ω |
|-------|-------|-------|----------|-------------|-------|--------|
| wrist | 6     | 30    | 16       | 0.1         | 0.4   | 3      |
| baby  | 4     | 15    | 16       | 0.1         | 0.4   | 3      |




## References / 参考文献

- [Eulerian Video Magnification for Revealing Subtle Changes in the World](https://people.csail.mit.edu/mrub/evm/)

## Contributors / 贡献者

- Hussem Ben Belgacem
