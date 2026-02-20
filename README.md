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

## Deployment (Docker) / 容器部署

The project supports multi-architecture Docker deployment (AMD64 & ARM64), making it easy to run on devices like standard PCs, Macs, and **Raspberry Pi** without any environment setup.

本项目原生支持多架构 Docker 部署 (AMD64 PC / ARM64 树莓派等)，无需任何环境配置即可跨平台运行。

### 1. Prerequisites / 前置条件
- Install [Docker](https://docs.docker.com/get-docker/) & Docker Compose on your target device.
- 在目标设备上安装 Docker 与 Docker Compose 插件。

### 2. Quick Start / 快速启动

```bash
# Clone the project / 拉取项目
git clone <repo-url> && cd Eulerian-Video-Magnification

# Start the background daemon / 启动后台守护进程
docker compose up -d

# Check logs / 查看运行日志
docker logs -f evm-cpu-runner
```

Container will automatically build and start the **background daemon** (`watcher.py`), which:  
容器将自动构建并启动**后台守护进程**，它会：

1. Scan `data/resources/` for any unprocessed `.mp4` files on startup / 启动时扫描所有未处理的视频
2. Automatically detect new files dropped into `data/resources/` / 实时监听新丢入的视频文件
3. Process them and save results as `magnified_<filename>.mp4` in `data/results/` / 处理后以 `magnified_` 前缀保存到结果目录

### 3. Directory Structure / 目录结构

```
data/
├── resources/          ← Drop source videos here / 将源视频放入此处
│   ├── face1.mp4
│   └── your_video.mp4
└── results/            ← Processed outputs appear here / 处理结果自动出现在此处
    ├── magnified_face1.mp4
    └── magnified_your_video.mp4
```

### 4. Algorithm Parameters / 算法参数配置

The daemon's processing parameters are configured in **`src/watcher.py`** (line 36–41):  
后台守护进程的**算法参数**在 **`src/watcher.py`** 文件的第 36–41 行配置：

```python
cmd = [
    "/app/.venv/bin/python", EVM_SCRIPT,
    "-v", input_path,
    "-s", output_path,
    "--accel", "cpu",     # Acceleration mode / 加速模式: cpu or cuda
    "-t", "4"             # CPU threads / CPU 线程数
]
```

You can add any algorithm arguments from the table below to customize magnification behavior.  
可以在此处添加下方参数表中的任何算法参数来自定义放大效果，例如：

```python
cmd = [
    "/app/.venv/bin/python", EVM_SCRIPT,
    "-v", input_path,
    "-s", output_path,
    "-m", "laplacian",    # Pyramid mode / 金字塔模式
    "-l", "4",            # Pyramid levels / 金字塔层数
    "-a", "15",           # Amplification factor / 放大系数
    "-lc", "16",          # Lambda cutoff / λ 截止值
    "-at", "0.1",         # Attenuation / 衰减系数
    "-lo", "0.4",         # Low frequency / 最低频率
    "-ho", "3",           # High frequency / 最高频率
    "--accel", "cpu",
    "-t", "4"
]
```

> **Note / 提示**: After modifying `watcher.py`, rebuild the image with `docker compose build` and restart with `docker compose up -d`.  
> 修改 `watcher.py` 后，需执行 `docker compose build` 重新构建镜像，再 `docker compose up -d` 重启。

### 5. Container Management / 容器管理

```bash
# Start daemon in background / 后台启动守护进程
docker compose up -d

# View live processing logs / 查看实时处理日志
docker logs -f evm-cpu-runner

# Stop the daemon / 停止守护进程
docker compose down

# Rebuild after code changes / 修改代码后重新构建
docker compose build && docker compose up -d
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
