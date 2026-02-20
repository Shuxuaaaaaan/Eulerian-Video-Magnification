# 使用官方轻量级 Python 3.10 镜像，原生支持 amd64 (PC) 和 arm64 (树莓派/Mac M系列)
FROM python:3.10-slim-bookworm

# 设置环境变量，提升 Python 运行效率
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 更新系统并清理缓存
# (使用 opencv-python-headless 之后，已经无需安装笨重的各类 libgl 视窗依赖库)
RUN apt-get update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 安装高效包管理器 uv
RUN pip install --no-cache-dir uv

# 复制项目依赖声明文件
COPY pyproject.toml uv.lock requirements.txt ./

# 运行 uv 安装环境，完全摒弃 CUDA 相关的包，直接打包最纯净的 CPU 运行环境
# 同步依赖时会自动在 /app 目录下创建 .venv 虚拟环境
RUN uv sync --no-install-project

# 配置环境变量，让系统默认使用虚拟环境中的 Python 和相关指令
ENV PATH="/app/.venv/bin:$PATH"

# 复制源代码
COPY src/ ./src/

# 设置入口命令为我们新编写的全自动监听程序的绝对路径
ENTRYPOINT ["python", "/app/src/watcher.py"]

