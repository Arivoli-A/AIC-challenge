curl -LsSf https://astral.sh/uv/install.sh | sh

export PATH="$HOME/.local/bin:$PATH"


# sudo apt-get update
# sudo apt-get install -y \
#     ffmpeg \
#     libavcodec-dev \
#     libavformat-dev \
#     libavdevice-dev \
#     libavfilter-dev \
#     libavutil-dev \
#     libswscale-dev \
#     libswresample-dev \
#     pkg-config \
#     build-essential \
#     python3-dev
    

# uv pip install cython

GIT_LFS_SKIP_SMUDGE=1 uv sync --extra tpu
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e . 