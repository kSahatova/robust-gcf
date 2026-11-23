#!/usr/bin/env bash
# Prepare a RunPod pod for training. Run it once per new pod (pip packages live on the
# container disk and are lost with the pod; code, data and outputs live on the network volume).
#
#   bash /workspace/robust-gcf/scripts/runpod_setup.sh            # first time: clone it first, see below
#   LOCAL_CACHE=1 bash .../runpod_setup.sh                       # also copy the cache to local NVMe
#
# First pod only:
#   cd /workspace && git clone https://github.com/kSahatova/robust-gcf.git
#
# Layout on the network volume (mounted at /workspace):
#   /workspace/robust-gcf                 repository (outputs/ inside it, so results persist)
#   /workspace/data/raw/celeba/...        img_align_celeba/ + list_*.csv
#   /workspace/data/cache/...             built on first training run
set -euo pipefail

REPO=/workspace/robust-gcf
VOLUME_DATA=/workspace/data
LOCAL_DATA=/root/data

cd "$REPO"
git pull --ff-only || echo "[setup] git pull skipped (local changes?)"

# Keep the template's CUDA build of torch/torchvision: install them only if missing.
python - <<'EOF' || pip install "torch>=1.11" "torchvision>=0.12"
import torch, torchvision
print(f"[setup] torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
EOF
pip install -q -e ".[data,train,viz,medmnist,test]"

# Unpack an uploaded CelebA archive, if present and not yet extracted.
if [ -f "$VOLUME_DATA/celeba_raw.tar" ] && [ ! -d "$VOLUME_DATA/raw/celeba/img_align_celeba" ]; then
    echo "[setup] extracting celeba_raw.tar ..."
    mkdir -p "$VOLUME_DATA/raw"
    tar -xf "$VOLUME_DATA/celeba_raw.tar" -C "$VOLUME_DATA/raw"
fi

# Data root: the network volume, or (LOCAL_CACHE=1) a local copy of the built caches, which is
# faster for the memory-mapped random reads. A complete cache never touches the raw files.
DATA_ROOT=$VOLUME_DATA
if [ "${LOCAL_CACHE:-0}" = "1" ] && [ -d "$VOLUME_DATA/cache" ]; then
    echo "[setup] copying caches to $LOCAL_DATA ..."
    mkdir -p "$LOCAL_DATA"
    cp -r "$VOLUME_DATA/cache" "$LOCAL_DATA/"
    DATA_ROOT=$LOCAL_DATA
fi

grep -q ROBUST_GCF_DATA ~/.bashrc 2>/dev/null && sed -i '/ROBUST_GCF_DATA/d' ~/.bashrc
echo "export ROBUST_GCF_DATA=$DATA_ROOT" >> ~/.bashrc
echo "[setup] ROBUST_GCF_DATA=$DATA_ROOT (open a new shell or: source ~/.bashrc)"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /workspace | tail -1
echo "[setup] shared memory: $(df -h /dev/shm | tail -1 | awk '{print $2}')"
