#!/usr/bin/env bash
# cloud-init runs this on first boot. Minimal setup for a Python dev box.
# Full output streamed to /var/log/bernoulli-cloud-init.log for debugging.
set -eux
exec > /var/log/bernoulli-cloud-init.log 2>&1

export DEBIAN_FRONTEND=noninteractive

# -- base packages ----------------------------------------------------------
apt-get update
apt-get install -y git tmux htop build-essential rsync jq curl ca-certificates pkg-config

# -- uv (for the ubuntu user) ----------------------------------------------
# Installs into ~/.local/bin which is already on PATH for Ubuntu 24.04 non-login shells.
sudo -u ubuntu bash -lc 'curl -LsSf https://astral.sh/uv/install.sh | sh'

# -- HF cache on instance-store NVMe ---------------------------------------
# The NVIDIA GPU Base AMI already mounts the g6.xlarge's 232GB NVMe at
# /opt/dlami/nvme via LVM (vg.01/lv_ephemeral). We just use it. If a future
# AMI stops pre-mounting it, fall back to the EBS root.
if [ -d /opt/dlami/nvme ] && mountpoint -q /opt/dlami/nvme; then
  HF_CACHE_DIR="/opt/dlami/nvme/hf-cache"
  # Convenience symlink so paths/doc examples stay stable regardless of AMI changes.
  ln -sfn /opt/dlami/nvme /opt/nvme
else
  HF_CACHE_DIR="/home/ubuntu/.cache/huggingface"
fi
mkdir -p "$HF_CACHE_DIR"
chown -R ubuntu:ubuntu "$HF_CACHE_DIR"

cat >> /home/ubuntu/.bashrc <<EOF

# bernoulli-dev defaults (set by cloud-init)
export HF_HOME="$HF_CACHE_DIR"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TRANSFORMERS_CACHE="\$HF_HOME"
EOF
chown ubuntu:ubuntu /home/ubuntu/.bashrc

# -- a working-directory for rsync target ----------------------------------
sudo -u ubuntu mkdir -p /home/ubuntu/bernoulli

# -- sanity: log GPU + driver versions so first SSH shows a known-good state
nvidia-smi > /var/log/bernoulli-nvidia-smi.log 2>&1 || echo "nvidia-smi failed" >> /var/log/bernoulli-nvidia-smi.log

echo "cloud-init complete at $(date -u)" >> /var/log/bernoulli-cloud-init.log
