# Bernoulli — Deploy a public URL on AWS GPU

Goal: starting from "I have SSH into an EC2 GPU instance", end with "my team can POST to `https://bernoulli.alice.example.com/v1/decide` over HTTPS with an API key".

This is the self-serve path. It does **not** depend on the private `bernoulli-ops` repo. If you have the OSS repo, SSH into a GPU box, and a domain name, you can finish this walkthrough.

---

## 1. Pick an instance

### Recommended: `g5.xlarge`

| | |
|---|---|
| GPU | 1 × NVIDIA A10G, 24 GB VRAM |
| vCPU / RAM | 4 / 16 GB |
| Price (us-east-1 on-demand) | ~$1.00 / hr |
| Fits the default Qwen2.5-VL-7B backbone | yes, with headroom |
| Fits the 32B production backbone | no — needs L40S 48 GB (`g6e.xlarge`) |

This is the same shape the project's dev-box lineage used; benchmarks and the Docker image are both tested here.

**Alternatives:**

| Instance | GPU | Hourly | When to pick it |
|---|---|---|---|
| `g6.xlarge` | L4 24 GB | ~$0.80 | g5 capacity scarce; same VRAM, newer card |
| `g6e.xlarge` | L40S 48 GB | ~$1.86 | You want to run `Qwen2.5-VL-32B-Instruct-AWQ` (~20 GB int4) with headroom for KV cache |
| `g5.2xlarge` / `g6.2xlarge` | A10G / L4 24 GB | ~$1.20 / ~$1.00 | More host RAM (32 GB) — useful if you intend to run eval harnesses alongside serving |

**AMI:** the "Deep Learning OSS Nvidia Driver AMI, Ubuntu 22.04" ships with NVIDIA driver + Docker + `nvidia-container-toolkit` pre-installed. That saves you the driver setup. Any Ubuntu 22.04 AMI works too — see §3 for the manual driver bits.

**Storage:** 150 GB gp3 root volume is enough for the OSS core (OS ~15 GB, Docker images ~20 GB, 7B weights ~15 GB, HF cache + room to grow). The 80 GB default is tight.

**Elastic IP:** attach one. You need a stable IP for the DNS record in §5.

---

## 2. SSH in and sanity-check the GPU

```bash
ssh -i ~/.ssh/alice-key.pem ubuntu@<elastic-ip>
nvidia-smi                                  # should show the GPU + driver version
docker info | grep -i runtime               # should mention 'nvidia' as a runtime
```

If `docker info` does not list the nvidia runtime, install it:

```bash
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

The last line should print the GPU table from inside a container. If it does, Docker + CUDA plumbing is good.

---

## 3. Install the repo and build the image

```bash
# Clone the repo
git clone https://github.com/shyamsfo/bernoulli.git
cd bernoulli

# Build the vLLM-based serving image (~5 min first time)
docker build -t bernoulli:latest .
```

That's the whole install. The image bakes torch + vLLM + transformers; your host does not need Python set up at all for the serving path.

**Pre-download the model weights to the host cache** (so the container doesn't do it on first request):

```bash
sudo apt-get install -y python3-pip
pip install --user "huggingface_hub[cli,hf_transfer]"
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download Qwen/Qwen2.5-VL-7B-Instruct \
  --revision cc594898137f460bfe9f0759e9844b3ce807cfb5
```

Weights land in `~/.cache/huggingface/`.

---

## 4. First boot — local port only

Generate an API key and start the server:

```bash
# A real random token, not "secret123"
API_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
echo "API_KEY=$API_KEY" | tee ~/.bernoulli.env         # save so you don't lose it

docker run -d --name bernoulli --restart unless-stopped \
  --gpus all -p 127.0.0.1:8000:8000 \
  -v $HOME/.cache/huggingface:/data/hf-cache \
  -e BERNOULLI_SCORER=vllm \
  -e BERNOULLI_MAX_MODEL_LEN=8192 \
  -e BERNOULLI_API_KEYS="$API_KEY" \
  bernoulli:latest
```

Note the bind to `127.0.0.1:8000` — the port is not exposed to the public internet yet. Caddy will handle that in §5.

**Watch the engine load** (first boot = ~90 s model load + ~30 s Triton JIT compile):

```bash
docker logs -f bernoulli
# ...wait for 'Uvicorn running on http://0.0.0.0:8000'...
```

**Verify locally** (still on the box):

```bash
curl http://127.0.0.1:8000/healthz
# → {"status":"ok"}        (/healthz is intentionally unauthenticated)

curl -sS -X POST http://127.0.0.1:8000/v1/decide \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"state":{"text":"GET RICH QUICK!!!"},
       "questions":[{"id":"c","type":"choice","prompt":"Category?",
                     "options":["safe","spam","abuse"]}]}'

# Omit the Authorization header → 401 {"detail":"missing bearer token"}
```

If both work, the server is sound. Now put HTTPS and a hostname in front.

---

## 5. Public URL with HTTPS — Caddy

Caddy automates Let's Encrypt and is a single-file config. Alternatives (nginx + certbot) work fine; Caddy is just the shortest path.

### DNS

Point an A record at the Elastic IP:

```
bernoulli.alice.example.com   A   <elastic-ip>
```

Wait for propagation (`dig +short bernoulli.alice.example.com` should return the IP).

### Security group

In the AWS console, on the instance's security group, allow inbound **TCP 80** and **TCP 443** from `0.0.0.0/0`. Do **not** expose 8000 publicly — the container is already bound to localhost in §4, but belt and suspenders.

### Install Caddy

```bash
sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | \
  sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | \
  sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt-get update && sudo apt-get install -y caddy
```

### Configure Caddy

```bash
sudo tee /etc/caddy/Caddyfile >/dev/null <<'EOF'
bernoulli.alice.example.com {
    reverse_proxy 127.0.0.1:8000
    encode gzip
    request_body {
        max_size 2MB
    }
}
EOF

sudo systemctl reload caddy
```

Caddy automatically obtains and renews a Let's Encrypt certificate for the hostname. Watch its journal if anything misbehaves:

```bash
sudo journalctl -u caddy -f
```

### Verify from your laptop

```bash
# Health check — no auth required
curl https://bernoulli.alice.example.com/healthz
# → {"status":"ok"}

# Real request — auth required
curl -sS -X POST https://bernoulli.alice.example.com/v1/decide \
  -H "Authorization: Bearer $YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"state":{"text":"Where is my package?"},
       "questions":[{"id":"i","type":"choice","prompt":"Intent?",
                     "options":["refund","tracking","other"]}]}'
```

You now have a public, HTTPS, authenticated Bernoulli endpoint.

---

## 6. Authentication — what you get and what you don't

### What is already in the repo

- **API-key auth via `BERNOULLI_API_KEYS`.** The server reads a comma-separated list of bearer tokens at startup (`bernoulli/server.py:_load_api_keys`). Every request except `GET /healthz` must carry `Authorization: Bearer <token>` where `<token>` matches one of the configured values. Unknown or missing tokens return `401`.
- **`/healthz` is intentionally unauthenticated.** So load balancers, uptime pingers, and Caddy itself can probe without a key.
- **Empty `BERNOULLI_API_KEYS` = auth disabled.** The right default for `localhost` dev and tests. **Never leave it empty on a public endpoint.**

### What is not in the repo

- **No per-key quotas, per-key logging, or rotation.** All keys in the comma-separated list are equivalent. Add/remove keys by restarting the container with a new value.
- **No usage metering.** If you need per-tenant billing or rate limiting, do it at the Caddy / reverse-proxy layer, or add middleware. The server itself is unmetered.
- **No mTLS / OAuth / JWT.** API keys only. If you need corporate SSO in front, run an auth-proxy (oauth2-proxy, Pomerium, etc.) between Caddy and the container.
- **No key-management UI.** Keys are strings in an env var. Rotate by editing the env and restarting.

### Rotating a key

```bash
NEW_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
docker rm -f bernoulli
docker run -d --name bernoulli --restart unless-stopped \
  --gpus all -p 127.0.0.1:8000:8000 \
  -v $HOME/.cache/huggingface:/data/hf-cache \
  -e BERNOULLI_SCORER=vllm \
  -e BERNOULLI_MAX_MODEL_LEN=8192 \
  -e BERNOULLI_API_KEYS="$NEW_KEY,$OLD_KEY" \
  bernoulli:latest
# ... migrate callers to NEW_KEY ...
# then restart again with -e BERNOULLI_API_KEYS="$NEW_KEY" to retire OLD_KEY.
```

Supporting both during the switchover is just a comma in the env var.

---

## 7. Running costs & stop/start

A `g5.xlarge` left running is ~$24 / day. Stop it when idle:

```bash
# From your laptop
aws ec2 stop-instances --instance-ids i-0abc...     # stops billing for compute
aws ec2 start-instances --instance-ids i-0abc...    # start again later
```

With an Elastic IP attached, the public IP survives stop/start. DNS keeps working. Caddy's cert survives too — stored under `/var/lib/caddy/`. The Docker container restarts automatically (`--restart unless-stopped`), and the HF cache is a host volume, so the only "cold start" cost is the model load (~90 s) + Triton JIT (~30 s) on first request after boot.

**EBS keeps billing while stopped** (~$12 / month for 150 GB gp3). The Elastic IP also charges a small fee if the instance isn't attached (free when attached and running).

---

## 8. Troubleshooting

- **Container exits during startup with CUDA OOM.** Lower `BERNOULLI_MAX_MODEL_LEN` further (e.g., 4096), or step up to a bigger GPU. Confirm with `nvidia-smi` that no zombie process is pinning VRAM — if there is, `sudo kill -9 <pid>` (see [`CLAUDE.md`](CLAUDE.md) Gotchas).
- **Caddy fails to issue a cert.** Check `sudo journalctl -u caddy -f`. Common causes: DNS hasn't propagated yet; port 80 blocked in the security group (Let's Encrypt uses HTTP-01 by default); the instance isn't publicly reachable.
- **`401 missing bearer token` on every request.** Caddy strips headers? Unlikely with the config above, but confirm by hitting `127.0.0.1:8000/v1/decide` directly on the box with the same token.
- **First real request after a long idle period is slow.** Expected — model + Triton JIT. Keep the container running (not stopped) between uses, or warm it with a dummy request after boot.
- **`422 Unprocessable Entity`.** Request body didn't validate against the pydantic model (strict `extra="forbid"`). The error body names the field.

See [`HTTP.md`](HTTP.md) for the full API surface and [`CLAUDE.md`](CLAUDE.md) for operational caveats.

---

## 9. What this does *not* set up

- **No autoscaling / multi-instance HA.** One box, one process. If this one dies, your endpoint dies. Add an ALB + ASG if you need redundancy.
- **No metrics / logs shipping.** Docker logs and `journalctl` only. Wire to CloudWatch / Prometheus / Loki as needed.
- **No request logging to a database.** The server does not persist requests. Add Caddy's `log` directive or a middleware if you need audit trails.
- **No backup of the HF cache.** It can be re-downloaded in 1–2 min via `hf_transfer`. Not worth snapshotting.

These are deliberate — the OSS core is "one box, serves well, authenticated". The private `bernoulli-ops` repo handles ALBs, Terraform, systemd, metrics, and multi-tenant key management for the project's own hosted endpoint, but none of that is required for Alice to run her own.

---

## Appendix: PyTorch CUDA pin

`pyproject.toml` pins torch wheels to the `pytorch-cu130` index on Linux x86_64, which requires an NVIDIA driver ≥580 (CUDA 13 runtime). The AWS g5/g6/g6e AMIs recommended above ship with recent-enough drivers.

On GPU hosts that ship an older driver (e.g. CUDA 12.8 only, driver 570-era), torch imports fail with "driver too old". Fix that at the host layer (newer driver, or launch an image with one) rather than by downgrading the project pin — on 2026-10-30 we tried swapping to `cu128` to accommodate such a host and the change cascaded through `torchvision`, `torchaudio`, `vllm`, and `triton`, pinning most of the stack to older versions. We reverted. If someone needs a cu128 branch to start from, see `m19-cu128-experiment`.
