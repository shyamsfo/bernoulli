# deploy/ — Bernoulli dev box

Terraform for a single AWS g6.xlarge GPU box used as an rsync target for dev + test.

## Layout

```
deploy/
├── main.tf         # VPC/SG/AMI data lookups + EC2 instance
├── variables.tf    # region, instance_type, keypair, etc.
├── outputs.tf      # public IP, SSH command, ssh-config stanza
├── user_data.sh    # cloud-init: uv, HF cache on NVMe, env
├── sync.sh         # rsync wrapper
└── .gitignore      # terraform state + plan files
```

**No Python CLI.** For a single long-lived dev box, pure Terraform is simpler than the Click-CLI pattern in `~/nuw/inference/kube-do/nvidia-launcher/`. If this project ever needs multiple instances / spot fallback / per-model presets, graduate to that pattern.

## What this is NOT

- Not production serving — that lives in M4 (`VLLMScorer` + FastAPI in Docker).
- Not auto-shutdown — the instance stays up until you `aws ec2 stop-instances` or `terraform destroy` it.
- Not a shared box — SG is scoped to the caller's /32 at apply time.

## Prerequisites

- Terraform ≥ 1.6
- `awscli` configured, or `AWS_PROFILE=<profile>` set
- `id_nuwire` keypair already imported into AWS in `us-east-1` (nuw already did this — nvidia-launcher uses the same keypair)
- `~/.ssh/id_nuwire` present locally (already is, per `~/.ssh/config`)

## Up

```bash
cd deploy
terraform init
terraform plan      # inspect first
terraform apply     # ~90s to running state, then ~2 min of cloud-init
```

After apply, grab the ssh stanza and paste into `~/.ssh/config`:

```bash
terraform output -raw ssh_config_stanza >> ~/.ssh/config
```

Then wait for cloud-init:

```bash
ssh bernoulli 'sudo cloud-init status --wait && nvidia-smi'
```

## Push code and run tests

```bash
./deploy/sync.sh                # rsync, no delete
./deploy/sync.sh --mirror       # mirror (deletes remote files)

ssh bernoulli 'cd /home/ubuntu/bernoulli && uv sync && uv run pytest'
```

## Pause (save money without destroying)

```bash
aws ec2 stop-instances --instance-ids $(terraform output -raw instance_id)
aws ec2 start-instances --instance-ids $(terraform output -raw instance_id)
```

**NOTE:** Public IP changes on stop/start. After `start`, re-paste the ssh stanza:

```bash
terraform refresh && terraform output -raw ssh_config_stanza
```

Also: the instance-store NVMe is wiped on stop/start, so HF cache is lost. First run after start re-pulls model weights (~9 min for Qwen2.5-VL-7B).

## Down

```bash
terraform destroy
```

## HF token

HF Hub access is set via env on the box:

```bash
ssh bernoulli
export HF_TOKEN=hf_...
# add to ~/.bashrc or ~/.hf-token if you want it persistent
```

For most Qwen VL models no token is needed (they're public, non-gated). Add one only if we move to a gated model.

## Cost

- g6.xlarge on-demand: ~$0.80/hr → ~$576/mo if left on 24/7
- Stopped instance: just EBS root (80GB gp3) ≈ $6.40/mo
- Rough dev usage estimate: 4 hr/day × 20 days/mo = ~$64/mo

## Troubleshooting

- cloud-init log: `ssh bernoulli 'sudo cat /var/log/bernoulli-cloud-init.log'`
- GPU check: `ssh bernoulli 'nvidia-smi'`
- NVMe mount: `ssh bernoulli 'df -h /opt/nvme'`
- The NVIDIA GPU-Optimized AMI must be subscribed-to once per AWS account via the AWS Marketplace: https://aws.amazon.com/marketplace/pp?sku=7wvc601jyn35n5w8b3kro66vi (already done if nvidia-launcher has run in this account).
