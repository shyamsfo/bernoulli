variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "aws_profile" {
  description = "AWS CLI profile name (leave 'default' to use the default profile or env credentials)"
  type        = string
  default     = "default"
}

variable "instance_type" {
  description = "EC2 instance type. g6e.xlarge = L40S 48GB, ~$1.86/hr on-demand. Was g6.xlarge (L4 24GB) through M3; stepped up for M4 production backbone."
  type        = string
  default     = "g6e.xlarge"
}

variable "instance_name" {
  description = "Name tag for the EC2 instance (also used as SG prefix)"
  type        = string
  default     = "bernoulli-dev"
}

variable "key_pair_name" {
  description = "Name of an existing EC2 key pair in the target region. Must be already imported."
  type        = string
  default     = "id_nuwire"
}

variable "root_volume_gb" {
  description = "Root EBS volume size in GB. Code + venv + driver cache + some weights. HF cache goes to instance-store NVMe at /opt/nvme/hf-cache."
  type        = number
  default     = 80
}

variable "ssh_cidr" {
  description = "CIDR allowed to SSH to the box. Empty = auto-detect caller's /32."
  type        = string
  default     = ""
}
