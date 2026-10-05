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
  description = "EC2 instance type. g5.xlarge = A10G 24GB, ~$1.01/hr. Equivalent memory to g6.xlarge's L4; chosen because g6 capacity in us-east-1 is exhausted today. Step up when g6e capacity frees up for the production backbone work in M4e."
  type        = string
  default     = "g5.xlarge"
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

variable "availability_zone" {
  description = "AZ to launch in. Override if the default hits InsufficientInstanceCapacity. g6e.xlarge is offered in us-east-1{a,b,c,d}; not in 1e/1f."
  type        = string
  default     = "us-east-1a"
}
