terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60"
    }
    http = {
      source  = "hashicorp/http"
      version = "~> 3.4"
    }
  }
}

provider "aws" {
  region  = var.region
  profile = var.aws_profile
}

# Fetch the caller's current public IP so SSH ingress is scoped to just us.
# Override with var.ssh_cidr if you want to pin a specific CIDR.
data "http" "my_ip" {
  url = "https://checkip.amazonaws.com"
}

locals {
  my_ip_cidr = "${trimspace(data.http.my_ip.response_body)}/32"
  ssh_cidr   = var.ssh_cidr != "" ? var.ssh_cidr : local.my_ip_cidr
}

# Default VPC + any one default subnet — matches nvidia-launcher convention.
data "aws_vpc" "default" {
  default = true
}

# AZs that actually offer the requested instance type (us-east-1e doesn't have g6.xlarge).
data "aws_ec2_instance_type_offerings" "instance_type_azs" {
  filter {
    name   = "instance-type"
    values = [var.instance_type]
  }
  location_type = "availability-zone"
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
  filter {
    name   = "default-for-az"
    values = ["true"]
  }
  filter {
    name   = "availability-zone"
    values = data.aws_ec2_instance_type_offerings.instance_type_azs.locations
  }
}

# Pin a specific AZ to avoid retrying forever on InsufficientInstanceCapacity.
# g6e.xlarge capacity moves around; us-east-1a is the current default. If AWS
# throws capacity for 1a, override with `terraform apply -var 'availability_zone=us-east-1b'`.
data "aws_subnet" "primary" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
  filter {
    name   = "default-for-az"
    values = ["true"]
  }
  filter {
    name   = "availability-zone"
    values = [var.availability_zone]
  }
}

# id_nuwire keypair — imported once by the nuw team, exists in us-east-1.
data "aws_key_pair" "ssh" {
  key_name = var.key_pair_name
}

# NVIDIA GPU Base CUDA on Ubuntu 24.04 (same AMI nvidia-launcher uses).
# Owner 679593333241 = NVIDIA. No marketplace fee (verified in nvidia-launcher).
data "aws_ami" "nvidia_gpu_base" {
  most_recent = true
  owners      = ["679593333241"]

  filter {
    name   = "name"
    values = ["NVIDIA GPU Base CUDA * on Ubuntu 24.04 *"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_security_group" "dev" {
  name_prefix = "${var.instance_name}-"
  description = "Bernoulli dev box: SSH + egress"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "SSH from caller IP"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [local.ssh_cidr]
  }

  egress {
    description = "All egress"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "${var.instance_name}-sg"
    Project = "bernoulli"
    Owner   = "nuw"
  }
}

resource "aws_instance" "dev" {
  ami                    = data.aws_ami.nvidia_gpu_base.id
  instance_type          = var.instance_type
  subnet_id              = data.aws_subnet.primary.id
  key_name               = data.aws_key_pair.ssh.key_name
  vpc_security_group_ids = [aws_security_group.dev.id]

  # Fail fast on capacity errors instead of terraform's default retry-forever.
  timeouts {
    create = "4m"
  }

  # Instance-store NVMe is ephemeral (232GB on g6.xlarge) — mounted in cloud-init
  # for the HF cache. Root EBS is for OS + code.
  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.root_volume_gb
    delete_on_termination = true
    encrypted             = true
  }

  user_data                   = file("${path.module}/user_data.sh")
  user_data_replace_on_change = false

  tags = {
    Name    = var.instance_name
    Project = "bernoulli"
    Owner   = "nuw"
  }

  # Protect against accidental terraform destroy wiping the box mid-dev-session.
  # Flip to true when you want to tear it down.
  lifecycle {
    ignore_changes = [
      ami, # don't recreate when NVIDIA ships a new base AMI; recreate manually when desired
    ]
  }
}
