output "instance_id" {
  description = "EC2 instance id"
  value       = aws_instance.dev.id
}

output "public_ip" {
  description = "Public IPv4 address — may change if the instance is stopped and started"
  value       = aws_instance.dev.public_ip
}

output "public_dns" {
  description = "Public DNS name — stable for the instance lifetime"
  value       = aws_instance.dev.public_dns
}

output "ssh_command" {
  description = "One-liner to SSH in"
  value       = "ssh -i ~/.ssh/id_nuwire ubuntu@${aws_instance.dev.public_ip}"
}

output "ssh_config_stanza" {
  description = "Add this to ~/.ssh/config for 'ssh bernoulli' to Just Work. Public IP in this stanza will go stale on stop/start — regenerate with 'terraform output' and re-paste."
  value       = <<-EOT
    Host bernoulli
        HostName ${aws_instance.dev.public_ip}
        User ubuntu
        IdentityFile ~/.ssh/id_nuwire
        StrictHostKeyChecking accept-new
  EOT
}

output "ami_id" {
  description = "AMI the instance was launched from"
  value       = data.aws_ami.nvidia_gpu_base.id
}

output "security_group_id" {
  description = "Security group id"
  value       = aws_security_group.dev.id
}
