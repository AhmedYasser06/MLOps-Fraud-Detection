variable "postgres_user" {
  description = "MLflow backend store DB user"
  type        = string
  default     = "mlflow"
}

variable "postgres_password" {
  description = "MLflow backend store DB password"
  type        = string
  sensitive   = true
  default     = "mlflow"
}

variable "minio_root_user" {
  type    = string
  default = "minioadmin"
}

variable "minio_root_password" {
  type      = string
  sensitive = true
  default   = "minioadmin"
}

variable "mlflow_image" {
  type    = string
  default = "ghcr.io/mlflow/mlflow:v2.16.0"
}

variable "network_name" {
  type    = string
  default = "fraud_detection_mlops_net"
}
