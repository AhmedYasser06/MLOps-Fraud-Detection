output "mlflow_ui_url" {
  value = "http://localhost:5000"
}

output "minio_console_url" {
  value = "http://localhost:9001"
}

output "postgres_container_name" {
  value = docker_container.postgres.name
}

output "network_name" {
  value = docker_network.mlops_net.name
}
