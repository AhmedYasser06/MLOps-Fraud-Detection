resource "docker_network" "mlops_net" {
  name = var.network_name
}

resource "docker_volume" "pgdata" {
  name = "fraud_detection_pgdata"
}

resource "docker_volume" "miniodata" {
  name = "fraud_detection_miniodata"
}

resource "docker_image" "postgres" {
  name = "postgres:15"
}

resource "docker_image" "minio" {
  name = "minio/minio:latest"
}

resource "docker_image" "mlflow" {
  name = var.mlflow_image

  build {
    context    = "${path.module}/../mlflow"
    dockerfile = "Dockerfile"
  }
}

resource "docker_container" "postgres" {
  name  = "fraud-detection-postgres"
  image = docker_image.postgres.image_id
  networks_advanced {
    name = docker_network.mlops_net.name
  }
  env = [
    "POSTGRES_USER=${var.postgres_user}",
    "POSTGRES_PASSWORD=${var.postgres_password}",
    "POSTGRES_DB=mlflow",
  ]
  volumes {
    volume_name    = docker_volume.pgdata.name
    container_path = "/var/lib/postgresql/data"
  }
  restart = "unless-stopped"
}

resource "docker_container" "minio" {
  name    = "fraud-detection-minio"
  image   = docker_image.minio.image_id
  command = ["server", "/data", "--console-address", ":9001"]
  networks_advanced {
    name = docker_network.mlops_net.name
  }
  env = [
    "MINIO_ROOT_USER=${var.minio_root_user}",
    "MINIO_ROOT_PASSWORD=${var.minio_root_password}",
  ]
  ports {
    internal = 9000
    external = 9000
  }
  ports {
    internal = 9001
    external = 9001
  }
  volumes {
    volume_name    = docker_volume.miniodata.name
    container_path = "/data"
  }
  restart = "unless-stopped"
}

resource "docker_image" "minio_mc" {
  name = "minio/mc:latest"
}

resource "docker_container" "minio_init" {
  name     = "fraud-detection-minio-init"
  image    = docker_image.minio_mc.image_id
  must_run = false # one-shot: create the bucket, then exit
  networks_advanced {
    name = docker_network.mlops_net.name
  }
  entrypoint = ["/bin/sh", "-c"]
  command = [
    "until (mc alias set local http://${docker_container.minio.name}:9000 ${var.minio_root_user} ${var.minio_root_password}) do sleep 2; done; mc mb -p local/mlflow-artifacts; exit 0"
  ]
  depends_on = [docker_container.minio]
}

resource "docker_container" "mlflow" {
  name  = "fraud-detection-mlflow"
  image = docker_image.mlflow.image_id
  networks_advanced {
    name = docker_network.mlops_net.name
  }
  env = [
    "MLFLOW_S3_ENDPOINT_URL=http://${docker_container.minio.name}:9000",
    "AWS_ACCESS_KEY_ID=${var.minio_root_user}",
    "AWS_SECRET_ACCESS_KEY=${var.minio_root_password}",
  ]
  command = [
    "mlflow", "server",
    "--host", "0.0.0.0",
    "--port", "5000",
    "--backend-store-uri", "postgresql://${var.postgres_user}:${var.postgres_password}@${docker_container.postgres.name}:5432/mlflow",
    "--default-artifact-root", "s3://mlflow-artifacts/",
  ]
  ports {
    internal = 5000
    external = 5000
  }
  restart    = "unless-stopped"
  depends_on = [docker_container.postgres, docker_container.minio, docker_container.minio_init]
}
