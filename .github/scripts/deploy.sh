#!/bin/bash
set -euxo pipefail
cd ~/forecasting-engine

echo "🧹 Running docker system prune..."
docker system prune -af || true

echo "🐋 Building and starting containers with Docker Compose..."
DOCKER_BUILDKIT=0 docker-compose down || true
DOCKER_BUILDKIT=0 docker-compose up -d --build

echo "🕵️ Waiting for MLflow to become ready..."
for i in $(seq 1 24); do
  if curl --fail --silent --max-time 3 http://localhost:5050; then
    echo "✅ MLflow is ready!"
    exit 0
  else
    echo "⏳ Attempt $i: MLflow not ready yet..."
    sleep 5
  fi
done

echo "❌ MLflow did not become ready in time"
docker-compose logs mlflow
exit 1