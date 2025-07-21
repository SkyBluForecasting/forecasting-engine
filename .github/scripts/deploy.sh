#!/bin/bash
set -euxo pipefail
cd ~/forecasting-engine

docker-compose down || true
docker-compose up -d --build

# Wait for MLflow to be ready (similar to before)
for i in $(seq 1 24); do
  if curl --fail --silent --max-time 3 http://localhost:5050; then
    echo "✅ MLflow is ready!"
    exit 0
  else
    echo "⏳ Waiting for MLflow to become ready..."
    sleep 5
  fi
done

echo "❌ MLflow did not become ready in time"
docker-compose logs mlflow
exit 1