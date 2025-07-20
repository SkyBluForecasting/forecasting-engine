#!/bin/bash
set -euxo pipefail

# Load secrets from .env
if [ -f ~/forecasting-engine/.env ]; then
  export $(grep -v '^#' ~/forecasting-engine/.env | xargs)
else
  echo "❌ .env file not found at ~/forecasting-engine/.env"
  exit 1
fi

# Install nginx + htpasswd tools (idempotent)
if command -v dnf >/dev/null; then
  sudo dnf install -y nginx httpd-tools
elif command -v apt-get >/dev/null; then
  sudo apt-get update
  sudo apt-get install -y nginx apache2-utils
fi

# Enable and start nginx
sudo systemctl enable nginx

# Write basic auth creds
sudo htpasswd -bcm /etc/nginx/.htpasswd "$MLFLOW_BASIC_AUTH_USER" "$MLFLOW_BASIC_AUTH_PASSWORD"

# Write nginx config
sudo tee /etc/nginx/conf.d/mlflow.conf > /dev/null << 'NGINXCONF'
server {
    listen 80;
    location / {
        auth_basic "Restricted";
        auth_basic_user_file /etc/nginx/.htpasswd;
        proxy_pass http://127.0.0.1:5050;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
NGINXCONF

# Restart nginx if config is valid
sudo nginx -t && sudo systemctl restart nginx
