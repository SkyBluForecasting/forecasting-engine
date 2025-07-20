#!/bin/bash
# Emit output every 30s to keep SSH alive
while true; do echo "🟢 installing..."; sleep 30; done &
PING_LOOP_PID=$!

set -euxo pipefail

# Optional: speeds up dnf for big metadata downloads
echo "max_parallel_downloads=10" | sudo tee -a /etc/dnf/dnf.conf

sudo dnf install -y nginx httpd-tools

kill $PING_LOOP_PID