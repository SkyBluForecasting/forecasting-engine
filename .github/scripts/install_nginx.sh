#!/bin/bash

# Emit output every 30s to keep SSH alive
while true; do echo "🟢 installing..."; sleep 30; done &
PING_LOOP_PID=$!

set -euxo pipefail

# Prevent metadata sync prompt
sudo dnf config-manager --setopt=metadata_timer_sync=0 --save

# Optional: boost download speed
echo "max_parallel_downloads=10" | sudo tee -a /etc/dnf/dnf.conf

# Try no weak deps + verbose output
sudo dnf -v install -y --setopt=install_weak_deps=False nginx httpd-tools

kill $PING_LOOP_PID