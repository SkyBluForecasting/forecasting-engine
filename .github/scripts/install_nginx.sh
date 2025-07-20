#!/bin/bash

set -euxo pipefail

# Keep SSH session alive by emitting output every 30 seconds
while true; do echo "🟢 installing..."; sleep 30; done &
PING_LOOP_PID=$!
trap "kill $PING_LOOP_PID 2>/dev/null" EXIT

# Optional: Speed up dnf by allowing more parallel downloads
echo "max_parallel_downloads=10" | sudo tee -a /etc/dnf/dnf.conf

# Disable weak dependencies and unnecessary metadata sync
sudo dnf config-manager --setopt=metadata_timer_sync=0 --save

# Install nginx and httpd-tools (used for htpasswd)
sudo dnf -v install -y --nogpgcheck --setopt=install_weak_deps=False nginx httpd-tools

echo "✅ NGINX installation complete"