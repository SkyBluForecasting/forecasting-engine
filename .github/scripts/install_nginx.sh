#!/bin/bash

set -euxo pipefail

# Keep SSH session alive by printing every 30 seconds
while true; do echo "🟢 installing..."; sleep 30; done &
PING_LOOP_PID=$!

# Ensure the loop is killed no matter how the script exits
trap "kill $PING_LOOP_PID 2>/dev/null" EXIT

# Optional: Speed up dnf
echo "max_parallel_downloads=10" | sudo tee -a /etc/dnf/dnf.conf

# Disable metadata timer
sudo dnf config-manager --setopt=metadata_timer_sync=0 --save

# Run installation
sudo dnf -v install -y --nogpgcheck --setopt=install_weak_deps=False nginx httpd-tools

echo "✅ NGINX installation complete"

# Done: kill keepalive loop
kill "$PING_LOOP_PID"
wait "$PING_LOOP_PID" 2>/dev/null || true
