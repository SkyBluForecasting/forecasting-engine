#!/bin/bash

set -euxo pipefail

# Keep the session alive in the background
(
  while true; do
    echo "🟢 installing..."
    sleep 30
  done
) &
PING_LOOP_PID=$!

# Kill the keepalive loop no matter how the script exits
trap "kill $PING_LOOP_PID 2>/dev/null || true" EXIT

# Improve DNF performance
echo "max_parallel_downloads=10" | sudo tee -a /etc/dnf/dnf.conf
sudo dnf config-manager --setopt=metadata_timer_sync=0 --save

# Install required packages
sudo dnf -v install -y --nogpgcheck --setopt=install_weak_deps=False nginx httpd-tools

echo "✅ NGINX installation complete"

# Stop the keepalive loop without causing failure
if kill "$PING_LOOP_PID" 2>/dev/null; then
  wait "$PING_LOOP_PID" 2>/dev/null || true
fi

exit 0
