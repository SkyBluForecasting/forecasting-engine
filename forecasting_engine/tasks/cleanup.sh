#!/bin/bash
echo "$(date): Starting cleanup..."

# 1. Delete all tmp* directories in /tmp
echo "Cleaning /tmp..."
rm -rf /tmp/tmp*

# 2. Clean old logs safely
echo "Cleaning logs..."
rm -rf /var/log/*.gz /var/log/*.[0-9] 2>/dev/null
rm -rf /var/log/journal/* 2>/dev/null

# 3. Docker cleanup (requires Docker socket mounted)
if [ -S /var/run/docker.sock ]; then
    echo "Cleaning stopped containers older than 24h..."
    docker ps -a --filter "status=exited" --filter "until=24h" -q | xargs -r docker rm -v

    echo "Cleaning dangling images older than 24h..."
    docker images -f "dangling=true" -f "until=24h" -q | xargs -r docker rmi

    echo "Cleaning unused volumes older than 24h..."
    docker volume ls -qf "dangling=true" | xargs -r docker volume rm
fi

# 4. Optional: prune build cache (if needed)
if [ -S /var/run/docker.sock ]; then
    echo "Pruning build cache..."
    docker builder prune -af
fi

echo "$(date): Cleanup complete."
