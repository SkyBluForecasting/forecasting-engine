#!/bin/bash
# Delete files and directories older than 24 hours in /tmp
echo "$(date): Cleaning old temp files and directories..."

# Delete old files
find /tmp -type f -mmin +1440 -delete

# Delete old directories (non-empty)
find /tmp -mindepth 1 -maxdepth 1 -type d -mtime +0 -exec rm -rf {} \;

echo "$(date): Cleanup complete."