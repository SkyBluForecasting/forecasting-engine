#!/bin/bash
# Delete files and directories older than 24 hours in /tmp
echo "$(date): Cleaning old temp files and directories..."

# Find tmp* directories older than 24h and remove them
find /tmp -type d -name 'tmp*' -mmin +1440 -exec rm -rf {} +

echo "$(date): Cleanup complete."