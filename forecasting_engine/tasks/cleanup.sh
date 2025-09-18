#!/bin/bash
# Delete files and directories older than 24 hours in /tmp
echo "$(date): Cleaning old temp files and directories..."

# Delete files older than 24h
find /tmp -type f -mmin +1440 -delete

# Delete all tmp* directories immediately
rm -rf /tmp/tmp*

echo "$(date): Cleanup complete."