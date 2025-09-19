#!/bin/bash
echo "$(date): Cleaning old temp files and directories..."

# Delete all tmp* directories immediately
rm -rf /tmp/tmp*

echo "$(date): Cleanup complete."