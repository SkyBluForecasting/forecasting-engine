#!/bin/bash
# Delete files older than 24 hours in /tmp
echo "$(date): Cleaning old temp files..."
find /tmp -type f -mmin +1440 -delete