#!/bin/bash
set -e

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Loading AWS credentials from .env..."
if [ -f .env ]; then
    set -a
    source .env
    set +a
else
    echo "Error: .env file not found!"
    exit 1
fi

echo "Getting CodeArtifact token..."
CODEARTIFACT_AUTH_TOKEN=$(aws codeartifact get-authorization-token \
    --domain skyblu \
    --domain-owner 591082451778 \
    --region $AWS_REGION \
    --query authorizationToken \
    --output text)

if [ -z "$CODEARTIFACT_AUTH_TOKEN" ]; then
    echo "Failed to get CodeArtifact token"
    exit 1
fi

# Construct pip index URL for CodeArtifact
CODEARTIFACT_INDEX_URL="https://aws:$CODEARTIFACT_AUTH_TOKEN@skyblu-591082451778.d.codeartifact.$AWS_REGION.amazonaws.com/pypi/forecasting-db/simple/"

echo "Installing Python dependencies..."
pip install -r requirements.txt --index-url $CODEARTIFACT_INDEX_URL --extra-index-url https://pypi.org/simple

echo "All dependencies installed!"
