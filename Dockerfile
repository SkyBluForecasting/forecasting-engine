# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Build arg for CodeArtifact token
ARG CODEARTIFACT_AUTH_TOKEN
ARG AWS_REGION

# Set working directory inside container
WORKDIR /app

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Install system dependencies if any (example: build-essential, git, etc.)
RUN apt-get update && apt-get install -y build-essential curl

# Copy your forecasting engine code into the container
COPY forecasting_engine ./forecasting_engine
COPY requirements.txt .
ENV PIP_DEFAULT_TIMEOUT=120

# Build wheels first and install deps
RUN python -m pip install --upgrade pip wheel setuptools && \
    mkdir -p /wheels && \
    pip wheel -r requirements.txt -w /wheels \
        --index-url https://aws:${CODEARTIFACT_AUTH_TOKEN}@skyblu-591082451778.d.codeartifact.${AWS_REGION}.amazonaws.com/pypi/forecasting-db/simple/ \
        --extra-index-url https://pypi.org/simple

# Install from wheels
RUN pip install --no-cache-dir /wheels/*

# Expose port 5050 (used by mlflow server)
EXPOSE 5050

# Default command (can be overridden in docker-compose)
CMD ["tail", "-f", "/dev/null"]

# Copy cleanup script into the image and make executable
COPY forecasting_engine/tasks/cleanup.sh /usr/local/bin/cleanup.sh
RUN chmod +x /usr/local/bin/cleanup.sh

# Setup supercronic to run services on schedules
RUN apt-get update && apt-get install -y curl && \
    curl -L -o /usr/local/bin/supercronic https://github.com/aptible/supercronic/releases/download/v0.2.24/supercronic-linux-amd64 && \
    chmod +x /usr/local/bin/supercronic