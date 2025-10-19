# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Build arg for CodeArtifact token
ARG CODEARTIFACT_AUTH_TOKEN

# Set working directory inside container
WORKDIR /app

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Install system dependencies if any (example: build-essential, git, etc.)
RUN apt-get update && apt-get install -y build-essential curl

# Copy your forecasting engine code into the container
COPY forecasting_engine ./forecasting_engine

# Install Python dependencies if you have requirements.txt
COPY requirements.txt .
ENV PIP_DEFAULT_TIMEOUT=120
RUN pip install --no-cache-dir --prefer-binary \
    --index-url https://aws:${CODEARTIFACT_AUTH_TOKEN}@skyblu-591082451778.d.codeartifact.${AWS_REGION}.amazonaws.com/pypi/forecasting-db/simple/ \
    --extra-index-url https://pypi.org/simple \
    -r requirements.txt

# Expose port 5050 (used by mlflow server)
EXPOSE 5050

# Default command (can be overridden in docker-compose)
CMD ["tail", "-f", "/dev/null"]

# Copy cleanup script into the image and make executable
COPY forecasting_engine/tasks/cleanup.sh /usr/local/bin/cleanup.sh
RUN chmod +x /usr/local/bin/cleanup.sh