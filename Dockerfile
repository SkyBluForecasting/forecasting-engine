# Dockerfile
FROM python:3.11-slim

# Set working dir
WORKDIR /app

# Copy code and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY forecasting_engine/ forecasting_engine/

# No ENTRYPOINT or CMD for now, just test that the code is there.
CMD ["python", "-c", "import forecasting_engine; print('Module OK')"]