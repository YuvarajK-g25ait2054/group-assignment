# Use slim Python image (smaller size)
FROM python:3.10-slim

# Set model name as build argument with default
ARG MODEL_NAME=YuvarajK-g25ait2054/distilbert-imdb-sentiment
ENV MODEL_NAME=${MODEL_NAME}

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy inference script
COPY inference.py .

# Run inference when container starts
CMD ["python", "inference.py"]
