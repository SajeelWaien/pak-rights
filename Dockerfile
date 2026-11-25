# Dockerfile for pak-rights RAG application
# Use a lightweight Python base image
FROM python:3.11-slim

# Install system dependencies (if any are needed for some packages)
# For example, gcc and libpq-dev are often required; adjust as needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy only requirements first for caching layer
COPY requirements.txt ./

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Ensure the storage directory exists (for the vector index)
RUN mkdir -p storage/vector_index

# Declare a volume so the vector store can be persisted on the host
VOLUME ["/app/storage/vector_index"]

# Expose the port that uvicorn will run on
EXPOSE 8000

# Use environment variables from .env if present (optional)
# You can also pass them at runtime with -e flags
ENV PYTHONUNBUFFERED=1

# Default command to run the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
