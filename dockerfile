# Builder stage: Install dependencies in a virtual environment
FROM python:3.10-slim AS builder
WORKDIR /app

# Install build tools if needed (optional)
RUN apt-get update && apt-get install -y build-essential

# Copy only requirements first
COPY requirements.txt .

# Create a virtual environment and install dependencies
RUN python -m venv /venv && \
    /venv/bin/pip install --upgrade pip && \
    /venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy application code (if needed during build, for compiling extensions etc.)
COPY . .

# Runtime stage: Use a minimal image and copy over the virtual environment
FROM python:3.10-slim
WORKDIR /app

# Install runtime libraries (like libGL for OpenCV)
RUN apt-get update && apt-get install -y libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder stage
COPY --from=builder /venv /venv

# Copy the application code
COPY . .

# Update PATH to use the virtual environment
ENV PATH="/venv/bin:$PATH"

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
