# Use an official lightweight Python image
FROM python:3.10-slim

# Set a working directory
WORKDIR /app

# Upgrade pip and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port your app runs on. DigitalOcean sets a PORT env variable.
EXPOSE 8000

# Run the FastAPI app using uvicorn. Use the PORT env variable if provided.
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}


# docker build -t my-fastapi-app .
# docker run -p 8000:8000 my-fastapi-app