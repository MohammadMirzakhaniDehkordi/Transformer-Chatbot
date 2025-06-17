# Base Python image
FROM python:3.10-slim-bookworm

# Update system packages to reduce vulnerabilities
RUN apt-get update && apt-get upgrade -y && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create app directory
WORKDIR /app

# Copy requirements (choose which to use)
COPY requirements_api.txt ./requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Default port (used by uvicorn/gradio)
EXPOSE 8000

# Set default entrypoint to FastAPI
CMD ["uvicorn", "chatbot_api:app", "--host", "0.0.0.0", "--port", "8000"]
