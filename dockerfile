FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY fastapi_enpoints.py .
COPY gradio_app.py .
COPY .env .

# Expose port
EXPOSE 8000

# Run the application
CMD ["sh", "-c", "python fastapi.py & python gradio_app.py"]