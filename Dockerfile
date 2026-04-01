# Use Bookworm (Debian 12) so system SQLite is >= 3.35 required by Chroma
FROM python:3.11-slim-bookworm

# Install system dependencies for Chroma / LangChain
RUN apt-get update && \
    apt-get install -y build-essential libsqlite3-dev libgl1 libglib2.0-0 curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code
COPY . .

# Expose port and run Gunicorn
# Run with AWS credentials, e.g.:
#   docker run -d -p 8000:8000 --env-file .env rag_app
#   or: docker run -d -p 8000:8000 -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_REGION rag_app
EXPOSE 8000
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]