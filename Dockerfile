# Use official Python base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY book_recommender.py /app/book_recommender.py
COPY books.csv /app/books.csv
COPY book_embeddings.npy /app/book_embeddings.npy
COPY model_cache/ /app/model_cache/

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "book_recommender.py", "--server.port=8501", "--server.address=0.0.0.0"]