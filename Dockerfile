# ============================================================
# SmartCampus RAG Knowledge Assistant - Production Dockerfile
# ============================================================

# Step 1: Base Image (Python 3.11 Slim for minimal attack surface and fast builds)
FROM python:3.11-slim

# Step 2: Set Safe Environment Variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TRANSFORMERS_NO_TF=1 \
    USE_TF=0 \
    USE_TORCH=1 \
    KERAS_HOME=/tmp/.keras \
    TOKENIZERS_PARALLELISM=false

# Step 3: Install Minimal OS Build Dependencies & Curl for Health Check
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Set Working Directory
WORKDIR /app

# Step 5: Install Python Dependencies with Layer Caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Step 6: Copy Application Files
COPY data/ /app/data/
COPY src/ /app/src/
COPY app.py /app/app.py
COPY show_chroma.py /app/show_chroma.py
COPY test_rag.py /app/test_rag.py
COPY README.md /app/README.md

# Step 7: Create Non-Root System User for Security Best Practices
RUN useradd -m -u 1001 -s /bin/bash appuser && \
    mkdir -p /app/chroma_db /tmp/.keras && \
    chown -R appuser:appuser /app /tmp/.keras

USER appuser

# Step 8: Expose Streamlit Port
EXPOSE 8501

# Step 9: Configure Streamlit Health Check
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Step 10: Launch Streamlit Application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
