FROM python:3.11-slim

# libgomp1 is required by torch's OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install the CPU-only torch wheel first (avoids the multi-GB CUDA packages),
# then the rest of the requirements.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Headless rendering for matplotlib inside the container
ENV MPLBACKEND=Agg

# Hugging Face Spaces exposes a single public port (7860)
EXPOSE 7860

# FastAPI backend on 8000 (internal) + Streamlit UI on 7860 (public)
CMD uvicorn main:app --host 127.0.0.1 --port 8000 & streamlit run EEG-Frontend.py --server.port 7860 --server.address 0.0.0.0 --server.headless true
