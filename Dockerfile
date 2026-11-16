FROM python:3.11-slim

# libgomp1 is required by torch's OpenMP runtime.
# nginx is the single public reverse proxy in front of FastAPI + Streamlit.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    nginx \
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

COPY nginx.conf.template /etc/nginx/nginx.conf.template

# Normalise line endings (the repo may be checked out on Windows) and make
# the entrypoint executable.
RUN sed -i 's/\r$//' entrypoint.sh && chmod +x entrypoint.sh

EXPOSE 7860

CMD ["/bin/sh", "entrypoint.sh"]
