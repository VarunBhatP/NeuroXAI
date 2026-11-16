#!/bin/sh
set -e

# Public port (Render sets PORT, defaults to 10000; HF Spaces uses 7860).
PORT="${PORT:-10000}"

# nginx is the single public listener. It forwards /api/* to FastAPI and
# everything else to the Streamlit UI.
sed "s/__PORT__/${PORT}/g" /etc/nginx/nginx.conf.template > /etc/nginx/nginx.conf

# Internal backend (inference + streaming upload).
uvicorn main:app --host 127.0.0.1 --port 8000 &

# Internal Streamlit UI.
streamlit run EEG-Frontend.py \
    --server.port 7860 \
    --server.address 127.0.0.1 \
    --server.headless true \
    --server.maxUploadSize 2048 &

# nginx stays in the foreground (daemon off) as the container's main process.
exec nginx
