import os
import time
import uuid
import tempfile
from typing import Optional

import numpy as np
import torch
import mne
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from model import SeizureModel
from inference import predict, explain


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SeizureModel()
model.load_state_dict(torch.load("artifacts/seizure_model.pth", map_location=device))
model.to(device)

stats = np.load("artifacts/train_stats.npy")
mean, std = stats[0], stats[1]

CHANNEL_NAMES = ['FP1-F7','F7-T7','T7-P7','P7-O1','FP1-F3','F3-C3',
                 'C3-P3','P3-O1','FP2-F4','F4-C4','C4-P4','P4-O2',
                 'FP2-F8','F8-T8','T8-P8-0','P8-O2','FZ-CZ','CZ-PZ',
                 'P7-T7','T7-FT9','FT9-FT10','FT10-T8']

# ---------------------------------------------------------------------------
# Uploaded file storage
#
# Large EEG files are streamed straight to disk (never held in RAM) and are
# referenced afterwards by a short-lived token. The EDF is then read with
# mne's preload=False so only the current 3-second window is ever loaded.
# ---------------------------------------------------------------------------
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "neuroxai_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
UPLOAD_TTL_SECONDS = 3600
_uploads = {}


def _prune_uploads():
    """Drop uploads older than the TTL (also frees disk space)."""
    now = time.time()
    for token, meta in list(_uploads.items()):
        if now - meta["created"] > UPLOAD_TTL_SECONDS:
            try:
                os.unlink(meta["path"])
            except OSError:
                pass
            _uploads.pop(token, None)


@app.get("/")
def root():
    return {"status": "NeuroXAI API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload(request: Request, name: str = Query("upload.edf")):
    """Stream the raw request body to disk and return a token for it."""
    _prune_uploads()

    token = uuid.uuid4().hex
    safe_name = os.path.basename(name) or "upload.edf"
    path = os.path.join(UPLOAD_DIR, f"{token}.edf")

    size = 0
    with open(path, "wb") as fh:
        async for chunk in request.stream():
            fh.write(chunk)
            size += len(chunk)

    if size == 0:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise HTTPException(status_code=400, detail="Empty upload.")

    _uploads[token] = {
        "path": path,
        "name": safe_name,
        "size": size,
        "created": time.time(),
    }
    return {"token": token, "name": safe_name, "size": size}


def _get_raw(token: str):
    """Return a non-preloaded Raw for the given upload token."""
    meta = _uploads.get(token)
    if not meta or not os.path.exists(meta["path"]):
        _uploads.pop(token, None)
        raise HTTPException(
            status_code=404,
            detail="Upload not found or expired. Please upload the file again.",
        )
    return mne.io.read_raw_edf(meta["path"], preload=False, verbose=False)


def _read_window(raw, second):
    """Read only the 3-second window for `second` from a non-preloaded raw."""
    start_sample = (second - 2) * 256
    stop_sample = (second + 1) * 256
    data = raw.get_data(start=start_sample, stop=stop_sample)[:22, :].astype(np.float32)
    return data


@app.post("/predict")
def predict_seizure(
    token: str = Query(...),
    second: Optional[int] = Query(None),
):
    raw = _get_raw(token)
    total_seconds = raw.n_times // 256

    if second is None:
        mid = total_seconds // 2
    else:
        mid = int(second)
    mid = max(2, min(mid, total_seconds - 1))

    window = _read_window(raw, mid)
    window = (window - mean) / std

    tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
    tensor.requires_grad_(True)

    prob, label = predict(model, tensor)

    cam, sv = explain(model, tensor)
    channel_importance = np.abs(sv).mean(axis=1)

    top3 = np.argsort(channel_importance)[::-1][:3]
    top3_names = [CHANNEL_NAMES[i] for i in top3]

    peak_sample = int(np.argmax(cam))
    peak_second = round(peak_sample / 256, 2)

    explanation = (
        f"The model predicted '{label}' with {round(prob*100, 1)}% confidence. "
        f"The most influential EEG channels were {top3_names[0]}, {top3_names[1]}, "
        f"and {top3_names[2]}. "
        f"The model's attention peaked at {peak_second} seconds into the 3-second "
        f"window based on GradCAM analysis."
    )

    return {
        "probability": round(prob, 4),
        "prediction": label,
        "explanation": explanation,
        "gradcam": cam.tolist(),
        "shap_values": sv.tolist(),
        "channel_importance": channel_importance.tolist()
    }


@app.post("/scan")
def scan_file(token: str = Query(...)):
    raw = _get_raw(token)
    total_seconds = raw.n_times // 256

    batch_size = 64
    all_probs = []
    model.eval()

    for sec_start in range(2, total_seconds, batch_size):
        sec_end = min(sec_start + batch_size, total_seconds)
        chunk_start = (sec_start - 2) * 256
        chunk_stop = sec_end * 256
        chunk = raw.get_data(start=chunk_start, stop=chunk_stop)[:22, :].astype(np.float32)

        windows = []
        for sec in range(sec_start, sec_end):
            local_start = (sec - (sec_start - 2)) * 256
            local_stop = local_start + 3 * 256
            w = chunk[:, local_start:local_stop]
            w = (w - mean) / std
            windows.append(w)

        windows_np = np.array(windows, dtype=np.float32)
        tensor_batch = torch.tensor(windows_np).to(device)

        with torch.no_grad():
            logits = model(tensor_batch)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()

        if probs.ndim == 0:
            all_probs.append(float(probs))
        else:
            all_probs.extend(probs.tolist())

    seconds = list(range(2, total_seconds))
    seizure_windows = [
        {"second": seconds[i], "probability": round(all_probs[i], 4)}
        for i in range(len(seconds))
        if all_probs[i] > 0.5
    ]

    events = []
    if seizure_windows:
        start = seizure_windows[0]["second"]
        end = seizure_windows[0]["second"]
        for w in seizure_windows[1:]:
            if w["second"] <= end + 2:
                end = w["second"]
            else:
                events.append({"start_second": start, "end_second": end,
                                "duration_seconds": end - start})
                start = w["second"]
                end = w["second"]
        events.append({"start_second": start, "end_second": end,
                        "duration_seconds": end - start})

    return {
        "total_seconds_scanned": total_seconds,
        "seizure_events_detected": len(events),
        "events": events,
        "explanation": (
            f"Scanned {total_seconds} seconds in batches. "
            f"Found {len(events)} potential seizure event(s). "
            + (f"First event starts at second {events[0]['start_second']}."
               if events else "No seizures detected.")
        )
    }
