from fastapi import FastAPI, UploadFile, File
from typing import Optional
import numpy as np
import torch
from model import SeizureModel
from inference import predict, explain
import io
import os
import tempfile
import mne
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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


@app.get("/")
def root():
    return {"status": "NeuroXAI API is running"}


def _read_window(raw, second):
    """Read only the 3-second window for `second` from a non-preloaded raw."""
    start_sample = (second - 2) * 256
    stop_sample = (second + 1) * 256
    data = raw.get_data(start=start_sample, stop=stop_sample)[:22, :].astype(np.float32)
    return data


def _edf_to_raw(contents):
    """Write EDF bytes to a temp file and return a non-preloaded Raw object."""
    tmp = tempfile.NamedTemporaryFile(suffix=".edf", delete=False)
    tmp.write(contents)
    tmp.close()
    raw = mne.io.read_raw_edf(tmp.name, preload=False, verbose=False)
    return raw, tmp.name


@app.post("/predict")
async def predict_seizure(file: UploadFile = File(...), second: Optional[int] = None):
    contents = await file.read()
    raw, tmp_path = _edf_to_raw(contents)
    try:
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
    finally:
        del raw
        os.unlink(tmp_path)

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
async def scan_file(file: UploadFile = File(...)):
    contents = await file.read()
    raw, tmp_path = _edf_to_raw(contents)
    try:
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
    finally:
        del raw
        os.unlink(tmp_path)

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
