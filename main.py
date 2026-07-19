from fastapi import FastAPI, UploadFile, File
import numpy as np
import torch
from model import SeizureModel
from inference import predict, explain_gradcam, explain_shap
import io
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

# load everything once at startup, not per request
model = SeizureModel()
model.load_state_dict(torch.load("artifacts/seizure_model.pth", map_location=device))
model.to(device)

stats = np.load("artifacts/train_stats.npy")
mean, std = stats[0], stats[1]

background = np.load("artifacts/background.npy")
background_tensor = torch.tensor(background, dtype=torch.float32).to(device)

@app.get("/")
def root():
    return {"status": "NeuroXAI API is running"}

@app.post("/predict")
async def predict_seizure(file: UploadFile = File(...), second: int = None):
    # 1. read bytes from uploaded file
    contents = await file.read()
    
    # 2. load EDF from bytes via MNE
    raw = mne.io.read_raw_edf(io.BytesIO(contents), preload=True, verbose=False)
    data = raw.get_data()[:22, :]
    
    # 3. extract one 3-second window from the middle of the file
    total_seconds = data.shape[1] // 256
    if second is None:
        mid = total_seconds // 2
    else:
        mid = int(second)

    mid = max(2, min(mid, total_seconds - 1))
    window = data[:, (mid-2)*256 : (mid+1)*256]
    
    # 4. standardize using saved stats
    window = (window - mean) / std 
    
    # 5. convert to tensor
    tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
    tensor.requires_grad_(True)
    
    # 6. predict
    prob, label = predict(model, tensor)
    print("\n" + "="*60)
    print("SECOND RECEIVED:", second)
    print("PROBABILITY:", prob)
    print("LABEL:", label)
    print("="*60 + "\n")
    
    # 7. explain
    cam = explain_gradcam(model, tensor)
    sv = explain_shap(model, tensor, background_tensor)
    channel_importance = np.abs(sv).mean(axis=1)
    
    # find top 3 channels by SHAP importance
    channel_names = ['FP1-F7','F7-T7','T7-P7','P7-O1','FP1-F3','F3-C3',
                 'C3-P3','P3-O1','FP2-F4','F4-C4','C4-P4','P4-O2',
                 'FP2-F8','F8-T8','T8-P8-0','P8-O2','FZ-CZ','CZ-PZ',
                 'P7-T7','T7-FT9','FT9-FT10','FT10-T8']

    top3 = np.argsort(channel_importance)[::-1][:3]
    top3_names = [channel_names[i] for i in top3]

    # find peak GradCAM time
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
    raw = mne.io.read_raw_edf(io.BytesIO(contents), preload=True, verbose=False)
    data = raw.get_data()[:22, :]
    total_seconds = data.shape[1] // 256

    # build all windows
    windows = []
    seconds = []
    for sec in range(2, total_seconds):
        window = data[:, (sec-2)*256 : (sec+1)*256]
        window = (window - mean) / std
        windows.append(window)
        seconds.append(sec)

    windows_np = np.array(windows, dtype=np.float32)
    tensor_all = torch.tensor(windows_np)

    # batch inference
    batch_size = 128
    all_probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(tensor_all), batch_size):
            batch = tensor_all[i:i+batch_size].to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            if probs.ndim == 0:
                all_probs.append(float(probs))
            else:
                all_probs.extend(probs.tolist())

    # find seizure windows
    seizure_windows = [
        {"second": seconds[i], "probability": round(all_probs[i], 4)}
        for i in range(len(seconds))
        if all_probs[i] > 0.5
    ]

    # group consecutive seconds into events
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

    # check against known ground truth if available
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
