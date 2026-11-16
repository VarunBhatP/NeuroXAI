import numpy as np
import torch
import matplotlib.pyplot as plt


def predict(model, input_window):
    """Returns probability and label for one window."""
    model.eval()
    with torch.no_grad():
        logit = model(input_window)
        prob = torch.sigmoid(logit).item()
    return prob, "Seizure" if prob > 0.5 else "Normal"


def explain(model, input_window):
    """Returns (gradcam, saliency) from a single forward+backward pass.

    gradcam  — temporal attention heatmap (768,)
    saliency — signed attribution per channel x time (22, 768),
               positive values push toward seizure, negative away.
    """
    model.train()
    model.dropout.p = 0

    store = {}

    def forward_hook(module, input, output):
        store['activations'] = output

    def backward_hook(module, grad_input, grad_output):
        store['gradients'] = grad_output[0]

    fwd_handle = model.conv1.register_forward_hook(forward_hook)
    bwd_handle = model.conv1.register_full_backward_hook(backward_hook)

    input_window.requires_grad_(True)
    output = model(input_window)
    output.squeeze().backward()

    # ---- GradCAM ----
    grads = store['gradients'].cpu().detach().numpy()
    acts = store['activations'].cpu().detach().numpy()
    avg = np.mean(grads, axis=2)
    expanded = np.expand_dims(avg, axis=-1)
    w_sum = np.sum(expanded * acts, axis=1)
    cam = np.maximum(0, w_sum)
    cam_upsampled = np.interp(
        np.linspace(0, 1, 768),
        np.linspace(0, 1, 744),
        cam.squeeze()
    )

    # ---- Saliency (input gradients as signed attribution) ----
    saliency = input_window.grad.squeeze().cpu().detach().numpy()

    model.dropout.p = 0.15
    fwd_handle.remove()
    bwd_handle.remove()

    return cam_upsampled, saliency


def explain_gradcam(model, input_window):
    """Backwards-compatible wrapper returning only the GradCAM heatmap."""
    cam, _ = explain(model, input_window)
    return cam


def explain_saliency(model, input_window):
    """Lightweight signed attribution (replaces SHAP GradientExplainer)."""
    _, saliency = explain(model, input_window)
    return saliency


def plot_explanations(raw_window, cam, shap_sv, channel_names):
    """raw_window shape (22, 768), cam shape (768,), shap_sv shape (22, 768)"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    axes[0].plot(raw_window[0], alpha=0.5, label="EEG ch0")
    ax2 = axes[0].twinx()
    ax2.plot(cam, color='red', label="GradCAM")
    axes[0].set_title("GradCAM overlaid on raw EEG")

    channel_importance = np.abs(shap_sv).mean(axis=1)
    axes[1].bar(channel_names, channel_importance)
    axes[1].set_xticklabels(channel_names, rotation=45, ha='right', fontsize=8)
    axes[1].set_title("Channel Importance (mean |attribution|)")

    im = axes[2].imshow(shap_sv, aspect='auto', cmap='RdBu_r',
                        vmin=-np.abs(shap_sv).max(), vmax=np.abs(shap_sv).max())
    axes[2].set_yticks(range(22))
    axes[2].set_yticklabels(channel_names, fontsize=7)
    axes[2].set_title("Channel x Time Attribution")
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    return fig
