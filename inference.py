import numpy as np
import torch
import shap
import matplotlib.pyplot as plt


def predict(model, input_window):
    """Returns probability and label for one window."""
    model.eval()
    with torch.no_grad():
        logit = model(input_window)
        prob = torch.sigmoid(logit).item()
    return prob, "Seizure" if prob > 0.5 else "Normal"


def explain_gradcam(model, input_window):
    """Returns upsampled GradCAM heatmap for one window."""
    store = {}
    def forward_hook(module, input, output):
        store['activations'] = output
    def backward_hook(module, grad_input, grad_output):
        store['gradients'] = grad_output[0]
        
    model.train()
    model.dropout.p = 0
    fwd_handle = model.conv1.register_forward_hook(forward_hook)
    bwd_handle = model.conv1.register_full_backward_hook(backward_hook)

    output = model(input_window)
    output.squeeze().backward()
    
    avg = np.mean(store['gradients'].cpu().detach().numpy(), axis=2)
    expanded = np.expand_dims(avg, axis=-1)
    
    w_sum = np.sum(expanded * store['activations'].cpu().detach().numpy(), axis=1)
    cam = np.maximum(0, w_sum)
    cam_upsampled = np.interp(
        np.linspace(0, 1, 768),
        np.linspace(0, 1, 744),
        cam.squeeze()
    )
    model.dropout.p = 0.15 
    fwd_handle.remove()
    bwd_handle.remove()
    return cam_upsampled
    


def explain_shap(model, input_window, background_tensor):
    """Returns SHAP values (22, 768) for one window."""
    model.train()
    model.dropout.p = 0
    with torch.backends.cudnn.flags(enabled=False):
        explainer = shap.GradientExplainer(model, background_tensor)
        shap_values = explainer.shap_values(input_window)
    model.dropout.p = 0.15
    sv = shap_values[0] if isinstance(shap_values, list) else shap_values
    return sv.squeeze()  # shape (22, 768)

def plot_explanations(raw_window, cam, shap_sv, channel_names):
    """raw_window shape (22, 768), cam shape (768,), shap_sv shape (22, 768)"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # GradCAM overlay
    axes[0].plot(raw_window[0], alpha=0.5, label="EEG ch0")
    ax2 = axes[0].twinx()
    ax2.plot(cam, color='red', label="GradCAM")
    axes[0].set_title("GradCAM overlaid on raw EEG")
    
    # SHAP channel importance bar
    channel_importance = np.abs(shap_sv).mean(axis=1)
    axes[1].bar(channel_names, channel_importance)
    axes[1].set_xticklabels(channel_names, rotation=45, ha='right', fontsize=8)
    axes[1].set_title("SHAP — Channel Importance")
    
    # SHAP heatmap
    im = axes[2].imshow(shap_sv, aspect='auto', cmap='RdBu_r',
                        vmin=-np.abs(shap_sv).max(), vmax=np.abs(shap_sv).max())
    axes[2].set_yticks(range(22))
    axes[2].set_yticklabels(channel_names, fontsize=7)
    axes[2].set_title("SHAP — Channel × Time Attribution")
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    return fig