import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

API_BASE_URL = "http://127.0.0.1:8000"

CHANNEL_NAMES = [
    'FP1-F7','F7-T7','T7-P7','P7-O1',
    'FP1-F3','F3-C3','C3-P3','P3-O1',
    'FP2-F4','F4-C4','C4-P4','P4-O2',
    'FP2-F8','F8-T8','T8-P8-0','P8-O2',
    'FZ-CZ','CZ-PZ','P7-T7','T7-FT9',
    'FT9-FT10','FT10-T8'
]

st.set_page_config(
    page_title="NeuroXAI",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 NeuroXAI - EEG Seizure Detection & Explainability")

st.markdown("""
Upload an EEG EDF file and let the model:

- Detect seizure activity
- Estimate confidence
- Highlight important EEG channels
- Explain predictions using SHAP
- Show model attention using GradCAM
""")

st.divider()

# =====================================================
# PREDICTION SECTION
# =====================================================

st.header("🔍 Single Window Analysis")

file_pred = st.file_uploader(
    "Upload EDF File",
    type=["edf"],
    key="predict_file"
)

time_seconds = st.number_input(
    "Analysis Second (Optional)",
    min_value=0,
    value=0,
    step=1
)

if st.button("Run Prediction", type="primary"):

    if file_pred is None:
        st.error("Please upload an EDF file.")
    else:
        try:

            with st.spinner("Running NeuroXAI Analysis..."):

                files = {
                    "file": (
                        file_pred.name,
                        file_pred.getvalue(),
                        file_pred.type
                    )
                }

                params = {}

                if time_seconds > 0:
                    params["second"] = int(time_seconds)

                response = requests.post(
                    f"{API_BASE_URL}/predict",
                    files=files,
                    params=params
                )

                if response.status_code != 200:
                    st.error(response.text)
                    st.stop()

                result = response.json()
                

            prediction = result["prediction"]
            probability = float(result["probability"])
            explanation = result["explanation"]

            gradcam = np.array(result["gradcam"])
            shap_values = np.array(result["shap_values"])
            channel_importance = np.array(result["channel_importance"])

            # ======================================
            # SUMMARY
            # ======================================

            st.success("Analysis Completed Successfully")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Prediction",
                    prediction
                )

            with col2:
                st.metric(
                    "Model prediction of seizure",
                    f"{probability*100:.2f}%"
                )

            st.progress(min(max(probability, 0.0), 1.0))

            st.divider()

            # ======================================
            # HUMAN REPORT
            # ======================================

            st.subheader("📋 AI Generated Report")

            if prediction == "Seizure":
                st.error(explanation)
            else:
                st.success(explanation)

            top_idx = np.argsort(channel_importance)[::-1][:3]

            st.markdown("### Key EEG Regions")

            st.write(
                f"🥇 **{CHANNEL_NAMES[top_idx[0]]}**"
            )

            st.write(
                f"🥈 **{CHANNEL_NAMES[top_idx[1]]}**"
            )

            st.write(
                f"🥉 **{CHANNEL_NAMES[top_idx[2]]}**"
            )

            st.divider()

            # ======================================
            # GRADCAM
            # ======================================

            st.subheader("🔥 GradCAM Temporal Attention")

            fig, ax = plt.subplots(figsize=(12, 4))

            time_axis = np.arange(len(gradcam)) / 256

            ax.plot(time_axis, gradcam)

            ax.set_title(
                "Model Attention Across EEG Window"
            )

            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Attention Score")

            st.pyplot(fig)

            # ======================================
            # SHAP CHANNEL IMPORTANCE
            # ======================================

            st.subheader("📊 SHAP Channel Importance")

            importance_df = pd.DataFrame({
                "Channel": CHANNEL_NAMES,
                "Importance": channel_importance
            })

            importance_df = importance_df.sort_values(
                by="Importance",
                ascending=False
            )

            fig, ax = plt.subplots(figsize=(12, 5))

            ax.bar(
                importance_df["Channel"],
                importance_df["Importance"]
            )

            ax.set_title(
                "Most Influential EEG Channels"
            )

            ax.set_ylabel(
                "Mean Absolute SHAP Value"
            )

            plt.xticks(rotation=60)

            st.pyplot(fig)

            st.dataframe(
                importance_df,
                use_container_width=True
            )

            # ======================================
            # SHAP HEATMAP
            # ======================================

            st.subheader("🌡 SHAP Attribution Heatmap")

            fig, ax = plt.subplots(figsize=(15, 8))

            vmax = np.abs(shap_values).max()

            im = ax.imshow(
                shap_values,
                aspect="auto",
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax
            )

            ax.set_title(
                "Channel × Time Attribution"
            )

            ax.set_xlabel("Time Samples")
            ax.set_ylabel("EEG Channels")

            ax.set_yticks(range(len(CHANNEL_NAMES)))
            ax.set_yticklabels(CHANNEL_NAMES)

            plt.colorbar(
                im,
                ax=ax,
                label="SHAP Value"
            )

            st.pyplot(fig)

            # ======================================
            # CLINICAL REPORT
            # ======================================

            st.subheader("🩺 Clinical Interpretation")

            if prediction == "Seizure":

                report = f"""
The model detected seizure-like activity with a confidence of
{probability*100:.2f}%.

The most influential EEG channels were:

• {CHANNEL_NAMES[top_idx[0]]}
• {CHANNEL_NAMES[top_idx[1]]}
• {CHANNEL_NAMES[top_idx[2]]}

GradCAM highlights the temporal region within the EEG window
that contributed most strongly to the prediction.

SHAP values indicate which EEG channels and time points
were responsible for the model's decision.
"""

                st.warning(report)

            else:

                report = f"""
The model classified this EEG segment as normal with
{probability*100:.2f}% confidence.

No strong seizure-like activity was identified in the selected
window.

GradCAM and SHAP visualizations are displayed for
model transparency and interpretability.
"""

                st.info(report)

        except Exception as e:
            st.error(f"Error: {str(e)}")

st.divider()

# =====================================================
# FULL FILE SCAN
# =====================================================

st.header("📡 Full EEG Scan")

file_scan = st.file_uploader(
    "Upload EDF File For Full Scan",
    type=["edf"],
    key="scan_file"
)

if st.button("Run Full Scan"):

    if file_scan is None:
        st.error("Please upload an EDF file.")
    else:

        try:

            with st.spinner("Scanning Entire EEG Recording..."):

                files = {
                    "file": (
                        file_scan.name,
                        file_scan.getvalue(),
                        file_scan.type
                    )
                }

                response = requests.post(
                    f"{API_BASE_URL}/scan",
                    files=files
                )

                if response.status_code != 200:
                    st.error(response.text)
                    st.stop()

                result = response.json()

            st.success("Scan Completed")

            st.metric(
                "Total Seconds Scanned",
                result["total_seconds_scanned"]
            )

            st.metric(
                "Seizure Events Detected",
                result["seizure_events_detected"]
            )

            st.info(result["explanation"])

            events = result["events"]

            if len(events) > 0:

                st.subheader("Detected Events")

                df = pd.DataFrame(events)

                st.dataframe(
                    df,
                    use_container_width=True
                )

            else:
                st.success(
                    "No seizure events detected."
                )

        except Exception as e:
            st.error(str(e))