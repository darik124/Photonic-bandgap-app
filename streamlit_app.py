import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import requests

# Read your backend URL from secrets (set this in Streamlit Cloud → Settings → Secrets)
# For local testing, secrets.toml can contain: BACKEND_URL = "http://localhost:8000"
BACKEND_URL = st.secrets.get("BACKEND_URL", "").rstrip("/")
API_URL = f"{BACKEND_URL}/bands" if BACKEND_URL else ""

st.title("Photonic Band Gap Visualizer (MPB-backed)")

epsilon = st.slider("Dielectric Permittivity (ε)", 1.0, 15.0, 3.5, 0.1)
r_over_a = st.slider("Rod radius ratio r/a", 0.05, 0.45, 0.20, 0.01)
num_bands = st.slider("Bands", 4, 16, 8, 1)
resolution = st.slider("Resolution", 16, 64, 32, 4)
lattice = st.selectbox("Lattice", ["square", "triangular"])
kpts = st.slider("k-points per segment", 8, 40, 16, 2)

if not API_URL:
    st.warning("Set BACKEND_URL in secrets to your FastAPI server (e.g., https://your-domain:8000).")
else:
    if st.button("Compute"):
        with st.spinner("Running MPB on backend…"):
            payload = {
                "epsilon": epsilon,
                "r_over_a": r_over_a,
                "num_bands": num_bands,
                "resolution": resolution,
                "k_points_per_segment": kpts,
                "lattice": lattice
            }
            resp = requests.post(API_URL, json=payload, timeout=180)
            resp.raise_for_status()
            data = resp.json()

        freqs = np.array(data["frequencies"])  # shape (k_count, num_bands)
        k_count, nb = freqs.shape

        fig, ax = plt.subplots()
        for b in range(nb):
            ax.plot(range(k_count), freqs[:, b], lw=1)
        ticks = [0, k_count//3, 2*k_count//3, k_count-1]
        ax.set_xticks(ticks)
        ax.set_xticklabels(data["k_path_labels"])
        ax.set_ylabel("Normalized frequency (ωa/2πc)")
        ax.set_title("Band Structure (MPB)")
        ax.grid(True)
        st.pyplot(fig)
