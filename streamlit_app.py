# streamlit_app.py (snippet)
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import requests
import streamlit as st
backend_url = st.secrets["BACKEND_URL"]

API_URL = st.secrets.get("MPB_API_URL", "http://<YOUR-MPB-SERVER>:8000/bands")

st.title("Photonic Band Gap Visualizer (MPB-backed)")

epsilon = st.slider("Dielectric Permittivity (ε)", 1.0, 15.0, 3.5, 0.1)
r_over_a = st.slider("Rod radius ratio r/a", 0.05, 0.45, 0.20, 0.01)
num_bands = st.slider("Bands", 4, 16, 8, 1)
resolution = st.slider("Resolution", 16, 64, 32, 4)
lattice = st.selectbox("Lattice", ["square", "triangular"])
kpts = st.slider("k-points per segment", 8, 40, 16, 2)

if st.button("Compute"):
    with st.spinner("Running MPB…"):
        payload = {
            "epsilon": epsilon,
            "r_over_a": r_over_a,
            "num_bands": num_bands,
            "resolution": resolution,
            "k_points_per_segment": kpts,
            "lattice": lattice
        }
        r = requests.post(API_URL, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()

    freqs = np.array(data["frequencies"])   # shape (k_count, num_bands)
    k_count, nb = freqs.shape

    # Plot band structure
    fig, ax = plt.subplots()
    for b in range(nb):
        ax.plot(range(k_count), freqs[:, b], lw=1)
    ax.set_xticks([0, k_count//3, 2*k_count//3, k_count-1])
    ax.set_xticklabels(data["k_path_labels"])
    ax.set_ylabel("Normalized frequency (ωa/2πc)")
    ax.set_title("Band Structure (MPB)")
    ax.grid(True)
    st.pyplot(fig)
    
