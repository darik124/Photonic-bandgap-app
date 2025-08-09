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
import requests
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# You already have: BACKEND_URL from secrets.toml

st.markdown("---")
st.header("Transmission (finite slab, Meep)")

colA, colB = st.columns(2)

with colA:
    eps_tx = st.number_input("Dielectric Permittivity (ε)", 1.1, 30.0, 3.5, 0.1)
    a_mm = st.slider("Lattice constant a (mm)", 3.0, 15.0, 7.0, 0.1)
    r_over_a = st.slider("Rod radius ratio r/a", 0.02, 0.40, 0.16, 0.01)

with colB:
    nx = st.slider("Rods along x", 4, 20, 10, 1)
    ny = st.slider("Rods along y", 4, 20, 8, 1)
    lattice_tx = st.selectbox("Lattice", ["square", "triangular"], index=0)

colC, colD, colE = st.columns(3)
with colC:
    fmin = st.number_input("fmin (GHz)", 1.0, 60.0, 5.0, 0.5)
with colD:
    fmax = st.number_input("fmax (GHz)", 1.0, 60.0, 35.0, 0.5)
with colE:
    nfreq = st.slider("Points", 50, 600, 300, 10)

res_tx = st.slider("Resolution (px per a)", 8, 64, 24, 1,
                   help="Higher = more accurate but slower")

if st.button("Compute Transmission"):
    payload = {
        "epsilon": float(eps_tx),
        "r_over_a": float(r_over_a),
        "a_mm": float(a_mm),
        "nx": int(nx),
        "ny": int(ny),
        "lattice": lattice_tx,
        "resolution": int(res_tx),
        "fmin_GHz": float(fmin),
        "fmax_GHz": float(fmax),
        "nfreq": int(nfreq)
    }
    with st.spinner("Running Meep (this can take ~10–60s)…"):
        r = requests.post(f"{BACKEND_URL}/transmission", json=payload, timeout=900)
        r.raise_for_status()
        data = r.json()

    fig, ax = plt.subplots()
    ax.plot(data["frequency_GHz"], data["transmission_dB"])
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Transmission (dB)")
    ax.set_title("Transmission Diagram (finite slab)")
    ax.grid(True)
    st.pyplot(fig)
