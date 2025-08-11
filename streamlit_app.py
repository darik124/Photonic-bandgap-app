import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import requests

# ---- tiny health helper (prevents blank page when backend is down) ----
def backend_ok(url: str) -> bool:
    if not url:
        return False
    try:
        r = requests.get(f"{url}/health", timeout=5)
        return r.ok
    except Exception:
        return False

# Read your backend URL from secrets (for local: BACKEND_URL = "http://localhost:8000")
BACKEND_URL = st.secrets.get("BACKEND_URL", "").rstrip("/")
API_URL = f"{BACKEND_URL}/bands" if BACKEND_URL else ""

st.title("Photonic Band Gap Visualizer (MPB-backed)")

# (optional) non-blocking banner if backend is not reachable yet
if not backend_ok(BACKEND_URL):
    st.info("Backend not reachable yet. Start the Docker backend, then try again.")

# =========================
# 1) Infinite crystal (MPB)
# =========================
epsilon   = st.slider("Dielectric Permittivity (ε)", 1.0, 15.0, 3.5, 0.1, key="mpb_eps")
r_over_a  = st.slider("Rod radius ratio r/a", 0.05, 0.45, 0.20, 0.01, key="mpb_r_over_a")
num_bands = st.slider("Bands", 4, 16, 8, 1, key="mpb_bands")
resolution = st.slider("Resolution", 16, 64, 32, 4, key="mpb_res")
lattice   = st.selectbox("Lattice", ["square", "triangular"], key="mpb_lattice")
kpts      = st.slider("k-points per segment", 8, 40, 16, 2, key="mpb_kpts")

if not API_URL:
    st.warning("Set BACKEND_URL in secrets to your FastAPI server (e.g., http://localhost:8000).")
else:
    if st.button("Compute bands", key="mpb_btn"):
        try:
            with st.spinner("Running MPB on backend…"):
                payload = {
                    "epsilon": epsilon,
                    "r_over_a": r_over_a,
                    "num_bands": num_bands,
                    "resolution": resolution,
                    "k_points_per_segment": kpts,
                    "lattice": lattice,
                }
                resp = requests.post(API_URL, json=payload, timeout=180)
                resp.raise_for_status()
                data = resp.json()

            freqs = np.array(data["frequencies"])  # shape (k_count, num_bands)
            k_count, nb = freqs.shape

            fig, ax = plt.subplots()
            for b in range(nb):
                ax.plot(range(k_count), freqs[:, b], lw=1)
            ticks = [0, k_count // 3, 2 * k_count // 3, k_count - 1]
            ax.set_xticks(ticks)
            ax.set_xticklabels(data["k_path_labels"])
            ax.set_ylabel("Normalized frequency (ωa/2πc)")
            ax.set_title("Band Structure (MPB)")
            ax.grid(True)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Bands error: {e}")

st.markdown("---")
st.header("Transmission (finite slab, Meep)")

# =========================
# 2) Finite slab (Meep)
# =========================
colA, colB = st.columns(2)
with colA:
    eps_tx    = st.number_input("Dielectric Permittivity (ε)", 1.1, 30.0, 3.5, 0.1, key="tx_eps")
    a_mm      = st.slider("Lattice constant a (mm)", 3.0, 15.0, 7.0, 0.1, key="tx_a_mm")
    r_over_a2 = st.slider("Rod radius ratio r/a", 0.02, 0.40, 0.16, 0.01, key="tx_r_over_a")
with colB:
    nx = st.slider("Rods along x", 4, 20, 10, 1, key="tx_nx")
    ny = st.slider("Rods along y", 4, 20, 8, 1, key="tx_ny")
    lattice_tx = st.selectbox("Lattice", ["square", "triangular"], index=0, key="tx_lattice")

colC, colD, colE = st.columns(3)
with colC:
    fmin = st.number_input("fmin (GHz)", 1.0, 60.0, 5.0, 0.5, key="tx_fmin")
with colD:
    fmax = st.number_input("fmax (GHz)", 1.0, 60.0, 35.0, 0.5, key="tx_fmax")
with colE:
    nfreq = st.slider("Points", 50, 600, 300, 10, key="tx_nfreq")

res_tx = st.slider("Resolution (px per a)", 8, 64, 24, 1,
                   help="Higher = more accurate but slower", key="tx_res")

if st.button("Compute Transmission", key="tx_btn"):
    payload = {
        "epsilon": float(eps_tx),
        "r_over_a": float(r_over_a2),
        "a_mm": float(a_mm),
        "nx": int(nx),
        "ny": int(ny),
        "lattice": lattice_tx,
        "resolution": int(res_tx),
        "fmin_GHz": float(fmin),
        "fmax_GHz": float(fmax),
        "nfreq": int(nfreq),
    }
    try:
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
    except Exception as e:
        st.error(f"Transmission error: {e}")

st.markdown("---")
st.header("Attenuation in Forbidden Band (Transmission vs Layers)")

# =========================
# 3) Attenuation vs layers
# =========================
col1, col2, col3 = st.columns(3)
with col1:
    att_eps = st.number_input("Dielectric Permittivity (ε)", 1.1, 30.0, 3.5, 0.1, key="att_eps")
with col2:
    att_a = st.slider("Lattice constant a (mm)", 3.0, 15.0, 7.0, 0.1, key="att_a")
with col3:
    att_r = st.slider("Rod radius ratio r/a", 0.02, 0.40, 0.16, 0.01, key="att_r")

col4, col5, col6 = st.columns(3)
with col4:
    att_ny = st.slider("Rods along y", 2, 30, 8, 1, key="att_ny")
with col5:
    att_nmax = st.slider("Max layers along x", 1, 30, 10, 1, key="att_nmax")
with col6:
    att_f0 = st.number_input("Probe frequency f0 (GHz)", 0.1, 100.0, 15.0, 0.1, key="att_f0")

att_res = st.slider("Resolution (px per a)", 8, 64, 24, 1, key="att_res")
att_lat = st.selectbox("Lattice", ["square", "triangular"], index=0, key="att_lat")

if st.button("Compute Attenuation", key="att_btn"):
    if not BACKEND_URL:
        st.warning('Set BACKEND_URL in .streamlit/secrets.toml (e.g., "http://localhost:8000").')
    else:
        payload = {
            "epsilon": float(att_eps),
            "r_over_a": float(att_r),
            "a_mm": float(att_a),
            "ny": int(att_ny),
            "nmax": int(att_nmax),
            "f0_GHz": float(att_f0),
            "lattice": att_lat,
            "resolution": int(att_res),
        }
        try:
            with st.spinner("Running attenuation sweep (Meep)…"):
                r = requests.post(f"{BACKEND_URL}/attenuation", json=payload, timeout=900)
                r.raise_for_status()
                data = r.json()

            layers = data["layers"]
            TdB = data["T_dB"]

            fig, ax = plt.subplots()
            ax.plot(layers, TdB, marker="o")
            ax.set_xlabel("Number of layers")
            ax.set_ylabel("Transmission (dB)")
            ax.set_title("Transmission vs Layers (at f0)")
            ax.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Attenuation error: {e}")
