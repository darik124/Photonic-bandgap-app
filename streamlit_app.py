import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import requests

# ---------- tiny health helper ----------
def backend_ok(url: str) -> bool:
    if not url:
        return False
    try:
        r = requests.get(f"{url}/health", timeout=5)
        return r.ok
    except Exception:
        return False

# ---------- config ----------
BACKEND_URL = st.secrets.get("BACKEND_URL", "").rstrip("/")
API_BANDS = f"{BACKEND_URL}/bands" if BACKEND_URL else None
API_TX    = f"{BACKEND_URL}/transmission" if BACKEND_URL else None
API_ATT   = f"{BACKEND_URL}/attenuation" if BACKEND_URL else None

st.set_page_config(page_title="Photonic Band Gap Visualizer", layout="centered")
st.title("Photonic Band Gap Visualizer (MPB-backed)")

# show a non-blocking notice if backend is down
if not backend_ok(BACKEND_URL):
    st.info("Backend not reachable yet. Start the Docker backend, then try again.")

# Ensure session slots exist so plots persist across reruns
for k in ("bands_data", "tx_data", "att_data"):
    st.session_state.setdefault(k, None)

# =========================================================
# 1) Infinite crystal (MPB band structure)
# =========================================================
epsilon   = st.slider("Dielectric Permittivity (ε)", 1.0, 15.0, 3.5, 0.1, key="mpb_eps")
r_over_a  = st.slider("Rod radius ratio r/a", 0.05, 0.45, 0.20, 0.01, key="mpb_r_over_a")
num_bands = st.slider("Bands", 4, 16, 8, 1, key="mpb_bands")
resolution = st.slider("Resolution", 16, 64, 32, 4, key="mpb_res")
lattice   = st.selectbox("Lattice", ["square", "triangular"], key="mpb_lattice")
kpts      = st.slider("k-points per segment", 8, 40, 16, 2, key="mpb_kpts")

col_mpb_btn = st.columns([1,3])[0]
with col_mpb_btn:
    if st.button("Compute bands", key="mpb_btn", use_container_width=True):
        if not API_BANDS:
            st.warning("Set BACKEND_URL in secrets (e.g., http://localhost:8000).")
        else:
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
                    resp = requests.post(API_BANDS, json=payload, timeout=180)
                    resp.raise_for_status()
                    st.session_state["bands_data"] = resp.json()
            except Exception as e:
                st.error(f"Bands error: {e}")

# Render MPB plot if we have data
if st.session_state["bands_data"]:
    data = st.session_state["bands_data"]
    freqs = np.array(data["frequencies"])  # (k_count, num_bands)
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

st.markdown("---")
st.header("Transmission (finite slab, Meep)")

# =========================================================
# 2) Finite slab (Meep transmission)
# =========================================================
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

res_tx = st.slider(
    "Resolution (px per a)", 8, 64, 24, 1,
    help="Higher = more accurate but slower", key="tx_res"
)

col_tx_btn = st.columns([1,3])[0]
with col_tx_btn:
    if st.button("Compute Transmission", key="tx_btn", use_container_width=True):
        if not API_TX:
            st.warning("Set BACKEND_URL in secrets (e.g., http://localhost:8000).")
        else:
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
                    r = requests.post(API_TX, json=payload, timeout=900)
                    r.raise_for_status()
                    st.session_state["tx_data"] = r.json()
            except Exception as e:
                st.error(f"Transmission error: {e}")

# Render transmission plot if we have data
if st.session_state["tx_data"]:
    data = st.session_state["tx_data"]
    fig, ax = plt.subplots()
    ax.plot(data["frequency_GHz"], data["transmission_dB"])
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Transmission (dB)")
    ax.set_title("Transmission Diagram (finite slab)")
    ax.grid(True)
    st.pyplot(fig)

st.markdown("---")
st.header("Attenuation in Forbidden Band (Transmission vs Layers)")

# =========================================================
# 3) Attenuation vs layers (Meep)
# =========================================================
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

col_att_btn = st.columns([1,3])[0]
with col_att_btn:
    if st.button("Compute Attenuation", key="att_btn", use_container_width=True):
        if not API_ATT:
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
                    r = requests.post(API_ATT, json=payload, timeout=900)
                    r.raise_for_status()
                    st.session_state["att_data"] = r.json()
            except Exception as e:
                st.error(f"Attenuation error: {e}")

# Render attenuation plot if we have data
if st.session_state["att_data"]:
    data = st.session_state["att_data"]
    layers = data.get("layers") or [d["layers"] for d in data.get("attenuation_data", [])]
    TdB    = data.get("T_dB")   or [d["transmission"] for d in data.get("attenuation_data", [])]

    fig, ax = plt.subplots()
    ax.plot(layers, TdB, marker="o")
    ax.set_xlabel("Number of layers")
    ax.set_ylabel("Transmission (dB)")
    ax.set_title("Transmission vs Layers (at f0)")
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)
