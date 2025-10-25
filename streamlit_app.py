import streamlit as st
import numpy as np
import pandas as pd
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
API_CAL   = f"{BACKEND_URL}/calibrate" if BACKEND_URL else None

st.set_page_config(page_title="Photonic Band Gap Visualizer", layout="centered")
st.title("Photonic Band Gap Visualizer (MPB-backed)")

# show a non-blocking notice if backend is down
if not backend_ok(BACKEND_URL):
    st.info("Backend not reachable yet. Start the Docker backend, then try again.")

# Ensure session slots exist so plots persist across reruns
for k in ("bands_data", "tx_data", "att_data", "calib_overlay"):
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
    ax.grid(True, alpha=0.3)
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
    nx = st.slider("Rods along x", 4, 40, 10, 1, key="tx_nx")
    ny = st.slider("Rods along y", 4, 40, 8, 1, key="tx_ny")
    lattice_tx = st.selectbox("Lattice", ["square", "triangular"], index=0, key="tx_lattice")

colC, colD, colE = st.columns(3)
with colC:
    fmin = st.number_input("fmin (GHz)", 1.0, 60.0, 5.0, 0.5, key="tx_fmin")
with colD:
    fmax = st.number_input("fmax (GHz)", 1.0, 60.0, 35.0, 0.5, key="tx_fmax")
with colE:
    nfreq = st.slider("Points", 50, 1000, 300, 10, key="tx_nfreq")

res_tx = st.slider(
    "Resolution (px per a)", 8, 96, 24, 1,
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
    # backend returns freq_GHz + trans_dB
    ax.plot(data["freq_GHz"], data["trans_dB"], lw=1.5, label="Simulation (Meep)")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Transmission (dB)")
    ax.set_title("Transmission Diagram (finite slab)")
    ax.grid(True, linestyle="--", alpha=0.4)

    # If a calibration overlay exists, draw it
    if st.session_state["calib_overlay"] is not None:
        ov = st.session_state["calib_overlay"]
        if "ref_df" in ov:
            ax.plot(ov["ref_df"]["freq_GHz"], ov["ref_df"]["trans_dB"], alpha=0.8, label="Reference (experimental)")
        if "sim_df" in ov:
            ax.plot(ov["sim_df"]["freq_GHz"], ov["sim_df"]["sim_dB"], alpha=0.9, label="Best-fit (Meep)")
        ax.legend()

    st.pyplot(fig)

# =========================================================
# 2b) Calibration to reference dataset (new)
# =========================================================
st.subheader("Match to reference (experimental)")

uploaded = st.file_uploader("Upload CSV with columns: freq_GHz, trans_dB", type=["csv"])
if uploaded is not None:
    df_ref = pd.read_csv(uploaded)
    # sanitize
    df_ref = df_ref.dropna()[["freq_GHz", "trans_dB"]].sort_values("freq_GHz")
    st.caption(f"Loaded {len(df_ref)} reference points.")
    st.line_chart(df_ref.set_index("freq_GHz"))

    with st.expander("Calibration search settings", expanded=False):
        a_min = st.number_input("a_min (mm)", value=7.0)
        a_max = st.number_input("a_max (mm)", value=12.0)
        a_steps = st.number_input("a_steps", value=11, step=1)
        nx_min = st.number_input("nx_min", value=8, step=1)
        nx_max = st.number_input("nx_max", value=28, step=1)
        nx_step = st.number_input("nx_step", value=4, step=1)
        calib_res = st.number_input("calibration resolution (px per a)", value=28, step=2)
        pts = st.number_input("points per sweep", value=400, step=100)
        ny_cal = st.number_input("ny (rods along y)", value=8, step=1)

    do_cal = st.button("Calibrate to dataset")
    if do_cal:
        if not API_CAL:
            st.warning("Set BACKEND_URL in secrets (e.g., http://localhost:8000).")
        else:
            payload = {
                "epsilon": float(eps_tx),
                "r_over_a": float(r_over_a2),
                "lattice": lattice_tx,
                "ny": int(ny_cal),
                "a_min_mm": float(a_min),
                "a_max_mm": float(a_max),
                "a_steps": int(a_steps),
                "nx_min": int(nx_min),
                "nx_max": int(nx_max),
                "nx_step": int(nx_step),
                "calib_resolution": int(calib_res),
                "points": int(pts),
                "reference": df_ref.to_dict(orient="records"),
            }
            try:
                with st.spinner("Searching best a, nx to fit reference…"):
                    r = requests.post(API_CAL, json=payload, timeout=1200)
                    r.raise_for_status()
                    res = r.json()
                    st.success(f"Best fit: a = {res['a_mm']:.2f} mm, nx = {res['nx']}  (MSE = {res['mse']:.4f})")

                    # overlay storage for the main plot
                    sim_df = pd.DataFrame({"freq_GHz": res["freq_GHz"], "sim_dB": res["sim_dB"]})
                    st.session_state["calib_overlay"] = {"ref_df": df_ref, "sim_df": sim_df}

                    # optionally push into the sliders for a one-click re-run at high res
                    st.session_state["tx_a_mm"] = float(res["a_mm"])
                    st.session_state["tx_nx"] = int(res["nx"])

            except Exception as e:
                st.error(f"Calibration error: {e}")

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
    att_ny = st.slider("Rods along y", 2, 60, 8, 1, key="att_ny")
with col5:
    att_nmax = st.slider("Max layers along x", 1, 60, 10, 1, key="att_nmax")
with col6:
    att_f0 = st.number_input("Probe frequency f0 (GHz)", 0.1, 100.0, 15.0, 0.1, key="att_f0")

att_res = st.slider("Resolution (px per a)", 8, 96, 24, 1, key="att_res")
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
