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

# notice if backend is down (non-blocking)
if not backend_ok(BACKEND_URL):
    st.info("Backend not reachable yet. Start the FastAPI server, then try again.")

# ensure state slots so plots persist across reruns
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
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

st.markdown("---")
st.header("Transmission (finite slab, Meep)")

# --- upload measured data (freq GHz, dB) ---
meas = None
meas_file = st.file_uploader(
    "Upload measured transmission CSV (two columns: Frequency_GHz, Transmission_dB)",
    type=["csv"],
    help="No header or with header is fine. We will autodetect the first numeric two columns."
)
if meas_file is not None:
    import pandas as pd
    df = pd.read_csv(meas_file)
    # try to pick the first two numeric columns
    numcols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    if len(numcols) >= 2:
        fx = df[numcols[0]].to_numpy(dtype=float)
        ty = df[numcols[1]].to_numpy(dtype=float)
        # simple cleanup
        ok = np.isfinite(fx) & np.isfinite(ty)
        meas = (fx[ok], ty[ok])
    else:
        st.error("Could not find two numeric columns in the CSV.")

colA, colB = st.columns(2)
with colA:
    eps_tx    = st.number_input("Dielectric Permittivity (ε)", 1.1, 30.0, 3.5, 0.1, key="tx_eps")
    a_mm      = st.slider("Lattice constant a (mm)", 3.0, 15.0, 7.0, 0.1, key="tx_a_mm")
    r_over_a2 = st.slider("Rod radius ratio r/a", 0.02, 0.40, 0.16, 0.01, key="tx_r_over_a")
with colB:
    nx = st.slider("Rods along x", 4, 120, 20, 1, key="tx_nx")
    ny = st.slider("Rods along y", 4, 60, 12, 1, key="tx_ny")
    lattice_tx = st.selectbox("Lattice", ["square", "triangular"], index=0, key="tx_lattice")

# Y boundary selector (2D periodic vs finite-height slab)
y_bound = st.selectbox(
    "Y boundary",
    ["periodic", "pml"],
    index=0,
    help="periodic = infinite height (2D Bloch k=0, closer to HFSS); pml = finite-height slab",
    key="tx_ybound"
)

colC, colD, colE = st.columns(3)
with colC:
    fmin = st.number_input("fmin (GHz)", 1.0, 60.0, 5.0, 0.5, key="tx_fmin")
with colD:
    fmax = st.number_input("fmax (GHz)", 1.0, 60.0, 35.0, 0.5, key="tx_fmax")
with colE:
    nfreq = st.slider("Points", 50, 1000, 400, 10, key="tx_nfreq")

# Depth booster (more accurate numerics)
depth_boost = st.toggle(
    "Depth booster (thicker PML, more padding, higher resolution)",
    value=True,
    help="Helps reach ≤ −12 dB like the poster notch, at the cost of runtime."
)

res_default = 40 if depth_boost else 24
res_tx = st.slider(
    "Resolution (px per a)", 8, 96, res_default, 1,
    help="Try 40–64 for deeper stop-bands; increase nx to 24–30 for even deeper.",
    key="tx_res"
)

# Preset that tends to produce the poster-like notch
if st.button("Load poster-like preset", use_container_width=True):
    st.session_state.update({
        "tx_eps": 3.5,        # change to ~9.6 if these were alumina rods
        "tx_a_mm": 7.0,
        "tx_r_over_a": 0.16,
        "tx_nx": 24,
        "tx_ny": 12,
        "tx_lattice": "square",
        "tx_ybound": "periodic",
        "tx_fmin": 5.0,
        "tx_fmax": 35.0,
        "tx_nfreq": 500,
        "tx_res": 48,
    })
    st.experimental_rerun()

# --- compute transmission ---
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
                "nx": int(nx if not depth_boost else max(nx, 24)),
                "ny": int(ny),
                "lattice": lattice_tx,
                "resolution": int(max(res_tx, 32 if depth_boost else res_tx)),
                "fmin_GHz": float(fmin),
                "fmax_GHz": float(fmax),
                "nfreq": int(max(nfreq, 400 if depth_boost else nfreq)),
                "y_boundary": y_bound,
            }
            try:
                with st.spinner("Running Meep…"):
                    r = requests.post(API_TX, json=payload, timeout=900)
                    r.raise_for_status()
                    st.session_state["tx_data"] = r.json()
            except Exception as e:
                st.error(f"Transmission error: {e}")

# --- coarse auto-fit to measured data (optional) ---
def _rmse(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    if y_true.size == 0 or y_pred.size == 0:
        return np.inf
    n = min(y_true.size, y_pred.size)
    return np.sqrt(np.mean((y_true[:n] - y_pred[:n])**2))

if meas is not None:
    st.subheader("Auto-fit to measurement (coarse)")
    c1, c2, c3 = st.columns(3)
    with c1:
        eps_lo = st.number_input("ε min", 1.1, 30.0, 3.0, 0.1, key="fit_eps_lo")
        eps_hi = st.number_input("ε max", 1.1, 30.0, 4.0, 0.1, key="fit_eps_hi")
        eps_steps = st.slider("ε steps", 2, 12, 5, 1, key="fit_eps_steps")
    with c2:
        ra_lo = st.number_input("r/a min", 0.02, 0.40, 0.14, 0.01, key="fit_ra_lo")
        ra_hi = st.number_input("r/a max", 0.02, 0.40, 0.18, 0.01, key="fit_ra_hi")
        ra_steps = st.slider("r/a steps", 2, 12, 5, 1, key="fit_ra_steps")
    with c3:
        a_lo = st.number_input("a (mm) min", 3.0, 15.0, 6.5, 0.1, key="fit_a_lo")
        a_hi = st.number_input("a (mm) max", 3.0, 15.0, 7.5, 0.1, key="fit_a_hi")
        a_steps = st.slider("a steps", 2, 12, 5, 1, key="fit_a_steps")

    if st.button("Run coarse fit (grid search)", use_container_width=True):
        f_meas, y_meas = meas
        # set search grid
        e_grid  = np.linspace(eps_lo, eps_hi, eps_steps)
        ra_grid = np.linspace(ra_lo,  ra_hi,  ra_steps)
        a_grid  = np.linspace(a_lo,   a_hi,   a_steps)

        best = {"rmse": np.inf}
        with st.spinner("Searching…"):
            for e in e_grid:
                for ra in ra_grid:
                    for aa in a_grid:
                        payload = {
                            "epsilon": float(e),
                            "r_over_a": float(ra),
                            "a_mm": float(aa),
                            "nx": int(max(nx, 20)),
                            "ny": int(ny),
                            "lattice": lattice_tx,
                            "resolution": int(max(res_tx, 32)),
                            "fmin_GHz": float(fmin),
                            "fmax_GHz": float(fmax),
                            "nfreq": int(max(nfreq, 300)),
                            "y_boundary": y_bound,
                        }
                        try:
                            r = requests.post(API_TX, json=payload, timeout=900)
                            r.raise_for_status()
                            d = r.json()
                            fx = np.array(d.get("freq_GHz") or d.get("frequency_GHz") or [])
                            ty = np.array(d.get("trans_dB") or d.get("transmission_dB") or [])
                            # interpolate sim onto measured frequencies
                            if fx.size > 2 and ty.size == fx.size:
                                sim_on_meas = np.interp(f_meas, fx, ty)
                                err = _rmse(y_meas, sim_on_meas)
                                if err < best["rmse"]:
                                    best = {"rmse": err, "eps": e, "ra": ra, "a": aa, "fx": fx, "ty": ty}
                        except Exception:
                            pass
        if best["rmse"] < np.inf:
            st.success(f"Best fit → ε={best['eps']:.3g}, r/a={best['ra']:.3g}, a={best['a']:.3g} mm (RMSE={best['rmse']:.2f} dB)")
            # store best curve
            st.session_state["tx_data"] = {"freq_GHz": best["fx"].tolist(), "trans_dB": best["ty"].tolist()}
        else:
            st.warning("Fit did not converge. Expand ranges/steps or increase resolution a bit.")

# --- plotting ---
if st.session_state.get("tx_data"):
    d = st.session_state["tx_data"]
    freq = np.array(d.get("freq_GHz") or d.get("frequency_GHz") or [])
    trdb = np.array(d.get("trans_dB") or d.get("transmission_dB") or [])

    # optional smoothing (simple moving average)
    smooth = st.toggle("Smooth measured curve (moving avg)", value=True)
    win = st.slider("Smoothing window (points)", 1, 31, 9, 2) if smooth and (meas is not None) else 1

    fig, ax = plt.subplots()
    # sim
    if freq.size and trdb.size:
        ax.plot(freq, trdb, lw=2.0, label="Simulation (Meep)")

    # measurement overlay
    if meas is not None:
        fx, ty = meas
        if smooth and win > 1:
            k = max(1, int(win))
            kernel = np.ones(k) / k
            ty = np.convolve(ty, kernel, mode="same")
        ax.plot(fx, ty, lw=1.2, alpha=0.85, label="Measure")

    ax.set_xlim(min(fmin, 5.0), max(fmax, 30.0))
    ax.set_ylim(-25, 1)              # match poster range
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Transmission (dB)")
    ax.set_title("Transmission Diagram (finite slab)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
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
    layers = data.get("layers", [])
    # prefer attenuation (positive), fall back to transmission dB
    y = data.get("atten_dB") or data.get("T_dB") or []
    ylabel = "Attenuation (dB)" if "atten_dB" in data else "Transmission (dB)"

    fig, ax = plt.subplots()
    ax.plot(layers, y, marker="o")
    ax.set_xlabel("Number of layers")
    ax.set_ylabel(ylabel)
    ax.set_title("Transmission vs Layers (at f₀)")
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)
