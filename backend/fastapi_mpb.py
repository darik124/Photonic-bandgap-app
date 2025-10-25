# backend/fastapi_mpb.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import meep as mp
import meep.mpb as mpb

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- health ----------
@app.get("/health")
def health():
    return {"ok": True}

# ---------- helpers ----------
C0 = 299_792_458.0

def _a_from_mm(a_mm: float) -> float:
    return a_mm * 1e-3  # meters

def _GHz_to_meep(f_GHz: float, a_m: float) -> float:
    f_Hz = f_GHz * 1e9
    return (a_m * f_Hz) / C0  # dimensionless f = a/λ

def _mse_against_reference(ref_xy: np.ndarray, sim_f: np.ndarray, sim_db: np.ndarray) -> float:
    sim_interp = np.interp(ref_xy[:, 0], sim_f, sim_db, left=np.nan, right=np.nan)
    m = ~np.isnan(sim_interp)
    if np.count_nonzero(m) < max(10, int(0.2 * len(ref_xy))):
        return np.inf
    return float(np.mean((sim_interp[m] - ref_xy[:, 1][m]) ** 2))

# ==========================================================
# 1) Infinite crystal (MPB band structure)
# ==========================================================
class BandInput(BaseModel):
    epsilon: float
    r_over_a: float
    num_bands: int = 8
    resolution: int = 32
    k_points_per_segment: int = 16
    lattice: str = "square"  # "square" | "triangular"

@app.post("/bands")
def compute_bands(inp: BandInput):
    if inp.lattice == "square":
        geometry_lattice = mp.Lattice(size=mp.Vector3(1, 1))
        G = mp.Vector3(0, 0); X = mp.Vector3(0.5, 0); M = mp.Vector3(0.5, 0.5)
        k_points = mp.interpolate(inp.k_points_per_segment, [G, X, M, G])
        labels = ["Γ", "X", "M", "Γ"]
    elif inp.lattice == "triangular":
        geometry_lattice = mp.Lattice(
            size=mp.Vector3(1, 1),
            basis1=mp.Vector3(1, 0),
            basis2=mp.Vector3(0.5, np.sqrt(3)/2),
        )
        G = mp.Vector3()
        M = mp.Vector3(0.5, 0.5/np.sqrt(3))
        K = mp.Vector3(1/3, 1/(3*np.sqrt(3)))
        k_points = mp.interpolate(inp.k_points_per_segment, [G, M, K, G])
        labels = ["Γ", "M", "K", "Γ"]
    else:
        return {"error": "unknown lattice"}

    radius = inp.r_over_a * 0.5  # normalized (a=1)
    geometry = [mp.Cylinder(radius=radius, height=mp.inf, material=mp.Medium(epsilon=inp.epsilon))]

    ms = mpb.ModeSolver(
        num_bands=inp.num_bands,
        k_points=k_points,
        geometry_lattice=geometry_lattice,
        geometry=geometry,
        resolution=inp.resolution,
        default_material=mp.Medium(epsilon=1.0),
    )
    ms.run_tm()  # TM/Ez
    freqs = np.asarray(ms.all_freqs).tolist()
    return {"k_path_labels": labels, "frequencies": freqs}

# ==========================================================
# 2) Finite crystal (Meep transmission spectrum)
# ==========================================================
class TxInput(BaseModel):
    epsilon: float
    r_over_a: float
    a_mm: float
    nx: int = 10
    ny: int = 8
    lattice: str = "square"
    resolution: int = 24
    fmin_GHz: float = 5.0
    fmax_GHz: float = 35.0
    nfreq: int = 300

def run_transmission_simulation(
    *, epsilon: float, r_over_a: float, a_mm: float,
    nx: int, ny: int, lattice: str,
    fmin: float, fmax: float, points: int, resolution: int
):
    """Return dict with freq_GHz[], Tlin[] (power) for a finite slab."""
    a_m = _a_from_mm(a_mm)
    fmin_mu = _GHz_to_meep(fmin, a_m)
    fmax_mu = _GHz_to_meep(fmax, a_m)
    fcen = 0.5 * (fmin_mu + fmax_mu)
    fwidth = (fmax_mu - fmin_mu)

    # Build geometry in a-units (a=1)
    r = r_over_a * 0.5
    rods = []
    if lattice == "triangular":
        dy = np.sqrt(3)/2
        height = dy * ny
        basis = [mp.Vector3(0, 0), mp.Vector3(0.5, dy)]
        for ix in range(nx):
            for iy in range(ny):
                for b in basis:
                    rods.append(mp.Cylinder(radius=r, height=mp.inf,
                                            material=mp.Medium(epsilon=epsilon),
                                            center=mp.Vector3(ix + b.x, iy*dy + b.y)))
    else:
        height = ny
        for ix in range(nx):
            for iy in range(ny):
                rods.append(mp.Cylinder(radius=r, height=mp.inf,
                                        material=mp.Medium(epsilon=epsilon),
                                        center=mp.Vector3(ix, iy)))

    # Cell and sources
    dpml = 1.0
    sx = nx + 2*dpml + 2.0
    sy = max(6.0, height + 2*dpml)
    cell = mp.Vector3(sx, sy, 0)
    src_x = -0.5*sx + dpml + 0.5
    src = [mp.Source(src=mp.GaussianSource(frequency=fcen, fwidth=fwidth),
                     component=mp.Ez,
                     center=mp.Vector3(src_x, 0),
                     size=mp.Vector3(0, sy - 2*dpml))]

    tran_fr = mp.FluxRegion(center=mp.Vector3(0.5*sx - dpml - 0.5, 0),
                            size=mp.Vector3(0, sy - 2*dpml))

    # With crystal
    sim = mp.Simulation(cell_size=cell, geometry=rods,
                        boundary_layers=[mp.PML(dpml)], sources=src,
                        resolution=resolution)
    tran = sim.add_flux(fcen, fwidth, points, tran_fr)
    sim.run(until=mp.stop_when_fields_decayed(50, mp.Ez, tran_fr.center, 1e-6))
    tran_spec = np.array(mp.get_fluxes(tran))

    # Reference (no crystal)
    sim.reset_meep()
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(dpml)],
                        sources=src, resolution=resolution)
    tran0 = sim.add_flux(fcen, fwidth, points, tran_fr)
    sim.run(until=mp.stop_when_fields_decayed(50, mp.Ez, tran_fr.center, 1e-6))
    tran0_spec = np.array(mp.get_fluxes(tran0))

    Tlin = tran_spec / (tran0_spec + 1e-12)  # power transmittance
    freq_GHz = np.linspace(fmin, fmax, points)
    return {"freq_GHz": freq_GHz, "Tlin": Tlin}

@app.post("/transmission")
def transmission(inp: TxInput):
    sim = run_transmission_simulation(
        epsilon=inp.epsilon, r_over_a=inp.r_over_a, a_mm=inp.a_mm,
        nx=inp.nx, ny=inp.ny, lattice=inp.lattice,
        fmin=inp.fmin_GHz, fmax=inp.fmax_GHz, points=inp.nfreq,
        resolution=inp.resolution,
    )
    TdB = 10.0*np.log10(np.clip(sim["Tlin"], 1e-12, None))
    return {"freq_GHz": sim["freq_GHz"].tolist(),
            "trans_dB": TdB.tolist(),
            "trans_lin": sim["Tlin"].tolist()}

# ==========================================================
# 3) Calibration against reference data
# ==========================================================
class RefPoint(BaseModel):
    freq_GHz: float
    trans_dB: float  # power dB (10*log10(T))

class CalibrateRequest(BaseModel):
    epsilon: float = 3.50
    r_over_a: float = 0.20
    lattice: str = "square"
    ny: int = 8
    a_min_mm: float = 7.0
    a_max_mm: float = 12.0
    a_steps: int = 11
    nx_min: int = 8
    nx_max: int = 28
    nx_step: int = 4
    calib_resolution: int = 28
    points: int = 400
    reference: List[RefPoint]

@app.post("/calibrate")
def calibrate(req: CalibrateRequest):
    ref = np.array([[p.freq_GHz, p.trans_dB] for p in req.reference], dtype=float)
    fmin, fmax = float(np.min(ref[:, 0])), float(np.max(ref[:, 0]))
    a_grid = np.linspace(req.a_min_mm, req.a_max_mm, req.a_steps)
    nx_grid = list(range(req.nx_min, req.nx_max + 1, req.nx_step))

    best = {"mse": float("inf")}
    for a_mm in a_grid:
        for nx in nx_grid:
            sim = run_transmission_simulation(
                epsilon=req.epsilon, r_over_a=req.r_over_a, a_mm=a_mm,
                nx=nx, ny=req.ny, lattice=req.lattice,
                fmin=fmin, fmax=fmax, points=req.points,
                resolution=req.calib_resolution,
            )
            Tdb = 10.0*np.log10(np.clip(sim["Tlin"], 1e-12, None))
            mse = _mse_against_reference(ref, sim["freq_GHz"], Tdb)
            if mse < best["mse"]:
                best = {
                    "mse": float(mse),
                    "a_mm": float(a_mm),
                    "nx": int(nx),
                    "freq_GHz": sim["freq_GHz"].tolist(),
                    "sim_dB": Tdb.tolist(),
                }
    return best
# ==========================================================
# 4) Built-in "reference / measurement" curves (for overlay)
#    A) /ref/list      -> list available built-ins
#    B) /ref/get       -> fetch a named dataset
#    C) /ref/generate  -> parametric "Soumia-style" curve with sliders
# ==========================================================
from typing import Optional

# A very light ~Soumia Fig.3 estimate (1–30 GHz), digitized/parameterized
def _Graph_Stimate(fmin=5.0, fmax=30.0, points=400):
    f = np.linspace(fmin, fmax, points)
    # Dip centered ~14.8 GHz, ~−12 dB depth, ~3.5 GHz width
    fc, depth, width = 14.8, 12.0, 3.5
    # smooth notch (Lorentzian-ish)
    dip = -depth / (1.0 + ((f - fc) / (0.5 * width)) ** 2)

    # right-side roll-off ~26–30 GHz down to ~−12 dB
    roll_start, roll_slope = 26.0, -3.5  # dB per GHz after roll_start
    roll = np.where(f > roll_start, (f - roll_start) * roll_slope, 0.0)

    # suppress below ~9 GHz a bit (tiny wiggle)
    left_wiggle = -0.6 * np.exp(-((f - 9.5) / 1.2) ** 2)

    tr_db = np.maximum(dip + roll + left_wiggle, -25.0)
    return f.tolist(), tr_db.tolist()

@app.get("/ref/list")
def ref_list():
    return {"datasets": [
        {"name": "Graph_Stimate", "desc": "Graph Stimate (1–30 GHz)"},
        {"name": "flat_0dB", "desc": "Flat 0 dB reference (sanity)"},
    ]}

@app.get("/ref/get")
def ref_get(name: str = "Graph_Stimate",
            fmin: float = 5.0, fmax: float = 30.0, points: int = 400):
    f = np.linspace(fmin, fmax, points)
    if name == "Graph_Stimate":
        freq, tr_db = _Graph_Stimate(fmin, fmax, points)
    elif name == "flat_0dB":
        freq, tr_db = f.tolist(), [0.0]*points
    else:
        return {"error": "unknown dataset"}
    return {"name": name, "freq_GHz": freq, "trans_dB": tr_db}

class RefGenParams(BaseModel):
    fmin: float = 5.0
    fmax: float = 30.0
    points: int = 400
    # Notch
    fc_GHz: float = 14.8      # dip center
    depth_dB: float = 12.0    # positive number -> depth
    width_GHz: float = 3.5    # approximate FWHM
    # Right roll-off
    roll_start_GHz: float = 26.0
    roll_slope_dB_per_GHz: float = -3.5
    # Left wiggle (small)
    left_mu_GHz: float = 9.5
    left_sigma_GHz: float = 1.2
    left_amp_dB: float = 0.6

@app.post("/ref/generate")
def ref_generate(p: RefGenParams):
    f = np.linspace(p.fmin, p.fmax, p.points)
    dip = -p.depth_dB / (1.0 + ((f - p.fc_GHz) / (0.5 * p.width_GHz)) ** 2)
    roll = np.where(f > p.roll_start_GHz, (f - p.roll_start_GHz) * p.roll_slope_dB_per_GHz, 0.0)
    left = -p.left_amp_dB * np.exp(-((f - p.left_mu_GHz) / p.left_sigma_GHz) ** 2)
    tr_db = np.maximum(dip + roll + left, -25.0)
    return {"freq_GHz": f.tolist(), "trans_dB": tr_db.tolist()}
