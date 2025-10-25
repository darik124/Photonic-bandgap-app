from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import meep as mp
import meep.mpb as mpb

# -------------------------
# App & CORS
# -------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Health
# -------------------------
@app.get("/health")
def health():
    return {"ok": True}

# -------------------------
# Helpers
# -------------------------
C0 = 299_792_458.0  # m/s

def _a_from_mm(a_mm: float) -> float:
    return a_mm * 1e-3  # meters

def _GHz_to_meep(f_GHz: float, a_m: float) -> float:
    """Dimensionless frequency f = a/λ for Meep/MPB (2πc factors handled by Meep)."""
    return (a_m * (f_GHz * 1e9)) / C0

def _build_rods_grid(r_over_a: float, eps: float, nx: int, ny: int, lattice: str,
                     periodic_y: bool = False):
    """
    Return (geometry, height_in_a, y_period_in_a) in a=1 units.

    periodic_y = True  -> build a single Y period only (k=0 Bloch), ignoring 'ny'
    periodic_y = False -> build ny rows (finite-height slab)
    """
    r = r_over_a * 0.5
    rods = []

    if lattice == "triangular":
        dy = np.sqrt(3) / 2  # y period
        if periodic_y:
            # One unit cell in Y: two basis rods per layer
            height = dy
            for ix in range(nx):
                cx = ix - 0.5*nx + 0.5
                # basis at y = ±dy/2 centered around 0
                rods.append(mp.Cylinder(radius=r, height=mp.inf,
                                        material=mp.Medium(epsilon=eps),
                                        center=mp.Vector3(cx + 0.0, +0.5*dy)))
                rods.append(mp.Cylinder(radius=r, height=mp.inf,
                                        material=mp.Medium(epsilon=eps),
                                        center=mp.Vector3(cx + 0.5, -0.0*dy)))  # (0.5,0) in conventional basis
        else:
            # Finite-height: ny rows, two basis rods per row
            height = dy * ny
            for ix in range(nx):
                for iy in range(ny):
                    cx = ix - 0.5*nx + 0.5
                    cy0 = (iy + 0.5) * dy - 0.5*height
                    rods.append(mp.Cylinder(radius=r, height=mp.inf,
                                            material=mp.Medium(epsilon=eps),
                                            center=mp.Vector3(cx + 0.0, cy0 + 0.5*dy)))
                    rods.append(mp.Cylinder(radius=r, height=mp.inf,
                                            material=mp.Medium(epsilon=eps),
                                            center=mp.Vector3(cx + 0.5, cy0 + 0.0)))
        yperiod = dy

    else:  # square
        dy = 1.0  # y period
        if periodic_y:
            # One unit cell in Y: a single rod per layer at y=0
            height = dy
            for ix in range(nx):
                cx = ix - 0.5*nx + 0.5
                rods.append(mp.Cylinder(radius=r, height=mp.inf,
                                        material=mp.Medium(epsilon=eps),
                                        center=mp.Vector3(cx, 0.0)))
        else:
            # Finite-height: ny rows
            height = ny
            for ix in range(nx):
                for iy in range(ny):
                    cx = ix - 0.5*nx + 0.5
                    cy = iy - 0.5*ny + 0.5
                    rods.append(mp.Cylinder(radius=r, height=mp.inf,
                                            material=mp.Medium(epsilon=eps),
                                            center=mp.Vector3(cx, cy)))
        yperiod = dy

    return rods, height, yperiod


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
    # Lattice & k-path
    if inp.lattice == "square":
        geometry_lattice = mp.Lattice(size=mp.Vector3(1, 1))
        G = mp.Vector3(0, 0); X = mp.Vector3(0.5, 0); M = mp.Vector3(0.5, 0.5)
        k_points = mp.interpolate(inp.k_points_per_segment, [G, X, M, G])
        labels = ["Γ", "X", "M", "Γ"]
    elif inp.lattice == "triangular":
        geometry_lattice = mp.Lattice(
            size=mp.Vector3(1, 1),
            basis1=mp.Vector3(1, 0),
            basis2=mp.Vector3(0.5, np.sqrt(3) / 2),
        )
        G = mp.Vector3()
        M = mp.Vector3(0.5, 0.5 / np.sqrt(3))
        K = mp.Vector3(1 / 3, 1 / (3 * np.sqrt(3)))
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
    ms.run_tm()  # TM (Ez) for rods-in-air
    freqs = np.asarray(ms.all_freqs).tolist()  # (num_k, num_bands)
    return {"k_path_labels": labels, "frequencies": freqs}

# -------------------------
# 2) Finite crystal (Meep transmission spectrum)
# -------------------------
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
    y_boundary: str = "periodic"   # "periodic" = 2D/infinite height (one Y period), "pml" = finite-Height

def _run_transmission(
    *, epsilon: float, r_over_a: float, a_mm: float,
    nx: int, ny: int, lattice: str,
    fmin_GHz: float, fmax_GHz: float, nfreq: int,
    resolution: int, y_boundary: str = "periodic"
):
    # --- frequency setup (Meep units) ---
    a_m = _a_from_mm(a_mm)
    fmin_mu = _GHz_to_meep(fmin_GHz, a_m)
    fmax_mu = _GHz_to_meep(fmax_GHz, a_m)
    fcen    = 0.5 * (fmin_mu + fmax_mu)
    fwidth  = (fmax_mu - fmin_mu)

    # --- build geometry ---
    periodic_y = (y_boundary or "periodic").lower() == "periodic"
    rods, height, yperiod = _build_rods_grid(
        r_over_a, epsilon, nx, ny, lattice, periodic_y=periodic_y
    )

    # --- numerics: thicker PML + more air padding improves deep stop-bands ---
    dpml    = 2.0          # was 1.0
    air_pad = 2.0          # was 1.0
    sx = nx + 2*dpml + 2*air_pad

    if periodic_y:
        # true 2D: one Y period, PML only in X
        sy   = yperiod
        cell = mp.Vector3(sx, sy, 0)
        bnd  = [mp.PML(dpml, direction=mp.X)]
        src_h = sy
        flx_h = sy
        k_point = mp.Vector3(0, 0, 0)  # Bloch k=0 (explicit)
    else:
        # finite-height: ny periods + PML top/bottom
        sy   = height + 2*dpml
        cell = mp.Vector3(sx, sy, 0)
        bnd  = [mp.PML(dpml)]
        src_h = sy - 2*dpml
        flx_h = sy - 2*dpml
        k_point = mp.Vector3(0, 0, 0)

    # place source and probe away from PML and structure
    src_x   = -0.5*sx + dpml + 0.75*air_pad
    probe_x = +0.5*sx - dpml - 0.75*air_pad

    src = [mp.Source(
        src=mp.GaussianSource(frequency=fcen, fwidth=fwidth),
        component=mp.Ez,
        center=mp.Vector3(src_x, 0),
        size=mp.Vector3(0, src_h),
    )]
    tran_fr = mp.FluxRegion(center=mp.Vector3(probe_x, 0),
                            size=mp.Vector3(0, flx_h))

    # ----------------------------
    # Reference FIRST (no crystal)
    # ----------------------------
    sim_ref = mp.Simulation(cell_size=cell,
                            boundary_layers=bnd,
                            sources=src,
                            resolution=resolution,
                            k_point=k_point)
    tran0 = sim_ref.add_flux(fcen, fwidth, nfreq, tran_fr)
    sim_ref.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, tran_fr.center, 1e-7))
    tran0_spec = np.array(mp.get_fluxes(tran0))
    # save spectral phase/amplitude of the incident wave
    ref_data = sim_ref.get_flux_data(tran0)

    # --------------------------------------
    # With crystal, using minus-flux loading
    # --------------------------------------
    sim = mp.Simulation(cell_size=cell,
                        geometry=rods,
                        boundary_layers=bnd,
                        sources=src,
                        resolution=resolution,
                        k_point=k_point)
    tran = sim.add_flux(fcen, fwidth, nfreq, tran_fr)
    # subtract the incident field so transmitted power is accurate in stop-bands
    sim.load_minus_flux_data(tran, ref_data)

    sim.run(until_after_sources=mp.stop_when_fields_decayed(80, mp.Ez, tran_fr.center, 1e-7))
    tran_spec = np.array(mp.get_fluxes(tran))

    # ratio (already incident-subtracted on numerator): still divide by |inc| for absolute T
    Tlin = tran_spec / (tran0_spec + 1e-18)
    freq_GHz = np.linspace(fmin_GHz, fmax_GHz, nfreq)
    return freq_GHz, Tlin



# ==========================================================
# 3) Attenuation in forbidden band (Transmission vs layers)
# ==========================================================
class AttenuationInput(BaseModel):
    epsilon: float
    r_over_a: float
    a_mm: float
    ny: int
    nmax: int
    f0_GHz: float
    lattice: str = "square"
    resolution: int = 24

@app.post("/attenuation")
def attenuation(inp: AttenuationInput):
    """Probe a single frequency f0 vs number of layers; return T and Attenuation."""
    a_m = _a_from_mm(inp.a_mm)
    f0 = _GHz_to_meep(inp.f0_GHz, a_m)
    df = 1e-6  # razor-thin DFT bin

    def run_for_layers(nx_layers: int, ref_flux=None):
        rods, height = _build_rods_grid(inp.r_over_a, inp.epsilon, nx_layers, inp.ny, inp.lattice)
        dpml = 1.0
        sx = nx_layers + 2*dpml + 2.0
        sy = max(6.0, height + 2*dpml)
        cell = mp.Vector3(sx, sy, 0)

        src_x = -0.5*sx + dpml + 0.5
        src = [mp.Source(mp.ContinuousSource(frequency=f0),
                         component=mp.Ez,
                         center=mp.Vector3(src_x, 0),
                         size=mp.Vector3(0, sy - 2*dpml))]

        tran_fr = mp.FluxRegion(center=mp.Vector3(+0.5*sx - dpml - 0.5, 0),
                                size=mp.Vector3(0, sy - 2*dpml))

        sim = mp.Simulation(cell_size=cell, geometry=rods,
                            boundary_layers=[mp.PML(dpml, direction=mp.X)],
                            sources=src, resolution=inp.resolution)
        fobj = sim.add_flux(f0, df, 1, tran_fr)
        sim.run(until=200)  # steady-state for continuous source
        phi = mp.get_fluxes(fobj)[0]
        if ref_flux is None:
            return phi
        return phi / (ref_flux + 1e-12)

    # reference (no crystal), then sweep N=1..nmax
    ref = run_for_layers(0, ref_flux=None)
    layers, Tdb = [], []
    for N in range(1, inp.nmax + 1):
        T = run_for_layers(N, ref_flux=ref)  # power ratio
        Tdb.append(10*np.log10(max(T, 1e-12)))
        layers.append(N)

    # include positive attenuation for convenience
    atten_dB = [-x for x in Tdb]
    T_lin = [10**(x/10.0) for x in Tdb]
    return {
        "layers": layers,
        "T_dB": Tdb,
        "atten_dB": atten_dB,
        "T_lin": T_lin
    }
