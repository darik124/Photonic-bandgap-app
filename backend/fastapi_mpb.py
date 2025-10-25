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

def _build_rods_grid(r_over_a: float, eps: float, nx: int, ny: int, lattice: str):
    """Return (geometry, height_in_a) for a centered rods-in-air slab. a=1 units."""
    r = r_over_a * 0.5
    rods = []
    if lattice == "triangular":
        dy = np.sqrt(3) / 2
        height = dy * ny
        basis = [mp.Vector3(0, 0), mp.Vector3(0.5, dy)]
        for ix in range(nx):
            for iy in range(ny):
                for b in basis:
                    cx = ix + b.x - 0.5*nx + 0.5
                    cy = iy*dy + b.y - 0.5*height + dy/2
                    rods.append(
                        mp.Cylinder(
                            radius=r, height=mp.inf,
                            material=mp.Medium(epsilon=eps),
                            center=mp.Vector3(cx, cy)
                        )
                    )
    else:  # square
        height = ny
        for ix in range(nx):
            for iy in range(ny):
                cx = ix - 0.5*nx + 0.5
                cy = iy - 0.5*ny + 0.5
                rods.append(
                    mp.Cylinder(
                        radius=r, height=mp.inf,
                        material=mp.Medium(epsilon=eps),
                        center=mp.Vector3(cx, cy)
                    )
                )
    return rods, height

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
    y_boundary: str = "periodic"   # "periodic" (infinite height) or "pml" (finite slab)

def _run_transmission(
    *, epsilon: float, r_over_a: float, a_mm: float,
    nx: int, ny: int, lattice: str,
    fmin_GHz: float, fmax_GHz: float, nfreq: int,
    resolution: int
):
    """Compute power transmittance T(f) for a finite slab, normalized to no-crystal."""
    a_m = _a_from_mm(a_mm)
    fmin_mu = _GHz_to_meep(fmin_GHz, a_m)
    fmax_mu = _GHz_to_meep(fmax_GHz, a_m)
    fcen = 0.5 * (fmin_mu + fmax_mu)
    fwidth = (fmax_mu - fmin_mu)

# Geometry in a-units (a=1)
rods, height = _build_rods_grid(r_over_a, epsilon, nx, ny, lattice)

dpml = 1.0
sx = nx + 2*dpml + 2.0

if y_boundary.lower() == "periodic":
    # infinite-height (2D) case
    sy = height
    cell = mp.Vector3(sx, sy, 0)
    bnd = [mp.PML(dpml, direction=mp.X)]
    src_size_y = sy
    tran_size_y = sy
else:
    # finite-height slab with PML above/below
    sy = height + 2*dpml
    cell = mp.Vector3(sx, sy, 0)
    bnd = [mp.PML(dpml)]  # PML in all directions
    src_size_y = sy - 2*dpml
    tran_size_y = sy - 2*dpml

src_x = -0.5*sx + dpml + 0.5
src = [mp.Source(src=mp.GaussianSource(frequency=fcen, fwidth=fwidth),
                 component=mp.Ez,
                 center=mp.Vector3(src_x, 0),
                 size=mp.Vector3(0, src_size_y))]

tran_fr = mp.FluxRegion(center=mp.Vector3(0.5*sx - dpml - 0.5, 0),
                        size=mp.Vector3(0, tran_size_y))

# With crystal
sim = mp.Simulation(cell_size=cell, geometry=rods,
                    boundary_layers=bnd, sources=src,
                    resolution=resolution)
tran = sim.add_flux(fcen, fwidth, nfreq, tran_fr)
sim.run(until=mp.stop_when_fields_decayed(50, mp.Ez, tran_fr.center, 1e-6))
tran_spec = np.array(mp.get_fluxes(tran))

# Reference (no crystal)
sim.reset_meep()
sim = mp.Simulation(cell_size=cell, boundary_layers=bnd,
                    sources=src, resolution=resolution)
tran0 = sim.add_flux(fcen, fwidth, nfreq, tran_fr)
sim.run(until=mp.stop_when_fields_decayed(50, mp.Ez, tran_fr.center, 1e-6))
tran0_spec = np.array(mp.get_fluxes(tran0))


    Tlin = tran_spec / (tran0_spec + 1e-12)  # power transmittance
    freq_GHz = np.linspace(fmin_GHz, fmax_GHz, nfreq)
    return freq_GHz, Tlin

@app.post("/transmission")
def transmission(inp: TxInput):
    freq_GHz, Tlin = _run_transmission(
        epsilon=inp.epsilon, r_over_a=inp.r_over_a, a_mm=inp.a_mm,
        nx=inp.nx, ny=inp.ny, lattice=inp.lattice,
        fmin_GHz=inp.fmin_GHz, fmax_GHz=inp.fmax_GHz, nfreq=inp.nfreq,
        resolution=inp.resolution,
    )
    Tdb = 10.0*np.log10(np.clip(Tlin, 1e-12, None))
    return {
        "freq_GHz": freq_GHz.tolist(),
        "trans_dB": Tdb.tolist(),
        "trans_lin": Tlin.tolist()
    }

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
