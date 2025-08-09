# backend/fastapi_mpb.py
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import meep as mp
import meep.mpb as mpb
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten later to your Streamlit domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Infinite crystal (MPB band structure) ----------

class BandInput(BaseModel):
    epsilon: float
    r_over_a: float
    num_bands: int = 8
    resolution: int = 32
    k_points_per_segment: int = 16
    lattice: str = "square"   # "square" | "triangular"

@app.post("/bands")
def compute_bands(inp: BandInput):
    # Lattice & k-path
    if inp.lattice == "square":
        geometry_lattice = mp.Lattice(size=mp.Vector3(1, 1))
        G = mp.Vector3(0, 0); X = mp.Vector3(0.5, 0); M = mp.Vector3(0.5, 0.5)
        k_points = mp.interpolate(inp.k_points_per_segment, [G, X, M, G])
        labels = ["Γ", "X", "M", "Γ"]
    elif inp.lattice == "triangular":
        geometry_lattice = mp.Lattice(size=mp.Vector3(1, 1),
                                      basis1=mp.Vector3(1, 0),
                                      basis2=mp.Vector3(0.5, np.sqrt(3)/2))
        G = mp.Vector3()
        M = mp.Vector3(0.5, 0.5/np.sqrt(3))
        K = mp.Vector3(1/3, 1/3/np.sqrt(3))
        k_points = mp.interpolate(inp.k_points_per_segment, [G, M, K, G])
        labels = ["Γ", "M", "K", "Γ"]
    else:
        return {"error": "unknown lattice"}

    radius = inp.r_over_a * 0.5  # normalized (a=1)
    geometry = [mp.Cylinder(radius=radius, height=mp.inf,
                            material=mp.Medium(epsilon=inp.epsilon))]

    ms = mpb.ModeSolver(num_bands=inp.num_bands,
                        k_points=k_points,
                        geometry_lattice=geometry_lattice,
                        geometry=geometry,
                        resolution=inp.resolution,
                        default_material=mp.Medium(epsilon=1.0))
    # TM (Ez) matches dielectric rods in air
    ms.run_tm()
    freqs = ms.all_freqs  # shape (num_k, num_bands)

    return {"k_path_labels": labels, "frequencies": np.asarray(freqs).tolist()}


# ---------- Finite crystal (Meep transmission) ----------

class TxInput(BaseModel):
    epsilon: float          # rod permittivity
    r_over_a: float         # radius / lattice constant
    a_mm: float             # lattice constant in mm (sets GHz scale)
    nx: int = 10            # rods along x (prop direction)
    ny: int = 8             # rods along y (height)
    lattice: str = "square" # or "triangular"
    resolution: int = 24    # Meep pixels per 'a'
    fmin_GHz: float = 5.0
    fmax_GHz: float = 35.0
    nfreq: int = 300        # spectrum samples

def _a_from_mm(a_mm: float) -> float:
    return a_mm * 1e-3   # meters

def _GHz_to_meep_freq(f_GHz: float, a_m: float) -> float:
    c0 = 299_792_458.0
    f_Hz = f_GHz * 1e9
    return (a_m * f_Hz) / c0  # dimensionless a/λ

@app.post("/transmission")
def transmission(inp: TxInput):
    a_m = _a_from_mm(inp.a_mm)

    # Frequency axis in Meep units
    fmin = _GHz_to_meep_freq(inp.fmin_GHz, a_m)
    fmax = _GHz_to_meep_freq(inp.fmax_GHz, a_m)
    fc = np.linspace(fmin, fmax, inp.nfreq)

    # Lattice footprint (in a=1)
    if inp.lattice == "triangular":
        height = np.sqrt(3)/2 * inp.ny
        basis = [mp.Vector3(0,0), mp.Vector3(0.5, np.sqrt(3)/2)]
    else:
        height = inp.ny
        basis = [mp.Vector3(0,0)]

    # Rod geometry (2D, TM/Ez)
    r = inp.r_over_a * 0.5
    mat = mp.Medium(epsilon=inp.epsilon)
    geometry = []
    for ix in range(inp.nx):
        for iy in range(inp.ny):
            for b in basis:
                x = ix + b.x
                y = (iy + b.y) if inp.lattice == "triangular" else iy
                geometry.append(
                    mp.Cylinder(radius=r, height=mp.inf, material=mat,
                                center=mp.Vector3(x, y, 0))
                )

    # Computational cell
    dpml = 1.0
    sx = inp.nx + 2*dpml + 2.0
    sy = height + 2*dpml
    cell = mp.Vector3(sx, sy, 0)

    # Source (broadband) at left
    src_x = -0.5*sx + dpml + 0.5
    src = [mp.Source(src=mp.GaussianSource(frequency=0.5*(fc[0]+fc[-1]),
                                           fwidth=(fc[-1]-fc[0])),
                     component=mp.Ez,
                     center=mp.Vector3(src_x, 0, 0),
                     size=mp.Vector3(0, sy-2*dpml, 0))]

    # Flux planes
    refl_fr = mp.FluxRegion(center=mp.Vector3(src_x + 0.4, 0, 0),
                            size=mp.Vector3(0, sy-2*dpml, 0))
    tran_fr = mp.FluxRegion(center=mp.Vector3(0.5*sx - dpml - 0.5, 0, 0),
                            size=mp.Vector3(0, sy-2*dpml, 0))

    # DFT band selection (center & width)
    fcen   = 0.5 * (fc[0] + fc[-1])
    fwidth = (fc[-1] - fc[0])

    # With crystal
    sim = mp.Simulation(cell_size=cell,
                        geometry=geometry,
                        boundary_layers=[mp.PML(dpml)],
                        sources=src,
                        resolution=inp.resolution)
    tran = sim.add_flux(fcen, fwidth, inp.nfreq, tran_fr)
    sim.add_flux(fcen, fwidth, inp.nfreq, refl_fr)  # optional: reflection
    sim.run(until=mp.stop_when_fields_decayed(50, mp.Ez, tran_fr.center, 1e-6))
    tran_spec = np.array(mp.get_fluxes(tran))

    # Reference (no crystal)
    sim.reset_meep()
    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=[mp.PML(dpml)],
                        sources=src,
                        resolution=inp.resolution)
    tran0 = sim.add_flux(fcen, fwidth, inp.nfreq, tran_fr)
    sim.run(until=mp.stop_when_fields_decayed(50, mp.Ez, tran_fr.center, 1e-6))
    tran0_spec = np.array(mp.get_fluxes(tran0))

    # Transmission ratio
    T = tran_spec / (tran0_spec + 1e-12)
    freq_GHz = np.linspace(inp.fmin_GHz, inp.fmax_GHz, inp.nfreq)
    TdB = 10*np.log10(np.clip(T, 1e-12, None))

    return {"frequency_GHz": freq_GHz.tolist(),
            "transmission_dB": TdB.tolist()}

@app.get("/health")
def health():
    return {"ok": True}
