from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# If you have meep/mpb installed with Python bindings:
import meep as mp
from meep import mpb

# --- FastAPI app ---
app = FastAPI()

# CORS so Streamlit (different domain) can call this API
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # for dev; restrict to your Streamlit domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BandInput(BaseModel):
    epsilon: float          # rod permittivity
    r_over_a: float         # r/a
    num_bands: int = 8
    resolution: int = 32    # MPB real-space resolution
    k_points_per_segment: int = 16
    lattice: str = "square" # "square" or "triangular"

@app.post("/bands")
def compute_bands(inp: BandInput):
    # --- lattice & k-path ---
    if inp.lattice == "square":
        geometry_lattice = mp.Lattice(size=mp.Vector3(1, 1))
        G = mp.Vector3(0, 0)
        X = mp.Vector3(0.5, 0)
        M = mp.Vector3(0.5, 0.5)
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
        K = mp.Vector3(1/3, 1/3/np.sqrt(3))
        k_points = mp.interpolate(inp.k_points_per_segment, [G, M, K, G])
        labels = ["Γ", "M", "K", "Γ"]
    else:
        return {"error": "unknown lattice"}

    # --- geometry: dielectric rod in air ---
    # lattice constant a = 1 (normalized). radius = r_over_a * a/2 if you prefer;
    # here we set absolute radius = r_over_a * 0.5 so r/a means diameter ~ r_over_a
    radius = inp.r_over_a * 0.5
    geometry = [
        mp.Cylinder(
            radius=radius,
            height=mp.inf,
            material=mp.Medium(epsilon=inp.epsilon),
            center=mp.Vector3(),
        )
    ]

    ms = mpb.ModeSolver(
        num_bands=inp.num_bands,
        k_points=k_points,
        geometry_lattice=geometry_lattice,
        geometry=geometry,
        resolution=inp.resolution,
        default_material=mp.Medium(epsilon=1.0),  # air
    )

    # TM polarization (common for rods in air). Use ms.run_te() for TE.
    ms.run_tm()
    freqs = ms.all_freqs  # shape: (num_k, num_bands), normalized ωa/2πc

    return {
        "k_path_labels": labels,
        "frequencies": np.asarray(freqs).tolist(),  # JSON serializable
    }
