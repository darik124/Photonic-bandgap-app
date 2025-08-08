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

class BandInput(BaseModel):
    epsilon: float
    r_over_a: float
    num_bands: int = 8
    resolution: int = 32
    k_points_per_segment: int = 16
    lattice: str = "square"   # "square" | "triangular"

@app.post("/bands")
def compute_bands(inp: BandInput):
    if inp.lattice == "square":
        geometry_lattice = mp.Lattice(size=mp.Vector3(1, 1))
        G = mp.Vector3(0, 0); X = mp.Vector3(0.5, 0); M = mp.Vector3(0.5, 0.5)
        k_points = mp.interpolate(inp.k_points_per_segment, [G, X, M, G])
        labels = ["Γ", "X", "M", "Γ"]
    elif inp.lattice == "triangular":
        geometry_lattice = mp.Lattice(size=mp.Vector3(1,1),
                                      basis1=mp.Vector3(1,0),
                                      basis2=mp.Vector3(0.5, np.sqrt(3)/2))
        G = mp.Vector3()
        M = mp.Vector3(0.5, 0.5/np.sqrt(3))
        K = mp.Vector3(1/3, 1/3/np.sqrt(3))
        k_points = mp.interpolate(inp.k_points_per_segment, [G, M, K, G])
        labels = ["Γ", "M", "K", "Γ"]
    else:
        return {"error": "unknown lattice"}

    radius = inp.r_over_a * 0.5  # normalized a=1
    geometry = [mp.Cylinder(radius=radius, height=mp.inf,
                            material=mp.Medium(epsilon=inp.epsilon))]

    ms = mpb.ModeSolver(num_bands=inp.num_bands,
                        k_points=k_points,
                        geometry_lattice=geometry_lattice,
                        geometry=geometry,
                        resolution=inp.resolution,
                        default_material=mp.Medium(epsilon=1.0))
    ms.run_tm()  # use ms.run_te() for TE
    freqs = ms.all_freqs  # (num_k, num_bands)

    return {"k_path_labels": labels, "frequencies": np.asarray(freqs).tolist()}
