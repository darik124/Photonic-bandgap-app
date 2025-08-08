from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np

import meep as mp
from meep import mpb   # <-- MPB bindings

app = FastAPI()

class BandInput(BaseModel):
    epsilon: float         # rod permittivity
    r_over_a: float        # r/a
    num_bands: int = 8
    resolution: int = 32   # MPB grid resolution
    k_points_per_segment: int = 16
    lattice: str = "square"   # (square|triangular)

@app.post("/bands")
def compute_bands(inp: BandInput):
    # Lattice
    if inp.lattice == "square":
        geometry_lattice = mp.Lattice(size=mp.Vector3(1, 1))
        # high-symmetry path Γ-X-M-Γ in reduced coords
        G = mp.Vector3(0, 0)
        X = mp.Vector3(0.5, 0)
        M = mp.Vector3(0.5, 0.5)
        k_points = mp.interpolate(inp.k_points_per_segment, [G, X, M, G])
    elif inp.lattice == "triangular":
        geometry_lattice = mp.Lattice(size=mp.Vector3(1, 1),
                                      basis1=mp.Vector3(1, 0),
                                      basis2=mp.Vector3(0.5, np.sqrt(3)/2))
        G = mp.Vector3()
        M = mp.Vector3(0.5, 0.5/np.sqrt(3))
        K = mp.Vector3(1/3, 1/3/np.sqrt(3))
        k_points = mp.interpolate(inp.k_points_per_segment, [G, M, K, G])
    else:
        return {"error": "unknown lattice"}

    # Geometry: dielectric rod in air
    r = inp.r_over_a * 0.5 * geometry_lattice.size.x  # since a = lattice size
    geometry = [mp.Cylinder(radius=r, material=mp.Medium(epsilon=inp.epsilon),
                            center=mp.Vector3(), height=mp.inf)]

    ms = mpb.ModeSolver(num_bands=inp.num_bands,
                        k_points=k_points,
                        geometry_lattice=geometry_lattice,
                        geometry=geometry,
                        resolution=inp.resolution,
                        default_material=mp.Medium(epsilon=1.0))  # air background

    # Choose polarization (TM here for rods in air; switch to ms.run_te() for TE)
    ms.run_tm()
    # Normalized frequencies (omega a / 2πc) for each k-point/band
    freqs = ms.all_freqs  # shape: (num_k, num_bands)

    # Return as lists (JSON serializable)
    return {
        "k_path_labels": ["G", "X", "M", "G"] if inp.lattice == "square" else ["G", "M", "K", "G"],
        "k_count": len(k_points),
        "num_bands": inp.num_bands,
        "frequencies": freqs.tolist()  # list of list [k_index][band]
    }
# backend/Dockerfile
FROM ubuntu:22.04

# System deps for meep/mpb (abridged; exact list depends on platform)
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev git build-essential \
    libfftw3-dev libhdf5-dev guile-3.0-dev liblapack-dev libblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip3 install fastapi uvicorn meep

WORKDIR /app
COPY main.py /app/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
docker build -t mpb-api backend/
docker run -p 8000:8000 mpb-api
