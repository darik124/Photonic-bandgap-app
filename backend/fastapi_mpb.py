from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import meep as mp
import math

app = FastAPI()

# ==============================================================
# Existing models & endpoints here (unchanged)...
# ==============================================================

# ---------------------------
# New AttenuationInput model
# ---------------------------
class AttenuationInput(BaseModel):
    epsilon: float          # rod permittivity
    r_over_a: float         # radius / lattice constant
    a_mm: float             # lattice constant (mm)
    ny: int                 # rods along y (height)
    nmax: int               # max layers along x to test
    f0_GHz: float           # single probe frequency (GHz)
    lattice: str = "square" # "square" | "triangular"
    resolution: int = 24    # px per a

# ---------------------------
# Attenuation endpoint
# ---------------------------
@app.post("/attenuation")
def attenuation(inp: AttenuationInput):
    a = inp.a_mm
    r = inp.r_over_a * a
    eps = inp.epsilon
    f0 = inp.f0_GHz / 1000.0  # GHz → THz conversion if needed

    # Build geometry
    geometry = []
    if inp.lattice == "square":
        for y in range(inp.ny):
            geometry.append(mp.Cylinder(radius=r, material=mp.Medium(epsilon=eps),
                                        center=mp.Vector3(0, y*a)))
    elif inp.lattice == "triangular":
        for y in range(inp.ny):
            offset = (y % 2) * (a / 2)
            geometry.append(mp.Cylinder(radius=r, material=mp.Medium(epsilon=eps),
                                        center=mp.Vector3(offset, y*(a*math.sqrt(3)/2))))

    # Transmission vs layers
    attenuation_data = []
    for n_layers in range(1, inp.nmax + 1):
        sx = n_layers * a + 2.0
        sy = inp.ny * a + 2.0

        cell = mp.Vector3(sx, sy)
        sources = [mp.Source(mp.ContinuousSource(frequency=f0),
                             component=mp.Ez,
                             center=mp.Vector3(-0.5*sx + 0.5, 0))]

        sim = mp.Simulation(cell_size=cell,
                            geometry=geometry,
                            boundary_layers=[mp.PML(1.0)],
                            sources=sources,
                            resolution=inp.resolution)

        tran_fr = mp.FluxRegion(center=mp.Vector3(0.5*sx - 0.5, 0))
        tran = sim.add_flux(f0, 0, 1, tran_fr)

        sim.run(until=200)

        flux_val = mp.get_fluxes(tran)[0]
        attenuation_data.append({"layers": n_layers, "transmission": flux_val})

    return {"attenuation_data": attenuation_data}
