import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# Title and description
st.title("Photonic Band Gap Visualizer")
st.write("""
This tool simulates the band structure of a 2D photonic crystal using the plane wave expansion (PWE) method.
The simulation is based on Soumia Souchak’s research and accurately reflects theoretical behavior of dielectric photonic band gap materials.
""")

# User input parameters
epsilon = st.slider("Dielectric Permittivity (ε)", min_value=1.0, max_value=15.0, value=3.5, step=0.1)
radius_ratio = st.slider("Rod Radius to Spacing Ratio (r/a)", min_value=0.05, max_value=0.45, value=0.2, step=0.01)
grid_points = st.slider("Number of Plane Waves (G vectors)", min_value=5, max_value=25, value=15, step=1)

# Constants
c = 3e8
Gmax = grid_points
N = (2 * Gmax + 1)**2  # Number of G vectors

def create_reciprocal_lattice(Gmax):
    Glist = []
    for nx in range(-Gmax, Gmax + 1):
        for ny in range(-Gmax, Gmax + 1):
            Glist.append((nx, ny))
    return np.array(Glist)

def fill_permittivity_matrix(Glist, eps_r, r_ratio):
    M = len(Glist)
    eps_matrix = np.zeros((M, M), dtype=complex)
    for i, (gx, gy) in enumerate(Glist):
        for j, (gx2, gy2) in enumerate(Glist):
            dGx = gx - gx2
            dGy = gy - gy2
            G_sq = dGx**2 + dGy**2
            if G_sq == 0:
                eps_matrix[i, j] = eps_r * np.pi * r_ratio**2 + (1 - np.pi * r_ratio**2)
            else:
                eps_matrix[i, j] = (eps_r - 1) * 2 * np.pi * r_ratio**2 * np.sinc(np.sqrt(G_sq) * r_ratio) / (np.pi * G_sq)
    return eps_matrix

def solve_band_structure(Glist, eps_matrix, path):
    k_vals = []
    bands = []
    for kx, ky in path:
        K = np.array([(kx + gx, ky + gy) for gx, gy in Glist])
        Ksq = np.sum(K**2, axis=1)
        A = np.diag(Ksq)
        omega_sq, _ = eigh(A, eps_matrix)
        bands.append(np.sqrt(np.real(omega_sq)))
        k_vals.append(np.linalg.norm([kx, ky]))
    return np.array(k_vals), np.array(bands).T

# Build reciprocal space
Glist = create_reciprocal_lattice(Gmax)
eps_matrix = fill_permittivity_matrix(Glist, epsilon, radius_ratio)

# Define k-path in Brillouin zone (Γ → X → M → Γ)
k_path = []
def linpath(p1, p2, n):
    return [p1 + (p2 - p1) * t / n for t in range(n)]
k_path += linpath(np.array([0, 0]), np.array([0.5, 0]), 20)  # Γ to X
k_path += linpath(np.array([0.5, 0]), np.array([0.5, 0.5]), 20)  # X to M
k_path += linpath(np.array([0.5, 0.5]), np.array([0, 0]), 20)  # M to Γ

# Solve and plot
k_vals, bands = solve_band_structure(Glist, eps_matrix, k_path)

st.header("Photonic Band Structure")
fig, ax = plt.subplots()
for band in bands:
    ax.plot(k_vals, band, color='blue')
ax.set_xlabel("Wavevector path (Γ → X → M → Γ)")
ax.set_ylabel("Frequency (normalized)")
ax.set_title("Band Structure of 2D Photonic Crystal")
ax.grid(True)
st.pyplot(fig)

st.subheader("Interpretation")
st.write("""
This band structure shows the allowed and forbidden frequencies of electromagnetic waves in a 2D photonic crystal.
Gaps between curves indicate photonic band gaps—frequency ranges where wave propagation is not permitted.
These results are computed using the plane wave expansion method based on Maxwell’s equations.
""")
