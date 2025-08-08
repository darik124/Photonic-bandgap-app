# Photonic Band Gap Visualizer

An interactive Streamlit web app for simulating and visualizing the behavior of 2D dielectric photonic band gap (PBG) materials.
This tool allows users to adjust structural parameters—such as rod diameter, spacing, and permittivity—and view how those affect electromagnetic wave transmission.

## Features
- Adjustable permittivity, diameter, spacing, and rod count
- Simulated transmission spectrum
- Easy deployment via Streamlit Cloud

## Author
Darikson Valenzuela

streamlit
matplotlib
numpy

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Title and description
st.title("Photonic Band Gap Visualizer")
st.write("""
This interactive tool simulates how the transmission of electromagnetic waves changes in a 2D photonic crystal.
Adjust the parameters below to see how the transmission spectrum is affected.
""")

# User input parameters
epsilon = st.slider("Dielectric Permittivity (ε)", min_value=1.0, max_value=15.0, value=3.5, step=0.1)
diameter = st.slider("Rod Diameter (mm)", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
spacing = st.slider("Rod Spacing (mm)", min_value=5.0, max_value=20.0, value=7.0, step=0.5)
rod_count = st.slider("Number of Rods", min_value=1, max_value=20, value=10)

# Simulate transmission data (placeholder for real computation)
frequencies = np.linspace(1, 20, 400)
transmission = np.exp(-((frequencies - (epsilon + rod_count / 5))**2) / (2 * (diameter / 5)**2))

# Plot
fig, ax = plt.subplots()
ax.plot(frequencies, transmission)
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Transmission")
ax.set_title("Simulated Transmission Spectrum")
ax.grid(True)
st.pyplot(fig)
