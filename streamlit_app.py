import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Title and description
st.title("Photonic Band Gap Visualizer")
st.write("""
This interactive tool simulates how the transmission of electromagnetic waves changes in a 2D photonic crystal.
The simulation is based on concepts presented in Soumia Souchak’s research, using estimated forbidden band locations derived from HFSS and Translight results.
""")

# User input parameters
epsilon = st.slider("Dielectric Permittivity (ε)", min_value=1.0, max_value=15.0, value=3.5, step=0.1)
diameter = st.slider("Rod Diameter (mm)", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
spacing = st.slider("Rod Spacing (mm)", min_value=5.0, max_value=20.0, value=7.0, step=0.5)
rod_count = st.slider("Number of Rods", min_value=1, max_value=20, value=10)

# Curve style toggle
show_hfss = st.checkbox("Show HFSS Curve (Solid Blue)", value=True)
show_translight = st.checkbox("Show Translight Curve (Dashed Red)", value=True)

# Transmission diagram (simulated using Gaussian dips)
st.header("Simulated Transmission Diagram")

freq = np.linspace(0.4, 32, 800)
transmission_db_hfss = np.zeros_like(freq)
transmission_db_translight = np.zeros_like(freq)

# Calculate dip positions based on effective spacing and permittivity (simplified scaling approximation)
dip1 = 30 * spacing / 7 * (3.5 / epsilon)  # high-frequency dip shifts left as ε increases
dip2 = 18 * spacing / 7 * (3.5 / epsilon)
dip3 = 10 * spacing / 7 * (3.5 / epsilon)
dips = [dip3, dip2, dip1]
depths = [-25, -20, -30]
widths = [1.5, 2.0, 1.0]

# Apply Gaussian dips for HFSS (solid)
for dip, depth, width in zip(dips, depths, widths):
    transmission_db_hfss += depth * np.exp(-((freq - dip)**2) / (2 * width**2))

# Apply similar dips for Translight but with slightly reduced depth (dashed)
for dip, depth, width in zip(dips, depths, widths):
    transmission_db_translight += (depth + 5) * np.exp(-((freq - dip - 0.5)**2) / (2 * (width + 0.3)**2))

# Adjust for rod count and permittivity scaling
scaling = (rod_count / 10) * (epsilon / 3.5)
transmission_db_hfss *= scaling
transmission_db_translight *= scaling

# Clamp to realistic dB range
transmission_db_hfss = np.clip(transmission_db_hfss, -30, 0)
transmission_db_translight = np.clip(transmission_db_translight, -30, 0)

# Plotting
fig, ax = plt.subplots()
if show_hfss:
    ax.plot(freq, transmission_db_hfss, label="HFSS", color='blue')
if show_translight:
    ax.plot(freq, transmission_db_translight, label="Translight", color='red', linestyle='--')
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Transmission (dB)")
ax.set_title("Transmission Diagram Based on Souchak’s Model")
ax.grid(True)
ax.set_ylim(-35, 5)
ax.legend()
st.pyplot(fig)

# Description and context
st.subheader("Interpretation")
st.write("""
In the frequency spectrum above, sharp drops (dips) represent forbidden bands, where electromagnetic waves cannot propagate due to interference from the periodic dielectric structure. 
These bands shift with changes in permittivity and spacing, which reflects real physical behavior shown in HFSS and Translight simulations from Soumia Souchak’s photonic band gap research.
""")

# Export graph data
export = st.selectbox("Export graph data as:", ["None", "CSV", "PNG"])
if export == "CSV":
    df_export = pd.DataFrame({"Frequency (GHz)": freq})
    if show_hfss:
        df_export["HFSS"] = transmission_db_hfss
    if show_translight:
        df_export["Translight"] = transmission_db_translight
    csv = df_export.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "transmission_data.csv", "text/csv")
elif export == "PNG":
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    st.download_button("Download PNG", buf.getvalue(), "transmission_plot.png", "image/png")
