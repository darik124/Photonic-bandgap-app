FROM python:3.10-slim

# Install system dependencies for MPB & Meep
RUN apt-get update && apt-get install -y \
    meep mpb libhdf5-dev libfftw3-dev \
    python3-mpi4py python3-h5py \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install fastapi uvicorn numpy meep

# Copy app files
WORKDIR /app
COPY . .

# Run FastAPI
CMD ["uvicorn", "fastapi_mpb:app", "--host", "0.0.0.0", "--port", "8000"]
