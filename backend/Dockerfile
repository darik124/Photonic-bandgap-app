FROM ubuntu:22.04

# System deps (abridged; you may adjust for your platform)
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev build-essential git \
    libfftw3-dev libhdf5-dev guile-3.0-dev liblapack-dev libblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Python deps
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# App
COPY fastapi_mpb.py /app/fastapi_mpb.py

EXPOSE 8000
CMD ["uvicorn", "fastapi_mpb:app", "--host", "0.0.0.0", "--port", "8000"]
