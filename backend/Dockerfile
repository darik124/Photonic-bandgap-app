FROM condaforge/miniforge3

# Use pymeep (Python bindings) + mpb, all from conda-forge
RUN mamba install -y -c conda-forge python=3.10 pymeep mpb fastapi uvicorn numpy && mamba clean -afy

WORKDIR /app
COPY fastapi_mpb.py /app/fastapi_mpb.py

EXPOSE 8000
CMD ["uvicorn", "fastapi_mpb:app", "--host", "0.0.0.0", "--port", "8000"]
