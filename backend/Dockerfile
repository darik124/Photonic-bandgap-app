# Use conda-forge so meep + mpb (and native libs) install cleanly
FROM condaforge/miniforge3

# Create an env with everything we need
RUN mamba install -y python=3.10 meep mpb fastapi uvicorn numpy && mamba clean -afy

WORKDIR /app
COPY fastapi_mpb.py /app/fastapi_mpb.py

EXPOSE 8000
CMD ["uvicorn", "fastapi_mpb:app", "--host", "0.0.0.0", "--port", "8000"]
