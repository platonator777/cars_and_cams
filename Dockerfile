FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime AS base
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 XFORMERS_DISABLED=1
WORKDIR /app
RUN python -m pip install --no-cache-dir pandas==2.2.3 numpy==2.1.3 Pillow==10.4.0

FROM base AS runtime
COPY infer.py evaluate.py verify.py benchmark.py /app/
COPY vehicle_reid/ /app/vehicle_reid/
COPY configs/inference.yaml /app/configs/
COPY vendor/dinov2/ /app/vendor/dinov2/
COPY official/evaluate.py /app/official/evaluate.py
COPY weights/ /app/weights/
ENTRYPOINT ["python", "infer.py"]

FROM base AS training
COPY train.py infer.py evaluate.py verify.py benchmark.py /app/
COPY pyproject.toml requirements-train.lock /app/
COPY vehicle_reid/ /app/vehicle_reid/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/
COPY official/ /app/official/
COPY vendor/dinov2/ /app/vendor/dinov2/
ENTRYPOINT ["python", "scripts/reproduce.py"]
