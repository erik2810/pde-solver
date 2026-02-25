# --- Build stage ---
FROM python:3.12-slim AS base

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY model.py pinn_model.py main.py index.html ./

# --- Training stage (optional — run with: docker build --target train) ---
FROM base AS train
RUN python pinn_model.py --no-show-plot --save-plot

# --- Production stage ---
FROM base AS production

# Copy pre-trained model (train locally or in CI, then COPY the .pth file)
COPY burgers_model.pth ./

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
