FROM python:3.12-slim

WORKDIR /app

# CPU-only torch keeps the image small; the model runs fine on CPU.
COPY pyproject.toml README.md ./
COPY groundcheck ./groundcheck
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu ".[serve,model]"

EXPOSE 8000
CMD ["uvicorn", "groundcheck.api:app", "--host", "0.0.0.0", "--port", "8000"]
