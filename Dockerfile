FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app (including ./artifacts if a trained model has been placed there).
COPY . .

# The detector loads from GROUNDCHECK_MODEL_PATH if set; otherwise it runs on the
# heuristic fallback so the container always starts.
ENV GROUNDCHECK_BACKEND=auto
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
