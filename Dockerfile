FROM python:3.13-slim-bookworm

# Copy uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy project files
COPY "pyproject.toml" "uv.lock" ./

# Sync dependencies
RUN uv sync --locked  

# Copy application files
COPY "main.py" "credit_card_fraud_detection.pkl" ./

EXPOSE 8000

# Run uvicorn with 0.0.0.0 to accept external connections
CMD ["uv", "run", "--", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]




