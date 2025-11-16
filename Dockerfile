FROM python:3.13-slim-bookworm

# RUN pip install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV PATH="/code/.venv/bin:$PATH"

COPY ".python-version" "pyproject.toml" "uv.lock" ./

RUN uv sync --locked  

COPY "main.py" "credit_card_fraud_detection.pkl" ./

EXPOSE 8000

ENTRYPOINT [ "uv", "run", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000" ]




