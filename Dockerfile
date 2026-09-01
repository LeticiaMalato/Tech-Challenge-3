FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen --no-dev

COPY src/app ./app
COPY model_artifacts ./model_artifacts

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

CMD ["uv", "run", "--frozen", "--no-dev", "uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
