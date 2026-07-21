FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requerments.txt .
RUN pip install --no-cache-dir -r requerments.txt

COPY api.py filter.py perception.py pred.py reg.py regression.py ./
COPY templates/ ./templates/
COPY data/ ./data/
COPY model_reg/ ./model_reg/

RUN test -f data/data_final.csv \
    && test -f data/data_final_2y.csv \
    && test -f data/neighborhood_ids.csv

EXPOSE 8050

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8050/')" || exit 1

CMD ["python", "reg.py", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8050"]
