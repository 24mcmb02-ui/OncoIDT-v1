FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY shared/ /app/shared/
COPY services/__init__.py /app/services/__init__.py
# Copy websocket-hub to a Python-importable path (no hyphens)
COPY services/websocket-hub/ /app/services/websocket_hub/

RUN pip install --no-cache-dir \
    fastapi>=0.111.0 \
    uvicorn[standard]>=0.29.0 \
    pydantic>=2.7.0 \
    pydantic-settings>=2.2.0 \
    redis[hiredis]>=5.0.0 \
    pyjwt>=2.8.0 \
    websockets>=12.0 \
    prometheus-fastapi-instrumentator>=6.1.0 \
    prometheus-client>=0.19.0

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8010

CMD ["uvicorn", "services.websocket_hub.main:app", "--host", "0.0.0.0", "--port", "8010"]
