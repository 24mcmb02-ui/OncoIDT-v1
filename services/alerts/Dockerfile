FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY shared/ /app/shared/
COPY services/__init__.py /app/services/__init__.py
COPY services/alerts/ /app/services/alerts/

RUN pip install --no-cache-dir \
    fastapi>=0.111.0 \
    uvicorn[standard]>=0.29.0 \
    pydantic>=2.7.0 \
    pydantic-settings>=2.2.0 \
    sqlalchemy[asyncio]>=2.0.0 \
    asyncpg>=0.29.0 \
    redis[hiredis]>=5.0.0 \
    httpx>=0.27.0 \
    prometheus-fastapi-instrumentator>=6.1.0 \
    prometheus-client>=0.19.0

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8005

CMD ["uvicorn", "services.alerts.main:app", "--host", "0.0.0.0", "--port", "8005"]
