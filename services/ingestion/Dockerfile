FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy shared and service code
COPY shared/ /app/shared/
COPY services/ingestion/ /app/services/ingestion/
COPY services/__init__.py /app/services/__init__.py

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi>=0.111.0 \
    uvicorn[standard]>=0.29.0 \
    pydantic>=2.7.0 \
    pydantic-settings>=2.2.0 \
    sqlalchemy[asyncio]>=2.0.0 \
    asyncpg>=0.29.0 \
    redis[hiredis]>=5.0.0 \
    httpx>=0.27.0 \
    pyjwt>=2.8.0 \
    numpy>=1.26.0 \
    hl7apy>=1.3.4 \
    python-multipart>=0.0.9 \
    prometheus-fastapi-instrumentator>=6.1.0 \
    prometheus-client>=0.19.0

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8001

CMD ["uvicorn", "services.ingestion.main:app", "--host", "0.0.0.0", "--port", "8001"]
