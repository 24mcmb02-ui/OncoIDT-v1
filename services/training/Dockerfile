FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY shared/ /app/shared/
COPY services/__init__.py /app/services/__init__.py
COPY services/training/ /app/services/training/

RUN pip install --no-cache-dir \
    fastapi>=0.111.0 \
    uvicorn[standard]>=0.29.0 \
    pydantic>=2.7.0 \
    pydantic-settings>=2.2.0 \
    sqlalchemy[asyncio]>=2.0.0 \
    asyncpg>=0.29.0 \
    redis[hiredis]>=5.0.0 \
    pyjwt>=2.8.0 \
    mlflow>=2.13.0 \
    numpy>=1.26.0 \
    scikit-learn>=1.4.0 \
    prometheus-fastapi-instrumentator>=6.1.0 \
    prometheus-client>=0.19.0

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8008

CMD ["uvicorn", "services.training.main:app", "--host", "0.0.0.0", "--port", "8008"]
