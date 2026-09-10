FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEMAND_INPUT_FOLDER=/input/demand
ENV HOST=0.0.0.0

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY weather_pipeline ./weather_pipeline
COPY uk_training_data_prep ./uk_training_data_prep
COPY ml_training ./ml_training
COPY ui ./ui
COPY scripts/start-render.sh ./scripts/start-render.sh
COPY README.md .

EXPOSE 8765

RUN chmod +x ./scripts/start-render.sh

CMD ["./scripts/start-render.sh"]
