# Dockerfile for MLFlow container

FROM python:3.9-slim

WORKDIR /app

RUN pip install mlflow

EXPOSE 5000

ENTRYPOINT ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
