# Dockerfile for MLFlow container

FROM python:3.8

WORKDIR /app

RUN pip install mlflow==2.15.1

EXPOSE 5000

ENTRYPOINT ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
