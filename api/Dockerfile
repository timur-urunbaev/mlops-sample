# Dockerfile for FastAPI

FROM python:3.8

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

EXPOSE 8000

RUN python model_register.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
