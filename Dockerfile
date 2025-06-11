FROM python:3.11

WORKDIR /app

COPY . /app

RUN pip install fastapi uvicorn mlflow scikit-learn pandas pydantic

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
