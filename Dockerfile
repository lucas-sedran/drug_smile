FROM python:3.10.6-buster

COPY requirements.txt requirements.txt
RUN pip install -r --no-cache-dir requirements.txt

COPY code._01_preprocessing code._01_preprocessing
COPY code.api code.api

CMD uvicorn code.api.api:app --host 0.0.0.0 --port $PORT
