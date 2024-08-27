FROM python:3.10.6-buster

WORKDIR /prod

RUN pip install --upgrade pip

COPY requirements_api.txt requirements_api.txt
RUN pip install --no-cache-dir -r requirements_api.txt

COPY drug_smile/api api
COPY drug_smile/params.py params.py


CMD uvicorn api.api:app --host 0.0.0.0 --port 8010
