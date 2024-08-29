FROM python:3.10.6-buster

WORKDIR /drug_smile

RUN pip install --upgrade pip

COPY requirements_api.txt requirements_api.txt
RUN pip install --no-cache-dir -r requirements_api.txt

COPY drug_smile/api drug_smile/api
COPY drug_smile/_01_preprocessing drug_smile/_01_preprocessing
COPY drug_smile/params.py drug_smile/params.py
COPY models models


CMD uvicorn drug_smile.api.api:app --host 0.0.0.0 --port 8010
