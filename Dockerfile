FROM python:3.10.6-buster

WORKDIR /prod

RUN pip install --upgrade pip

COPY requirements_api.txt requirements_api.txt
RUN pip install -r requirements_api.txt

COPY drug_smile drug_smile
COPY models models

CMD uvicorn drug_smile.api.api:app --host 0.0.0.0 --port 8010
