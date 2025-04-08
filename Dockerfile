FROM python:3.10

WORKDIR /app

COPY requirements.txt /app/requirements.txt

COPY ./surya /app/surya

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

