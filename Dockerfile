FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN python -m venv venv
RUN venv/bin/avtivate && pip install --no-cache-dir -r requirements.txt
RUN . venv/bin/activate && python -m pip install --upgrade pip setuptools
RUN /bin/bash

COPY model.pkl ./
COPY encoders.pkl ./
COPY app.py ./
COPY key.json ./

ENV PORT=8080
EXPOSE 8080

CMD ["gunicorn", "--bind", ":8080", "--workers", "1", "--threads", "8", "--timeout", "0", "app:app"]
