FROM python:3.11.9-slim

WORKDIR /usr/src/myapp

COPY prediction-api/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY prediction-api/ .

COPY models/model.pkl ./models/model.pkl

EXPOSE 5000

ENV MODEL_REPO=/usr/src/myapp/models

CMD ["python3", "app.py"]