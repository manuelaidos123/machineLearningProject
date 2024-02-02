FROM python:latest

WORKDIR /app

COPY . /app

RUN pip install -r requeriments.txt

CMD ["python", "main.py"]