FROM python:3.11-slim-buster

WORKDIR /code

COPY ./req.txt /code/req.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /code/req.txt

COPY ./app /code/app

ENV PYTHONUNBUFFERED 1

EXPOSE 84

CMD ["uvicorn", "app.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "84"]