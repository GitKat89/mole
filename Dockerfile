FROM python:3.6-buster
ENV PYTHONUNBUFFERED 1
WORKDIR /mole_django
COPY requirements.txt /mole_django/
RUN pip install -r requirements.txt
COPY . /mole_django/