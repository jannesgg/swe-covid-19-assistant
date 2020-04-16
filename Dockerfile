# Covid-19 Assistant Sweden
# author: Jannes Germishuys

FROM python:3.6-slim

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git


RUN mkdir -p /usr/src/covid_app
COPY /app /usr/src/covid_app/app
COPY /data /usr/src/covid_app/data
COPY /requirements.txt /usr/src/covid_app/requirements.txt

WORKDIR /usr/src/covid_app/
RUN pip3 install -r requirements.txt

WORKDIR /usr/src/covid_app/app/
EXPOSE 5000
ENTRYPOINT [ "python" ]
CMD [ "app.py" ]
