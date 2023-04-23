FROM python:3.8-slim-buster

WORKDIR /app

RUN pip3 install numpy scipy flask 

COPY . .

EXPOSE 8888

CMD [ "python3", "main.py"]