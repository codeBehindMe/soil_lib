FROM python:3.9-buster

RUN mkdir app

COPY . app/

WORKDIR /app/

RUN pip install -r requirements.txt

RUN python train.py

ENTRYPOINT [ "streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0" ]