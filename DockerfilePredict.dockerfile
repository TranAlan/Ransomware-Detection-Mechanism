FROM python:3.8

WORKDIR /usr/RDM/
RUN mkdir -p ./data/processed
RUN mkdir  ./src

COPY ./requirements.txt /requirements.txt
RUN pip3 install -r requirements.txt

COPY ./models ./
COPY ./data/processed/processed.csv ./data/processed/processed.csv
COPY ./src/utils ./src/
COPY ./src/models ./src/

WORKDIR ./src/models

CMD [ "python3", "predict_model.py" ]
