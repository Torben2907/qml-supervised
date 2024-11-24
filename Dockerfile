FROM python:3.13

WORKDIR /tests/

COPY ./requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r ./requirements.txt

CMD ["sh", "-c", "pytest -v"]