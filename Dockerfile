FROM python:3.9

RUN pip install --upgrade pip
RUN pip install poetry

WORKDIR /opt
COPY poetry.lock pyproject.toml /opt/

RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction

COPY . /opt
EXPOSE 8002

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
