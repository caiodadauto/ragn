FROM tensorflow/tensorflow:2.16.1-gpu
WORKDIR /app
RUN mkdir ragn
COPY ./poetry.lock ./pyproject.toml /app/
COPY ./ragn /app/ragn/
EXPOSE 5000
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:${PATH}"
RUN poetry install
