FROM tensorflow/tensorflow:2.16.1-gpu
RUN mkdir -p /app/scripts
COPY ./poetry.lock ./pyproject.toml /app/
COPY ./ragn /app/ragn/
COPY ./scripts/ragn_config.yaml ./scripts/run_training.py /app/scripts/
# EXPOSE 5000
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:${PATH}"
WORKDIR /app/scripts
RUN poetry install
RUN poetry run pip install tensorboard_plugin_profile
