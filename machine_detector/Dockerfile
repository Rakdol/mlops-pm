FROM python:3.10-slim

ENV PROJECT_DIR=machine_model
WORKDIR /${PROJECT_DIR}
ADD ./requirements.txt /${PROJECT_DIR}/
RUN apt-get -y update && \
    apt-get -y install apt-utils gcc curl && \
    pip install --no-cache-dir -r requirements.txt


COPY ./src/ /${PROJECT_DIR}/src/
COPY ./model/ /${PROJECT_DIR}/model/

ENV MODEL_DIRECTORY=/${PROJECT_DIR}/model/
ENV ONNX_FILE_NAME=machine_rf_0.onnx


COPY ./run.sh /${PROJECT_DIR}/run.sh
RUN chmod +x /${PROJECT_DIR}/run.sh
CMD ["./run.sh"]