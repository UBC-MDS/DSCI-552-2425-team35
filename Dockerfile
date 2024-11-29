FROM quay.io/jupyter/minimal-notebook:afe30f0c9ad8

COPY conda-linux-64.lock /tmp/conda-linux-64.lock

RUN mamba update --quiet --file /tmp/conda-linux-64.lock
RUN mamba clean --all -y -f
RUN pip install deepchecks==0.18.1 seaborn==0.13.2 altair-ally==0.1.1
RUN fix-permissions "${CONDA_DIR}"
RUN fix-permissions "/home/${NB_USER}"

RUN echo "Done Building!!"