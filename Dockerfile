FROM quay.io/jupyter/minimal-notebook:afe30f0c9ad8

# Copy the conda lock file into the container
COPY conda-linux-64.lock /tmp/conda-linux-64.lock

# Switch to root user to install additional packages
USER root

# Install lmodern and make for Quarto PDF rendering and build tools
RUN apt update && apt install -y lmodern make \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Fix permissions
RUN fix-permissions "/home/${NB_USER}"

# Switch back to the notebook user
USER $NB_UID

# Update mamba environment and clean up
RUN mamba update --quiet --file /tmp/conda-linux-64.lock \
    && mamba clean --all -y -f \
    && fix-permissions "${CONDA_DIR}"

# Install Python packages with pip
RUN pip install --no-cache-dir \
    deepchecks==0.18.1 \
    seaborn==0.13.2 \
    altair-ally==0.1.1

# Log container build completion
RUN echo "Done Building Container!!"
