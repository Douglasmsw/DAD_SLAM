FROM continuumio/miniconda3

WORKDIR /app

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Demonstrate the environment is activated:
RUN echo "Make sure torch is installed:"
RUN python -c "import torch"

# The code to run when container is started:
COPY train.py .
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "myenv", "python", "run.py"]