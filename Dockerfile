FROM python:3.8.10 AS builder

# Install cmake and other build tools
RUN apt-get update && \
    apt-get install -y cmake build-essential && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY ./requirements.txt .

# Create virtual environment
RUN python -m venv /venv

# Activate venv and install requirements
RUN . /venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

# ----- STAGE 2: Runtime -----
FROM python:3.8.10

# Install bash (base image may not include it)
RUN apt-get update && \
    apt-get install -y bash libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only the necessary parts
COPY --from=builder /venv /venv
COPY dataset dataset
COPY model model
COPY custom.py custom.py

# Set ENV to use the venv
ENV VIRTUAL_ENV=/venv
ENV PATH="/venv/bin:$PATH"

# Start a shell by default (or override with docker-compose)
CMD ["bash", "-c", "source /venv/bin/activate && exec bash"]