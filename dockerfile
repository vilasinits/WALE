FROM python:3.12-slim

# Environment setup
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies, including gfortran for CAMB
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    git \
    libfftw3-dev \
    libgomp1 \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY README.md .

# Install Python dependencies, including CAMB
RUN pip install --upgrade pip && pip install .[dev]

# Copy the rest of the code
COPY . .

# Expose Jupyter port
EXPOSE 8888

# Default shell
CMD ["/bin/bash"]
