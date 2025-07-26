FROM python:3.12-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends curl git npm && rm -rf /var/lib/apt/lists/*

# Install uv (Python package manager by Astral)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Make uv globally accessible
ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR /app
COPY . /app

CMD [ "bash" ]