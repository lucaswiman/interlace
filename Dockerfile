FROM ubuntu:24.04
ARG UID=1000
ARG GID=1000
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y curl ca-certificates gnupg && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_22.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list && \
    apt update && apt install -y \
    build-essential \
    rustc \
    cargo \
    nodejs \
    python3 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv globally
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin/:$PATH"

# Safely handle the ubuntu user
RUN if getent passwd ubuntu; then userdel -f ubuntu; fi && \
    if getent group ubuntu; then groupdel ubuntu; fi && \
    groupadd -g $GID ubuntu && \
    useradd -m -u $UID -g $GID ubuntu

RUN npm install -g @google/gemini-cli

USER ubuntu
# Re-install uv for ubuntu user or use absolute path
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/ubuntu/.local/bin/:$PATH"

WORKDIR /app
RUN uv venv --python=3.13
