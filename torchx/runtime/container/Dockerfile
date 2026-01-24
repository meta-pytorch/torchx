FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files early so we don't have to redownload dependencies on code changes
COPY pyproject.toml uv.lock /app/

# Install dependencies (without the project itself)
RUN uv sync --frozen --no-install-project --extra dev

COPY . /app

# Install the project
RUN uv sync --frozen --extra dev

# Add the virtual environment to PATH so commands use the installed dependencies
ENV PATH="/app/.venv/bin:$PATH"
