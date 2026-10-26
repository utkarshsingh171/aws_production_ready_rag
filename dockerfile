
# syntax=docker/dockerfile:1

FROM python:3.12-slim

WORKDIR /app

# Copy dependencies separately so application changes
# do not invalidate the dependency installation layer.
COPY requirements.txt .

# Reuse pip's downloaded package cache across builds.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy application files after installing dependencies.
COPY app.py agent.py guardrails.py gateway.py ./

EXPOSE 3000

CMD ["python", "app.py"]
