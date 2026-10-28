
FROM python:3.12-slim

WORKDIR /app


COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY app.py agent.py guardrails.py gateway.py ./

EXPOSE 3000

CMD ["python", "app.py"]
