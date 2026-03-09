FROM python:3.11-slim

WORKDIR /app

# Install build dependencies if needed (e.g., for NumPy)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir awscli

# Increase timeout for pip install
RUN pip install --no-cache-dir --timeout=300 -r requirements.txt

CMD ["python3", "app.py"]