FROM python:3.10-slim

# Install system deps for wheels (may be needed for catboost)
RUN apt-get update && apt-get install -y build-essential wget ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files (for inference only)
COPY . /app

RUN pip install --upgrade pip setuptools
RUN pip install -r requirements.txt

CMD ["python", "src/predict_with_catboost.py", "--input", "data/sample_input.csv", "--output", "results/preds.csv"]
