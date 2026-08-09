FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY app.py .
COPY cache ./cache

ENV PYTHONPATH=/app/src

EXPOSE 7860

CMD ["python", "app.py"]
