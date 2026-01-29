# Use a lightweight Python base image
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --default-timeout=100 --retries 5 -r requirements.txt

# Copy the rest of the project files
COPY . .

# Set environment variables (optional for logging)
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
