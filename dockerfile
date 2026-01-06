# Use the official Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY predict.py .
COPY xgb_model_2.pkl .

# Expose port 5000 for Flask
EXPOSE 5000

# Set environment variables
ENV PORT=5000

# Run the Flask app
CMD ["python", "predict.py"]
