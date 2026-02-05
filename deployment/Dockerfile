# Production Dockerfile for Yield Prediction API
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create non-root user for security
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ /app/

# Copy models
COPY models/ /app/models/

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port (Railway uses dynamic PORT)
EXPOSE 8000

# Set default port (Railway overrides with $PORT)
ENV PORT=8000

# Run with dynamic port from Railway
CMD uvicorn main:app --host 0.0.0.0 --port $PORT --workers 4
