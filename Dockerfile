# -------------------------------
# Base image
# -------------------------------
FROM python:3.10-slim

# -------------------------------
# Environment settings
# -------------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Streamlit config (important!)
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# -------------------------------
# System dependencies
# -------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------
# Working directory
# -------------------------------
WORKDIR /app

# -------------------------------
# Install Python dependencies
# -------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------
# Copy project files
# -------------------------------
COPY . .

# -------------------------------
# Expose Streamlit port
# -------------------------------
EXPOSE 8501

# -------------------------------
# Run Streamlit
# -------------------------------
CMD ["streamlit", "run", "app.py"]