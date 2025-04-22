# Use an official Python runtime as a parent image
# Choose a version compatible with your dependencies (e.g., 3.10 or 3.11)
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed system dependencies (if any)
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*

# Install Python dependcencies
# --no-cache-dir reduces image size
# --upgrade pip ensures you have the latest pip
RUN pip install --no-cache-dir --upgrade pip -r requirements.txt

# Copy the rest of your application code into the container at /app
# This includes your main script (e.g., app.py), src/, config/ directories
COPY . .

# Make port 8501 available to the world outside this container (Streamlit's default port)
EXPOSE 8501

# Define environment variables (Optional: can be overridden at runtime)
# It's generally better to pass sensitive keys at runtime
# ENV OLLAMA_HOST="http://host.docker.internal:11434" # Example if Ollama runs on host

# Command to run the Streamlit application
# Use 0.0.0.0 to make it accessible from outside the container
# Assumes your script is named app.py
CMD ["streamlit", "run", "deepseek_rag_agent.py", "--server.port=8501", "--server.address=0.0.0.0"]