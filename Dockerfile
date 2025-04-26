# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# Set the port the application will run on
ENV PORT 8000
# Set the working directory in the container
WORKDIR /app

# Install system dependencies if needed (e.g., for certain libraries)
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Copy only requirements first to leverage Docker cache
COPY requirements.txt .
# Ensure pip is up-to-date and install requirements
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY ./app ./app/
# Copy the tests directory
COPY ./tests ./tests/
# Copy other necessary files like the example env file to the WORKDIR
COPY .env.example .

# Expose the port the app runs on
EXPOSE $PORT

# Define the command to run the application
# Use 0.0.0.0 to make it accessible from outside the container
# The number of workers can be adjusted based on the server resources
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
# Using reload for development purposes as specified in earlier tasks,
# but for production, remove --reload and potentially increase workers.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]