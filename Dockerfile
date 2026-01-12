# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/home/appuser" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

# Create directory for Streamlit config with proper permissions
RUN mkdir -p /home/appuser/.streamlit && chown -R appuser:appuser /home/appuser

# Copy and install dependencies first to leverage Docker layer caching
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy the source code into the container.
COPY . .

# Ensure appuser can read the application files
RUN chown -R appuser:appuser /app

# Switch to the non-privileged user to run the application.
USER appuser

# Expose the port that the application listens on.
EXPOSE 8502

# Run the application.
CMD ["python", "-m", "streamlit", "run", "streamlit_apps/app.py", "--server.port", "8502", "--server.address", "0.0.0.0", "--server.headless", "true"]
