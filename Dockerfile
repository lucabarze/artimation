FROM registry.suse.com/bci/python:3.10

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install OS dependencies
RUN zypper install -y Mesa-libGL1 libglib-2_0-0 libgthread-2_0-0

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . /app/

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the application:
CMD ["gunicorn", "-b", "0.0.0.0:8000", "--worker-class", "gevent", "app:app"]
