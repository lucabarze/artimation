FROM registry.suse.com/bci/python:3.6

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

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
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]
