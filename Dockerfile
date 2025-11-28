# Use Python 3.9
FROM python:3.9

# Set working directory
WORKDIR /code

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your files (app.py, models, templates)
COPY . .

# Grant permissions to the folder (Crucial for Hugging Face)
RUN chmod -R 777 /code

# Open the Cloud Port
EXPOSE 7860

# Run the App
CMD ["python", "app.py"]