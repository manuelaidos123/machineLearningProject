# Base image
FROM python:latest

# Working directory
WORKDIR /app

COPY requeriments.txt ./

RUN pip install -r requeriments.txt

RUN pip install pyarrow

# Copy the Python script
COPY . .

VOLUME ["/credit/crx.data:/app/file1"]
VOLUME ["/adult/adult.data:/app/file2"]

# Command to run the script
CMD ["python", "main.py", "--data_dir1", "/app/file1", "--data_dir2", "/app/file2"]