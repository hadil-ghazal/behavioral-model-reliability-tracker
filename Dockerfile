#No AI was used to generate this content authored by HG on 11/21/25
#accidentally typed config content into docker, this v2 version corrects and updates syntax by HG 11/25/25

#Base Python image
FROM python:3.10-slim

#Preventing Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

#Creating and then switching to the working directory
WORKDIR /app

#Installing system level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

#Copying requirements and installing Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#Copying the rest of the project into the image
COPY . .

#Cloud Run / local container port
ENV PORT=8080
EXPOSE 8080

#Starting the FastAPI server
CMD ["uvicorn", "codes.api:app", "--host", "0.0.0.0", "--port", "8080"]

