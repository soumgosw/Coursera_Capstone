
FROM python:3.7.5-stretch

RUN apt-get update && apt-get install -y \
python3-dev \
build-essential    
        
# Container configurations
WORKDIR /app
ADD . /app
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 80
ENV NAME AICapstoneTest

# Run main-app.py when the container launches
CMD ["python", "main-app.py"]
