FROM python:3.6
WORKDIR /app
COPY requirements.txt /app
RUN apt update -y
RUN apt upgrade -y
RUN apt-get install build-essential cmake pkg-config libx11-dev libatlas-base-dev libgtk-3-dev libboost-python-dev -y
RUN pip install -r ./requirements.txt
COPY . /app
ENV PYTHONPATH "${PYTHONPATH}:$(pwd)"
CMD ["python", "face_detection/flask_app/app.py"]~