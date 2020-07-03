FROM ubuntu:20.04

COPY . /app
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.7 \
    python3-pip \
    nano \
    libglib2.0-0 \
    ffmpeg \
    libsm6 libxext6 libxrender-dev

# got some troubles when try to install numpy and scikit from requirements.txt
RUN pip3 install scikit-learn
RUN pip3 install numpy
RUN pip3 install numba==0.48
RUN pip3 install -r requirements.txt
RUN pip3 install opencv-python
RUN pip3 install pathlib

CMD python3 ./main.py

