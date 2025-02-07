FROM nvcr.io/nvidia/tritonserver:24.09-py3

COPY ./models /models
COPY ./download.py /download.py

RUN apt-get update
RUN apt-get install ffmpeg -y
RUN apt-get install git-lfs -y

RUN pip install resemble-enhance --upgrade

RUN python3 /download.py