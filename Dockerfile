FROM nvcr.io/nvidia/tensorflow:20.10-tf2-py3

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install opencv-python==4.5.3.56 pandas matplotlib
EXPOSE 8888

CMD ["jupyter-notebook"]
