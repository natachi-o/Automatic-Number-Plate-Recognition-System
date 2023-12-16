FROM python:3.11

ADD main.py .

RUN pip install torch torchvision
RUN pip install opencv-python-headless
RUN pip install cvzone
RUN pip install easyocr
RUN pip install ultralytics

CMD ["python3", "./main.py"]