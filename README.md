# Automated Number Plate Recognition (ANPR) System

Hi there, this is an Automated Number Plate Recognition System that I built as a personal project. This application represents my expertise in integrating machine learning models with real-time image processing techniques to create a robust and efficient system for vehicle license plate detection and recognition.

## Overview

The ANPR system is designed to detect and read vehicle license plates in real-time using webcam feeds. It employs state-of-the-art deep learning models for object detection and optical character recognition (OCR), along with tracking algorithms to maintain consistency across video frames.

## Technical Breakdown

### 1. Vehicle and License Plate Detection

- **YOLO Models**: The system uses two YOLO (You Only Look Once) models: one for detecting vehicles and another specifically trained for detecting license plates. These convolutional neural networks excel in real-time object detection due to their speed and accuracy.

### 2. Vehicle Tracking

- **SORT Algorithm**: To track the detected vehicles across frames, the SORT (Simple Online and Realtime Tracking) algorithm is implemented. This enhances the efficiency of the system by reducing redundant detections and maintaining identity consistency.

### 3. License Plate Recognition

- **EasyOCR**: Once a license plate is detected, EasyOCR is used to extract the alphanumeric text from the plate. This OCR engine is highly effective in reading texts from images under various conditions.

### 4. Image Processing

- **OpenCV and cvzone**: These libraries are utilized for image manipulation and drawing utilities. OpenCV processes the video feed, while cvzone is used for drawing bounding boxes around detected vehicles and license plates.

### 5. Custom Logic for Text Validation

- **Text Formatting and Validation**: The system includes custom logic to format and validate the extracted license plate text. This ensures that the recognized text adheres to standard license plate formats.

## Key Features

- Real-time processing with high accuracy.
- Robust detection and tracking of multiple vehicles simultaneously.
- Effective recognition of license plate text under varied conditions.
- Custom validation logic for recognized plate numbers.

## How I Built It

The ANPR system is a result of my extensive research and hands-on experience in machine learning, computer vision, and software development. I carefully selected and integrated each component to ensure that the system is not only accurate but also efficient in a real-time environment. Special attention was given to the interplay between the detection, tracking, and OCR components to create a seamless workflow.

## Results

Below is the results of running the software on a sample video:

![](https://github.com/natachi-o/Automatic-Number-Plate-Recognition-System/blob/main/ezgif-3-aeb0657e8f.gif)

The video depicts the car license plates being accurately detected with accompanying visualiation to further emphasise the detected license plates.

## Future Enhancements

I plan to continuously improve this project by integrating more advanced models, optimizing the code for better performance, and adding new features like different environment adaptability.

## License

This project is open-sourced under the MIT License.

---


