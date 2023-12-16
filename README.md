# Automated Number Plate Recognition (ANPR) System

Welcome to my portfolio project showcasing an Automated Number Plate Recognition (ANPR) system. This project represents my expertise in integrating machine learning models with real-time image processing techniques to create a robust and efficient system for vehicle license plate detection and recognition.

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

## Installation and Usage

Please refer to the Installation section for setting up the project. To run the system, execute the `main_combined.py` script.

## Future Enhancements

I plan to continuously improve this project by integrating more advanced models, optimizing the code for better performance, and adding new features like different environment adaptability.

## License

This project is open-sourced under the MIT License.

---

This project is a testament to my skills in AI, machine learning, and system integration, showcasing my ability to create complex, real-world applications. It stands as an example of my dedication to continuous learning and applying cutting-edge technology to solve challenging problems.
