from ultralytics import NAS
from super_gradients.training import models
from super_gradients.common.object_names import Models
import cv2
from sort import *
from cvzone import cornerRect

# Load models
net = models.get(Models.YOLO_NAS_S, pretrained_weights="coco")

# Initialize tracker
mot_tracker = Sort()

# Load video
cap = cv2.VideoCapture("sample.mp4")

vehicles = [2, 3, 5, 7]  # specify the class IDs for vehicles

# Read frames
frame_number = -1
ret = True
while ret:
    frame_number += 1
    ret, frame = cap.read()

    # Frame resizing
    scale_percent = 50  # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    # Run detection and tracking
    if ret and frame_number % 1 == 0:  # Frame skipping
        # Detect vehicles
        detections = coco_model(frame)[0]
        car_boxes = []
        car_ids = []

        # Track vehicles
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])
        track_ids = mot_tracker.update(np.asarray(detections_))

        for track_id in track_ids:
            x1, y1, x2, y2, car_id = track_id
            car_boxes.append([x1, y1, x2, y2])
            car_ids.append(car_id)

        for car_box in car_boxes:
            x1, y1, x2, y2 = map(int, car_box)
            w = x2 - x1
            h = y2 - y1
            bbox = (x1, y1, w, h)
            cornerRect(frame, bbox, 70, 10, rt=0)

        frame = cv2.resize(frame, (1280, 720))
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # break the loop if 'q' is pressed
            break

# Close all OpenCV windows and release the video capture
cv2.destroyAllWindows()
cap.release()