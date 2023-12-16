from ultralytics import YOLO
import cv2
import cvzone
from sort import *
import string
import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}

# Define utility functions
def licence_complies_format(text):
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
            (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
            (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
            (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
            (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
            (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
            (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    else:
        return False

def format_licence(text):
    licence_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            licence_plate_ += mapping[j][text[j]]
        else:
            licence_plate_ += text[j]

    return licence_plate_

def read_licence_plate(licence_plate_crop):
    detections = reader.readtext(licence_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')

        if licence_complies_format(text):
            return format_licence(text), score

    return None, None


def get_car(licence_plate, vehicle_track_ids):
    x1, y1, x2, y2, score, class_id = licence_plate

    found = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_index = j
            found = True
            break

    if found:
        return vehicle_track_ids[car_index]

    return -1, -1, -1, -1, -1

# Load models
coco_model = YOLO("yolov8n.pt")
license_plate_detector = YOLO("license_plate_detector.pt")

# Initialize tracker
mot_tracker = Sort()

# Initialize results dictionary
results = {}

# Load video
cap = cv2.VideoCapture(0)

vehicles = [2, 3, 5, 7]

# Read frames
frame_number = -1
ret = True
while ret:
    frame_number += 1
    ret, frame = cap.read()

    # Frame resizing
    scale_percent = 50 # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

    # Run detection and tracking
    if ret and frame_number % 1 == 0:  # Frame skipping
        results[frame_number] = {}

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

        # Detect licence plates
        licence_plates = license_plate_detector(frame)[0]
        for licence_plate in licence_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = licence_plate

            # Assign licence plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(licence_plate, track_ids)

            if car_id != -1:
                # Crop licence plate
                licence_plate_crop = frame[int(y1): int(y2), int(x1): int(x2), :]

                # Process licence plate
                licence_plate_crop_gray = cv2.cvtColor(licence_plate_crop, cv2.COLOR_BGR2GRAY)
                _, licence_plate_crop_thresh = cv2.threshold(licence_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # Save the processed licence plate image to a file
                cv2.imwrite('licence_plate_thresh.png', licence_plate_crop_thresh)

                # Read licence plate number
                licence_plate_text, licence_plate_text_score = read_licence_plate(licence_plate_crop_thresh)

                if licence_plate_text is not None:
                    car_boxes.append([xcar1, ycar1, xcar2, ycar2])
                    car_ids.append(car_id)
                    results[frame_number][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                     'licence_plate': {'bbox': [x1, y1, x2, y2],
                                                                       'text': licence_plate_text,
                                                                       'bbox_score': score,
                                                                       'text_score': licence_plate_text_score}}
        for car_box in car_boxes:
            x1, y1, x2, y2 = map(int, car_box)
            w = x2 - x1
            h = y2 - y1
            bbox = (x1, y1, w, h)
            cvzone.cornerRect(frame, bbox, 70, 20, rt=0)

        for licence_plate in licence_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = licence_plate
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # red rectangle for licence plates

        frame = cv2.resize(frame, (1280, 720))
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # break the loop if 'q' is pressed

            break

# Close all OpenCV windows and release the video capture
cv2.destroyAllWindows()
cap.release()
