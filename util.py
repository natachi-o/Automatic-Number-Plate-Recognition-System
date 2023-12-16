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


def write_csv(results, output_path):
    print(f"Writing results to {output_path}")
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_number', 'car_id', 'car_bbox',
                                                'licence_plate_bbox', 'licence_plate_bbox_score', 'licence_number',
                                                'licence_number_score'))

        for frame_number in results.keys():
            for car_id in results[frame_number].keys():
                print(results[frame_number][car_id])
                if 'car' in results[frame_number][car_id].keys() and \
                   'licence_plate' in results[frame_number][car_id].keys() and \
                   'text' in results[frame_number][car_id]['licence_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_number,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_number][car_id]['car']['bbox'][0],
                                                                results[frame_number][car_id]['car']['bbox'][1],
                                                                results[frame_number][car_id]['car']['bbox'][2],
                                                                results[frame_number][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_number][car_id]['licence_plate']['bbox'][0],
                                                                results[frame_number][car_id]['licence_plate']['bbox'][1],
                                                                results[frame_number][car_id]['licence_plate']['bbox'][2],
                                                                results[frame_number][car_id]['licence_plate']['bbox'][3]),
                                                            results[frame_number][car_id]['licence_plate']['bbox_score'],
                                                            results[frame_number][car_id]['licence_plate']['text'],
                                                            results[frame_number][car_id]['licence_plate']['text_score'])
                            )
        f.close()

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
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_

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