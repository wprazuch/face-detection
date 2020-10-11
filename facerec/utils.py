import cv2
import face_recognition


def annotate_image(image, boxes, names):
    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    return image


def get_face_data(image, detection_method):
    boxes = face_recognition.face_locations(image, model=detection_method)
    encodings = face_recognition.face_encodings(image, boxes)
    return boxes, encodings


def load_image(path):
    image = cv2.imread(path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb


def recognize_face(encoding, database):
    matches = face_recognition.compare_faces(database['encodings'], encoding)
    name = "Unknown"

    if True in matches:
        matched_idxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}

        for i in matched_idxs:
            name = database["names"][i]
            counts[name] = counts.get(name, 0) + 1

        name = max(counts, key=counts.get)

    return name


def to_bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
