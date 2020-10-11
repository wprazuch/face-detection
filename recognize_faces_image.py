# python recognize_faces_image.py --encodings encodings.pickle --image examples/example_01.png

import face_recognition
import argparse
import pickle
import cv2

from facerec.utils import load_image, get_face_data, to_bgr, recognize_face, annotate_image

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--encodings", required=True,
                    help="path to serialized db of facial encodings")
    ap.add_argument("-i", "--image", required=True,
                    help="path to serialized db of facial encodings")
    ap.add_argument("-d", "--detection_method", type=str, default="cnn",
                    help="face detection model to use: either 'hog' or 'cnn'")
    args = vars(ap.parse_args())
    return args


def main():
    args = parse_args()

    logging.info("Loadings encodings...")

    # encodings are our database of people
    with open(args['encodings'], 'rb') as handle:
        database = pickle.load(handle)

    image_rgb = load_image(args['image'])

    logging.info("Recognizing faces...")

    # detect faces, get their encodings and get bounding box cords
    boxes, encodings = get_face_data(image_rgb, args['detection_method'])

    names = []

    # for each face, try to recognize the person
    for encoding in encodings:
        name = recognize_face(encoding, database)
        names.append(name)

    # for cv2 image showing
    image_bgr = to_bgr(image_rgb)

    # put bounding boxes and names around the faces
    image = annotate_image(image_bgr, boxes, names)

    cv2.imshow("Image", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
