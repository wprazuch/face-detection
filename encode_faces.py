# python encode_faces.py --input dataset --output encodings.pickle

from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

from facerec.utils import load_image, get_face_data

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def parse_args():
    ap = argparse.ArgumentParser('Script for encoding faces from a dataset')
    ap.add_argument("-i", "--input", required=True,
                    help="path to input directory of faces + images")
    ap.add_argument("-o", "--output", required=True,
                    help="path to serialized db of facial encodings")
    ap.add_argument("-d", "--detection_method", type=str, default="cnn",
                    help="face detection model to use: either 'hog' or cnn")

    args = vars(ap.parse_args())
    return args


def main():
    args = parse_args()
    logging.info("Quantifying faces...")

    image_paths = list(paths.list_images(args["input"]))

    known_encodings = []
    known_names = []

    for (i, image_path) in enumerate(image_paths):
        logging.info("Processing image {}/{}".format(i + 1, len(image_paths)))

        name = image_path.split(os.path.sep)[-2]

        image_rgb = load_image(image_path)

        _, encodings = get_face_data(image_rgb, args['detection_method'])

        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(name)

    logging.info(f"Serializaing encodings into pickle file {args['output']}")
    data = {"encodings": known_encodings, "names": known_names}

    with open(args['output'], 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
