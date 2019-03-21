import os
import sys
import csv

import face_recognition
import numpy as np
from PIL import Image
from face_detection.misc.image_helpers import ImageHelpers


def main():
    images_directory = sys.argv[1]
    face_directory = sys.argv[2]
    scanned_images_registry = sys.argv[3]
    files = os.listdir(images_directory)
    output_files = os.listdir(face_directory)
    scanned_images = load_scanned_images_names(scanned_images_registry)

    for file_name in files:
        if file_name.endswith("jpg") \
                and file_name not in output_files\
                and file_name not in scanned_images:
            file_path = os.path.join(images_directory, file_name)
            append_scanned_image_name_to_file(scanned_images_registry, file_name)
            print("file_path: " + file_path)
            image = face_recognition.load_image_file(file_path)
            print(np.array(image).shape)
            face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="hog")
            print("I found {} face(s) in this photograph.".format(len(face_locations)))
            for face_location in face_locations:
                # Print the location of each face in this image
                diagonal_len = ImageHelpers.get_diagonal(face_location)
                print("Diagonal length: " + str(diagonal_len))
                if diagonal_len >= 300:
                    top, right, bottom, left = ImageHelpers.expand_image(image, face_location)
                    print("A face is located at pixel location "
                          "Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
                    face_image = image[top:bottom, left:right]
                    pil_image = Image.fromarray(face_image)
                    pil_image.save(os.path.join(face_directory, file_name))


def load_scanned_images_names(csv_file_path):
    scanned_images = []
    with open(csv_file_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count > 0:
                scanned_images.append(row["file_name"])
            line_count += 1
    return scanned_images


def append_scanned_image_name_to_file(csv_file_path, image_name):
    with open(csv_file_path, 'a') as csv_file:
        csv_file.write(image_name)


if __name__ == "__main__":
    main()
