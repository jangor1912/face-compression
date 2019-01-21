import os
import sys

import face_recognition
import numpy as np
from PIL import Image
from face_detection.misc.image_helpers import ImageHelpers


def main():
    images_directory = sys.argv[1]
    face_directory = sys.argv[2]
    useless_path = sys.argv[3]
    files = os.listdir(images_directory)
    output_files = os.listdir(face_directory)
    useless_files = os.listdir(useless_path)
    for file_name in files:
        if file_name.endswith("jpg") \
                and file_name not in output_files\
                and file_name not in useless_files:
            file_path = os.path.join(images_directory, file_name)
            print("file_path: " + file_path)
            image = face_recognition.load_image_file(file_path)
            print(np.array(image).shape)
            face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")
            print("I found {} face(s) in this photograph.".format(len(face_locations)))
            if len(face_locations) > 0:
                useless_image = os.path.join(useless_path, file_name)
                ImageHelpers.touch(useless_image)
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
                else:
                    useless_image = os.path.join(useless_path, file_name)
                    ImageHelpers.touch(useless_image)


if __name__ == "__main__":
    main()
