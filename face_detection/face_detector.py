import face_recognition

from face_detection.misc.image_helpers import ImageHelpers


class FaceDetector(object):
    def __init__(self, model="hog", upsampling_times=0):
        self.model = model
        self.upsampling_times = upsampling_times

    def detect_faces(self, image):
        face_locations = face_recognition.face_locations(image,
                                                         number_of_times_to_upsample=self.upsampling_times,
                                                         model=self.model)
        locations = []
        for face_location in face_locations:
            location = ImageHelpers.expand_image(image, face_location)
            locations.append(location)
        return locations
