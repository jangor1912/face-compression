import io

import face_recognition
import flask as flask

from face_detection.misc.image_helpers import ImageHelpers

app = flask.Flask(__name__)


@app.route("/detect_face", methods=["POST"])
def detect_face():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False,
            "face_locations": []}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = face_recognition.load_image_file(io.BytesIO(image))
            face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")
            print("I found {} face(s) in this photograph.".format(len(face_locations)))
            for face_location in face_locations:
                face_location = ImageHelpers.expand_image(image, face_location)
                top, right, bottom, left = face_location
                print("A face is located at pixel location "
                      "Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
                data["face_locations"].append(face_location)
                # indicate that the request was a success
                data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("Flask starting server..."
           "please wait until server has fully started"))
    app.run(host='0.0.0.0')
