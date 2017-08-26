import sys

import dlib
import os
from skimage import io

detector = dlib.get_frontal_face_detector()
win = dlib.image_window()
directory = "/home/lucasx/Documents/face_data/training_set"


def detect_face(face_filepath):
    img = io.imread(face_filepath)
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    # dets = detector(img, 1)
    dets, scores, idx = detector.run(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for i, d in enumerate(dets):
        print("Left: {} Top: {} Right: {} Bottom: {} score: {} face_type:{}".format(d.left(), d.top(), d.right(),
                                                                                    d.bottom(), scores[i], idx[i]))

    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(dets)


if __name__ == '__main__':
    detect_face("/home/lucasx/Documents/talor.jpg")

# Finally, if you really want to you can ask the detector to tell you the score
# for each detection.  The score is bigger for more confident detections.
# The third argument to run is an optional adjustment to the detection threshold,
# where a negative value will return more detections and a positive value fewer.
# Also, the idx tells you which of the face sub-detectors matched.  This can be
# used to broadly identify faces in different orientations.

# if len(os.listdir(directory)) > 0:
#     img = io.imread(sys.argv[1])
#     dets, scores, idx = detector.run(img, 1, -1)
#     for i, d in enumerate(dets):
#         print("Detection {}, score: {}, face_type:{}".format(
#             d, scores[i], idx[i]))
