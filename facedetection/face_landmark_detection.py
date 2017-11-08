"""
face landmarks detection powered by dlib
"""
import dlib
from skimage import io
import cv2

predictor_path = "/home/lucasx/Documents/PretrainedModels/shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
win = dlib.image_window()


def detect_face_landmarks(face_filepath):
    img = io.imread(face_filepath)

    win.clear_overlay()
    win.set_image(img)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                  shape.part(1)))
        # Draw the face landmarks on the screen.
        win.add_overlay(shape)

    win.add_overlay(dets)
    dlib.hit_enter_to_continue()


def draw_landmark(image_path):
    img = cv2.imread(image_path)
    faces = detector(img, 1)
    if len(faces) > 0:
        for k, d in enumerate(faces):
            cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (255, 255, 255))
            shape = predictor(img, d)
            for i in range(68):
                cv2.circle(img, (shape.part(i).x, shape.part(i).y), 2, (255, 191, 0), -1, 8)
                # cv2.putText(img, str(i), (shape.part(i).x, shape.part(i).y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 2555, 255))
    cv2.imshow('Frame', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    # detect_face_landmarks("/home/lucasx/Documents/talor.jpg")
    draw_landmark("/media/lucasx/Document/DataSet/Face/SCUT-FBP/Data_Collection/Data_Collection/SCUT-FBP-2.jpg")
