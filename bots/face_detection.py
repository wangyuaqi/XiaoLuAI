import cv2
import sys

"""
    This code is provided for face detection!
"""

imagePath = '../res/dxx.jpg'
cascPath = '../res/haarcascade_frontalface_default.xml'
EYE_CASCADE_PATH = '../res/haarcascade_eye.xml'


def face_detect():
    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)

    # Read the image
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)

    result = {}

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            result['eye_x'] = ex
            result['eye_y'] = ey

    print(result)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    face_detect()
