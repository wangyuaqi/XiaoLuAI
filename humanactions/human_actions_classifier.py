import os
import cv2
import numpy as np


def video_to_images(avi_filepath, image_dir):
    if os.path.exists(image_dir):
        os.rmdir(image_dir)
    os.makedirs(image_dir)
    cap = cv2.VideoCapture(avi_filepath)
    frame_index = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        for _ in range(len(frame)):
            cv2.imwrite(os.path.join(image_dir, '%d.jpg') % frame_index, frame)
            frame_index += 1

            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('frame', gray)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_to_images('/home/lucasx/Documents/Dataset/HumanActions/running/person01_running_d1_uncomp.avi',
                    '/tmp/humanactions/')
