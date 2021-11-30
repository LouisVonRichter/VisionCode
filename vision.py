import numpy as np
import cv2
import RPi.GPIO as GPIO
import time

# Note: OpenCV uses BGR color space rather than RGB


def threshold():
    # Load image
    cap = cv2.VideoCapture(0)
    # set the width and height
    frame_width = cap.get(3)
    frame_height = cap.get(4)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    cv2.namedWindow("Threshold", 0)

    print("Press Q to quit")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if frame is not None:
            print("frame")
        else:
            print("No frame")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(
            frame, (0, 125, 125), (80, 255, 255)
        )  # Set to stanley yellow
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        frame = frame & mask_rgb
        cv2.imshow("Threshold", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.DestroyWindow("Threshold")
            break

    cap.release()
    cv2.destroyAllWindows()


threshold()
