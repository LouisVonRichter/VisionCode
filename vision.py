import numpy as np
import cv2

import RPi.GPIO as GPIO
import time

Motor1_pos = 21
Motor1_neg = 20
Motor2_pos = 26
Motor2_neg = 19
# Note: OpenCV uses BGR color space rather than RGB
#
GPIO.setmode(GPIO.BCM)
GPIO.setup(Motor1_pos, GPIO.OUT)
GPIO.setup(Motor1_neg, GPIO.OUT)
GPIO.setup(Motor2_pos, GPIO.OUT)
GPIO.setup(Motor2_neg, GPIO.OUT)

# GPIO.output(Motor1_pos, True)
# GPIO.output(Motor1_neg, False)


def forward():
    GPIO.output(Motor1_pos, True)
    GPIO.output(Motor1_neg, False)
    GPIO.output(Motor2_pos, True)
    GPIO.output(Motor2_neg, False)


def spin():
    GPIO.output(Motor1_pos, False)
    GPIO.output(Motor1_neg, True)
    GPIO.output(Motor2_pos, True)
    GPIO.output(Motor2_neg, False)


def stop():
    GPIO.output(Motor1_pos, False)
    GPIO.output(Motor1_neg, False)
    GPIO.output(Motor2_pos, False)
    GPIO.output(Motor2_neg, False)


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

    lower_limit_threshold = np.array([5, 100, 120])
    upper_limit_threshold = np.array([77, 255, 231])
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if frame is not None:
            print("frame")
        else:
            print("No frame")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(
            frame, lower_limit_threshold, upper_limit_threshold
        )  # Set to yellow
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        frame = frame & mask_rgb

        # find contours
        (cnts, __) = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # draw one box around the biggest contour
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # place circle in centre of box
            cv2.circle(frame, (x + int(w / 2), y + int(h / 2)), 5, (0, 0, 255), -1)

            # display the coordinates of the centre of the box
            cv2.putText(
                frame,
                "x: " + str(x + int(w / 2)) + " y: " + str(y + int(h / 2)),
                (x + int(w / 2), y + int(h / 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
        cv2.imshow("Threshold", frame)
        # If contours are found, move the motors
        if len(cnts) > 0:
            forward()
        else:
            spin()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.DestroyWindow("Threshold")
            stop()
            break

    cap.release()
    cv2.destroyAllWindows()


threshold()
