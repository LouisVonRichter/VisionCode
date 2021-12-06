import numpy as np
import cv2

# import RPi.GPIO as GPIO
import time

# Note: OpenCV uses BGR color space rather than RGB
# 

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
        (__, cnts, __) = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

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

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.DestroyWindow("Threshold")
            break

    cap.release()
    cv2.destroyAllWindows()


threshold()

# from picamera import PiCamera
# from picamera.array import PiRGBArray
# import time
# import cv2

# # initialize the camera and grab a reference to the raw camera capture
# camera = PiCamera()
# rawCapture = PiRGBArray(camera)

# # allow the camera to warmup
# time.sleep(2)

# # grab an image from the camera
# while(True):
#     rawCapture = PiRGBArray(camera)

#     camera.capture(rawCapture, format="bgr")
#     image = rawCapture.array

#     # Convert to a normal RGB space
#     im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # display the image on screen and wait for a keypress
#     cv2.imshow("Image", im)
#     cv2.waitKey(0)
