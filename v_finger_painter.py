import cv2
import numpy as np
import time
import os
import hand_tracking_module as htm

cam_width = 1280
cam_height = 720

cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)

detector = htm.HandDetector(detection_confidence=0.85)

while True:
    success, img = cap.read()
    # Rotating and flipping the image to make drawing more intuitive
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.flip(img, 1)

    # Finding hands and landmarks
    img = detector.find_hands(img)
    landmarks_list = detector.find_position(img, draw=False)

    if len(landmarks_list) != 0:
        print(landmarks_list)

        # x1, y1 will be the coordinates of the index finger tip
        x1, y1 = landmarks_list[8][1:]
        # x2, y2 will be the coordinates of the middle finger tip
        x2, y2 = landmarks_list[12][1:]

        # Getting list of fingers that are up
        fingers = detector.fingers_up()

        if fingers[1] and fingers[2]:
            print("Selection mode activated")

        if fingers[1] and fingers[2] == False:
            print("Drawing mode activated")

    cv2.imshow("Image", img)
    cv2.waitKey(1)