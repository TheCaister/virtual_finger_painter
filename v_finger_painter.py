import cv2
import numpy as np
import time
import os
import hand_tracking_module as htm

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    success, img = cap.read()

    cv2.imshow("Image", img)
    cv2.waitKey(1)