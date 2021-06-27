import cv2
import numpy as np
import time
import os
import hand_tracking_module as htm

# Selected drawing colour
draw_colour = (255, 0, 255)

# Thickness of the brush and eraser
brush_thickness = 15
eraser_thickness = 40

cam_width = 1280
cam_height = 720

cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)

detector = htm.HandDetector(detection_confidence=0.85)
x_previous, y_previous = 0

# Creating a canvas
img_canvas = np.zeros((cam_height, cam_width, 3), np.uint8)

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

        # If index and middle are up, use selection mode
        if fingers[1] and fingers[2]:
            print("Selection mode activated")
            # When the index is above a certain height, depending on its x position,
            # change the colour accordingly
            if y1 < 125:
                if 250 < x1 < 450:
                    draw_colour = (255, 0, 255)
                elif 550 < x1 < 750:
                    draw_colour = (255, 0, 0)
                elif 800 < x1 < 950:
                    draw_colour = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    draw_colour = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), draw_colour, cv2.FILLED)

        # If index is up but middle is down, use drawing mode
        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), (x2, y2), (255, 0, 255), cv2.FILLED)
            print("Drawing mode activated")

            # On first draw, just draw a point
            if x_previous == 0 and y_previous == 0:
                x_previous, y_previous = x1, y1

            # If the "colour" is the eraser, make the size bigger
            if draw_colour == (0, 0, 0):
                cv2.line(img, (x_previous, y_previous), (x1, y1), draw_colour, eraser_thickness)
                cv2.line(img_canvas, (x_previous, y_previous), (x1, y1), draw_colour, eraser_thickness)
            else:
                # After that, continuously draw lines between index position and previous index position
                cv2.line(img, (x_previous, y_previous), (x1, y1), draw_colour, brush_thickness)
                cv2.line(img_canvas, (x_previous, y_previous), (x1, y1), draw_colour, brush_thickness)

            # Constantly update x_previous and y_previous
            x_previous, y_previous = x1, y1

    cv2.imshow("Image", img)
    cv2.imshow("Canvas", img_canvas)
    cv2.waitKey(1)
