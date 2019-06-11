import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

color = True

while True:
    ret, orig = cap.read()
    frame=cv2.resize(orig, None, fx=.5, fy=.5)
    cv2.imshow('frame', frame)

    k = cv2.waitKey(20) & 0xFF
    k_char = chr(k)

    if k_char == 's':
        color = not color
    if not color:
        frame = gray
    if k_char == 'q':
        break

cap.release()
cv2.destroyAllWindows()
