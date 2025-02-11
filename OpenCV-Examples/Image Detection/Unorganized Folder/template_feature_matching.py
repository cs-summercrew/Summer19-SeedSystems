import cv2
import numpy as np
from matplotlib import pyplot as plt
def template_matching():

    img_rgb = cv2.imread('messi5.jpg')
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    template = cv2.imread('template.jpg',0)
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where( res >= threshold)

    print("before loop")
    for pt in zip(*loc[::-1]):
        print("Draw rectangle")
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)
    print("after loop")
    cv2.imshow("Detected", img_rgb)