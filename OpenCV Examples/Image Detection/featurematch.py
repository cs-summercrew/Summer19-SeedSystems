# Authors: CS-World Domination Summer19 - CB
import numpy as np
import cv2
import matplotlib.pyplot as plt

#this is the number of connections made between the two images
NUM_EDGES=5
NUM_NODES=NUM_EDGES*2

img1 = cv2.imread('Images/pillow.jpg',0)
img2 = cv2.imread('Images/objects.jpg',0)

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:NUM_NODES],None, flags=2)
plt.imshow(img3)
plt.show()
