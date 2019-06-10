import cv2
import numpy as np
img = cv2.imread("messi5.jpg", cv2.IMREAD_GRAYSCALE)

sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()
#increase nfeatures to show more points
orb = cv2.ORB_create(nfeatures=1500)

keypoints_sift, descriptors = sift.detectAndCompute(img, None)
keypoints_surf, descriptors = surf.detectAndCompute(img, None)
keypoints_orb, descriptors = orb.detectAndCompute(img, None)

#change keypoints_orb to any of the above
img = cv2.drawKeypoints(img, keypoints_orb, None)
cv2.imshow("Image", img)

#press any key to close image window
cv2.waitKey(0)
cv2.destroyAllWindows()