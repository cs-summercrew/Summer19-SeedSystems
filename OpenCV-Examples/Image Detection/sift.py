import cv2
import numpy as np

def find_image_keypoints(image):

    sift = cv2.xfeatures2d.SIFT_create()
    kp, descriptors = sift.detectAndCompute(image,None)

    new_img=cv2.drawKeypoints(image,kp, None)

    cv2.imshow("Image",new_img)
    cv2.imwrite('sift_keypoints.jpg',new_img)

def match_images(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

    cv2.imshow("Img1", img1)
    cv2.imshow("Img2", img2)
    cv2.imshow("Matching result", img3)
    cv2.imwrite('sift_matching.jpg', img3)


# https://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html