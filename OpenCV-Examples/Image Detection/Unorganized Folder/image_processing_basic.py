import cv2

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def get_image( S ):
    """
        Given a string S, use OpenCV methods to open that image
        and return an image object
        input: string that represents a filename
        output: OpenCV image object 
    """
    raw_image = cv2.imread(S, cv2.IMREAD_COLOR)
    ret_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

    return ret_image


def show_image(image):
    """
        Displays an image using matplotlib functions
        input: OpenCV image object
        output: N/A
    """
    plt.imshow(image)
    plt.show()


def rotate_90_left(image):
    """
        Rotates an imagem object 90 degrees to the left using
        OpenCV
        input: OpenCV image object
        output: OpenCV image object
    """
    num_rows, num_cols = image.shape[:2]
    center = (num_rows/2, num_cols/2)
    M = cv2.getRotationMatrix2D(center, 90, 1.0)
    rotated90 = cv2.warpAffine(image, M, (num_rows, num_cols))
    return rotated90


def mirror_image(image):
    """
        Takes in an image object and mirrors it horizontally
        input: OpenCV image object
        output: OpenCV image object
    """
    horizontal_img = cv2.flip( image, 1 )
    return horizontal_img


def rotate_90_right(image):
    """
        Takes in an image object and rotates it 90 degrees to the right
        input: OpenCV image object
        output: OpenCV image object
    """
    num_rows, num_cols, num_chans = image.shape
    center = (num_cols/2, num_rows/2)
    M = cv2.getRotationMatrix2D(center, 270, 1.0)
    rotated90 = cv2.warpAffine(image, M, (num_rows, num_cols))
    return rotated90


def rotate_180(image):
    """
        Takes in an image object and rotates it 180 degrees
        input: OpenCV image object
        output: OpenCV image object
    """
    num_rows, num_cols, num_chans = image.shape
    center = (num_cols/2, num_rows/2)
    M = cv2.getRotationMatrix2D(center, 180, 1.0)
    rotated180 = cv2.warpAffine(image, M, (num_cols, num_rows))
    return rotated180


def new_image():
    """
        Creates and returns a new blank image
        input: OpenCV image object
        output: OpenCV image object
    """
    new_image = np.zeros((new_row,new_col,3), dtype=int)
    return new_image


def filter_inv( image ):
    """ 
        an example of a pixel-by-pixel filter 
        input: an r, g, b image
        output: a transformed r, g, b image
    """
    new_image = image.copy()
    num_rows, num_cols, num_chans = new_image.shape
    for row in range(num_rows):
        for col in range(num_cols):
            r, g, b = image[row,col]
            new_image[row,col] = [255-r, 255-g, 255-b]
    return new_image 


def filter_bgr( image ):
    """ 
        an example of a pixel-by-pixel filter 
        input: an r, g, b image
        output: a transformed r, g, b image
    """
    new_image = image.copy()
    num_rows, num_cols, num_chans = new_image.shape
    for row in range(num_rows):
        for col in range(num_cols):
            r, g, b = image[row,col]
            new_image[row,col] = [g, b, r]
    return new_image


def two_image_filter( image1, image2 ):
    """
        filter that takes the average of the pixel values between two
        pictures and sets them as the pixels of a new image
        input: OpenCV image object
        output: OpenCV image object
    """
    num_rows1, num_cols1, num_chans1 = image1.shape
    num_rows2, num_cols2, num_chans2 = image2.shape
    new_row = min(num_rows1,num_rows2)
    new_col = min(num_cols1,num_cols2)
    new_image = np.zeros((new_row,new_col,3), dtype=int)
    for row in range(new_row):
        for col in range(new_col):
            r1, g1, b1 = image1[row,col]
            r2, g2, b2 = image2[row,col]
            new_r = (r1+r2)/2
            new_g = (g1+g2)/2
            new_b = (b1+b2)/2
            new_image[row,col] = [new_r , new_g, new_b]
    return new_image
