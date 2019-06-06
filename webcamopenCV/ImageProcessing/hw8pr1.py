# coding: utf-8

# # Introduction to OpenCV and pixel-processing (cs35 spring 19)

# Check that you have opencv:
import cv2

# Check that you have Pillow:
from PIL import Image


# if these doesn't work, try
"""
conda install -c anaconda opencv
pip install Pillow
"""

# documentation for OpenCV:   http://docs.opencv.org/3.1.0/
# tutorials are in many places, e.g., http://docs.opencv.org/3.1.0/d6/d00/tutorial_py_root.html


# other imports!
import numpy as np
from matplotlib import pyplot as plt
import cv2

# run 
# %matplotlib 
# 
# for smoother image-display...



# ### Images as pictures: reading, writing, and resize

def opencv_test1():
    """ some initial trials... read and show images """
    # options include cv2.IMREAD_COLOR # color, with no alpha channel
    # also cv2.IMREAD_GRAYSCALE (== 0)
    # or cv2.IMREAD_UNCHANGED (includes alpha)

    # Reading and color-converting an image to RGB
    raw_image = cv2.imread('messi5.jpg',cv2.IMREAD_COLOR) 
    #raw_image = cv2.imread('monalisa.jpg',cv2.IMREAD_COLOR) 

    # convert an OpenCV image (BGR) to an "ordinary" image (RGB) 
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

    plt.imshow(image)       #cv2.imshow("img", image)      
    plt.xticks([]), plt.yticks([])  # to hide axis labels
    plt.show()              #cv2.destroyAllWindows()  
    input("Hit enter to continue...")

    # In[ ]:

    # let's get the flag image next
    raw_image = cv2.imread('flag.jpg',cv2.IMREAD_COLOR)
    flag = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    plt.imshow(flag)
    plt.show()
    print("flag.shape is", flag.shape)
    input("Hit enter to continue...")

    return flag   # for use further downstream...



def opencv_test1b():
    """ some initial trials... read and show images """
    # options include cv2.IMREAD_COLOR # color, with no alpha channel
    # also cv2.IMREAD_GRAYSCALE (== 0)
    # or cv2.IMREAD_UNCHANGED (includes alpha)

    # Reading and color-converting an image to RGB
    raw_image = cv2.imread('spam.png',cv2.IMREAD_COLOR) 
    #raw_image = cv2.imread('monalisa.jpg',cv2.IMREAD_COLOR) 

    # convert an OpenCV image (BGR) to an "ordinary" image (RGB) 
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

    plt.imshow(image)    
    plt.show()
    # plt.xticks([]), plt.yticks([])  # to hide axis labels
    input("Hit enter to continue...")

    # In[ ]:

    # let's get the flag image next
    raw_image = cv2.imread('flag.jpg',cv2.IMREAD_COLOR)
    flag = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    plt.imshow(flag) 
    print("flag.shape is", flag.shape)
    input("Hit enter to continue...")

    return flag   # for use further downstream...




def opencv_test2():
    """ a second set of tests... resize and image and write to file... """
    raw_image = cv2.imread('flag.jpg',cv2.IMREAD_COLOR)
    flag = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    plt.imshow(flag)
    plt.show()
    print("flag.shape is", flag.shape)
    input("Hit enter to continue...")

    # want to resize a new image - no problem:
    small_flag = cv2.resize(flag, dsize=(50,50), interpolation=cv2.INTER_AREA)  
    plt.imshow(small_flag) 
    plt.show()
    print("flag.shape is", small_flag.shape)
    input("Hit enter to continue...")
    # there are three resize algorithms (cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC), described here:
    # http://docs.opencv.org/3.1.0/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d
    raw_image = cv2.cvtColor(small_flag, cv2.COLOR_RGB2BGR) # convert back!
    
    cv2.imwrite( "small_flag.png", raw_image )
    print("The file small_flag.png was written...")
    print("Also returning that image...")

    return small_flag






def opencv_test3():
    """ a third set of tests... reread image and access its pixels and channels... """

    # re-read to check the small image:
    raw_image = cv2.imread('small_flag.png',cv2.IMREAD_COLOR)
    small_flag = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    plt.imshow(small_flag) 
    input("Hit enter to continue...")


    # ### Images as pixels: accessing their individual components
    print("Here are the first four pixels in the top row:")
    print(small_flag[0,0]) # row, col == 0,0  r,g,b == 37, 138, 175
    print("Its r, g, and b components:", small_flag[0,0,0], ",", small_flag[0,0,1], ",", small_flag[0,0,2])
    print(small_flag[0,1]) # row, col
    print(small_flag[0,2]) # row, col
    print(small_flag[0,3]) # row, col

    if True:
        print("printing all pixels of the small_flag image")
        NUMROWS, NUMCOLS, NUMCOLORS  = small_flag.shape
        for r in range(NUMROWS):
            for c in range(NUMCOLS):
                print(small_flag[r,c], end=" ")
            print()
        print("end of printing all pixels")

    return small_flag   # for downstream processing...








def opencv_test4():
    """ an example of a single-image filter
        a filter is an image-in -> image-out transformation
    """
    # blur - a common photoshop filter...
    raw_image = cv2.imread('spam.png',cv2.IMREAD_COLOR) 
    spam = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

    plt.figure()
    plt.imshow(spam)
    plt.title("Unblurred")
    plt.show()
    input("Hit enter to continue...")
    blurred_spam = cv2.blur( spam, ksize=(5,5) )  # average of 5x5 box (each direction)
    plt.figure()
    plt.imshow(blurred_spam) 
    plt.title("Blurred")
    plt.show()







def opencv_test5():
    """ more image-filter examples """
    # but we want to be able to go _beyond_ Photoshop (as needed...)
    raw_image = cv2.imread('flag.jpg',cv2.IMREAD_COLOR) 
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

    #image_r, image_g, image_b = cv2.split( image )  # the three channels OR
    image_r = image[:,:,0]  # red
    image_g = image[:,:,1]  # gre
    image_b = image[:,:,2]  # blu

    # here is a 1d layout of images:
    fig1, ax1 = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False)
    ax1[0].imshow(image);  
    ax1[0].set_title("orig");  
    ax1[0].axis('off')
    ax1[1].imshow(image_b, cmap='gray');   
    ax1[1].set_title("blue channel alone"); 
    ax1[1].axis('off')
    plt.show(fig1)

    # here is a 2d layout of grayscale images:
    image_r = image[:,:,0]  # red
    image_g = image[:,:,1]  # gre
    image_b = image[:,:,2]  # blu

    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False)
    ax[0,0].imshow(image);  
    ax[0,0].set_title("orig")
    ax[0,1].imshow(image_b, cmap='gray');  
    ax[0,1].set_title("red")
    ax[1,0].imshow(image_r, cmap='gray');  
    ax[1,0].set_title("green")
    ax[1,1].imshow(image_g, cmap='gray');  
    ax[1,1].set_title("blue")
    for row in range(2):
        for col in range(2):
            ax[row,col].axis('off')
    plt.show(fig)




def opencv_test6():
    """ more image-filter examples - base yours from these examples... """
    # #### an example of a color conversion in OpenCV
    # but we want to be able to go _beyond_ Photoshop (as needed...)
    raw_image = cv2.imread('flag.jpg',cv2.IMREAD_COLOR) 
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

    # color conversions are here:
    # http://docs.opencv.org/3.1.0/d7/d1b/group__imgproc__misc.html#gsc.tab=0
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    plt.imshow(gray_image, cmap='gray', vmin=0, vmax=255)
    plt.title("Grayscale")
    plt.show()
    input("Hit return to continue...")

    # what will happen here?
    gray2 = gray_image//2
    plt.imshow(gray2, cmap='gray', vmin=0, vmax=255)
    plt.title("HALF grayscale")
    plt.show()
    input("Hit return to continue...")

    # example of thresholding!
    # numpy arrays are cleverer than plain lists...
    a = np.array( [ [1,2,3], [4,5,6], [7,8,9] ] )
    print(a)
    indices = a < 4.5
    print(indices)
    a[indices] = 0
    print(a)

    # types are important!
    # numpy types: http://docs.scipy.org/doc/numpy-1.10.1/user/basics.types.html
    a = a.astype(float)
    print(a)
    a = a.astype(int)
    print(a)


    # #### let's try with images
    # here are some thresholded images
    image_gt142_indices = gray_image > 142
    image_gt142 = gray_image.copy()
    image_gt142[image_gt142_indices] = 255
    image_gt142[np.logical_not(image_gt142_indices)] = 0
    # note that this shows how to take logical operators of index arrays...

    plt.imshow(image_gt142, cmap='gray', vmin=0, vmax=255)
    plt.title("Thresholded at gray level == 142")
    input("Hit return to continue...")

    # here are some other thresholded images
    image_lt42_indices = gray_image < 42
    image_lt42 = image.copy()
    image_lt42[image_lt42_indices] = 255
    image_lt42[np.logical_not(image_lt42_indices)] = 0
    plt.imshow(image_lt42, cmap='gray', vmin=0, vmax=255)
    plt.show()
    input("Hit return to continue...")

    image_not_btw_42_142_indices =   np.logical_or(image_gt142_indices,image_lt42_indices)
    image_btw_42_142 = image.copy()
    image_btw_42_142[image_not_btw_42_142_indices] = 0
    image_btw_42_142[np.logical_not(image_not_btw_42_142_indices)] = 255
    plt.imshow(image_btw_42_142, cmap='gray', vmin=0, vmax=255)
    plt.show()
    input("Hit return to continue...")

    # here is a 2d layout of images:
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False)
    ax[0,0].imshow(image, cmap='gray');  #ax[0,0].set_title("orig")
    ax[0,1].imshow(image_gt142, cmap='gray');  #ax[1,1].set_title("blue")
    ax[1,0].imshow(image_btw_42_142, cmap='gray');  #ax[1,0].set_title("green")
    ax[1,1].imshow(image_lt42, cmap='gray');  #ax[0,1].set_title("red")

    for row in range(2):
        for col in range(2):
            ax[row,col].axis('off')
    plt.show()







#
# Here are several example filters on which to base your own...
#
def example_filter( image ):
    """ an example of a pixel-by-pixel filter 
        input: an r, g, b image
        output: a transformed r, g, b image
    """
    new_image = image.copy()
    num_rows, num_cols, num_chans = new_image.shape
    for row in range(num_rows):
        for col in range(num_cols):
            r, g, b = image[row,col]
            new_image[row,col] = [b, b, b]  # [0,0,b]
    return new_image


from matplotlib import pyplot as plt
#
# try_filter!
#
def try_filter( image ):
    """ try out a filter! """
    from matplotlib import pyplot as plt
    new_image = example_filter( image )
    plt.imshow(new_image)
    plt.show() 


# Problem #1 - a few example filters
#
# The starter file provides these -- your task: to create two additional filters: one
# that transforms a single image; one that composited two images together (in some way)
# 



# for loops over the image
def example_filter_bgr( image ):
    """ an example of a pixel-by-pixel filter 
        input: an r, g, b image
        output: a transformed r, g, b image
    """
    new_image = image.copy()
    num_rows, num_cols, num_chans = new_image.shape
    for row in range(num_rows):
        for col in range(num_cols):
            r, g, b = image[row,col]
            new_image[row,col] = [b, g, r]
    return new_image


def example_filter_gbr( image ):
    """ an example of a pixel-by-pixel filter 
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


def example_filter_inv( image ):
    """ an example of a pixel-by-pixel filter 
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

def example_filter_ig( image ):
    """ an example of a pixel-by-pixel filter 
        input: an r, g, b image
        output: a transformed r, g, b image
    """
    new_image = image.copy()
    num_rows, num_cols, num_chans = new_image.shape
    for row in range(num_rows):
        for col in range(num_cols):
            r, g, b = image[row,col]
            new_image[row,col] = [r, 255-g, b]
    return new_image

def example_filter_delbot2( image ):
    """ an example of a pixel-by-pixel filter 
        input: an r, g, b image
        output: a transformed r, g, b image
    """
    new_image = image.copy()
    num_rows, num_cols, num_chans = new_image.shape
    for row in range(num_rows):
        for col in range(num_cols):
            r, g, b = image[row,col]
            new_image[row,col] = [r>>2 << 2, g>>2<<2, b>>2 << 2]
    return new_image

def example_filter_deltop2( image ):
    """ an example of a pixel-by-pixel filter 
        input: an r, g, b image
        output: a transformed r, g, b image
    """
    new_image = image.copy()
    num_rows, num_cols, num_chans = new_image.shape
    for row in range(num_rows):
        for col in range(num_cols):
            r, g, b = image[row,col]
            new_image[row,col] = [r>>2 , g>>2, b>>2 ]
    return new_image




# ## Problem 1:  write-your-own-filter
# 
# Part A will be at least one filter that takes in a single image, transforms its pixels in an unusual way (something not in Photoshop's menu options, perhaps?), and then runs that filter on at least two of your own images.
# 
# Part B will ask you to create at least one filter -- but one that combines _two_ images, meshing their pixels in a creative way, and outputting a single image. It's OK to resize the images so that they;re the same shape.
# use the above examples as a starting point...



def new_filter( image ):
    """ Gray-ifies the image by taking the average of a pixel's rgb value to replace r, g, and b"""
    new_image = image.copy()
    num_rows, num_cols, num_chans = new_image.shape
    for row in range(num_rows):
        for col in range(num_cols):
            r, g, b = image[row,col]
            gray = ((r+g+b)//3)
            new_image[row,col] = [gray, gray, gray]
    return new_image






def two_image_filter( image1, image2 ):
    """ Produces a combined image that changes between the two inputs every 15 rows"""
    imageA = image1.copy()
    imageB = image2.copy()
    # num_rowsA, num_colsA, num_chansA = imageA.shape
    # num_rowsB, num_colsB, num_chansB = imageB.shape
    new_rows = min(imageA.shape[0],imageB.shape[0])
    new_cols = min(imageA.shape[1],imageB.shape[1])
    new_image = np.zeros((new_rows,new_cols,3), dtype=int)
    
    flipflopA = 0
    flipflopB = 0
    for row in range(new_rows):
        if (row%5) == 0:
            flipflopA+=1

        if (flipflopA%2) == 0:
            for col in range(new_cols):
                if (col%5) == 0:
                    flipflopB+=1
                if (flipflopB%2) == 0:
                    r, g, b = imageA[row,col]
                    new_image[row,col] = [r, g, b]
                else:
                    r, g, b = imageB[row,col]
                    new_image[row,col] = [r, g, b]
        else:
            for col in range(new_cols):
                if (col%5) == 0:
                    flipflopB+=1
                if (flipflopB%2) == 0:
                    r, g, b = imageB[row,col]
                    new_image[row,col] = [r, g, b]
                else:
                    r, g, b = imageA[row,col]
                    new_image[row,col] = [r, g, b]
    return new_image

def main():
    print("Start of main()\n")
    # opencv_test1()
    # opencv_test1b()
    # opencv_test2()    # Resize image
    # opencv_test3()    # Acess Pixels
    # opencv_test4()    # Blur
    # opencv_test5()    # Different Color Channels
    # opencv_test6()

    # raw_pic = cv2.imread('monalisa.jpg',cv2.IMREAD_COLOR) 
    # pic = cv2.cvtColor(raw_pic, cv2.COLOR_BGR2RGB)
    # pic0 = example_filter(pic)
    # pic1 = example_filter_gbr(pic)
    # pic2 = example_filter_bgr(pic)
    # pic3 = example_filter_inv(pic)
    # pic4 = example_filter_ig(pic)
    # pic5 = example_filter_deltop2(pic)
    # pic6 = example_filter_delbot2(pic)
    # pic7 = new_filter(pic)
    # fig, ax = plt.subplots(nrows=2, ncols=4, sharex=False, sharey=False)
    # ax[0,0].imshow(pic);  
    # ax[0,0].set_title("orig")
    # ax[0,1].imshow(pic0);  
    # ax[0,1].set_title("example")
    # ax[0,2].imshow(pic1);  
    # ax[0,2].set_title("gbr")
    # ax[0,3].imshow(pic2);  
    # ax[0,3].set_title("bgr")
    # ax[1,0].imshow(pic3);  
    # ax[1,0].set_title("inv")
    # ax[1,1].imshow(pic4);  
    # ax[1,1].set_title("ig")
    # ax[1,2].imshow(pic5);  
    # ax[1,2].set_title("deltop")
    # ax[1,3].imshow(pic7);  
    # ax[1,3].set_title("MYFILTER")
    # for row in range(2):
    #     for col in range(4):
    #         ax[row,col].axis('off')
    # plt.show(fig)

    raw_picA = cv2.imread('monalisa.jpg',cv2.IMREAD_COLOR) 
    picA = cv2.cvtColor(raw_picA, cv2.COLOR_BGR2RGB)
    raw_picB = cv2.imread('flag.jpg',cv2.IMREAD_COLOR) 
    picB = cv2.cvtColor(raw_picB, cv2.COLOR_BGR2RGB)
    new_pic = two_image_filter(picA, picB)
    Apic = new_filter(picA)
    plt.imshow(Apic)
    plt.xticks([]), plt.yticks([])  # to hide axis labels
    plt.show()

    print("End of main()\n")

if __name__ == "__main__":
    main()