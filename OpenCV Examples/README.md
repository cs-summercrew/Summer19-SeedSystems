## OpenCVExamples

`Image Detection/` directory containing image detection examples using OpenCV's haarCascade and feature matching with orb, sift, and suft.  

`Image Histograms/` directory containing examples involving the color histograms of images.

`Image Processing/` directory containing examples that process images (saving, labeling, and cropping).

`loadImagesTest.py` contains examples of loading images to help wrap your head around how OpenCV stores images in BGR format. When working with OpenCV, unless you start messing with images on the pixel level, you should be fine and not have to worry about converting between RGB and BGR. In OpenCV4, they added a button to close their windows, don't use it (your code will continue running), just press any key to close the windows. However, you can close the matplotlib windows with the red x button.

`small_flag.png` the image opened by `loadImagesTest.py`.