## Image Detection
`featurematch.py` matches the similarties between `objects.jpg` and `pillow.jpg` using the orb algorithm.  

`orb.py` an example comparing the orb, sift, and suft algorithms. For this file to work, you will need to use either an older version of OpenCV, or to download the opencv_contrib package. See this [link](https://www.pyimagesearch.com/2015/07/16/where-did-sift-and-surf-go-in-opencv-3/) for more information.  

`Images/` directory storing the images used by `featuremath.py` and `orb.py`  

`cam.py` is a file that will turn on your camera and detects your face with OpenCV's pretrained model in `haarcascade_frontalface_default.xml`. The controls are the following:
```
g: BGR->RGB
f: vertical flip
d: horizontal flip
esc: quit
``` 