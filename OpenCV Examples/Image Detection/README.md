## Image Detection
`featuremath.py` matches the similarties between `objects.jpg` and `pillow.jpg` using the orb algorith.

`cam.py` is a file that will turn on your camera and detects your face with OpenCV's pretrained model in `haarcascade_frontalface_default.xml`. The controls are the following:
```
g: BGR->RGB
f: vertical flip
d: horizontal flip
esc: quit
``` 