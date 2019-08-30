`image-labeler.py` opens every image in the `Banana` folder individually and gives you the option to press **Y** (1), **N** (0), or **Esc**(N/A) for each. Depending on the key you pressed, the results get saved to `BanannaData.csv`. This is usefule if you are making a machine learning classifier and need to label a large amount of images quickly. 

`mouse-crop.py` allows you to crop the image and save it if you want. The line `FILE_NAME = "frame448.png"` controls which file in `/Bannana` is opened, you may want to change this. The controls are the following:
```
mouse click n'drag: creates a rectangle representing the crop region
escape: Exits the first window without saving if no crop window, displays cropped image otherwise.
s: If you are in the crop-window, pressing s, and then escape will save your image to the bannana directory as frame448_crop.png
``` 

`simeple-recording.py` will turn on your camera, records video, displays the current frame, and lets you switch in and out of color. It also allows capturing images and stores them to a `temp` folder. The controls are the following:
```
spacebar: Saves the current frame as an image.
c: Saves the next 30 continuous frames.
s: Switches color display (grayscale <--> rgb).
esc: Quits (do not press the x button on the window).
``` 
