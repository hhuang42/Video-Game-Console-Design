#Lab Report 5

##Introduction

In this lab, an OpenCV implementation of a hand-detection algorithm was modified to recognize the location of a specific held marker in a manner useful to perform actions in an application.

##Design

The codeflow was initially modified from C to C++ in order to allow for the GPU matrices that were supported under the C++ library. In order to do so, the equivalent functions in the C version of the library were replaced with C++ versions. After that was done so, the GPU forms of the data structure were used in order to prompt the opencv library to use GPU processing.

The marker was chosen since it could be detected with a slight modification onto the existing code, modifying the desired colors and removing other components of the algorithm entirely, such as finger detection. As a result, it was fairly simple to implement and test since no component had to be created from scratch.

Furthermore, since the HSV color values were selected linearly, the use of a blue marker ensured that the desired range lay in a continuous range, whereas the color red would comprise colors with Hue approximately 0 and Hue approximately 255 in 2 distinct ranges.

Finally, the brightness of the marker cap made it easy to distinguish from background colors, and a marker was also extemely easy to hold and move as a user.

Although gestures are not explicitly classified in this lab, the ability to find the location of the marker head allows for sufficient data as a controllable point to manipulate as desired, such as in lab 6.

## Testing

The design was tested primarily through checking if the demonstration was able to run properly, as well as if the program was able to effectively mark the top of the marker.

The section of the program designed to mark the user's hand was retained in order to allow for verification that the program was able to detect the marker top. Furthermore, it was evident from the thresholded image if there was some unwanted noise from the background that was being detected.

## Results and Discussion

![alt-text](https://github.com/hhuang42/Video-Game-Console-Design/blob/master/Lab-5/raw_screenshot.png)

The program was able to accurately detect the marker tip against the background.

The marker did need to be relatively close to the camera, and latency/framerate were as poor as in the original demo, but the accuracy of the marker tip detection was high, and the location could steadily be tracked.

## Conclusion

In this lab, motion controls were implemented allowing the tracking of a blue marker across the screeen.

8 hours were spent on this lab.
