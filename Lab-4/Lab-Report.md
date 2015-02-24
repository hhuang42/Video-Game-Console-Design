#Lab Report 4

##Introduction

In this lab, the CUDA sample code was manipulated in order to perform motion blur in addition to the uniform blur already present
in the sample code.

## Design

In order to perform motion blur, it seemed simple enough to use faded sets of previous frames in order to form a trailing afterimage of what had just appeared.

Since the intent of the lab is to make use of the gpu, the set of previous frames was stored in an array stored on
the gpu, as `uchar4* motion_blur_buffer`. `uchar4` was chosen since that was the format of the pixels used in `cudaProcess()`, so it would be easy to manipulate this motion blur buffer in the same way. Furthermore, it was initialized in main.cpp in order to be initialized in the same process as another device resource, `cuda_dest_resource`, which was allocated and freed in predefined functions `initCUDABuffers()` and `FreeResources`. As a result, it was possible to use additional cuda resources without disrupting the flow of the program, as would happen through the use of global variables, which may not necessary be instantiated as needed.

The motion blur effect was performed through the algorithm where the buffer stored the motion blurred effect of the previous frame. The image for the current  frame is then blended with the motion blurred buffer through the use of weighted averaging with fixed constants. As a result, the final pixel values are weighted most heavily by the value of the current frame, with diminished influence from the previous frame before that, such that the frames have exponentially decreasing influence as they get older.

Finally, it was ensured that the threads were synced with each other before continuing with the blurring effect that was laid on afterwords.

## Testing

The motion blur effect was tested by simply compiling and running the program. In particular, it was tested whether the motion blur incurred any artificial patterns on the image, such as rectangles based on the blocks that were used in the cuda calculations.

## Results and Discussion

The image of the spinning teapot was motion blurred as expected, and furthermore, the existing blurred effect was able to be performed on top of the motion blur.

Although at one point, there were inconsistencies in the colors of rectangular tiles corresponding to the thread block size, but this was resolved after the threads were forced to synchronized.

## Conclusion

In this lab, the GPU code for enabling blur on an image was modified to also provide motion blur.

5 hours were spent on this lab.

## Source Code

[Source Code Repository](https://github.com/hhuang42/e190u/tree/master/cuda/postProcessGL)

