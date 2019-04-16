# Image Style transfer using CNN
In this project, we are going to create remarkable style transfer effects. In order to do so, we will have to get a deeper understanding of how Convolutional Neural Networks and its layers work. By the end of this project, you will be able to create a style transfer application that is able to apply a new style to an image while still preserving its original content.
## Style Transfer
Before we go to our Style Transfer application, let’s clarify what we are striving to achieve.

Let’s define a style transfer as a process of modifying the style of an image while still preserving its content.

Given an input image and a style image, we can compute an output image with the original content but a new style. It was outlined in Leon A. Gatys’ paper, A Neural Algorithm of Artistic Style, which is a great publication, and you should definitely check it out.
## How does it work?
1.We take input image and style images and resize them to equal shapes.
2.We load a pre-trained Convolutional Neural Network.
3.Knowing that we can distinguish layers that are responsible for the style (basic shapes, colors etc.) and the ones responsible for the content (image-specific features), we can separate the layers to independently work on the content and style.
4.Then we set our task as an optimization problem where we are going to minimize:
**content loss** (distance between the input and output images - we strive to preserve the content)
**style loss** (distance between the style and output images - we strive to apply a new style)
**total variation loss** (regularization - spatial smoothness to denoise the output image)
5. Finally, we set our gradients and optimize with the L-BFGS algorithm.
