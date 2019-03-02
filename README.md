# finding-lanes
Simple Lane Detection Algorithm

What I Learned

- first time using opencv, transforming rgb image into b & w reduces amount of channels and runtime 
- easy to generate solid Gradient Image using GaussianBlur and Canny
- Hough Transformation allows efficiently fitting lines, needs to be averaged out though
- create masks and use bitwise and to only focus on the relevant parts of the image and thereby save ressources 
- segmenting video into single frames allows us to process videos 
- unstable when only dashed lines are in the middle, doesn't work on curved roads
- a lot more potential will probably be possible with aastly different approach
