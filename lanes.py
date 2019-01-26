import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore", category = RuntimeWarning)

#from matplotlib import pyplot as plt

def make_coord(image,line_parameters):  #create coordinates to plot lines
      slope, y_intercept = line_parameters
      y1  = int(image.shape[0]) #represents start height
      #y2 = int(y1*(1/2.4))
      y2 = int(y1*(3/5))   #end height
      x1 = int((y1 - y_intercept)/slope)
      x2 = int((y2 - y_intercept)/slope)
      return  [[x1,y1,x2,y2]]

def av_slope_inter(image,lines): #average out lines to create 2 clear lines
    left_fit = []  #line on the left
    right_fit= []  #line on the right
    if lines is 0:
        return None
    for line in lines:
        for x1,y1,x2,y2 in line:
         parameters = np.polyfit((x1,x2),(y1,y2),1)  #fit x degree polynomial and returns vector that stores m and b
        #print(parameters)
         slope = parameters[0]  #m
         y_intercept = parameters[1] #b
        #lines on the right : positive slope
         if slope <0:
            left_fit.append((slope,y_intercept))
         else:
            right_fit.append((slope,y_intercept))
    if len(left_fit) and len(right_fit):
     left_fit_average = np.average(left_fit,axis=0) #operate along axis 1 to create average
     right_fit_average = np.average(right_fit,axis=0)
     left_line = make_coord(image,left_fit_average)
     right_line = make_coord(image,right_fit_average)
     return np.array([left_line,right_line])

def canny(image):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)  #changes RGB PIC to single channel picture for much faster processing
    kernel = 5
    blur = cv2.GaussianBlur(gray,(kernel,kernel),0) #performs last value is the deviation // 5x5 typical kernel size
    canny = cv2.Canny(gray,50,150)   #creates gradient image, low threshold and high threshold
    return canny

def disp_lines(image,lines):
    line_im = np.zeros_like(image)
    if lines is not None:           #checks if lines were detected
        for line in lines:    #loop through Lines
            for x1, y1, x2, y2 in line:   #iterate through lines, reshape from 2D to 1D
              cv2.line(line_im,(x1,y1),(x2,y2),(255,0,0),10)        #draws line from (x1,y1) to (x2,y2),next is color,last is line thickness
    return line_im

def region_of_interest(canny):  #focus on relevant area
    heigth = canny.shape[0]
    width = canny.shape[1]
    mask = np.zeros_like(canny)  #create mask, completely black
    polygons = np.array([ [(200, heigth) , (1100 , heigth) , (550,250)]],np.int32)
    cv2.fillPoly(mask, polygons, 255)         #data type has to be int32 when using alternative 2D array
    masked_image = cv2.bitwise_and(canny,mask)  #bitwise comparison to blend out not relevant area
    return masked_image

# image = cv2.imread('test_image.jpg')
# lane_im = np.copy(image) #copy is important to not change original
# canny_im = canny(lane_im)
# #performs gradient in every direction, outputs image with spots with highest gradient marked bright
#
# show = region_of_interest(canny_im)
# #Hough Transformation
# #(Transform possible lines into m and b, select pixel with most intersections for best line,...)
# lines = cv2.HoughLinesP(show,2, np.pi/180,100, np.array([]),minLineLength=100,maxLineGap=118)
# #2 is precision of pixels, 3:Angle, 4: minimum of intersections to accept line
# #5: minimum Line Length to be accepted, 6: Maximum gap between Lines segments
# averaged_lines = av_slope_inter(lane_im , lines)
# line_im= disp_lines(lane_im,averaged_lines)
# combo_im = cv2.addWeighted(lane_im,0.8,line_im,1,1)
# #2nd is decreasing brightnesss of the real image to see lines better
# #last value is a scaler argument
# cv2.imshow("result",combo_im)
# cv2.waitKey(0)
# #plt.imshow(combo_im)
# #plt.show()


cap = cv2.VideoCapture('test2.mp4')
i = 0
while(cap.isOpened() and i<1255):             #same as pic just loop through frames
    ret, frame = cap.read()
    if ret == True:
     canny_im = canny(frame)
    #performs gradient in every direction, outputs image with spots with highest gradient marked bright
     i = i+1  #counts frames, to close window before the end of the video (no error in the end)

     cropped = region_of_interest(canny_im)
#Hough Transformation
#(Transform possible lines into m and b, select pixel with most intersections for best line,...)
     lines = cv2.HoughLinesP(cropped,2, np.pi/180,100, np.array([]),minLineLength=40,maxLineGap=5)
#2 is precision of pixels, 3:Angle, 4: minimum of intersections to accept line
#5: minimum Line Length to be accepted, 6: Maximum gap between Lines segments
     averaged_lines = av_slope_inter(frame , lines)
     line_im = disp_lines(frame,averaged_lines)
     combo_im = cv2.addWeighted(frame,0.8,line_im,1,1)
#2nd is decreasing brightnesss of the real image to see lines better
#last value is a scaler argument
     cv2.imshow("result",combo_im)
     if cv2.waitKey(1) == ord('q'):
         break
cap.release()
cv2.destroyAllWindows()
#plt.imshow(combo_im)
#plt.show()
