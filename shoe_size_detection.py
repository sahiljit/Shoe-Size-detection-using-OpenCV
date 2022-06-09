
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils
import math
# %matplotlib inline

# To import image from google drive to google colab
# from google.colab import drive
# drive.mount('/content/drive/')

def display(img,cmap=None):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap=cmap)

image= cv2.imread('/path/to/image')
display(image)


# Apply Median Blurring
blur_image = cv2.medianBlur(image,25)
#display(image)

gray_image = cv2.cvtColor(blur_image,cv2.COLOR_BGR2GRAY)
#display(gray_image,cmap='gray')


#Binary Threshold
ret, thresh = cv2.threshold(gray_image,160,255,cv2.THRESH_BINARY_INV)
#display(thresh)

new_img=cv2.subtract(255, thresh)
display(new_img)


# Getting the countour with largest area
cnts = cv2.findContours(new_img.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

c = max(cnts, key=cv2.contourArea)


# finding extreme values of countour
extLeft = tuple(c[c[:, :, 0].argmin()][0])
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])

cv2.drawContours(image, [c], -1, (230, 30, 30), 30)


# getting length of remote in pixels
print(extLeft,extRight,extTop,extBot)

length_of_remote= math.sqrt(((extLeft[0]- extRight[0])**2)+((extLeft[1]-extRight[1])**2) )

print(length_of_remote)

display(image)


# now finding length of foot in pixels
min_YCrCb = np.array([0,133,77],np.uint8)
max_YCrCb = np.array([235,173,127],np.uint8)

image2 = cv2.imread("/path/to/image")
imageYCrCb = cv2.cvtColor(image2,cv2.COLOR_BGR2YCR_CB)
image3 = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)


# Finding the countour with maximum area
cnts1 = cv2.findContours(image3.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts1 = imutils.grab_contours(cnts1)
c1 = max(cnts1, key=cv2.contourArea)


# determine the most extreme points along the contour
extLeft1 = tuple(c1[c1[:, :, 0].argmin()][0])
extRight1 = tuple(c1[c1[:, :, 0].argmax()][0])
extTop1 = tuple(c1[c1[:, :, 1].argmin()][0])
extBot1 = tuple(c1[c1[:, :, 1].argmax()][0])

print(extLeft1,extRight1, extTop1,extBot1)

cv2.drawContours(image2, [c1], -1, (0, 255, 255), 2)


display(image2)   #skin is segmented



length_of_foot= math.sqrt(((extTop1[0]- extBot1[0])**2)+((extTop1[1]-extBot1[1])**2) )

print(length_of_foot)



# draw line on foot
width, height = 800, 600
x1, y1 = extTop1[0] ,extTop1[1]
x2, y2 = extBot1[0]-500,extBot1[1]-500
image = np.ones((height, width)) * 255

image4=image2.copy()
line_thickness = 20
cv2.line(image4, (x1, y1), (x2, y2), (0, 255, 0), thickness=line_thickness)

display(image4)



#Finding actual legnth of foot

actual_length_of_remote = 13.5    # in centimeters

actual_length_of_foot =  (actual_length_of_remote/length_of_remote)*length_of_foot

print(actual_length_of_foot)



# write text on image
font = cv2.FONT_HERSHEY_SIMPLEX
org=(1000,1000)
color = (230, 230,84)
thickness = 10
fontScale = 9

size= "actual_length_of_foot = " +str(round(actual_length_of_foot,2)) +'cm'



cv2.putText(image4, size ,org, font,  
                   fontScale, color, thickness, cv2.LINE_AA)

display(image4)

