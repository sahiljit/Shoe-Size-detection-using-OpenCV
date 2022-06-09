
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils
import math
from imutils import contours
from imutils import perspective
from scipy.spatial import distance as dist
# %matplotlib inline

# from google.colab import drive
# drive.mount('/content/drive/')

def display(img,cmap=None):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap=cmap)

old_image= cv2.imread('/path/to/image')
display(old_image)

# image= cv2.rotate(old_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

display(image)

# Apply Median Blurring
test_image = image.copy()
blur_image = cv2.medianBlur(test_image,25)
#display(image)

gray_image = cv2.cvtColor(blur_image,cv2.COLOR_BGR2GRAY)
#display(gray_image,cmap='gray')

#Binary Threshold
ret, thresh = cv2.threshold(gray_image,160,255,cv2.THRESH_BINARY_INV)
#display(thresh)

new_img=cv2.subtract(255, thresh)
#display(new_img)

# getting the countours
cnts = cv2.findContours(new_img.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

(cnts, _) = contours.sort_contours(cnts)    # countours are sorted from left to right


# ignoring the countours with less area
max_countours = []
for c in cnts:
	if cv2.contourArea(c) > 200000:
		max_countours.append(c)


pixelsPerMetric = None
width= 2.3  # width of the coin (left-most object)

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)



for c in max_countours:

	# compute the rotated bounding box of the contour
	orig = image.copy()
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")

	# order the points in the contour such that they appear
	# in top-left, top-right, bottom-right, and bottom-left
	# order, then draw the outline of the rotated bounding
	# box
	box = perspective.order_points(box)
	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

	# loop over the original points and draw them
	for (x, y) in box:
		cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

	# unpack the ordered bounding box, then compute the midpoint
	# between the top-left and top-right coordinates, followed by
	# the midpoint between bottom-left and bottom-right coordinates
	(tl, tr, br, bl) = box
	(tltrX, tltrY) = midpoint(tl, tr)
	(blbrX, blbrY) = midpoint(bl, br)

	# compute the midpoint between the top-left and top-right points,
	# followed by the midpoint between the top-righ and bottom-right
	(tlblX, tlblY) = midpoint(tl, bl)
	(trbrX, trbrY) = midpoint(tr, br)

	# draw the midpoints on the image
	cv2.circle(orig, (int(tltrX), int(tltrY)), 50, (255, 0, 0), 50)
	cv2.circle(orig, (int(blbrX), int(blbrY)), 50, (255, 0, 0), 50)
	cv2.circle(orig, (int(tlblX), int(tlblY)), 50, (255, 0, 0), 50)
	cv2.circle(orig, (int(trbrX), int(trbrY)), 50, (255, 0, 0), 50)

	# draw lines between the midpoints
	cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
		(255, 0, 255), 2)
	cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
		(255, 0, 255), 2)

	# compute the Euclidean distance between the midpoints
	dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
	dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

	# if the pixels per metric has not been initialized, then
	# compute it as the ratio of pixels to supplied metric
	# (in this case, inches)
	if pixelsPerMetric is None:
		pixelsPerMetric = dB / width

	# compute the size of the object
	dimA = dA / pixelsPerMetric
	dimB = dB / pixelsPerMetric

	print(dimA,dimB)

	display(orig)



# now finding length of foot in pixels

min_YCrCb = np.array([0,133,77],np.uint8)
max_YCrCb = np.array([235,173,127],np.uint8)

old_image2 = cv2.imread("/path/to/image")
image2= cv2.rotate(old_image2, cv2.ROTATE_90_COUNTERCLOCKWISE)
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

cv2.drawContours(image2, [c1], -1, (250, 0, 0), 10)

display(image2)   #skin is segmented



#computing length of foot in pixels
length_of_foot= math.sqrt(((extTop1[0]- extBot1[0])**2)+((extTop1[1]-extBot1[1])**2) )

print(length_of_foot)


# drawing line on foot
width, height = 800, 600
x1, y1 = extTop1[0] ,extTop1[1]
x2, y2 = extBot1[0],extBot1[1]
image = np.ones((height, width)) * 255

image4=image2.copy()
line_thickness = 20
cv2.line(image4, (x1, y1), (x2, y2), (0, 255, 0), thickness=line_thickness)

display(image4)


# computing actual length of foot
actual_length_of_foot =  length_of_foot/pixelsPerMetric
print(actual_length_of_foot)


# writing text on image 
font = cv2.FONT_HERSHEY_SIMPLEX
org=(1000,1000)
color = (230, 230,84)
thickness = 10
fontScale = 9
size= "actual_length_of_foot = " +str(round(actual_length_of_foot,2)) +'cm'

cv2.putText(image4, size ,org, font,  
                   fontScale, color, thickness, cv2.LINE_AA)

display(image4)

