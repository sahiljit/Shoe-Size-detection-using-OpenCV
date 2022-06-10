
# Shoe Size Detection using OpenCV

**Task** : Find size of foot in a camera image

**Approach**: If we have a reference object in an image whose actual size is known, then size of foot in image can be found. In our case reference object is  AC's remote. Following are the steps:

1. First only remote is segmented using opencv. Different techniques like blurring, edge detection, binary thresholding etc are experimented and best combination is chosen. 

2. Once the remote is segmented, the contours are found and contour with maximum area will be the segmented remote
 
3. Extreme points of segmented area are then found and hence length of remote in pixels is found.

4. Now skin is segmented using YCrCb color space. It works better than HSV. 

5. On segmented image, step-2 and step-3 are repeated and hence length of foot in pixel is found.

6. Finding actual length of foot is now simple math 

## Tech Stack

- Python
- OpenCV
- Numpy
- Matplotlib




## Results

#### Input Image
![foot](/images/foot-1.jpg)

#### Output Image and Segmentations
![output](/images/output.png)
![remote segmentation](/images/remote_segmentation_binary.png)


#### Using Coin
![foot](/images_coin/coin_detection.png)
![foot](/images_coin/output.png)





