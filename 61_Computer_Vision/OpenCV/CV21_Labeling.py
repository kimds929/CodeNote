import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt

import cv2 as cv

path = r'D:\Python\★★Python_POSTECH_AI\Dataset'
origin_path = os.getcwd()
os.chdir(path)



# 0
image = cv.imread("test.png", cv.IMREAD_COLOR)
cv.imshow("result", image)
cv.waitKey(0)

blurred = cv.GaussianBlur(image, (5, 5), 0)        # Noise Remove

gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)    # Gray Scale
cv.imshow("result", gray)
cv.waitKey(0)

edge = cv.Canny(gray, 50, 150)                     # Edge Detection
cv.imshow("result", edge )
cv.waitKey(0)


# 1
edge = cv.bitwise_not(edge)                        # Reverse Color
cv.imshow("result", edge )
cv.waitKey(0) 


contours = cv.findContours(edge.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)    # Contour
cv.drawContours(edge, contours[0], -1, (0, 0, 0), 1)
cv.imshow("result", edge )
cv.waitKey(0) 


# 2
nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(edge)


for i in range(nlabels):

	if i < 2:
		continue

	area = stats[i, cv.CC_STAT_AREA]
	center_x = int(centroids[i, 0])
	center_y = int(centroids[i, 1]) 
	left = stats[i, cv.CC_STAT_LEFT]
	top = stats[i, cv.CC_STAT_TOP]
	width = stats[i, cv.CC_STAT_WIDTH]
	height = stats[i, cv.CC_STAT_HEIGHT]


	if area > 50:
		cv.rectangle(image, (left, top), (left + width, top + height), 
				(0, 0, 255), 1)
		cv.circle(image, (center_x, center_y), 5, (255, 0, 0), 1)
		cv.putText(image, str(i), (left + 20, top + 20), 
				cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2);


cv.imshow("result", image)
cv.waitKey(0) 