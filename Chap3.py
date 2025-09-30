import numpy as np
import cv2 as cv

#----------------Aspect Ratio-----------
img = cv.imread('test.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
ret, thresh = cv.threshold(img,127,255,0)
contours, hierarchy = cv.findContours(thresh, 1, 2)

cnt = contours[0]
x,y,w,h = cv.boundingRect(cnt)
aspect_ratio = float(w)/h

#-------------Extent---------------
area = cv.contourArea(cnt)
x,y,w,h = cv.boundingRect(cnt)
rect_area = w*h
extent = float(area)/rect_area

#-----------------Solidity-------------------
area = cv.contourArea(cnt)
hull = cv.convexHull(cnt)
hull_area = cv.contourArea(hull)
solidity = float(area)/hull_area

#---------------Equivalent Diameter------------
area = cv.contourArea(cnt)
equi_diameter = np.sqrt(4*area/np.pi)

#--------------Orientation---------------
(x,y), (MA, ma), angle = cv.fitEllipse

#---------------Mask and Pixel Points-----------
mask = np.zeros(imgray.shape,np.uint8)
cv.drawContours(mask,[cnt],0,255,-1)
pixelpoints = np.transpose(np.nonzero(mask))
#pixelpoints = cv.findNonZero(mask)

#-----------Maximum Value, Minimum Value and their locations---------
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(imgray,mask = mask)

#-----Mean Color or Mean Intensity-------------
mean_val = cv.mean(im,mask = mask)

#---------Extreme Points----------
leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])