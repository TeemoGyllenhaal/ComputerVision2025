import numpy as np
import cv2 as cv

#-------------- Moments---------------
img = cv.imread('test.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
ret, thresh = cv.threshold(img,127,255,0)
contours, hierarchy = cv.findContours(thresh, 1, 2)

cnt = contours[0]
M = cv.moments (cnt)
print (M)

# (cx, cy) là tọa độ trọng tâm của contour (M_00 là diện tích)
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

#------------Contour Area---------------
area = cv.contourArea(cnt)

#---------------Contour Perimeter------------
perimeter = cv.arcLength (cnt, True)

#---------------Contour Approximation--------
epsilon = 0.1*cv.arcLength (cnt, True)
approx = cv.approxPolyDP (cnt, epsilon, True)

#-------------Convex Hull---------------
hull = cv.convexHull(points[, hull[, clockwise[, returnPoint]]])

hull = cv.convexHull(cnt)
#Còn quay lại

#--------------Checking Convexity-----------
#Lồi
k = cv.isContourConvex(cnt)

#------------------Bounding Rectangle---------
##-----------------Straight Bounding Rectangle-----
x,y,w,h = cv.boundingRect(cnt)
cv.rectangle(img,(x, y),(x+w, y+h),(0,255,0),2)

#------------------Rotated Rectangle----------
rect = cv.minAreaRect #Tìm BR xoay
box = cv.boxPoints(rect) #Tìm 4 đỉnh hình chữ nhật
box = np.int0(box) #Ép kiểu int
cv.drawContours(img, [box], 0, (0,0,255),2) #Vẽ hình chữ nhật đỏ

#--------------------Minimum Enclosing Circle-----------
#Tìm đường tròn ngoại tiếp vật thể

(x,y),radius = cv.minEnclosingCircle(cnt) #Đường tròn ngoại tiếp min
center = (int(x), int(y)) 
radius = int(radius)
cv.circle(img,center,radius,(0,255,0),2) #ảnh,tâm,bán kính,màu,độ nét

#------------Fitting an Ellipse----------
ellipse = cv.fitEllipse(cnt)
cv.ellipse(img,ellipse,(0,255,0),2)

#------------Fitting a Line-------
rows, cols = img.shape[:2]
[vx,vy,x,y] = cv.fitLine(cnt, cv.DIST_L2,0,0.01,0.01)
lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx)+y)
cv.line(img,(cols-1, righty),(0,lefty),(0,255,0),2)


