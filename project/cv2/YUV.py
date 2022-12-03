
import cv2
import numpy as np

fruit = cv2.imread("abc1.jpg")

fruit = cv2.cvtColor(fruit,cv2.COLOR_BGR2YUV)
Y,U,V = cv2.split(fruit)
#Blueberry = cv2.inRange(U,130,255)
print(V)
Strawberry = cv2.inRange(V,140,180)
#cv2.imshow("blueberry",Blueberry)
cv2.imshow("strawberry",Strawberry)
cv2.imshow("strawberry1",fruit)
cv2.waitKey(0)
