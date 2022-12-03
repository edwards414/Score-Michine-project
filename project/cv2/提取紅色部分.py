import cv2
import numpy as np
img = cv2.imread("100n6.png")
#img = cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
#img = cv2.resize(img,(800,600))
#cv2.imwrite('//project//n268jpg',img)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)#轉HSV
Y,U,V = cv2.split(hsv)
mask = cv2.inRange(V,150,255)
contours,hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#只檢查外輪廓
print('countours',len(contours))
filename=0
cv2.imshow("test",mask)
cv2.imshow("test1",img)

for i in range(0,len(contours)):
    x,y,w,h = cv2.boundingRect(contours[i])
    if w*h>2000:
        crop_img = img[y+7:y+h-7, x+7:x+w-7]
        cv2.imwrite("number//red_boder//"+str(filename)+'.jpg', crop_img)
        filename+=1
print(filename)
cv2.imwrite('number//red_boder//n286.jpg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()