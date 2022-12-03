
import cv2
import numpy as np
#BGR
lower = np.array([100,95,132])  # 轉換成 NumPy 陣列，範圍稍微變小 ( 55->30, 70->40, 252->200 )
upper = np.array([165,130,255]) # 轉換成 NumPy 陣列，範圍稍微加大 ( 70->90, 80->100, 252->255 )
img = cv2.imread('abc.jpg')

mask = cv2.inRange(img, lower, upper)             # 使用 inRange
output = cv2.bitwise_and(img, img, mask = mask )  # 套用影像遮罩
#cv2.imwrite('output.jpg', output)
cv2.imshow("ww",output)
cv2.imshow("ww1",img)
cv2.waitKey(0)                                    # 按下任意鍵停止
cv2.destroyAllWindows()