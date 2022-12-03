import cv2
for i in range(0,2):
    img = cv2.imread('number//'+str(i)+'.jpg')
    k=str("img"+str(i))
    cv2.imshow(k,img)
cv2.waitKey(0)
cv2.destroyAllWindows()