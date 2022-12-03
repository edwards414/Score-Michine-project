import cv2
import numpy as np

# 讀取中文路徑圖檔(圖片讀取為BGR)
def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    return cv_img

# 點擊欲判定HSV值的圖片位置(以滑鼠左鍵單擊)
def mouse_click(event, x, y, flags, para):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("BGR:", img[y, x])
        b,g,r  =  img[y, x]

        #print("GRAY:", gray[y, x])
        #print("HSV:", hsv[y, x])
        #print('='*30)

if __name__ == '__main__':
    # 讀取圖檔
    img = cv_imread('abc1.jpg')
    img = cv2.resize(img, (1000, 1000))
    # 轉換成gray與HSV
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.namedWindow("img")
    cv2.setMouseCallback("img", mouse_click)
    while True:
        cv2.imshow('img',img)
        if cv2.waitKey() == ord('q'):
            break
    cv2.destroyAllWindows()