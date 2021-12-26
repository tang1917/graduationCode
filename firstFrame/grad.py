import cv2
import numpy as np

def computeGrad(img):
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    x=cv2.Scharr(gray_img,cv2.CV_16S,1,0)
    y=cv2.Scharr(gray_img,cv2.CV_16S,0,1)
    absX=cv2.convertScaleAbs(x)
    absY=cv2.convertScaleAbs(y)
    dst=cv2.addWeighted(absX,0.5,absY,0.5,0)
    return dst


if __name__ == '__main__':
    img_root = r'D:\graduationStudy\experiment\MOT16\train\MOT16-02\img1\000001.jpg'
    img = cv2.imread(img_root)
    print(img.shape)
    computeGrad(img)
    print(img.shape)