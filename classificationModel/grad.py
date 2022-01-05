import cv2
import numpy as np

def computeGrad(img):
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imwrite(r'D:\report\materials\12_28\gray01.jpg',gray_img)
    x=cv2.Scharr(gray_img,cv2.CV_16S,1,0)
    y=cv2.Scharr(gray_img,cv2.CV_16S,0,1)
    absX=cv2.convertScaleAbs(x)
    absY=cv2.convertScaleAbs(y)
    return absX,absY


if __name__ == '__main__':
    img_root = r'D:\graduationStudy\experiment\MOT16\train\MOT16-02\img1\000001.jpg'
    img = cv2.imread(img_root)
    xim,yim = computeGrad(img)
    cv2.namedWindow('im',0)
    cv2.imshow('im',xim)
    cv2.waitKey(0)
    cv2.imshow('im',yim)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(r'D:\report\materials\12_28\xim01.jpg',xim)
    cv2.imwrite(r'D:\report\materials\12_28\yim01.jpg',yim)