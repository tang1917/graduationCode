import numpy as np
from utils import read_mot_results,unzip_objs
import copy
import cv2

class baseBox:
    def __init__(self,tlbr) -> None:
        self.tlbr = tlbr
        self.overlay = []
class motBoxs:
    def __init__(self) -> None:
        self.boxs = []
    def update(self,tlwhs,img):
        tlbrs = self.to_tlbrs(tlwhs)
        x1 = tlbrs[:,0]
        y1 = tlbrs[:,1]
        x2 = tlbrs[:,2]
        y2 = tlbrs[:,3]
        index = np.asarray([i for i in range(len(tlbrs))])
        self.boxs = [baseBox(tlbr) for tlbr in tlbrs]
        while index.size>0:
            i = index[0]
            #计算当前矩形框与其他矩形框相交的坐标
            xx1 = np.maximum(x1[i],x1[index[1:]])
            yy1 = np.maximum(y1[i],y1[index[1:]])
            xx2 = np.minimum(x2[i],x2[index[1:]])
            yy2 = np.minimum(y2[i],y2[index[1:]])
            #计算相交框的面积，注意矩形框不相交时w或h算出来的是负数，用0代替
            w = np.maximum(0.0,xx2-xx1)
            h = np.maximum(0.0,yy2-yy1)
            inter = w*h
            #计算有重叠的矩形框索引
            inds = np.where(inter>0)[0]
            #得到当前矩形框与其他矩形框重合的所有矩形区域
            if inds.size>0:
                coordi = zip(xx1[inds],yy1[inds],xx2[inds],yy2[inds])
                tls = np.asarray(list(coordi))
                self.boxs[i].overlay.extend(tls)
            #得到其他矩形框与当前矩形框重合的区域
            for num in inds:
                tl = np.asarray([xx1[num],yy1[num],xx2[num],yy2[num]])
                self.boxs[index[num+1]].overlay.append(tl)
            #更新矩形框索引
            index = index[1:]
    def displayInter(self,img):
        im = np.copy(img)
        interColor = (0,0,255)
        interThickness = 2
        boxColor = (0,255,0)
        boxThickness = 2

        for box in self.boxs:
            
            if len(box.overlay)>0:
                for tlbr in box.overlay:
                    x1,y1,x2,y2 = tlbr
                    intbox = tuple(map(int,(x1,y1,x2,y2)))
                    cv2.rectangle(im,intbox[0:2],intbox[2:4],color=interColor,thickness=interThickness)
            '''
            x1,y1,x2,y2 = box.tlbr
            intbox = tuple(map(int,(x1,y1,x2,y2)))
            cv2.rectangle(im,intbox[0:2],intbox[2:4],color=boxColor,thickness=boxThickness)
            '''
        return im
    def to_tlbrs(self,tlwhs):
        ret = tlwhs.copy()
        ret[:,2:] += ret[:,:2]
        return ret

if __name__=='__main__':
    filename = r'D:\graduationStudy\experiment\MOT16\train\MOT16-02\det\det.txt'
    #filename = r'D:\graduationStudy\experiment\MOT16\train\MOT16-02\gt\gt.txt'
    img_root = r'D:\graduationStudy\experiment\MOT16\train\MOT16-02\img1\000001.jpg'
    is_ignore = False
    is_gt = False
    dets = read_mot_results(filename,is_gt,is_ignore)
    first_dets = dets.get(20,[])
    tlwhs,_,scores = unzip_objs(first_dets)
    boxs = motBoxs()
    boxs.update(tlwhs)
    img = cv2.imread(img_root)
    im = boxs.displayInter(img)
    cv2.namedWindow('img',0)
    cv2.imshow('img',im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()