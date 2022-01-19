import numpy as np
import cv2
from utils.io import read_mot_results,unzip_objs,getGrad,get_color
from utils import bbox as bbox_utils
class baseBox:
    def __init__(self,tlbr) -> None:
        self.tlbr = tlbr
        self.overlay = []
        self.belong = []
        self.no_sub_tlbrs = []
        self.boxColor = None
class motBoxs:
    def __init__(self) -> None:
        pass
        #self.boxs = []
    def update(self,tlwhs,img):
        img_shape = img.shape
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        x=cv2.Scharr(gray_img,cv2.CV_16S,1,0)
        y=cv2.Scharr(gray_img,cv2.CV_16S,0,1)
        xIm = cv2.convertScaleAbs(x)
        yIm = cv2.convertScaleAbs(y)
        boxes= self.to_tlbrs(tlwhs)
        #裁剪目标框，保证都在图像边界内
        tlbrs = bbox_utils.clip_boxes(boxes,img_shape)
        x1 = tlbrs[:,0]
        y1 = tlbrs[:,1]
        x2 = tlbrs[:,2]
        y2 = tlbrs[:,3]
        index = np.asarray([i for i in range(len(tlbrs))])
        boxs = [baseBox(tlbr) for tlbr in tlbrs]
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
                coordi = list(zip(xx1[inds],yy1[inds],xx2[inds],yy2[inds]))
                otherboxs = [boxs[index[j+1]] for j in inds]
                self.detectInter(boxs[i],otherboxs,coordi,xIm,yIm)
                #tls = np.asarray(list(coordi))


                #self.boxs[i].overlay.extend(tls)
            #得到其他矩形框与当前矩形框重合的区域
            #for num in inds:
                #tl = np.asarray([xx1[num],yy1[num],xx2[num],yy2[num]])
                #self.boxs[index[num+1]].overlay.append(tl)
            #更新矩形框索引
            index = index[1:]
        return boxs
    def displayInter(self,img,boxs):
        im = np.copy(img)
        boxThickness = 2
        for ind,box in enumerate(boxs):
            boxColor = get_color(ind)
            box.boxColor = boxColor
            x1,y1,x2,y2 = box.tlbr
            intbox = tuple(map(int,(x1,y1,x2,y2)))
            #cv2.rectangle(im,intbox[0:2],intbox[2:4],color=boxColor,thickness=boxThickness)
            for i,tlbr in enumerate(box.overlay):
                if box.belong[i]:
                    x1,y1,x2,y2 = tlbr
                    intbox = tuple(map(int,(x1,y1,x2,y2)))
                    cv2.rectangle(im,intbox[0:2],intbox[2:4],color=box.boxColor,thickness=boxThickness)
        return im
    def to_tlbrs(self,tlwhs):
        ret = np.asarray(tlwhs)
        m,n = ret.shape
        if m>1:
            ret[:,2:] += ret[:,:2]
        else:
            ret[2:] += ret[:2]
        return ret
    def detectInter(self,box,otherBoxs,tls,xGradIm,yGradIm):
        for i,othBox in enumerate(otherBoxs):
            tlbr = tls[i]
            tlGrad,brGrad = getGrad(xGradIm,yGradIm,tlbr)
            if tlGrad>brGrad:
                flag = True
            else:
                flag = False
            x1,y1,x2,y2 = tlbr
            np_tlbr = np.asarray(tlbr)
            #去除一个位置被另一个位置完全包含的情况
            if np.all(np_tlbr==box.tlbr) or np.all(np_tlbr==otherBoxs[i].tlbr):
                #print('np_tlbr=',np_tlbr)
                continue
            #找出相交部分
            box.overlay.append(tlbr)
            otherBoxs[i].overlay.append(tlbr)
            #判断相交部分归属情况
            if x1 in box.tlbr and y1 in box.tlbr:
                if flag:
                    box.belong.append(True)
                    otherBoxs[i].belong.append(False)
                    otherBoxs[i].no_sub_tlbrs.append(tlbr)
                else:
                    box.belong.append(False)
                    box.no_sub_tlbrs.append(tlbr)
                    otherBoxs[i].belong.append(True)
            else:
                if flag:
                    box.belong.append(False)
                    box.no_sub_tlbrs.append(tlbr)
                    otherBoxs[i].belong.append(True)
                else:
                    box.belong.append(True)
                    otherBoxs[i].belong.append(False)
                    otherBoxs[i].no_sub_tlbrs.append(tlbr)
    def cutRegion(self,box):
        margin = 10
        tlbr = box.tlbr
        x1,y1,x2,y2 = tlbr

        if x1+1>=x2 or y1+1>=y2:
            print('before=',tlbr)
        no_tlbrs = box.no_sub_tlbrs
        for sub_tlbr in no_tlbrs:
            a1,b1,a2,b2 = sub_tlbr
            if a1 ==x1 and a2==x2 :
                if y1 == b1:
                    if b2+margin<y2:
                        y1 = b2
                else:
                    if y1+margin<b1:
                        y2 = b1
            elif a1 == x1:
                if a2+margin<x2:
                    x1 = a2
            elif a2==x2:
                if a1>x1+margin:
                    x2 = a1
            elif y1==b1 and y2==b2:
                if a1-x1>x2-a2:
                    if a2+margin<x2:
                        x1 = a2
                else:
                    if a1>x1+margin:
                        x2 = a1
            elif y1==b1:
                if b2+margin<y2:
                    y1 = b2
            elif y2 == b2:
                if b1>y1+margin:
                    y2 = b1
            #else:
                #x1,y1,x2,y2 = a1,b1,a2,b2
        cut_tlbr = [x1,y1,x2,y2]
    
        if x1+1>=x2 or y1+1>=y2:
            print('box.tlbr=',box.tlbr)
            print('box.no_sub_tlbrs=',box.no_sub_tlbrs)
            print('after=',cut_tlbr)
        return cut_tlbr
    def updateRegion(self,tlwhs,img):
        boxs = self.update(tlwhs,img)
        cut_regions = []
        for box in boxs:
            cut_r = self.cutRegion(box)
            cut_regions.append(cut_r)
        cut_regions = np.asarray(cut_regions)
        return cut_regions


def cutOverLap(tlwhs,img):
    boxs = motBoxs()
    tlbrs = boxs.updateRegion(tlwhs,img)
    return tlbrs
if __name__=='__main__':
    filename = r'/home/xd/graduate/MOTDT/data/MOT16/train/MOT16-02/det/det.txt'
    #filename = r'D:\graduationStudy\experiment\MOT16\train\MOT16-02\gt\gt.txt'
    #img_root = r'D:\graduationStudy\experiment\MOT16\train\MOT16-02\img1\000001.jpg'
    img_root = r'/home/xd/Pictures/000001.jpg'
    is_ignore = False
    is_gt = False
    dets = read_mot_results(filename,is_gt,is_ignore)
    first_dets = dets.get(1,[])
    tlwhs,_,scores = unzip_objs(first_dets)
    boxs = motBoxs()
    img = cv2.imread(img_root)
    #tlwhs = np.asarray([[614,430,92,289],[678,448,77,227]])
    #bs = boxs.update(tlwhs,img)
    tlbrs = boxs.updateRegion(tlwhs,img)
    print(len(tlbrs))
    for tlbr in tlbrs:
        x1,y1,x2,y2 = tlbr
        intbox = tuple(map(int,(x1,y1,x2,y2)))
        cv2.rectangle(img,intbox[0:2],intbox[2:4],color=(0,255,0),thickness=2)
    #im = boxs.displayInter(img,bs)
    cv2.imwrite(r'/home/xd/graduate/materials/week3/detect_cut.jpg',img)
    cv2.namedWindow('img',0)
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()