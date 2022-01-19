import numpy as np
import math
import copy
import cv2
from torch.utils.data import dataloader
from scipy import integrate
import time
from datasets.mot_seq import get_loader
from models.classification.classifier import PatchClassifier
from models.reid import load_reid_model, extract_reid_features
from scipy.spatial.distance import cdist

TRANS_X_STD = 0.5
TRANS_Y_STD = 1.0
#TRANS_S_STD = 0.001
TRANS_S_STD = 0.01
SHOW_ALL = 0
SHOW_SELECTED = 1

A1 = 2.0
A2 = -1.0
B0 = 1.0000



class Particle(object):
    def __init__(self, x=0, y=0, s=1.0, xp=0, yp=0, sp=1.0, x0=0, y0=0, width=0, height=0, w=0) -> None:
        self.x = x
        self.y = y
        self.s = s
        self.xp = xp
        self.yp = yp
        self.sp = sp
        self.x0 = x0
        self.y0 = y0
        self.width = width
        self.height = height
        self.w = w


class ParticleFilter(object):
    def __init__(self) -> None:
        #self.particles = []
        self.numParticles = 100

    def initiate(self, measurement):
        # measurement:tlwh
        particles = []
        # print('measurement=',measurement)
        ret = copy.deepcopy(measurement)
        ret[:2] = ret[:2]+ret[2:]/2
        x_ = ret[0]
        y_ = ret[1]
        width = ret[2]
        height = ret[3]

        #a = width/height
        #mean = np.asarray([x_,y_,a,height],dtype=float)
        particle = Particle(x_, y_, 1.0, x_, y_, 1.0, x_, y_, width, height)
        for i in range(self.numParticles):
            particles.append(particle)
        # print('mean=',mean)
        return particles

    def calTransition(self, p, w, h):
        pn = Particle()
        x = A1*(p.x-p.x0) + A2*(p.xp-p.x0)+B0 * \
            np.random.normal(0, TRANS_X_STD) + p.x0
        y = A1*(p.y-p.y0) + A2*(p.yp-p.y0)+B0 * \
            np.random.normal(0, TRANS_Y_STD) + p.y0
        s = p.s +np.random.normal(0, TRANS_S_STD)
        #s = A1*(p.s-1.0) + A2*(p.sp-1.0)+B0*np.random.normal(0, TRANS_S_STD)+1.0
        pn.x = max(0.0, min(w-1.0, x))
        pn.y = max(0.0, min(h-1.0, y))

        pn.s = max(0.9*p.s, min(s, 1.1*p.s))
    
        #pn.s = 1.0
        pn.xp = p.x
        pn.yp = p.y
        pn.sp = p.s
        pn.x0 = p.x0
        pn.y0 = p.y0
        pn.width = p.width*pn.s
        pn.height = p.height*pn.s
        pn.w = 0
        return pn

    def transition(self, particles, w, h):
        for i in range(self.numParticles):
            particles[i] = self.calTransition(particles[i], w, h)
        return particles
    '''
    打分模型计算粒子权重
    '''
    def updateweight(self, particles, classifier,reid_model,feature,image,min_score):
        tlbrs = []
        for p in particles:
            x = p.x
            y = p.y
            width = p.width
            height = p.height
            x = x-0.5*width
            y = y-0.5*height
            tlbrs.append([x, y, x+width, y+height])
        tlbrs = np.asarray(tlbrs)
        scores = classifier.predict(tlbrs)
        #只保留分数较高的粒子
        index = np.where(scores>min_score)[0]
        count = len(index)
        # 标记为跟踪丢失
        if count==0:
            return False, particles
        particles = [particles[i] for i in index]
        tlbrs  = [tlbrs[i] for i in index]
        #print('tlbrs=',tlbrs)
        pFeatures = extract_reid_features(reid_model, image, tlbrs)
        pFeatures = pFeatures.cpu().numpy()
        feature = feature[np.newaxis,:]
        #根据特征相似性确定粒子权重
        scores = 1/cdist(feature,pFeatures,metric="cosine")
        scores = scores[0]
        for i, p in enumerate(particles):
            p.w = scores[i]
        total = scores.sum()
        #print('total=', total)
        for i in range(count):
            particles[i].w = float(particles[i].w/total)
        return True,particles

    def resample(self, particles):
        n = self.numParticles
        k = 0
        particles = sorted(particles, key=lambda x: x.w, reverse=True)
        particles_num = len(particles)

        new_particles = copy.deepcopy(particles)
        for i in range(particles_num):
            np = round(particles[i].w*particles_num)
            #print('np=',np)
            for j in range(np):
                new_particles[k] = particles[i]
                k = k+1
                if k == particles_num:
                    break
            if(k == particles_num):
                break
        
        while k < particles_num:
            new_particles[k] = particles[0]
            k = k+1

        num = n-particles_num
        for i in range(num):
            new_particles.append(particles[0])
        return new_particles

    def update(self, particles, classfier,reid_model,feature,image):
        co,particles = self.updateweight(particles, classfier,reid_model,feature,image)
        if co:
            particles = self.resample(particles)
        return particles