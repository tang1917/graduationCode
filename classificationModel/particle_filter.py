from warnings import filters
import numpy as np
import math
import copy
import cv2
from torch.utils.data import dataloader
from scipy import integrate
import time
from datasets.mot_seq import get_loader
from models.classification.classifier import PatchClassifier
TRANS_X_STD = 0.5
TRANS_Y_STD = 1.0
#TRANS_S_STD = 0.001
TRANS_S_STD = 0.01
MAX_PARTICLES = 50
SHOW_ALL = 0
SHOW_SELECTED = 1

A1 = 2.0
A2 = -1.0
B0 = 1.0000


#STD_Y = 10
#STD_X = 10
SIGMA = 3


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
        ret[2:] =ret[2:]-ret[:2]
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
        if pn.s < 0.95:
            pn.s = 0.95
        elif pn.s > 1.15:
            pn.s = 1.15
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
    def updateweight(self, particles, classifier):
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
        #print('scores=',scores)
        for i, p in enumerate(particles):
            p.w = scores[i]
        total = scores.sum()
        print('total=', total)
        for i in range(self.numParticles):
            particles[i].w = float(particles[i].w/total)
        return particles

    def resample(self, particles):
        n = self.numParticles
        k = 0
        #print('sorted', len(particles))
        particles = sorted(particles, key=lambda x: x.w, reverse=True)
        new_particles = copy.deepcopy(particles)
        width = particles[0].width
        height = particles[0].height
        for i in range(n):
            np = round(particles[i].w*n)
            for j in range(np):
                new_particles[k] = particles[i]
                k = k+1
                if k == n:
                    break
            if(k == n):
                break
        while k < n:
            new_particles[k] = particles[0]
            k = k+1
        return new_particles

    def update(self, particles, classfier):
        particles = self.updateweight(particles, classfier)
        particles = self.resample(particles)
        return particles

    def displayParticle(self, img, p, color):
        # print(p)
        x0 = max(0, round(p.x-0.5*p.width))
        y0 = max(0, round(p.y-0.5*p.height))
        x1 = x0+round(p.width)
        y1 = y0+round(p.height)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2, 8, 0)

    def displayParticles(self, img, particles, nColor, hColor, param):
        if param == SHOW_ALL:
            for i in range(self.numParticles-1, -1, -1):
                if i == 0:
                    color = hColor
                else:
                    color = nColor
                self.discircles(img, particles[i], color)
        elif param == SHOW_SELECTED:
            color = hColor
            self.displayParticle(img, particles[0], color)

    def discircles(self, img, p, color):
        cv2.circle(img, (round(p.x), round(p.y)),
                   round(100*p.w), (0, 255, 0), 2, 8, 0)


if __name__ == '__main__':
    classifier = PatchClassifier()
    data_root = '/home/xd/graduate/MOTDT/data/MOT16/train'
    det_root = None
    seq = 'MOT16-02'
    loader = get_loader(data_root, det_root, seq)
    pFilter = ParticleFilter()
    measurement = [[455,433,550,728],[542,442,589,577],[573,402,675,717],[652,454,713,650],[723,447,765,575],[1019,424,1058,546],[1098,433,1130,554],
    [1254,447,1288,549],[1361,412,1478,774],[1478,415,1598,775]]
    #measurement = [[455,433,550,728]]
    measurement = np.asarray(measurement, dtype=float)
    for frame_id, batch in enumerate(loader):
        print('***********',frame_id,'***************')
        frame, det_tlwhs, det_scores, _, _ = batch
        img = frame.copy()
        h, w, _ = frame.shape
        classifier.update(frame)
        if frame_id == 0:
            filters= []
            for tlbr in measurement:         
                particles = pFilter.initiate(tlbr)
                filters.append(particles)
            #particles = pFilter.initiate(measurement)
            # print(len(particles))
            #particles = pFilter.update(particles, classifier)
        else:
            size = len(filters)
            for i in range(size):
                filters[i] = pFilter.transition(filters[i],w,h)
                filters[i] = pFilter.update(filters[i], classifier)
            #for particles in filters:
                #particles = pFilter.transition(particles, w, h)
                #particles = pFilter.update(particles, classifier)
        # print('...',particles[0])
                pFilter.displayParticles(
                    img, filters[i], (0, 0, 255), (0, 255, 0), SHOW_SELECTED)
        cv2.imshow('img', img)
        cv2.waitKey(1)
