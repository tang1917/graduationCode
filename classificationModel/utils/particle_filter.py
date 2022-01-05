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
        ret[:2] += ret[2:]/2
        x_ = ret[0]
        y_ = ret[1]
        width = ret[2]
        height = ret[3]
        #mean = np.asarray([x_,y_,a,height],dtype=float)
        particle = Particle(x_, y_, 1.0, x_, y_, 1.0, x_, y_, width, height)
        for i in range(self.numParticles):
            particles.append(particle)
        # print('mean=',mean)
        #tlwh = np.asarray([x_,y_,width,height],dtype = float)
        return particles

    def calTransition(self, p, w, h):
        pn = Particle()
        x = A1*(p.x-p.x0) + A2*(p.xp-p.x0)+B0 * \
            np.random.normal(0, TRANS_X_STD) + p.x0
        y = A1*(p.y-p.y0) + A2*(p.yp-p.y0)+B0 * \
            np.random.normal(0, TRANS_Y_STD) + p.y0
        s = p.s +np.random.normal(0, TRANS_S_STD)
        #s = A1*(p.s-1.0) + A2*(p.sp-1.0)+B0*np.random.normal(0, TRANS_S_STD)+1.0
        #print('s=',s)
        pn.x = max(0.0, min(w-1.0, x))
        pn.y = max(0.0, min(h-1.0, y))

        pn.s = max(0.9*p.s, min(s, 1.1*p.s))
        if pn.s < 0.9:
            pn.s = 0.9
        elif pn.s > 1.1:
            pn.s = 1.1
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
        #print(scores)
        for i, p in enumerate(particles):
            p.w = scores[i]
        total = scores.sum()
        #print('total=', total)
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