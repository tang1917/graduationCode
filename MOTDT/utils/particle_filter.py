import numpy as np
import math
import copy
import cv2
from torch.utils.data import dataloader
from scipy import integrate
import time
#from mot_seq import get_loader

TRANS_X_STD = 0.5
TRANS_Y_STD = 1.0
TRANS_S_STD = 0.001
MAX_PARTICLES=50
SHOW_ALL=0
SHOW_SELECTED=1

A1=2.0
A2= -1.0
B0=1.0000


#STD_Y = 10
#STD_X = 10
SIGMA = 3

class Particle(object):
    def __init__(self,x=0,y=0,s=1.0,xp=0,yp=0,sp=1.0,x0=0,y0=0,width=0,height=0,w=0) -> None:
        self.x=x
        self.y=y
        self.s =s 
        self.xp=xp
        self.yp=yp
        self.sp=sp
        self.x0=x0
        self.y0=y0
        self.width=width
        self.height=height
        self.w=w
class ParticleFilter(object):
    def __init__(self) -> None:
        #self.particles = []
        self.numParticles = 20
    def initiate(self,measurement):
        #measurement:tlwh
        particles = []
        #print('measurement=',measurement)
        ret = copy.deepcopy(measurement)
        ret[:2] += ret[2:]/2
        x_ = ret[0]
        y_ = ret[1]
        width = ret[2]
        height = ret[3]
        a = width/height
        mean = np.asarray([x_,y_,a,height],dtype=float)
        particle = Particle(x_,y_,1.0,x_,y_,1.0,x_,y_,width,height)
        for i in range(self.numParticles):
            particles.append(particle)
        #print('mean=',mean)
        return mean,particles
    def calTransition(self,p,w,h):
        pn = Particle()
        x  = A1*(p.x-p.x0) + A2*(p.xp-p.x0)+B0*np.random.normal(0,TRANS_X_STD) + p.x0
        y = A1*(p.y-p.y0) + A2*(p.yp-p.y0)+B0*np.random.normal(0,TRANS_Y_STD) + p.y0
        s = A1*(p.s-1.0) + B0*np.random.normal(0,TRANS_S_STD) + 1.0

        pn.x = max(0.0,min(w-1.0,x))
        pn.y = max(0.0,min(h-1.0,y))

        pn.s = max(0.9*p.s,min(s,1.1*p.s))
        if pn.s<0.8:
            pn.s = 0.8
        elif pn.s>1.2:
            pn.s = 1.2
        #pn.s = 1.0
        pn.xp = p.x
        pn.yp = p.y
        pn.sp = p.s 
        pn.x0 = p.x0
        pn.y0 = p.y0
        pn.width = p.width
        pn.height = p.height
        pn.w = 0
        return pn
    def transition(self,particles,w,h):
        for i in range(self.numParticles):
            particles[i] = self.calTransition(particles[i],w,h)
        return particles
    def fun(self,x,y,tlwh,p):
        #sum = 0.0
        retT = np.asarray(tlwh,dtype=float).copy()
        retT[:2] += retT[2:]/2
        u1,u2 = retT[:2]
        stdXT = retT[2]/(2*SIGMA)
        stdYT = retT[3]/(3*SIGMA)
        n = -0.5*(math.pow((x-u1),2)/math.pow(stdXT,2)+math.pow(y-u2,2)/math.pow(stdYT,2))
        valT =math.exp(n)/(2*np.pi*stdXT*stdYT)

        m1,m2 = p.x,p.y
        stdXP = p.width/(2*SIGMA)
        stdYP = p.height/(3*SIGMA)
        v = -0.5*(math.pow(x-m1,2)/math.pow(stdXP,2)+math.pow(y-m2,2)/math.pow(stdYP,2))
        valP =math.exp(v)/(2*np.pi*stdXP*stdYP)
        if valP>valT:
            return valT
        return valP
    def compute(self,tlwh,p):

        start = time.clock()
        x = p.x
        y = p.y
        xLeft = x-(p.width/2)
        xRight = x+(p.width/2)
        yLow = y-(p.height/2)
        yUp = y+(p.height/2)
        #v,err = integrate.dblquad(f1,float("-inf"),float("inf"),float("-inf"),float("inf"),args=(tlwh,p))
        v, err = integrate.dblquad(
            self.fun, yLow, yUp, xLeft,xRight, args=(tlwh, p))
        print('time:',time.clock()-start)
        return v
    def updateweight(self,particles,tlwh):
        sum = 0.0
        for i in range(self.numParticles):
            particles[i].w = self.compute(tlwh,particles[i])
            sum += particles[i].w
        if sum==0.0:
            _,particles = self.initiate(tlwh)
        else:
            for i in range(self.numParticles):
                particles[i].w = float(particles[i].w/sum)
        return particles
    def resample(self,particles):
        n = self.numParticles
        k = 0
        particles = sorted(particles,key = lambda x:x.w,reverse=True)
        #print('****')
        #for i in range(5):
            #print(i)
            #print(self.particles[i].w)
            #print(self.particles[i].x,self.particles[i].y)
        #print('****')
        new_particles = copy.deepcopy(particles)
        width= particles[0].width
        height = particles[0].height
        s = particles[0].s
        sp = particles[0].s
        for i in range(n):
            np = round(particles[i].w*n)
            for j in range(np):
                new_particles[k] = particles[i]
                new_particles[k].width = width
                new_particles[k].height = height
                new_particles[k].s = s
                new_particles[k].sp = sp
                k = k+1
                if(k==n):
                    break
            if(k==n):
                break
        while k<n:
            new_particles[k] = particles[0]
            k = k+1
        return new_particles
    def update(self,particles,tlwh):
        particles = self.updateweight(particles,tlwh)
        particles = self.resample(particles)
        x = particles[0].x
        y = particles[0].y
        a = particles[0].width/particles[0].height
        h = particles[0].height
        mean = np.asarray([x,y,a,h],dtype=float)
        return mean,particles
        
    def displayParticle(self,img,p,color):
        x0 = max(0,round(p.x-0.5*p.width))
        y0 = max(0,round(p.y-0.5*p.height))
        x1 = x0+round(p.width)
        y1 = y0+round(p.height)
        cv2.rectangle(img,(x0,y0),(x1,y1),color,2,8,0)
    def displayParticles(self,img,nColor,hColor,param):
        if param == SHOW_ALL:
            for i in range(self.numParticles-1,-1,-1):
                if i==0:
                    color = hColor
                else:
                    color = nColor
                self.discircles(img,self.particles[i],color)
        elif param==SHOW_SELECTED:
            color = hColor
            self.displayParticle(img,self.particles[0],color)
    def discircles(self,img,p,color):
        cv2.circle(img,(round(p.x),round(p.y)),round(100*p.w),(0,255,0),2,8,0)


'''
def main(data_root='/home/xd/graduate/MOTDT/data/MOT16/train',det_root=None,seq='MOT16-02' ):
    loader = get_loader(data_root,det_root,seq)
    #粒子初始化
    #measurement = [1359.1,413.27,120.26,362.77]
    measurement = [435,442,110,300]
    measurement = np.asarray(measurement,dtype=float)
    particlefilter = ParticleFilter()
    particlefilter.initiate(measurement)
    for frame_id,batch in enumerate(loader):
        print('==============================')
        frame,det_tlwhs,det_scores,_,_=batch
        h,w,_ = frame.shape
        #粒子传播
        particlefilter.transition(w,h)
        #粒子权重更新
        particlefilter.updateweight(det_tlwhs)
        #重采样
        particlefilter.resample()
        #显示
        particlefilter.displayParticles(frame,(0,0,255),(0,255,0),SHOW_SELECTED)
        cv2.imshow('img',frame)
        cv2.waitKey(1000)

if __name__=='__main__':
    main()
'''