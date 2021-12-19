import numpy as np
from scipy import integrate
import math
import time
import copy

SIGMA = 3


def f1(x,y,tlwh,p):
    
    retT = np.asarray(tlwh,dtype=float).copy()
    retT[:2] += retT[2:]/2
    u1,u2 = retT[:2]
    stdXT = retT[2]/(2*SIGMA)
    stdYT = retT[3]/(3*SIGMA)
    n = -0.5*(math.pow((x-u1),2)/math.pow(stdXT,2)+math.pow(y-u2,2)/math.pow(stdYT,2))
    valT =math.exp(n)/(2*np.pi*stdXT*stdYT)

    retP = np.asarray(p,dtype=float).copy()
    retP[:2] += retP[2:]/2
    m1,m2 = retP[:2]
    stdXP = retP[2]/(2*SIGMA)
    stdYP = retP[3]/(3*SIGMA)
    v = -0.5*(math.pow(x-m1,2)/math.pow(stdXP,2)+math.pow(y-m2,2)/math.pow(stdYP,2))
    valP =math.exp(v)/(2*np.pi*stdXP*stdYP)
    if valP>valT:
        return valT
    return valP
def compute(tlwh,p):
    start = time.clock()
    
    x = p[0]+p[2]/2
    y = p[1]+p[3]/2
    
    x_left = x-(p[2]/2)
    x_right = x+(p[2]/2)
    y_low = y-(p[3]/2)
    y_up = y+(p[3]/2)

    #v,err = integrate.dblquad(f1,float("-inf"),float("inf"),float("-inf"),float("inf"),args=(tlwh,p))
    v, err = integrate.dblquad(
        f1, y_low, y_up, x_left, x_right, args=(tlwh, p))
    print('time:',time.clock()-start)
    return v

def f2(x):
    n = -math.pow(x,2)/2
    v = math.exp(n)/np.sqrt(2*np.pi)
    return v
v,err = integrate.quad(f2,-3,3)
print(v)
tlwh =[55,65,100,100]
p = [110,110,100,100]

val = compute(tlwh,p)
print(val)
