import numpy as np

def Jaccard(u1,u2,threshold=0.8):
    intersection = 0
    union        = 0
    
    u1.flatten
    u2.flatten
    
    for i in range(len(u1)) :
        if (u1[i] >threshold) and (u2[i]>threshold) :
            intersection += 1 
        if (u1[i] >threshold) or  (u2[i]>threshold) :
            union        += 1
    return intersection/union


def Sorensen(u1,u2,threshold=0.8):
    intersection = 0
    U1           = 0
    U2           = 0
    
    u1.flatten
    u2.flatten
    
    for i in range(len(u1)): 
        if (u1[i] >threshold) and (u2[i]>threshold) :
            intersection += 1
        if (u1[i] >threshold) :
            U1           += 1
        if (u2[i] >threshold) :
            U2           += 1
        
    return 2*intersection/(U1+U2)




