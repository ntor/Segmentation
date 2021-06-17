import numpy as np
from hausdorff import hausdorff_distance
from skimage import measure

def Jaccard(u1, u2, threshold=0.8):
    intersection = 0
    union = 0

    u1 = np.reshape(u1, np.size(u1))
    u2 = np.reshape(u2, np.size(u2))

    for i in range(len(u1)):
        a = u1[i]
        b = u2[i]
        if (a > threshold) and (b > threshold):
            intersection += 1
        if (a > threshold) or (b > threshold):
            union += 1
    return intersection / union


def Sorensen(u1, u2, threshold=0.8):
    intersection = 0
    U1 = 0
    U2 = 0

    u1 = np.reshape(u1, np.size(u1))
    u2 = np.reshape(u2, np.size(u2))

    for i in range(len(u1)):
        a = u1[i]
        b = u2[i]
        if (a > threshold) and (b > threshold):
            intersection += 1
        if a > threshold:
            U1 += 1
        if b > threshold:
            U2 += 1

    return 2 * intersection / (U1 + U2)



def Hausdorff(u1,u2,threshold=0.1):

    contours1 = np.concatenate(measure.find_contours(u1, threshold))
    contours2 = np.concatenate(measure.find_contours(u2, threshold))
    
    return hausdorff_distance(contours1,contours2)
    
    
    
    
