import numpy as np
import cv2
import math
img = cv2.imread('lena.jpg',cv2.IMREAD_GRAYSCALE)
out=img.copy()

#cv2.imshow('input image',img)

kernel = [[1,2,1],[1,1,2],[2,1,1]]
kernel1 = [[-1,-2,-1],[1,1,2],[2,1,1]]
for i in range(1, img.shape[0]-1):
    for j in range(1, img.shape[1]-1):
        w = 0
        for s in range(-1,2):
            for t in range(-1,2):
                w += kernel1[s+1][t+1] * img.item(i-s,j-t)
        out.itemset((i,j), w/4)#w=12 for kernel
        
cv2.imshow('output image',out)

cv2.imshow('input image',img)

cv2.waitKey(0)
cv2.destroyAllWindows()

