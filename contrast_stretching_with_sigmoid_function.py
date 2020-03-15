import numpy as np
import cv2
import math
import pandas as pd

img = cv2.imread('lena.jpg',0)

out = img.copy()

mid = int(input('Enter mid: '))

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        x = img.item(i,j)
        y = 0.047 * x - 6 #scaling between -6 and 6
        sigm = (math.exp(y - mid) - math.exp(mid - y))/(math.exp(y - mid) + math.exp(mid - y)) #putting in tanh funtion
        out.itemset((i,j), ((sigm + 1)/2) * 255) #scaling from (-1 to 1 ) to (0 - 255)
        
cv2.imshow('input image', img)
cv2.imshow('output image', out)

cv2.waitKey(0)
cv2.destroyAllWindows()