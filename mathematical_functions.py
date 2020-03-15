import numpy
import cv2
import math

img = cv2.imread('kuet.jpg', 0)
out = img.copy()
out_log = img.copy()
out_gama1 = img.copy()
out_gama2 = img.copy()


cv2.imshow('input image', img)

print(img.max())
print(img.min())

print(img.shape)
for i in range (img.shape[0]):
    for j in range(img.shape[1]):
        a = img.item(i,j)
        out.itemset((i,j),255-a)
        out_log.itemset((i,j), 31.875 * math.log(a+1))
        gama = 2
        gama1 = 0.8
        scale_factor = 255/math.pow(255,gama)
        scale_factor1 = 255/math.pow(255,gama1)
        out_gama1.itemset((i,j), scale_factor * math.pow(a,gama))
        out_gama2.itemset((i,j), scale_factor1 * math.pow(a,gama1))
cv2.imshow('negative image', out)
cv2.imshow('logarithmik image', out_log)
cv2.imshow('gama image 1', out_gama1)
cv2.imshow('gama image 2', out_gama2)


cv2.waitKey(0)
cv2.destroyAllWindows()