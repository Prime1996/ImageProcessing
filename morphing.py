import numpy
import cv2
import math

input1 = cv2.imread('orange.jpg', 0)
input2 = cv2.imread('one.jpg', 0)
out = input1.copy()
B = 20
interval = 1/(2 * B)

cv2.imshow('input image 1', input1)
cv2.imshow('input image 2', input2)
print(input1.shape)
print(input2.shape)

w = input1.shape[1] / 2

for i in range(input1.shape[0]):
    a = 1
    b = 0
    for j in range(input1.shape[1]):
        a1 = input1.item(i,j)
        a2 = input2.item(i,j)
        if j < (w - B):
            out.itemset((i,j),a1)
        elif j > (w + B):
            out.itemset((i,j),a2)
        else:
            out.itemset((i,j), a*a1 + b*a2)
            a -= interval
            b += interval
            
cv2.imshow('output', out) 

cv2.waitKey(0)
cv2.destroyAllWindows()