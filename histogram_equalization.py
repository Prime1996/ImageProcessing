import numpy as np
import cv2
import matplotlib.pyplot as plt

inp = cv2.imread('lena.jpg', 0)
out_array = np.zeros((inp.shape[0],inp.shape[1]))
out = inp.copy()
cv2.imshow('Input Image', inp)

n = plt.hist(inp.ravel(),256,[0,256])
histogram = np.histogram(inp.flatten(),256,[0,256])
plt.show()

total_pixel = inp.shape[0] * inp.shape[1]

pdf = np.zeros(histogram[0].shape[0]+1)

for i in range(histogram[0].shape[0]):
    pdf[i] = float(histogram[0][i]) / float(total_pixel)

for i in range(1,histogram[0].shape[0]):
    pdf[i] = pdf[i]+pdf[i-1]

print("Input cdf")
plt.plot(histogram[1], pdf, color = 'green')
plt.show()

for i in range(inp.shape[0]):
    for j in range(inp.shape[1]):
        a = inp.item(i,j)
        out_array[i][j] = 255 * pdf[a]
        out.itemset((i,j), int(out_array[i][j]))
        
cv2.imshow('Output Image', out)
n = plt.hist(out.ravel(),256,[0,256])
histogram = np.histogram(out.flatten(),256,[0,256])
plt.show()

for i in range(histogram[0].shape[0]):
    pdf[i] = float(histogram[0][i]) / float(total_pixel)

for i in range(1,histogram[0].shape[0]):
    pdf[i] = pdf[i]+pdf[i-1]

print("Output cdf")

plt.plot(histogram[1], pdf, color = 'green')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()