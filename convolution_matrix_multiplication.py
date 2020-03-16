import numpy as np
from scipy import signal
import cv2

def toeplitize(F,col):
    out = np.zeros((F.shape[0],col), dtype=int)
    k=0
    while(k<F.shape[0]):
        i=k
        j=0
        while(i<F.shape[0] and j<col):
            out[i][j]=F[k]
            i=i+1
            j=j+1
        k=k+1
   # print(out)
    return out

def vector_to_maxtrix(vector,m_shape):
    row = m_shape[0]
    col = m_shape[1]
    output = np.zeros(m_shape, dtype=int)
    i = row-1
    j=0
    for k in range(len(vector)):
        output[i,j]=vector[k]
        j=j+1
        if(j>=col):
            j=0
            i=i-1
    return output

def gauss_weighted_average(n):
    out_x = []
    out_y = []
    val = 1
    for i in range(n//2):
        out_x.append(val)
        out_y.append([val])
        val = val * 2
    out_x.append(val)
    out_y.append([val])
    for j in range(n//2):
        val = val / 2
        out_x.append(val)
        out_y.append([val])
    out_x = [out_x]
    out = np.matmul(out_y, out_x)
    return out

#I = np.array([[0,0,0],[0,1,0],[0,0,0]])
n = int(input("size of gaussian kernel = "))
I = cv2.imread('lena.jpg',cv2.IMREAD_GRAYSCALE)
I = cv2.resize(I, (200,200), interpolation = cv2.INTER_AREA)
Fil = I.copy()
F = gauss_weighted_average(n)
sum_ = 0
for i in F:
    for j in i:
        sum_=sum_+j

cv2.imshow('input_image',I)

'''print(I.shape)
print(F.shape)'''

out_row = I.shape[0]+F.shape[0]-1
out_column = I.shape[1]+F.shape[1]-1

Fil = cv2.resize(I, (out_row,out_column), interpolation = cv2.INTER_AREA)

out = np.zeros((out_row,out_column), dtype=int)


F = np.pad(F,((out_row-F.shape[0],0),(0,out_column-F.shape[1])),'constant',constant_values = 0)

toeplitz = []
count = 0

for i in range(F.shape[0]-1,-1,-1):
    toep = toeplitize(F[i,:], I.shape[1])
    count+=1
    toeplitz.append(toep)

'''print(toeplitz[0].shape[0]*count)
print(toeplitz[0].shape[1]*I.shape[0])'''

dbl = np.zeros((toeplitz[0].shape[0]*count,toeplitz[0].shape[1]*I.shape[0]), dtype=int)

'''F.shape[0] = toeplitz[0].shape[0]*count
col = toeplitz[0].shape[1]*I.shape[0]'''
b_h = toeplitz[0].shape[0]
b_w = toeplitz[0].shape[1]
k=0
while(k<count):
    i=k
    j=0
    while(i<count and j<I.shape[0]):
        start_i = int(i*b_h)
        start_j = int(j*b_w)
        end_i = int(start_i+b_h)
        end_j = int(start_j+b_w)
        dbl[start_i:end_i,start_j:end_j] = toeplitz[k]
        i=i+1
        j=j+1
    k=k+1

'''b_h = toeplitz_shape[0]
    b_w = toeplitz_shape[1]

    for i in range(dl_indices.shape[0]):
        for j in range(dl_indices.shape[1]):
            start_i = int(i*b_h)
            start_j = int(j*b_w)
            end_i = int(start_i+b_h)
            end_j = int(start_j+b_w)
            dbmatrix[start_i:end_i,start_j:end_j] = toeplitz_list[dl_indices[i,j]-1]'''


'''for i in range(count):
    for j in range(I.shape[0]):
        start_i = int(i*b_h)
        start_j = int(j*b_w)
        end_i = int(start_i+b_h)
        end_j = int(start_j+b_w)
        dbl[start_i:end_i,start_j:end_j] = toeplitz[i]'''

'''print(dbl)'''

row ,col = I.shape
output = []
for i in range(row-1,-1,-1):
    for j in range(0,col):
        output.append(I[i][j])

'''print(output)'''

result_vector = np.matmul(dbl,output)

'''print(result_vector)'''

result_matrix = vector_to_maxtrix(result_vector,(out_row,out_column))
        
print(result_matrix)

for i in range(Fil.shape[0]):
    for j in range(Fil.shape[1]):
        Fil[i][j] = result_matrix[i][j] / sum_

cv2.imshow("output", Fil)



#res = signal.convolve2d(I,F)

#print(res)

#visit https://github.com/sabertooth9/CSE-4128-Image-Processing-Lab/tree/1f10d06e6db40fb73f05774ac085d3fe5861e617 for support

cv2.waitKey(0)
cv2.destroyAllWindows()



