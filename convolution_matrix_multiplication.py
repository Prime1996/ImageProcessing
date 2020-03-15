import numpy as np
from scipy import signal

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

I = np.array([[0,0,0],[0,1,0],[0,0,0]])
F = np.array([[1,2,3],[4,5,6],[7,8,9]])

'''print(I.shape)
print(F.shape)'''

out_row = I.shape[0]+F.shape[0]-1
out_column = I.shape[1]+F.shape[1]-1

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

res = signal.convolve2d(I,F)

print(res)

#visit https://github.com/sabertooth9/CSE-4128-Image-Processing-Lab/tree/1f10d06e6db40fb73f05774ac085d3fe5861e617 for support




