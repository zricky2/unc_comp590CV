
"""
COMP590 Assignment1
Linear Interpolation
Ricky Zheng
"""

import numpy as np

def nn_resize(im, dim): #cv format (col,row)
    r_ratio = (im.shape[0])/(dim[1])
    c_ratio = (im.shape[1])/(dim[0])
    newImg = np.ndarray((dim[1],dim[0],3), dtype=int)
    for i in range(dim[1]): #iterate through the row 576
        for j in range(dim[0]): #iterate through the col 768
            x = int(np.floor(j*c_ratio))
            y = int(np.floor(i*r_ratio))
            newImg[i,j] = im[y,x]
    newImg = newImg.astype(np.uint8)
    return newImg

def bilinear_resize(im, dim):
    r_ratio = (im.shape[0])/(dim[1])
    c_ratio = (im.shape[1])/(dim[0])
    newImg = np.ndarray((dim[1],dim[0],3), dtype= int)
    for i in range(dim[1]): #iterate through each row 576
        for j in range(dim[0]): #iterate through each col 768
            x = int(np.floor(j*c_ratio))
            y = int(np.floor(i*r_ratio))
            x_diff = ((c_ratio * j)-x)
            y_diff = ((r_ratio * i)-y)
            if (x == 0 or x == (im.shape[1]-1) or y == 0 or y == (im.shape[0]-1)):
                newImg[i,j] = [0,0,0]
            else:
                newImg[i,j] = (im[y,x] * (1-x_diff)*(1-y_diff)) + (im[y, x+1] * (1-y_diff) * (x_diff)) + (im[y+1, x]*(y_diff)*(1-x_diff)) + (im[y+1, x+1]*(y_diff)*(x_diff))
                
    return newImg


