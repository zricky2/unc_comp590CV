from myresize import *
import cv2
im = cv2.imread("dogsmall.jpg")
dim = (im.shape[1]*4, im.shape[0]*4)
a = bilinear_resize(im, dim)
cv2.imwrite("dog4x-bl.jpg", a)