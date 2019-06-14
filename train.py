import cv2 as cv
import glob
import os
import numpy as np

imgs = glob.glob('*.jpeg')
imgs.extend(glob.glob('*.jpg'))

print(imgs[0])
print(imgs[1])
imgarray=[]
imglabel=[]

for i in imgs: 
    img=cv.imread(i,0).reshape(120000,1)
    imgarray.append(img)
    imglabel.append(ord(i[0])-99)

imgtrain=imgarray[:161]
imgtrainlabel=imglabel[:161]

imgtest=imgarray[161:]
imgtestlabel=imglabel[161:]


svm = cv.ml.SVM_create()
svm.setKernel(cv.ml.SVM_LINEAR)
svm.setType(cv.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.383)
svm.train(imgtrain,imgtrainlabel)
svm.save('svm_data.dat')
