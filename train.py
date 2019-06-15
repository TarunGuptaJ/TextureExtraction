import cv2 as cv
import glob
import os
import numpy as np

imgs = glob.glob('*.jpeg')
imgs.extend(glob.glob('*.jpg'))

imgarray=[]
imglabel=[]

for i in imgs: 
    img=cv.imread(i,0).reshape(120000,)
    img=np.float32(img)
    img=img/255
    imgarray.append(img)
    imglabel.append(ord(i[0])-99)

f=open("imgarray.txt",'w')
print(imgarray,file=f)
f.close()

imgtrain=np.asarray(imgarray[:161])
imgtrainlabel=np.asarray(imglabel[:161])

imgtest=np.asarray(imgarray[161:])
imgtestlabel=np.asarray(imglabel[161:])

if(__name__=="__main__"):
    train_data=cv.ml.TrainData_create(imgtrain,cv.ml.ROW_SAMPLE,imgtrainlabel)
    svm = cv.ml.SVM_create()
    svm.setKernel(cv.ml.SVM_LINEAR)
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setC(2.67)
    svm.setGamma(5.383)
    svm.train(train_data)
    svm.save('svm_data.dat')