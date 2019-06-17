import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import Counter

def get_colors(img,no):
    Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    # Stops execution when efficiency is 1 or 30 iterations whichever comes first
    _,label,centre=cv2.kmeans(Z,no,None,criteria,30,cv2.KMEANS_RANDOM_CENTERS)
    return label,centre

def display_graph(img,no):
    label,centre=get_colors(img,no)
    colours=[[i[2]/255,i[1]/255,i[0]/255] for i in centre]
    label=label.flatten()
    label_count=Counter(label)
    plt.bar(label_count.keys(),label_count.values(),color=colours)
    plt.show()

def display_colors(img,no,lno):
    label,centre=get_colors(img,no)
    label[label!=lno]=no
    centre = np.append(centre,[[255,255,255]],axis=0)
    centre = np.uint8(centre)
    res = centre[label.flatten()]
    res2 = res.reshape(img.shape)
    cv2.imshow("Img",res2)

img = cv2.imread('img2.jpeg')
cv2.imshow("Image",img)
display_graph(img,2)
cv2.waitKey(0)
cv2.destroyAllWindows()