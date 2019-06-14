import cv2
import glob
import os

# Get images

imgs = glob.glob('*.jpeg')
imgs.extend(glob.glob('*.jpg'))

print('Found files:')
print(imgs)

width = 300
height=400

print('Resizing all images be %d pixels wide' % width)
folder = 'resized'
if not os.path.exists(folder):
   os.makedirs(folder)

# Iterate through resizing and saving
for img in imgs:
   pic = cv2.imread(img, cv2.IMREAD_UNCHANGED)
   pic = cv2.resize(pic, (width, height))
   cv2.imwrite(folder + '/' + img, pic)
