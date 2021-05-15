import cv2
import os,glob
from PIL import Image
import numpy as np
import copy
from timeit import default_timer as timer

sample_image = '/disk2/taekwang/fire_dataset/video_frame/hand_held_nasa_training2_01670.jpg'
sample_image = '/disk2/taekwang/fire_dataset/video_frame/HouseFive_00504.jpg'

def image_load_time_test(imgpath, channel_multiplier):
    img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    if channel_multiplier is not None:
        img_tmp = img.astype(np.float32)
        img_tmp[:,:,0] *=1.2
        #img_tmp[:,:,1] *=1
        #img_tmp[:,:,2] *=1
        return img_tmp
    else:
        return img

time1 = timer()
a = image_load_time_test(sample_image,None)
time2 = timer()
print("no channel_multiplier", time2- time1)
time3 = timer()
b = image_load_time_test(sample_image,True)
time4 = timer()
print("channel_multiplier", time4- time3)



#image = Image.open(sample_image,).convert('RGB')
#image.show()

img_color= cv2.imread(sample_image,cv2.IMREAD_COLOR)

img_crop = img_color[243:431, 400:551]

img_b, img_g, img_r = cv2.split(img_color)
zeros = np.zeros((img_color.shape[0], img_color.shape[1]), dtype="uint8")

img_b = cv2.merge([img_b, zeros, zeros])
img_b_crop = img_b[243:431, 400:551]
img_g = cv2.merge([zeros, img_g, zeros])
img_g_crop = img_g[243:431, 400:551]
img_r = cv2.merge([zeros, zeros, img_r])
img_r_crop = img_r[243:431, 400:551]

print(img_r.shape)
print(type(img_r))
img_r_2 = img_r
img_r_2 = img_r_2.astype(np.float32)
img_r_2[:,:,2] *=1.2
print(img_r_2[:,:,0])

img_crop[:,:,0]/=255
img_crop[:,:,1]/=255
img_crop[:,:,2]/=255

"""
cv2.imwrite('./img_r.jpg',img_r_crop)
cv2.imwrite('./img_g.jpg',img_g_crop)
cv2.imwrite('./img_b.jpg',img_b_crop)
"""
cv2.imwrite('./img_origin.jpg',img_crop/255)

cv2.imshow("AA",img_crop)
#cv2.imshow("BGR", img_color)
#cv2.imshow("B", img_b)
#cv2.imshow("G", img_g)
#cv2.imshow("R", img_r)
#cv2.imshow("R2",img_r_2)

cv2.waitKey(0)

cv2.destroyAllWindows()


