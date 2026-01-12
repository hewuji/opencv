import numpy as np
import cv2


image = cv2.imread('K:/buzhihuowu.png')
cv2.imshow("image", image)
#  BGR色彩空间转RGB色彩空间
images = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#  RGB色彩空间转GRAY色彩空间
images1 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#RGB色彩空间转YCrCb色彩空间
images2 = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
#RGB色彩空间转HSV色彩空间
images3 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
cv2.imshow("images", images)
cv2.imshow("images1", images1)
cv2.imshow("images2", images2)
cv2.imshow("images3", images3)
cv2.waitKey(0)
cv2.destroyAllWindows()