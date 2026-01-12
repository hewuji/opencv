import cv2
import matplotlib.pyplot as plt


image = cv2.imread('K:/buzhihuowu.png')
# 计算其统计直方图信息
hist = cv2.calcHist([image],[0],None,[256],[0,256])
image = image.ravel()
plt.hist(image , 256)
arr1  =  [1,1.2,1.5,1.6,2,2.5,2.8,3.5,4.3]
arr2  =  [5,4.5,4.3,4.2,3.6,3.4,3.1,2.5,2.1,1.5]
plt.plot(arr1)
plt.plot(arr2, "--r")
plt.plot(hist)
cv2.waitKey()
cv2.destroyAllWindows()
plt.show()