import cv2
import matplotlib.pyplot as plt
import numpy as np

#  读取一幅灰度图像
image = cv2.imread('K:/buzhihuowu.png' ,0)
#最大值
imageMax = np.max(image)
#最小值
imageMin = np.min(image)
#确定输出最大灰度级与最小灰度级
min_1 = 0
max_1 = 255
#  计算m、n的值
m  =  float(max_1-min_1)/(imageMax-imageMin)
n =min_1 - imageMin  * m
#  矩阵的线性变换
image1 = m*image + n
#  数据类型转换
image2 = image1.astype(np.uint8)
#显示原始图像
cv2.imshow('image',image)
plt.figure("原始直方图")
#转化为一维数组
plt.hist(image.ravel(),256)
#显示正规化后的图像
plt.figure("正规化之后")
plt.hist(image.ravel(),256)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()



