import cv2
import matplotlib.pyplot as plt
import numpy as np
import math


#  计算图像灰度直方图
def calchist(image):

    # 灰度图像矩阵的宽高
    rows,cols = image.shape

    #  存储灰度直方图
    grayhist = np.zeros([256] , np.uint32)
    for i in range(rows):
        for j in range(cols):
            grayhist[image[i][j]] += 1
    return grayhist

#  直方图均衡化
def equalHist ( image ) :
    # 灰度图像矩阵的宽高
    rows, cols = image.shape

    # 计算灰度直方图
    grayhist = calchist ( image )

    # 计算累加灰度直方图
    zerocumumoment = np.zeros ( [256], np.uint32 )

    for i in range ( 256 ) :
        if i == 0 :
            zerocumumoment[i] = grayhist[0]
        else :
            zerocumumoment[i] = zerocumumoment[i - 1] + grayhist[i]

    # 根据直方图均衡化得到的输入灰度级和输出灰度级的映射
    output_q = np.zeros ( [256], np.uint8 )
    cofficient = 256.0 / (rows * cols)

    for i in range ( 256 ) :
        j = cofficient * float ( zerocumumoment[i] ) - 1
        if j >= 0 :
            output_q[i] = math.floor ( j )
        else :
            output_q[i] = 0

    # 得到直方图均衡化后的图像
    equalhistimage = np.zeros ( image.shape, np.uint8 )
    for r in range ( rows ) :
        for c in range ( cols ) :
            equalhistimage[r][c] = output_q[image[r][c]]

    return equalhistimage
image  =  cv2.imread('K:/buzhihuowu.png',cv2.IMREAD_GRAYSCALE)
#  直方图均衡化
dst  =  equalHist(image)
#  显示原图像
cv2.imshow("image",  image)
#  显示均衡化图像
cv2.imshow("dst",dst)
#  显示原始图像直方图
plt.figure("原始直方图")
plt.hist(image.ravel(),256)
#  显示均衡化后的图像直方图
plt.figure("均衡化直方图")
plt.hist(dst.ravel(),256)
plt.show()
cv2.waitKey()
cv2.destroyAllWindows()

#更简单的方法
#  读取一幅图像
image  =  cv2.imread('K:/buzhihuowu.png',  cv2.IMREAD_GRAYSCALE)
cv2.imshow("huowu",  image)                  #  显示原始图像
equ  =  cv2.equalizeHist(image)                #  直方图均衡化处理
cv2.imshow("huo",  equ)                #  显示均衡化后的图像
plt.figure("原始直方图")                        #  显示原始图像直方图
plt.hist(image.ravel(),256)
plt.figure("均衡化直方图")                    #  显示均衡化后的图像直方图
plt.hist(equ.ravel(),256)
plt.show()
cv2.waitKey()
cv2.destroyAllWindows()











