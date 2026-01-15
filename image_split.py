import cv2
import numpy as np
import  matplotlib.pyplot  as  plt


"""
    dst  =  cv2.  distanceTransform  (src,distanceType,maskSize[,  dstType])
    其中：
    dst表示计算得到目标函数图像。
    src表示原始图像，必须是8通道的二值图像。
    distanceType表示距离类型。
    maskSize表示掩模的尺寸大小。
    dstType表示目标函数的类型，默认为CV_F。
"""
#   读取图像，转换为灰度
image = cv2.imread('K:/1.png',0)
images = cv2.imread('K:/1.png',cv2.IMREAD_COLOR)
#   对灰度图进行OTSU阈值处理
ret , thresh = cv2.threshold(image , 0 , 255 , cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#   设定卷积
k = np.ones((4,4) , np.uint8)
#   对二值图像进行开运算
imageopen = cv2.morphologyEx(thresh , cv2.MORPH_OPEN , k , iterations = 2)
#对开运算后的图像进行膨胀操作，得到确定背景
bg  =  cv2.dilate(imageopen,k,iterations=3)
#   计算欧氏距离
distTransform = cv2.distanceTransform(imageopen, cv2.DIST_L2 , 5 )
#   对距离图像进行阈值处理
rst , fore = cv2.threshold(distTransform , 0.005*distTransform.max(),255,0)
#   调整对距离图像阈值处理的结果
ore  =  np.uint8(fore)
fore  =  np.uint8(fore)
#   确认未知区域
un = cv2.subtract(bg , ore)
"""
ret,labels  =  cv2.  connectedComponents  (image)
其中：
    ret表示标注的数量。
    labels表示标注的结果图像。
    image表示原始图像，必须是8通道的图像。

"""
#   对阈值处理结果进行标注
ret0,  labels  =  cv2.connectedComponents(fore)
#   输出标记的数量
print(ret0)
plt.subplot(121)
#   显示前景图像
plt.imshow(fore)
#   关闭坐标轴的显示
plt.axis('on')
plt.subplot(122)
#   显示标注结果
plt.imshow(labels)
plt.axis('on')
plt.show()
"""
img=  cv2.  watershed  (image,markers)
其中：
    img表示分水岭操作的结果。
    image表示输入的8位三通道图像。
    markers表示32位单通道标注结果。

"""
img  =  cv2.watershed(images,  labels)
plt.subplot(121)
plt.imshow(images)                                #  显示原始灰度图
plt.axis('off')
plt.subplot(122)
plt.imshow(img)                                          #  显示分水岭操作结果
plt.axis('off')
plt.show()
"""
dst=  cv2.  pyrDown  (src[,  dstsize[,  borderType]])
其中：
    dst表示目标图像。
    src表示输入的原始图像。
    dstsize表示目标图像的大小。
    borderType表示边界类型，默认值为BORDER_DEFAULT。
"""
#   第一次下采样
img1 = cv2.pyrDown(image)
#   第二次下采样
img2 = cv2.pyrDown(img1)
#   #第三次下采样
img3 = cv2.pyrDown(img2)
#显示第一次下采样图像
cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('img3', img3)
#  显示原始图像的大小
print("image.shape",image.shape)
#  显示第一次采样后图像的大小
print("img1.shape",img1.shape)
#  显示第二次采样后图像的大小
print("img2.shape",img2.shape)
#  显示第三次采样后图像的大小
print("img3.shape",img3.shape)
"""
dst=  cv2.  pyrUp  (src[,  dstsize[,  borderType]])
其中：
    dst表示目标图像。
    src表示输入的原始图像。
    dstsize表示目标图像的大小。
    borderType表示边界类型，默认值为BORDER_DEFAULT。
    图像不能是奇数
"""
#  第0层拉普拉斯金字塔
I0 = image - cv2.resize(cv2.pyrUp(img1), (image.shape[1], image.shape[0]))
#  第1层拉普拉斯金字塔
I1 = img1 - cv2.resize(cv2.pyrUp(img2), (img1.shape[1], img1.shape[0]))
#  第2层拉普拉斯金字塔
I2 = img2 - cv2.resize(cv2.pyrUp(img3), (img2.shape[1], img2.shape[0]))
#  显示第0层拉普拉斯金字塔图像
cv2.imshow("I0",I0)
#  显示第1层拉普拉斯金字塔图像
cv2.imshow("I1",I1)
#  显示第2层拉普拉斯金字塔图像
cv2.imshow("I2",I2)
#  恢复高精度图像
M0  =  I0  +  cv2.pyrUp(img1)
M1  =  I1  +  cv2.resize(cv2.pyrUp(img2), (img1.shape[1], img1.shape[0]))
M2  =  I2  +  cv2.resize(cv2.pyrUp(img3), (img2.shape[1], img2.shape[0]))
#  输出图像
cv2.imshow("image",image)
cv2.imshow("M0",  M0)
cv2.imshow("M1",  M1)
cv2.imshow("M2",  M2)


#   显示原始灰度图
cv2.imshow('image',image)
#   显示开运算图像
cv2.imshow("imageopen",imageopen)
#   显示确定背景图像
cv2.imshow("bg",bg)
#   显示距离图像
cv2.imshow("distTransform",distTransform)
#  显示距离图像阈值处理结果
cv2.imshow('fore',fore)
#   显示未知区域
cv2.imshow("un",un)

cv2.waitKey(0)
cv2.destroyAllWindows()