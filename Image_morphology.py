import cv2
import  numpy  as  np


"""
dst  =  cv2.  erode  (src,k[,  anchor[,  iterations[,  boderType[,  boderValue]]]])
其中：
    dst表示返回的腐蚀处理结果。
    src表示原始图像，即需要被腐蚀的图像。
    k表示腐蚀操作时所要采取的结构类型。它由两种方式得到，第一种是通过自定义得到，第二种是通过cv2.getStructuringElement()函数得到。
    anchor表示锚点的位置，默认为（-1,-1），表示在结构元的中心。
    iterations表示腐蚀擦操作的迭代次数。
    boderType表示边界样式，一般默认使用BORDER_CONSTANT。
    boderValue表示边界值，一般使用默认值。
"""

image = cv2.imread('K:/buzhihuowu.png')
cv2.imshow('image',image)
#构建3×3的矩形结构元
k = np.ones((3,3),np.uint8)
#  腐蚀操作，迭代1次
img = cv2.erode(image,k,iterations=1)
cv2.imshow('img',img)

"""
dst  =  cv2.  dilate  (src,k[,  anchor[,  iterations[,  boderType[,  boderValue]]]])
其中：
    dst表示返回的膨胀处理结果。
    src表示原始图像，即需要被膨胀的图像。
    k表示膨胀操作时所要采取的结构类型。它由两种方式得到，第一种是通过自定义得到，第二种是通过cv2.getStructuringElement()函数得到。
    anchor表示锚点的位置，默认为（-1,-1），表示在结构元的中心。
    iterations表示膨胀操作的迭代次数。
    boderType表示边界样式，一般默认使用BORDER_CONSTANT。
    boderValue表示边界值，一般使用默认值。

"""

img1  =  cv2.dilate(image,  k,iterations=1)
img2  =  cv2.dilate(image,  k,iterations=2)
img3  =  cv2.dilate(image,  k,iterations=2)
cv2.imshow("dilate1",  img1)
cv2.imshow("dilate2",  img2)
cv2.imshow("dilate3",  img3)
"""
    开运算，先腐蚀后膨胀
    闭运算，先膨胀后腐蚀
    可以利用cv2.erode()函数和cv2.dilate()函数来实现
    OpenCV提供了更方便的函数cv2.morphologyEx()来直接实现图像的开运算与闭运算。
    当将op参数设置为cv2.MORPH_OPEN和cv2.MORPH_CLOSE时，可以对图像实现开运算与闭运算的操作。
"""

#  构建10×10的矩形结构元
k1 = np.ones((10,10),np.uint8)
#  设置参数，实现图像的开运算
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, k1)
#  设置参数，实现图像的闭运算
closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, k1)
cv2.imshow("opening", opening)
cv2.imshow("closed", closed)

"""
    黑帽运算是原始图像减去闭运算结果 Bhat(I)=I·S-I
    它可以获得比原始图像边缘更加黑暗的边缘部分，或者获得图像内部的小孔。
    礼帽运算是原始图像减去开运算结果 That(I)=I·S-I
    它可以获得图像的噪声信息或者比原始图像边缘更亮的边缘部分。
"""
bhing = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, k1)
bhing1 = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, k1)
cv2.imshow("bhing", bhing)
cv2.imshow("bhing1", bhing1)
"""
    dst  =  cv2.  morphologyEx  (src,op,  k[,anchor[,iterations[,boderType[,boderValue]]]])
    其中：
    dst表示返回梯度运算的结果。
    src表示原始图像。
    op表示操作类型，当设置为cv2.MORPH_GRADIENT时，表示对图像进行梯度运算。
    参数k、anchor、iterations、boderType和boderValue与cv2.dilate()函数的参数用法一致。

"""
#构建一个5×5的结构元
k2 = np.ones((5,5),np.uint8)
#  实现图像的梯度运算
r1 = cv2.morphologyEx(image,cv2.MORPH_GRADIENT,k)
r2 = cv2.morphologyEx(image,cv2.MORPH_GRADIENT,k2)
cv2.imshow('r1',r1)
cv2.imshow('r2',r2)



cv2.waitKey(0)
cv2.destroyAllWindows()