import cv2
import numpy as np

""""
    高斯滤波
dst  =  cv2.GassianBlur  (src,ksize,sigmaX,  sigmaY,borderType)
其中：
    dst表示返回的高斯滤波处理结果。
    src表示原始图像，该图像不限制通道数目。
    ksize表示滤波卷积核的大小，需要注意的是滤波卷积核的数值必须是奇数。
    sigmaX表示卷积核在水平方向上的权重值。
    sigmaY表示卷积核在水平方向上的权重值。如果sigmaY被设置为0，则通过sigmaX的值得到，但是如果两者都为0，则通过如下方式计算得到：
"""
image = cv2.imread('K:/buzhihuowu.png')
cv2.imshow('image',image)
#  定义卷积和为5*5，采用自动计算权重的方式实现高斯滤波
gauss = cv2.GaussianBlur(image,(5,5),0)
cv2.imshow('gauss',gauss)
cv2.waitKey(0)
cv2.destroyAllWindows()

""""
均值滤波
dst  =  cv2.  blur  (src,ksize,anchor,borderType)
其中：
    dst表示返回的均值滤波处理结果。
    src表示原始图像，该图像不限制通道数目。
    ksize表示滤波卷积核的大小。
    anchor表示图像处理的锚点，其默认值为（-1,-1），表示位于卷积核中心点。
    borderType表示以哪种方式处理边界值。
"""
cv2.imshow("blur",  image)
#定义卷积和为5×5
means5  =  cv2.blur(image,  (5,5))
#定义卷积和为10×10
means10  =  cv2.blur(image,  (10,10))
#定义卷积和为20×20
means20  =  cv2.blur(image,  (20,20))
cv2.imshow("means5",  means5)
cv2.imshow("means10",  means10)
cv2.imshow("means20",  means20)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
    方框滤波
dst  =  cv2.boxFilter  (src,depth,ksize,anchor,normalize,borderType)
其中：
    dst表示返回的方框滤波处理结果。
    src表示原始图像，该图像不限制通道数目。
    depth表示处理后图像的深度，一般使用-1表示与原始图像相同的深度。
    ksize表示滤波卷积核的大小。
    anchor表示图像处理的锚点，其默认值为（-1,-1），表示位于卷积核中心点。
    normalize表示是否进行归一化操作。
    borderType表示以哪种方式处理边界值。
"""
cv2.imshow("boxFilter", image)
#  定义卷积和为5*5，normalize=0不进行归一化
box5_0  =  cv2.boxFilter(image,  -1,  (5,5),normalize=False)
#  定义卷积和为2*2，normalize=0不进行归一化
box2_0 = cv2.boxFilter(image, -1, (2,2) , normalize=False)
#  定义卷积和为5*5，normalize=1进行归一化
box5_1 = cv2.boxFilter(image, -1, (5,5) , normalize=True)
#  定义卷积和为2*2，normalize=1进行归一化
box2_1 = cv2.boxFilter(image, -1, (2,2) , normalize=True)
cv2.imshow("box5_0", box5_0)
cv2.imshow("box2_0", box2_0)
cv2.imshow("box5_1", box5_1)
cv2.imshow("box2_1", box2_1)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
    中值滤波
dst  =  cv2.  medianBlur  (src,ksize)
其中：
    dst表示返回的方框滤波处理结果。
    src表示原始图像，该图像不限制通道数目。
    ksize表示滤波卷积核的大小。

"""
cv2.imshow("median", image)
#  使用卷积核为5*5的中值滤波
median = cv2.medianBlur(image, 5)
cv2.imshow("medianBlur", median)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
    双边滤波
dst  =  cv2.bilateralFilter  (src,d,sigmaColor,sigmaSpace,borderType)
其中：
    dst表示返回的双边滤波处理结果。
    src表示原始图像，该图像不限制通道数目。
    d表示在滤波时选取的空间距离参数，表示以当前像素点为中心点的半径。在实际应用中一般选取5。
    sigmaColor表示双边滤波时选取的色差范围。
    sigmaSpace表示坐标空间中的sigma值，它的值越大，表示越多的点参与滤波。
    borderType表示以何种方式处理边界。
"""
cv2.imshow("bilateralFilter", image)
#   高斯
gauss  =  cv2.GaussianBlur(image,(55,55),0,0)
#   双边
bilateral  =  cv2.bilateralFilter(image,55,100,100)
cv2.imshow("gauss", gauss)
cv2.imshow("bilateralFilter", bilateral)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
    自定义卷积
dst  =  cv2.  filter2D  (src,  d,  kernel,  anchor,  delta,  borderType)
其中：
    dst表示返回的双边滤波处理结果。
    src表示原始图像，该图像不限制通道数目。
    d表示处理结果图像的图像深度，一般使用-1表示与原始图像使用相同的图像深度。
    kernel表示一个单通道的卷积核。
    anchor表示图像处理的锚点，其默认值为（-1,-1），表示位于卷积核中心点。
    delta表示修正值，可选。如果该值存在，会在滤波的基础上加上该值作为最终的滤波处理结果。
    borderType表示以何种情况处理边界。

"""
cv2.imshow("filter2D", image)
#  设置13×13的卷积核 总和分别为3和2 → 加权滤波器
#  乘以 3/(13*13)。这里13*13是169，所以每个元素都是1 * 3/169 = 3/169。
#  这样，整个卷积核的所有元素之和为 169 * (3/169) = 3。
#  为了归一化
k1  =  np.ones((13,13),  np.float32)*3/(13*13)
#  设置9×9的卷积核 总和分别为3和2 → 加权滤波器
k2  =  np.ones((9,9),np.float32)*2/81
#  设置5×5的卷积核 总和为1 → 均值滤波器
k3  =  np.ones((5,5),np.float32)/25
out = cv2.filter2D(image,-1,k1)
out1 = cv2.filter2D(image,-1,k2)
out2 = cv2.filter2D(image,-1,k3)
cv2.imshow("out", out)
cv2.imshow("out1", out1)
cv2.imshow("out2", out2)
cv2.waitKey(0)
cv2.destroyAllWindows()