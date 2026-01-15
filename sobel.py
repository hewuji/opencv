import cv2
import math
import  numpy  as  np
from  scipy  import  signal


"""
    Scharr算子
dst  =  cv2.  Sobel  (src,ddepth,dx,dy[,  ksize[,  scale[,  delta[,  borderType]]]])
其中：
    dst表示计算得到目标函数图像。
    src表示原始图像。
    ddepth表示输出图像的深度。
    dx表示x方向上求导的阶数。
    dy表示y方向上求导的阶数。
    ksize表示Sobel核的大小。
    scale表示计算导数时的缩放因子，默认值是1。
    delta表示在目标函数上所附加的值，默认为0。
    borderType表示边界样式。
"""
image = cv2.imread('K:/buzhihuowu.png',0)
images = cv2.imread('K:/huowu.png',1)
imagess = cv2.imread('K:/1.png',)
#  设置参数dx=1,dy=0，得到图像水平方向上的边缘信息
sobelx = cv2.Sobel(image , cv2.CV_64F, 1, 0)
#  对计算结果取绝对值
sobelx = cv2.convertScaleAbs(sobelx)
#  设置参数dx=0,dy=1，得到图像垂直方向上的边缘信息
sobely  =  cv2.Sobel(image,  cv2.CV_64F,  0,  1)
#  对计算结果取绝对值
sobely = cv2.convertScaleAbs(sobely)
#  设置参数dx=1,dy=1，得到图像垂直方向上的边缘信息
sobelys  =  cv2.Sobel(image,  cv2.CV_64F,  1,  1)
sobelys = cv2.convertScaleAbs(sobelys)
#  利用加权函数addWeighted对Sobel算子水平和垂直方向上进行加权计算
Sobelys_my  =  cv2.addWeighted(sobelx,  0.5,  sobely,  0.5,  0)
#  显示图像
#  显示原始图像
cv2.imshow("image",  image)
#  显示水平方向上的边缘图像
cv2.imshow("Sobelx",  sobelx)
#  显示垂直方向上的边缘图像
cv2.imshow("Sobely",  sobely)
#  显示水平和垂直方向的边缘图像
cv2.imshow("sobelys",  sobelys)
#  显示水平和垂直加权的图像
cv2.imshow("Sobelys_my",  Sobelys_my)

"""
    Canny边缘检测
edg=  cv2.  Canny  (src,threshould1,  threshould2  [,  apertureSize[,  L2gradient]])
其中：
    edg表示计算得到的边缘信息。
    src表示输入的8位图像。
    threshould1表示第一个阈值。
    threshould2表示第二个阈值。
    apertureSize表示Sobel算子的大小。
    L2gradient表示计算图像梯度幅度的标识，默认为False。
"""

edg = cv2.Canny(images,30,100)
edg1 = cv2.Canny(images,100,200)
edg2 = cv2.Canny(images,200,255)
cv2.imshow("images",images)
cv2.imshow("edg",edg)
cv2.imshow("edg1",edg1)
cv2.imshow("edg2",edg2)

"""
    Laplacian算子的计算
dst  =  cv2.  Laplacian  (src,ddepth[,ksize  [,  scale[,  delta[,  borderType]]]])
其中：
    dst表示计算得到的目标函数图像。
    src表示原始图像。
    ddepth表示输出图像的深度。
    ksize表示二阶导数核的大小，必须是正奇数。
    scale表示计算导数时的缩放因子，默认值是1。
    delta表示在目标函数上所附加的值，默认为0。
    borderType表示边界样式。

"""

#  使用拉普拉斯算子计算边缘信息
laplacian =cv2.Laplacian(imagess,cv2.CV_64F)
#  对计算结果取绝对值
laplacian  =  cv2.convertScaleAbs(laplacian)
cv2.imshow("imagess",imagess)
cv2.imshow("laplacian",  laplacian)

#构建LoG  算子
def create_log_kernel(sigma,ksize):
    #  LoG  算子的宽和高，且两者均为奇数
    win_h,win_w  =  ksize
    log_kernel  =  np.zeros(ksize,np.float32)
    #方差
    sigma_square  =  pow(sigma,2.0)
    #  LoG  算子的中心
    center_h  =  (win_h-1)/2
    center_w  =  (win_w-1)/2
    for  r  in  range(win_h):
        for  c  in  range(win_w):
            norm2  =  pow(r-center_h,2.0)  +  pow(c-center_w,2.0)
            log_kernel[r][c]  =  1.0/sigma_square*(norm2/sigma_square  -  2)*math.exp(-norm2/(2*sigma_square))
    return  log_kernel
#  LoG卷积，一般取_boundary  =  'symm'
def  log(image,sigma,ksize,_boundary='fill',_fillvalue  =  0):
    #构建LoG  卷积核
    log_kernel = create_log_kernel ( sigma, ksize )
    # 图像与LoG  卷积核卷积
    img_conv_log = signal.convolve2d ( image, log_kernel, 'same', boundary=_boundary )
    return img_conv_log


def edge_binary ( img ) :
    edge = np.copy ( img )
    edge[edge >= 0] = 0
    edge[edge < 0] = 255
    #   调整对距离图像阈值处理的结果
    edge = edge.astype ( np.uint8 )
    return edge

#  LoG卷积
img1 = log ( image, 2, (7, 7), 'symm' )
img2 = log ( image, 2, (11, 11), 'symm' )
img3 = log ( image, 2, (13, 13), 'symm' )
# 边缘的二值化显示
L1 = edge_binary ( img1 )
L2 = edge_binary ( img2 )
L3 = edge_binary ( img3 )
#  显示LoG边缘检测结果
cv2.imshow ( "L1", L1 )
cv2.imshow ( "L2", L2 )
cv2.imshow ( "L3", L3 )

cv2.waitKey()
cv2.destroyAllWindows()


