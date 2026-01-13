import cv2
import  numpy  as  np


"""
    阈值函数
ret,  dst  =  cv2.  threshold  (src,  thresh,  maxval,  type)
其中：
    ret表示返回的阈值。
    dst表示输出的图像。
    src表示要进行阈值分割的图像，可以是多通道的图像。
    thresh表示设定的阈值。
    maxval表示type参数为THRESH_BINARY或THRESH_BINARY_INV类型时所设定的最大值。在显示二值化图像时，一般设置为255
"""
#  创建一个6×6的随机像素矩阵
image = np.random.randint(0,256, size=(6, 6),dtype=np.uint8)
#  使用threshold函数进行二值化阈值处理
th , rst = cv2.threshold(image,125,255,cv2.THRESH_BINARY)
#截断阈值处理
ths , rsts = cv2.threshold(image,100,255,cv2.THRESH_TRUNC)

print("image=\n",image)
print("rst=\n",rst)
print("rsts=\n",rsts)

#对图像进行二值化阈值处理
images = cv2.imread('K:/buzhihuowu.png')
#阈值处理参数设置：thresh=127，maxval=255，type=THRESH_BINARY
ret , dst  =  cv2.threshold(images,  127,255,cv2.THRESH_BINARY)
cv2.imshow('images',images )
cv2.imshow('dst',dst )
#反二值化阈值处理
#阈值处理参数设置：thresh=100，maxval=255，type=THRESH_TOZERO_INV
rets , dsts  =  cv2.threshold(images,  100,255,cv2.THRESH_BINARY_INV)
cv2.imshow('dsts',dst )

#超阈值零处理
images1 = cv2.imread('K:/huowu.png')
#阈值处理参数设置：thresh=127，maxval=255，type=THRESH_TOZERO_INV
ret1 , dst1 = cv2.threshold(images1, 127,255,cv2.THRESH_TOZERO_INV)
cv2.imshow('images1',images1)
cv2.imshow('dst1',dst1 )
print('images1', images1)
print('dst1', dst1)
#低阈值处理
rst2 , dst2 = cv2.threshold(images, 127,255,cv2.THRESH_TOZERO)
cv2.imshow('dst2',dst2)

"""
    自适应阈值处理
dst  =  cv2.  adaptiveThreshold  (src,  maxValue,  adaptiveMethod,  thresholdType,  blockSize,c)
其中：
    dst表示输出的图像。
    src表示需要进行处理的原始图像，与threshold函数不同的是，该图像必须是8位单通道的图像。
    maxValue表示最大值。
    adaptiveMethod表示自适应方法。
    thresholdType表示阈值处理方式。
    blockSize表示块大小。
    c是常量。
    输入必须是单通道灰度图,彩色图需先转换,blockSize必须是奇数(确保窗口有中心像素）。
"""
#对一幅图像使用全局阈值处理和局部阈值处理两种方法
images2 = cv2.imread('K:/buzhihuowu.png',0)
#  使用局部阈值处理，参数设置为maxval=255，
#  adaptiveMethod=ADAPTIVE_THRESH_MEAN_C
#  thresholdType=THRESH_BINARY，blockSize=5，c=3
rst3, dst3 = cv2.threshold(images2, 127, 255, cv2.THRESH_BINARY)
#邻域权重相同方式处理图像
admean = cv2.adaptiveThreshold(images2, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                            cv2.THRESH_BINARY, 5, 3)
#高斯方程方式处理图像
adguass = cv2.adaptiveThreshold(images2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY, 5, 3)
cv2.imshow('dst3',dst3)
cv2.imshow('admean',admean)
cv2.imshow('adguass',adguass)

#Ostu方法实现图像的阈值分割
#  使用threshold函数实现图像的二值化阈值处理
t1,  thd  =  cv2.threshold(images2,  150,  255,  cv2.THRESH_BINARY)
#  使用threshold函数实现图像的Ostu阈值处理
t2,  Ostu  =  cv2.threshold(images2,  0,  255,  cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("thd",  thd)
#显示二值化阈值处理图像
cv2.imshow('Ostu',Ostu)
#阈值处理图像
#  输出阈值
print("二值化阈值处理的阈值是：%s"  %  t1)
#  输出阈值
print("Ostu阈值处理的阈值是：%s"  %  t2)

cv2.waitKey(0)
cv2.destroyAllWindows()


