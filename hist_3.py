import cv2
import  matplotlib.pyplot  as  plt


image = cv2.imread('K:/buzhihuowu.png', cv2.IMREAD_ANYCOLOR)
#  将彩色图像转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#  创建CLAHE均衡化对象 clipLimit对比度限制 tileGridSize每个小网格的大小默认(8,8)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#  将 CLAHE 应用于原始图像，生成一个对比度增强后的图像。
dst = clahe.apply(gray_image)
#  显示图像
cv2.imshow('image', gray_image)
cv2.imshow('dst', dst)

plt.figure('hist')
#把多维数组转化为一维，范围为(0, 256)
plt.hist(gray_image.ravel(), 256, (0, 256))
plt.figure('clahehist')
#把多维数组转化为一维，范围为(0, 256)
plt.hist(dst.ravel(), 256, (0, 256))

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()