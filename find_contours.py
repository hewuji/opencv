import cv2
import numpy as np


image = cv2.imread ( 'K:/2.png', 0 )
images = cv2.imread ( 'K:/buzhihuowu.png', 0 )
# 检查图像是否成功加载
if image is None :
    print ( "错误：无法加载图像 K:/2.png" )
    exit ()

cv2.imshow ( 'Original Grayscale', image )

# 对灰度图进行二值化阈值处理
ret, binary = cv2.threshold ( image, 127, 255, cv2.THRESH_BINARY )
cv2.imshow ( 'Binary', binary )

# 查找图像中的外部轮廓
contours, hierarchy = cv2.findContours ( binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
print ( "轮廓类型：", type ( contours ) )
print ( "找到的外部轮廓个数：", len ( contours ) )

# 制作掩模
mask = np.zeros ( image.shape, np.uint8 )
# 绘制图像中的轮廓（填充）
mask = cv2.drawContours ( mask, contours, -1, (255, 255, 255), -1 )
cv2.imshow ( "Mask", mask )

# 使用掩模提取前景
logimg = cv2.bitwise_and ( image, mask )
cv2.imshow ( "Foreground", logimg )

# --- 第二部分：对另一个图像进行Canny和轮廓绘制 ---
images_color = cv2.imread ( 'K:/buzhihuowu.png', cv2.IMREAD_COLOR )
# 检查图像是否成功加载
if images_color is None :
    print ( "错误：无法加载图像 K:/buzhihuowu.png" )
    exit ()

cv2.imshow ( 'Original Color', images_color )

# Canny边缘检测
binaryImg = cv2.Canny ( images_color, 50, 200 )
cv2.imshow ( "Canny Edges", binaryImg )

# 寻找轮廓
# 在OpenCV 4.x中, findContours返回2个值。用 _ 忽略第二个返回值(hierarchy)
contours_canny, _ = cv2.findContours ( binaryImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE )

# 创建白色幕布
temp = np.ones ( binaryImg.shape, np.uint8 ) * 255
# 画出轮廓
cv2.drawContours ( temp, contours_canny, -1, (0, 255, 0), 1 )
cv2.imshow ( "Contours on White", temp )

# --- 第三部分：修正错误，为第一个图像的轮廓绘制最小外包矩形 ---
# 使用之前找到的二值图 'binary'
contours1, hierarchy1 = cv2.findContours ( binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE )

# 创建一个彩色图像副本，用于绘制矩形
img_with_rects = cv2.cvtColor ( images, cv2.COLOR_GRAY2BGR )

print ( f"为 {len ( contours1 )} 个轮廓计算最小外包矩形..." )

# 遍历所有找到的轮廓
for c in contours1 :
    # 过滤掉太小的轮廓，避免产生很多小框
    if cv2.contourArea ( c ) < 10 :
        continue

    # 对当前的单个轮廓 c 计算最小外包矩形
    rect = cv2.minAreaRect ( c )
    # 获取矩形的四个顶点坐标
    box = cv2.boxPoints ( rect )
    # 将坐标转换为整数
    box = np.int32 ( box )
    # 在副本图像上画出这个矩形框
    cv2.drawContours ( img_with_rects, [box], 0, (0, 255, 0), 2 )  # 用绿色、粗一点的线条

cv2.imshow ( "Image with MinAreaRects", img_with_rects )

cv2.waitKey ( 0 )
cv2.destroyAllWindows ()