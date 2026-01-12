import cv2
import  numpy  as  np


#  定义两个随机的4×4矩阵，范围在[0,255]之间
images1 =np.random.randint(0,256 , size = [4,4] , dtype=np.uint8)
images2 =np.random.randint(0,256 , size = [4,4] , dtype=np.uint8)
print("image1=\n",images1)
print("image2=\n",images2)
print("images3 =\n",images1+images2)
#subtract方法 减法
images4 = cv2.subtract(images1,images2)
print("images4 =\n",images4)

#  定义一个随机的3×4矩阵，范围在[0,255]之间
images5 =np.random.randint(0,256 , size = [3,4] , dtype=np.uint8)
#  定义一个随机的4×3矩阵，范围在[0,255]之间
images6 =np.random.randint(0,256 , size = [4,3] , dtype=np.uint8)
#乘法
images8 = np.dot(images5,images6)
print("image5=\n",images5)
print("image6=\n",images6)
print("images8 =\n",images1+images2)
#点乘
images9 = cv2.multiply(images1 , images2)
print("image9=\n",images9)
#点除
images10 = cv2.divide(images1 , images2)
print("image10=\n",images10)


