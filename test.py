#杂
#encoding=utf-8

#2018.03.31
#测试函数作为参数传递时是否可以自身带参数
#可
'''
def fun1(x=1):
    while True:
        yield x

def fun11(x):
    while True:
        yield x

def func2(func):
    print(next(func))

def func3(x):
    func2(fun11(x))

func3(4)
'''
#仿射变换测试
import cv2
import numpy as np


def rotate_and_scale(image, angle, center=None, scale=1.0):
    '''
    desc:对图像进行旋转和scale操作，返回对应的转换矩阵和转换后的图像
    '''
    (h, w) = image.shape[:2]
    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)
    #得到仿射变换矩阵
    M = cv2.getRotationMatrix2D(center, angle, scale)
    #进行仿射变换  这个参数的目的是控制顺逆时针及放大缩小？
    rotscal = cv2.warpAffine(
        image,
        M,
        (w, h),
    )  #flags=cv2.WARP_INVERSE_MAP)

    return M, rotscal


img = cv2.imread('C:\\Users\\oldhen\\Pictures\\4.jpg')
#缩小为原图的1/10
M = np.asarray([0.1, 0, 0, 0, 0.1, 0]).reshape((2, 3))
#注意cv2.warpAffine在不指定flag时是会先求M的逆矩阵然后再进行计算的，因为是要根据目标图像中的每个像素点坐标来求得原图像中对应的像素点坐标以便进行插值计算.
ret = cv2.warpAffine(img, M, (int(img.shape[1] / 10), int(img.shape[0] / 10)))
print(img.shape, ret.shape)
cv2.imwrite('t6.jpg', ret)
'''
#传参测试
def func(x):
    x-=1

x = 10
func(x)
print(x)
'''

print(type(b'shit'))

x = False
if x:
    print(x)

#查看错误
#import traceback
#traceback.print_exc()

#append测试
'''
1. python不允许程序员选择采用传值还是传引用。Python参数传递采用的肯定是“传对象引用”的方式。实际上，这种方式相当于传值和传引用的一种综合。如果函数收到的是一个可变对象（比如字典或者列表）的引用，就能修改对象的原始值——相当于通过“传引用”来传递对象。如果函数收到的是一个不可变对象（比如数字、字符或者元组）的引用，就不能直接修改原始对象——相当于通过“传值'来传递对象。

2. 当人们复制列表或字典时，就复制了对象列表的引用同，如果改变引用的值，则修改了原始的参数。
'''
'''
l=[]
x = [1,2,3]
l.append(x)
x[0]=2
print(l[0])
'''