#杂


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
    rotscal = cv2.warpAffine(image, M, (w, h), )#flags=cv2.WARP_INVERSE_MAP)

    return M, rotscal
'''
img=cv2.imread('C:\\Users\\oldhen\\Pictures\\4.jpg')
ret = rotate_and_scale(img,0,scale=0.5)
print(img.shape,ret[1].shape)
cv2.imwrite('t5.jpg',ret[1])
'''

def func(x):
    x-=1

x = 10
func(x)
print(x)