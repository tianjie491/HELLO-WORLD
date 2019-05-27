import cv2
import numpy as np


# images = cv2.imread('000001.jpg')
# print(images.shape)
# cv2.imshow('hand000001', images)

# rand_x = np.random.randint(-50, 50, 1)[0]
# rand_y = np.random.randint(-50, 50, 1)[0]
# M = np.float32([[1, 0, rand_x], [0, 1, rand_y]])
# print(M)
# images = cv2.warpAffine(images, M, (images.shape[1], images.shape[0]))
# cv2.imshow('hand000001-1', images)
# img_size = [224, 224]
# images = cv2.resize(images, (img_size[1], img_size[0]), interpolation=cv2.INTER_CUBIC)
# images = images / 255
# print(images.max())
# cv2.imshow('hand000001-2', images)
# cv2.imwrite('hand000001-1.jpg', images)
# cv2.waitKey(0)
#
# gray_img = cv2.imread('000001.jpg', cv2.IMREAD_GRAYSCALE)
# print(gray_img.shape)
# cv2.imshow('hand000001-1', gray_img)

# images_200x200 = cv2.resize(images, (200, 200))
# print(images_200x200.shape)
# images_100x100 = cv2.resize(images_200x200, (0, 0), fx=0.5, fy=0.5)
# print(images_100x100.shape)
# images_200x100 = cv2.copyMakeBorder(images_100x100, 50, 50, 0, 0, cv2.BORDER_CONSTANT,value=(0, 0, 0))
#
# img_hsv = cv2.cvtColor(images, cv2.COLOR_BGR2HSV)
# turn_green_hsv = img_hsv.copy()
# turn_green_hsv[:, :, 0] = (turn_green_hsv[:, :, 0]+15) % 180
# turn_green_img = cv2.cvtColor(turn_green_hsv, cv2.COLOR_HSV2BGR)
# cv2.imwrite('turn_green.jpg', turn_green_img)


# darker_hsv = img_hsv.copy()
# darker_hsv[:, :, 2] = 0.5 * darker_hsv[:, :, 2]
# darker_img = cv2.cvtColor(darker_hsv, cv2.COLOR_HSV2BGR)
# cv2.imwrite('darker.jpg', darker_img)

#
# # 定义一块宽600，高400的画布，初始化为白色
# canvas = np.zeros((400, 600, 3), dtype=np.uint8) + 255
#
# # 画一条纵向的正中央的黑色分界线
# cv2.line(canvas, (300, 0), (300, 399), (0, 0, 0), 2)
#
# # 画一条右半部份画面以150为界的横向分界线
# cv2.line(canvas, (300, 149), (599, 149), (0, 0, 0), 2)
#
# # 左半部分的右下角画个红色的圆
# cv2.circle(canvas, (200, 300), 75, (0, 0, 255), 5)
#
# # 左半部分的左下角画个蓝色的矩形
# cv2.rectangle(canvas, (20, 240), (100, 360), (255, 0, 0), thickness=3)
#
# # 定义两个三角形，并执行内部绿色填充
# triangles = np.array([
#     [(200, 240), (145, 333), (255, 333)],
#     [(60, 180), (20, 237), (100, 237)]])
# cv2.fillPoly(canvas, triangles, (0, 255, 0))
#
# # 画一个黄色五角星
# # 第一步通过旋转角度的办法求出五个顶点
# phi = 4 * np.pi / 5
# rotations = [[[np.cos(i * phi), -np.sin(i * phi)], [i * np.sin(phi), np.cos(i * phi)]] for i in range(1, 5)]
# pentagram = np.array([[[[0, -1]] + [np.dot(m, (0, -1)) for m in rotations]]], dtype=np.float)
#
# # 定义缩放倍数和平移向量把五角星画在左半部分画面的上方
# pentagram = np.round(pentagram * 80 + np.array([160, 120])).astype(np.int)
#
# # 将5个顶点作为多边形顶点连线，得到五角星
# cv2.polylines(canvas, pentagram, True, (0, 255, 255), 9)
#
# # 按像素为间隔从左至右在画面右半部份的上方画出HSV空间的色调连续变化
# for x in range(302, 600):
#     color_pixel = np.array([[[round(180*float(x-302)/298), 255, 255]]], dtype=np.uint8)
#     line_color = [int(c) for c in cv2.cvtColor(color_pixel, cv2.COLOR_HSV2BGR)[0][0]]
#     cv2.line(canvas, (x, 0), (x, 147), line_color)
#
# # 如果定义圆的线宽大于半斤，则等效于画圆点，随机在画面右下角的框内生成坐标
# np.random.seed(42)
# n_pts = 30
# pts_x = np.random.randint(310, 590, n_pts)
# pts_y = np.random.randint(160, 390, n_pts)
# pts = zip(pts_x, pts_y)
#
# # 画出每个点，颜色随机
# for pt in pts:
#     pt_color = [int(c) for c in np.random.randint(0, 255, 3)]
#     cv2.circle(canvas, pt, 3, pt_color, 5)
#
# # 在左半部分最上方打印文字
# cv2.putText(canvas,
#             '打印的文字just english',
#             (5, 15),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.5,
#             (0, 0, 0),
#             1)
#
# cv2.imshow('窗口名称', canvas)
# cv2.waitKey()

gray = cv2.imread('000001.jpg', 0)  # 读取灰度图
img = cv2.GaussianBlur(gray, (3, 3), 0)  # 高斯滤波器去噪
dst = cv2.Canny(img, 50, 50)  # Canny算法边缘检测
cv2.imshow('gray', gray)
cv2.imshow('gauss', img)
cv2.imshow('canny', dst)
cv2.waitKey()
