from PIL import Image, ImageDraw
import numpy as np

im = Image.new('1', (370, 370))

draw = ImageDraw.Draw(im)
draw.polygon([20, 20, 20, 140, 60, 60], fill='white')

A = np.array([[20, 20, 1], [20, 140, 1], [60, 60, 1]])
B = np.array([[230, 130, 1], [350, 190, 1], [290, 110, 1]])
T = np.linalg.solve(A, B).transpose()

T = np.linalg.inv(T)
im = im.transform(im.size, Image.AFFINE,
                  (T[0, 0], T[0, 1], T[0, 2], T[1, 0], T[1, 1], T[1, 2]))

# x1, y1, _ = np.matmul(T, np.array([20, 20, 1]))
# x2, y2, _ = np.matmul(T, np.array([20, 140, 1]))
# x3, y3, _ = np.matmul(T, np.array([60, 60, 1]))
#
# # draw.polygon([x1, y1, x2, y2, x3, y3], fill='white')

im.save('triangle2.png')