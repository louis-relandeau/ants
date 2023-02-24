import numpy as np
import math, random
import matplotlib.pyplot as plt

# pos = [0,0]
# a = np.arange(25).reshape(5,5)
# print(a)
# view_range = 1
# b = np.pad(a, view_range, mode='wrap')
# print(b)
# xmin = pos[0]
# xmax = pos[0] + 2*view_range +1
# ymin = pos[1]
# ymax = pos[1] + 2*view_range +1

# view = b[int(xmin):int(xmax), int(ymin):int(ymax)]
# print(view)

# print()
# view = np.array([[1,1,1],[1,1,2],[1,1,1]])
# print(view)
# vec = np.array([0.,0.])
# box = np.shape(view)[0]
# for i in range(box):
#     for j in range(box):
#         vec += view[i,j] * np.array([i-box//2,j-box//2])
# print(vec)
# ang = np.arctan2(vec[1],vec[1])
# print(ang*180/math.pi)


# probas = []
# for _ in range(1000):
#     a = 0.7
#     b = random.sample([-a,a], k=1)[0]
#     c = random.gauss(b,0.2)
#     probas.append(c)

# # plt.plot(probas)
# plt.hist(probas)
# plt.show()

a = np.array([45,120])
print(a)
b = a % 100
print(b)