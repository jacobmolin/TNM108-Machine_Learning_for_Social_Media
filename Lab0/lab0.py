import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from numpy import array as nparray

# def sum_numpy():
#     start = time.time()
#     X = np.arange(10000000)
#     Y = np.arange(10000000)
#     Z = X + Y
#     return time.time() - start

# print('time sum numpy:', sum_numpy(), 'array = ', nparray([1,2,3], float)) 

# thelist = [3,4,5,6]

# print('type: ', type(thelist), ', size: ', len(thelist))

# thelist.extend([7])
# thelist.append(8)

# print('list: ', thelist, 'type: ', type(thelist), ', size: ', len(thelist))

# arr = np.array([1,2,4], float)
# arr1 = arr
# arr2 = arr.copy()
# arr[0] = 0
# print(arr, arr1, arr2)

# print(np.random.permutation(3))

# matrix = np.identity(5, dtype=float)
# print(matrix)

# matrix = np.eye(5, k=2, dtype=float)
# print(matrix)

# matrix = np.ones((2,3), dtype=float)
# print(matrix)

# matrix = np.zeros(6, dtype=float)
# print(matrix)

# matrix = np.array([[13 ,32, 31], [64, 25, 76]], dtype=float)
# print(matrix)
# mat2 = np.zeros_like(matrix)
# print(mat2)

# arr1 = np.array([1,3,2])
# arr2 = np.array([3,4,6])
# arr3 = np.vstack([arr1,arr2])
# print(arr3)

# arr = np.array([2., 6., 5., 5.])
# # np.random.shuffle(arr)
# print(np.array_equal(arr, np.array([2., 6., 5., 5.])))

# matrix = np.array([[4., 5., 6.], [2, 3, 6]], float)
# print(matrix[0,2])

# arr1 = np.array([1,2])
# arr2 = np.array([3,4])

# print(arr1**arr2)

# obj = pd.Series([3,5,-2,1])
# print(obj, '\n')

# # print(obj.values)
# # print(obj.index)
# print(obj*2, '\n')

# data = pd.read_csv("ad.data", header=None, low_memory=False)

# data.columns

plt.plot([10,5,2,4], color='green', label='line 1', linewidth=5)
plt.ylabel('y', fontsize=40)
plt.xlabel('x', fontsize=40)
plt.axis([0,3, 0,15])
plt.show()