import numpy as np

a = [1, 2, 3]
b = [2, 3, 4]
dot_product = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

print(dot_product)

dot_product_zip = 0
for pair_a, pair_b in zip(a, b):
    dot_product_zip += pair_a * pair_b

print(dot_product_zip)


numpy_dot_product = np.dot(a, b)

print(numpy_dot_product)
